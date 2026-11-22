"""Frozen pilot E2/V3 comparison on the ordered complement of their original val64."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from diagnose_e2_information_path import read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize, write_csv
from run_e2_cross_scale import seal_result, verify_result
from run_e2_local_correction_v3 import PILOT
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT
from run_e2_source_objective import image_scores, save_images
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, verify_images, write_json
from utils.dataset_os import digest_splits, manifest_pairs, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parent
RUNS = {"e2": ORIGINAL_LOSS_PILOT, "candidate": PILOT}
PRIOR = "QXS-E2-LOCAL-CORRECTION-V3-VAL64-REFERENCE-FMI"
GROUP = "QXS-E2-LOCAL-CORRECTION-V3-VAL936-REFERENCE-FMI"
OUTPUT = Path("test_result/e2_local_correction_pilot_v3/val936/seed42")


def remainder(full, pilot):
    validate_manifest_splits(full)
    validate_manifest_splits(pilot)
    original, excluded = full["splits"]["val"], set(pilot["splits"]["val"])
    if (len(original) != 1000 or len(excluded) != 64 or not excluded.issubset(original)
            or len(pilot["splits"]["train"]) != 1024
            or not set(pilot["splits"]["train"]).issubset(full["splits"]["train"])):
        raise ValueError("Expected the archived train1024/val64 subset of the parent split")
    return [scene for scene in original if scene not in excluded]


def prepare():
    entries = {r["id"]: r for r in read_json(ROOT / "EXPERIMENT_REGISTRY.json")["experiments"]}
    pilot = read_json(PILOT.manifest)
    full = read_json(MANIFEST)
    ids = remainder(full, pilot)
    parent_hash = sha256_file(MANIFEST)
    if parent_hash != pilot["source_manifest_sha256"] or pilot != read_json(ORIGINAL_LOSS_PILOT.manifest):
        raise ValueError("The two pilots or their parent manifests differ")
    checkpoints = {}
    for name, exp in RUNS.items():
        entry = entries[f"{PRIOR}-{name}"]
        digest = sha256_file(exp.final)
        if (Path(entry["checkpoint"]) != exp.final or entry["checkpoint_sha256"] != digest
                or entry["split_manifest_sha256"] != sha256_file(exp.manifest)
                or entry["metric_profile"] != DEFAULT_PROFILE or entry["output_policy"] != "clamp_then_uint8"):
            raise ValueError(f"Prior registered validation provenance mismatch: {name}")
        checkpoints[name] = dict(path=exp.final.as_posix(), sha256=digest)
    splits = dict(train=[], val=ids, test=[])
    manifest = dict(dataset="QXS-SAROPT", optical_dir=full["optical_dir"], sar_dir=full["sar_dir"],
                    splits=splits, splits_sha256=digest_splits(splits), source_manifest=MANIFEST.as_posix(),
                    source_manifest_sha256=parent_hash, excluded_val64=pilot["splits"]["val"],
                    pilot_manifest_sha256=sha256_file(PILOT.manifest))
    sources = ("run_e2_v3_val936.py", "run_e2_aspr_pilot.py", "run_e2_source_objective.py",
               "diagnose_e2_information_path.py", "direct_fusion.py", "net.py", "lwpd.py", "e2_base_anchor.py",
               "e2_local_correction.py", "e2_local_correction_v2.py", "e2_local_correction_v3.py",
               "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = dict(scene_ids=ids, dataset="QXS-SAROPT", split="val", sample_count=len(ids),
                checkpoints=checkpoints, pilot_group=PRIOR, metric_profile=DEFAULT_PROFILE,
                output_policy="clamp_then_uint8", device=str(device), torch=str(torch.__version__),
                source_sha256={s: sha256_file(ROOT / s) for s in sources})
    return manifest, meta, device


def load_models(device):
    models = {}
    for name, exp in RUNS.items():
        state = torch.load(exp.final, map_location="cpu", weights_only=False)
        if (state["variant"], state["epoch"], state["stage"]) != (exp.variant, 15, "fusion"):
            raise ValueError(f"Expected the original final fusion checkpoint for {name}")
        model = exp.build_model()
        model.load_state_dict(state["model"], strict=True)
        models[name] = model.to(device).eval()
    return models


@torch.inference_mode()
def evaluate_scene(folder, pair, models, device):
    scene = pair[0].stem
    path = folder / "scenes" / f"{scene}.json"
    inputs = [sha256_file(p) for p in pair]
    if path.exists():
        record = read_json(path)
        if record["inputs"] != inputs or record["scene"] != scene:
            raise ValueError(f"Saved scene/input mismatch: {scene}")
        if any(sha256_file(folder / f) != h for f, h in record["images"].items()):
            raise ValueError(f"Saved scene images changed: {scene}")
        return record["scores"]
    optical, sar = read_pair(pair)
    o, s = (to_tensor(image, device) for image in (optical, sar))
    values, artifacts = {}, {}
    for key, model in models.items():
        fused = model(o, s)
        if not torch.isfinite(fused).all():
            raise FloatingPointError(f"Non-finite fused image: {scene}/{key}")
        gray, scores, sources = image_scores(fused, optical, sar)
        save_images(folder / key, pair, gray, optical, sar)
        values[key] = [float(x) if np.isfinite(x) else None for x in (*scores, *sources)]
        for category in ("gray", "rgb", "preview"):
            f = Path(key) / category / f"{scene}.png"
            artifacts[f.as_posix()] = sha256_file(folder / f)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".pending")
    write_json(tmp, dict(scene=scene, inputs=inputs, scores=values, images=artifacts))
    tmp.replace(path)
    return values


def reports(folder, ids, records):
    fields = (*METRICS, *SOURCE_METRICS)
    arrays = {k: np.asarray([r[k] for r in records], dtype=float) for k in RUNS}
    results = {k: [(v[:len(METRICS)], v[len(METRICS):], ()) for v in a] for k, a in arrays.items()}
    summaries = summarize(folder, ids, results, [("e2", "candidate")], diagnostic_fields=())
    counts = []
    for j, name in enumerate(fields):
        keep = np.isfinite(arrays["candidate"][:, j]) & np.isfinite(arrays["e2"][:, j])
        diff = arrays["candidate"][keep, j] - arrays["e2"][keep, j]
        benefit = -diff if name == "Nabf" else diff
        counts.append((name, len(diff), int((benefit > 0).sum()), int((benefit == 0).sum()), int((benefit < 0).sum())))
    write_csv(folder / "paired_counts.csv", ("metric", "common_count", "better", "equal", "worse"), counts)
    return summaries


def register(meta):
    scores = read_json(OUTPUT / "registry_metrics.json")
    entries = []
    for name, exp in RUNS.items():
        folder = OUTPUT / name
        entries.append(dict(id=f"{GROUP}-{name}", comparable_group=GROUP, task="fusion",
                            status="development_validation", method=exp.method, dataset=meta["dataset"],
                            split="val", offset=0, sample_count=meta["sample_count"],
                            checkpoint=exp.final.as_posix(), checkpoint_sha256=meta["checkpoints"][name]["sha256"],
                            split_manifest=(OUTPUT / "split_manifest.json").as_posix(),
                            split_manifest_sha256=sha256_file(OUTPUT / "split_manifest.json"),
                            result_dir=folder.as_posix(), metadata_path=(OUTPUT / "run_metadata.json").as_posix(),
                            metric_profile=DEFAULT_PROFILE, output_policy=meta["output_policy"], metrics=scores[name],
                            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                                               match={"image":"mean"}, columns={k:k for k in scores[name]}),
                            notes="两项 train1024 小试原权重；固定 val1000 排除原 val64，保留父清单顺序。"))
    register_many(entries)


def bind(folder, filename, data):
    path = folder / filename
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Existing run has different provenance: {path}")
    else:
        tmp = path.with_suffix(".pending")
        write_json(tmp, data)
        tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "val"), nargs="?", default="check")
    action = parser.parse_args().action
    os.chdir(ROOT)
    manifest, meta, device = prepare()
    ids = meta["scene_ids"]
    print(f"val1000 - original val64 = {len(ids)} paired scenes; epoch 16; {device}; first={ids[0]}", flush=True)
    if action == "check":
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT.with_suffix(".lock")), timeout=0):
        if OUTPUT.exists():
            verify_result(OUTPUT, meta, image_keys=RUNS)
            if read_json(OUTPUT / "split_manifest.json") != manifest:
                raise ValueError("Completed manifest differs")
            if action == "val":
                register(meta)
            return
        folder = OUTPUT.with_suffix(".work")
        folder.mkdir(exist_ok=True)
        bind(folder, "split_manifest.json", manifest)
        bind(folder, "run_metadata.json", meta)
        pairs = manifest_pairs(DATA, folder / "split_manifest.json")[0]["val"]
        if [p.stem for p, _ in pairs] != ids:
            raise ValueError("Resolved scene order differs from the frozen split")
        models = load_models(device)
        if action == "smoke":
            record = evaluate_scene(folder, pairs[0], models, device)
            # Exercise the same CSV/export path in an isolated smoke subfolder.
            smoke = folder.parent / "smoke_first_scene"
            for key in RUNS:
                (smoke / key).mkdir(exist_ok=True, parents=True)
            reports(smoke, ids[:1], [record])
            print(f"Single scene {ids[0]} verified and cached; no full evaluation started.")
            return
        records = []
        for n, pair in enumerate(pairs, 1):
            records.append(evaluate_scene(folder, pair, models, device))
            print(f"paired {n}/{len(ids)}: {ids[n-1]}", flush=True)
        for key in RUNS:
            for category in ("gray", "rgb", "preview"):
                verify_images(folder / key / category, ids)
        write_json(folder / "registry_metrics.json", reports(folder, ids, records))
        seal_result(folder)
        folder.rename(OUTPUT)
        register(meta)
        print(f"Results: {OUTPUT / 'paired_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
