"""Separate the frozen V3 coarse readout from its correction on the original val64."""
import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from filelock import FileLock
from PIL import Image

from diagnose_e2_information_path import read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize, write_csv
from run_e2_cross_scale import read_rows, seal_result, verify_result
from run_e2_local_correction_v3 import GROUP as PRIOR_GROUP, PILOT
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT as BASE
from run_e2_v3_val936 import bind, evaluate_scene
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.dataset_os import digest_splits, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("test_result/e2_local_correction_v3_readouts_v1")
CACHE = OUTPUT / "cache"
PAIRS = (("e2", "coarse"), ("coarse", "final"), ("e2", "final"))
FIELDS = (*METRICS, *SOURCE_METRICS)


class CoarseReadout(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, optical, sar):
        features = self.model.encode_pair(optical, sar)
        base, detail = self.model.fuse_features(*features)
        return self.model.decoder(None, base, detail)[0]


def prepare():
    arch = PILOT.result
    checks = read_json(arch / "result_checksums.json")
    def checked(rel):
        path = arch / rel
        if sha256_file(path) != checks[rel]:
            raise ValueError(f"Changed archived result: {path}")
        return path

    original = read_json(checked("run_metadata.json"))
    pairs, manifest = manifest_pairs(DATA, PILOT.manifest)
    ids = manifest["splits"]["val"]
    if (len(ids) != 64 or original["scene_ids"] != ids or [a.stem for a, _ in pairs["val"]] != ids
            or original["split_manifest_sha256"] != sha256_file(PILOT.manifest)
            or original["metric_profile"] != DEFAULT_PROFILE or original["output_policy"] != "clamp_then_uint8"):
        raise ValueError("Original validation protocol does not match")
    registry = {r["id"]: r for r in read_json(ROOT / "EXPERIMENT_REGISTRY.json")["experiments"]}
    values = {}
    for key, old, exp in (("e2", "e2", BASE), ("final", "candidate", PILOT)):
        entry = registry[f"{PRIOR_GROUP}-{old}"]
        if (entry["checkpoint_sha256"] != sha256_file(exp.final)
                or original["checkpoint_hashes"][old] != entry["checkpoint_sha256"]
                or Path(entry["checkpoint"]) != exp.final or Path(original["checkpoints"][old]) != exp.final
                or entry["split_manifest_sha256"] != original["split_manifest_sha256"]):
            raise ValueError(f"Checkpoint/manifest mismatch: {key}")
        metrics = read_rows(checked(f"{old}/metrics.csv"), METRICS, ids)
        sources = read_rows(checked(f"{old}/diagnostics.csv"), SOURCE_METRICS, ids)
        values[key] = dict(zip(ids, [m+s for m, s in zip(metrics, sources)]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filenames = ("run_e2_v3_readouts.py", "run_e2_v3_val936.py", "run_e2_aspr_pilot.py",
                 "run_e2_source_objective.py", "e2_local_correction.py", "e2_local_correction_v2.py",
                 "e2_local_correction_v3.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py",
                 "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py")
    meta = dict(checkpoint_hashes=original["checkpoint_hashes"], archive=arch.as_posix(),
                archive_checksums_sha256=sha256_file(arch / "result_checksums.json"),
                parent_manifest_sha256=original["split_manifest_sha256"], device=str(device), torch=str(torch.__version__),
                metric_profile=DEFAULT_PROFILE, output_policy=original["output_policy"],
                sources={f: sha256_file(ROOT / f) for f in filenames})
    return meta, original, dict(zip(ids, pairs["val"])), values, manifest, device


def report(folder, ids, rows):
    for key in rows:
        (folder / key).mkdir(parents=True)
    arrays = {key: np.array([values[i] for i in ids], dtype=float) for key, values in rows.items()}
    results = {key: [(v[:len(METRICS)], v[len(METRICS):], ()) for v in matrix] for key, matrix in arrays.items()}
    means = summarize(folder, ids, results, PAIRS, diagnostic_fields=())
    finite = np.isfinite(np.stack(list(arrays.values()))).all(axis=0)
    counts = []
    for a, b in PAIRS:
        for j, key in enumerate(FIELDS):
            delta = (arrays[b][:, j] - arrays[a][:, j])[finite[:, j]]
            benefit = -delta if key == "Nabf" else delta
            counts.append((a, b, key, len(delta), int((benefit > 0).sum()),
                           int((benefit == 0).sum()), int((benefit < 0).sum())))
    write_csv(folder / "paired_counts.csv", ("control", "candidate", "metric", "common_count", "better", "equal", "worse"), counts)
    write_json(folder / "registry_metrics.json", means)


def register(folder, ids, meta):
    single = len(ids) == 1
    group = "QXS-E2-V3-READOUTS-V1-" + ("SCENE4880" if single else "VAL64")
    means = read_json(folder / "registry_metrics.json")
    entries = []
    for key, exp, description in (("e2", BASE, "Archived pilot E2"), ("coarse", PILOT, "V3 before local correction"),
                                  ("final", PILOT, "V3 after local correction")):
        path = folder / key
        scores = means[key]
        entries.append(dict(id=f"{group}-{key}", comparable_group=group, task="fusion",
                            status="diagnostic_single_scene" if single else "development_validation",
                            method=description, dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(ids),
                            checkpoint=exp.final.as_posix(), checkpoint_sha256=meta["checkpoint_hashes"]["e2" if key=="e2" else "candidate"],
                            split_manifest=(folder/"split_manifest.json").as_posix(),
                            split_manifest_sha256=sha256_file(folder/"split_manifest.json"),
                            metric_profile=DEFAULT_PROFILE, output_policy=meta["output_policy"],
                            result_dir=path.as_posix(), metadata_path=(folder/"run_metadata.json").as_posix(), metrics=scores,
                            metric_source=dict(type="csv_row", path=(path/"metrics.csv").as_posix(), match={"image":"mean"},
                                               columns={k:k for k in scores}),
                            notes="Three-way common finite sample set; coarse and final share the same V3 checkpoint."))
    register_many(entries)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "val"), default="check", nargs="?")
    action = parser.parse_args().action
    os.chdir(ROOT)
    meta, original, pairs, rows, manifest, device = prepare()
    print(f"Matched E2 and V3 epoch 16, {len(pairs)} archived pairs; {device}", flush=True)
    if action == "check":
        return
    ids = ["4880"] if action == "smoke" else original["scene_ids"]
    output = OUTPUT / ("scene4880" if action == "smoke" else "val64")
    split = dict(manifest, splits=dict(train=[], val=ids, test=[]))
    split["splits_sha256"] = digest_splits(split["splits"])
    metadata = dict(meta, scene_ids=ids, split="val", coarse_images=(CACHE / "coarse").as_posix(),
                    archived_final_images=(PILOT.result / "candidate/gray").as_posix(),
                    archived_e2_images=(Path(original["baseline_archive"]) / "e2/gray").as_posix(),
                    inputs={i: original["inputs"][i] for i in ids})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / ".lock"), timeout=0):
        CACHE.mkdir(exist_ok=True)
        bind(CACHE, "protocol.json", meta)
        if output.exists():
            verify_result(output, metadata)
            if read_json(output / "split_manifest.json") != split:
                raise ValueError("Completed split differs")
            register(output, ids, meta)
            return
        state = torch.load(PILOT.final, map_location="cpu", weights_only=False)
        if (state["variant"], state["epoch"], state["stage"]) != (PILOT.variant, 15, "fusion"):
            raise ValueError("Unexpected V3 checkpoint")
        core = PILOT.build_model().to(device).eval()
        core.load_state_dict(state["model"], strict=True)
        model = CoarseReadout(core).eval()
        rows["coarse"] = {}
        for n, scene in enumerate(ids, 1):
            pair = pairs[scene]
            if [sha256_file(p) for p in pair] != original["inputs"][scene]:
                raise ValueError(f"Source images changed: {scene}")
            rows["coarse"][scene] = evaluate_scene(CACHE, pair, {"coarse": model}, device)["coarse"]
            # Verify that the exported coarse tensor is the real pre-correction readout.
            if scene == "4880":
                o, s = (to_tensor(x, device) for x in read_pair(pair))
                final, features = core(o, s, return_features=True)
                for image, path in ((final, PILOT.result / "candidate/gray/4880.png"),
                                    (features["coarse_image"], CACHE / "coarse/gray/4880.png")):
                    actual = image[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
                    with Image.open(path) as archive:
                        if not np.array_equal(actual, np.asarray(archive)):
                            raise ValueError(f"Normal model forward and saved readout differ: {path}")
            print(f"{n}/{len(ids)} {scene}", flush=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as tmp:
            folder = Path(tmp) / "result"
            report(folder, ids, rows)
            write_json(folder / "split_manifest.json", split)
            write_json(folder / "run_metadata.json", metadata)
            seal_result(folder)
            folder.rename(output)
        register(output, ids, meta)
        print(f"Paired results: {output / 'paired_comparison.csv'}")


if __name__ == "__main__":
    main()
