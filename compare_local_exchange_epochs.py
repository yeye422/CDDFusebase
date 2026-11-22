"""Paired full-training E2/exchange checkpoint diagnostics on the fixed pilot val64."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, summarize, write_csv
from run_e2_cross_scale import read_rows, seal_result, verify_result
from run_e2_local_correspondence_full import FULL
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import BASE_ANCHOR, ORIGINAL_LOSS_PILOT
from run_e2_v3_val936 import bind, evaluate_scene
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.dataset_os import digest_splits, manifest_pairs, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("test_result/e2_full_local_exchange_epoch_comparison_v1/val64")
RUNS = {"e2": BASE_ANCHOR, "candidate": FULL}
EPOCHS = (8, 12, 16)


def prepare():
    source = read_json(FULL.manifest)
    pilot = read_json(ORIGINAL_LOSS_PILOT.manifest)
    validate_manifest_splits(source)
    validate_manifest_splits(pilot)
    ids = pilot["splits"]["val"]
    if (len(ids) != 64 or len(source["splits"]["val"]) != 1000
            or pilot["source_manifest_sha256"] != sha256_file(FULL.manifest)
            or not set(ids).issubset(source["splits"]["val"])):
        raise ValueError("Archived pilot/full split mismatch")
    splits = dict(train=[], val=ids, test=[])
    manifest = dict(dataset="QXS-SAROPT", optical_dir=source["optical_dir"], sar_dir=source["sar_dir"],
                    splits=splits, splits_sha256=digest_splits(splits), parent_sha256=sha256_file(FULL.manifest),
                    selected_from=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
                    selection_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest))
    archive = read_json(FULL.result / "run_metadata.json")
    if (archive["scene_ids"] != source["splits"]["val"] or archive["split_manifest_sha256"] != manifest["parent_sha256"]
            or archive["metric_profile"] != DEFAULT_PROFILE or archive["output_policy"] != "clamp_then_uint8"
            or archive["split"] != "val" or archive["offset"] != 0):
        raise ValueError("Epoch 16 archive protocol differs")
    final = {}
    hashes = {}
    for key, exp in RUNS.items():
        if archive["checkpoint_hashes"][key] != sha256_file(exp.final) or Path(archive["checkpoints"][key]) != exp.final:
            raise ValueError(f"Archived final checkpoint differs: {key}")
        path = FULL.result / key / "metrics.csv"
        hashes[key] = sha256_file(path)
        data = dict(zip(archive["scene_ids"], read_rows(path, METRICS, archive["scene_ids"])))
        final[key] = [data[i] for i in ids]
    return manifest, final, hashes


def compare(folder, ids, records):
    matrices = {key: np.asarray(rows, dtype=float) for key, rows in records.items()}
    results = {k: [(row, (), ()) for row in rows] for k, rows in matrices.items()}
    for key in RUNS:
        (folder / key).mkdir(exist_ok=True)
    means = summarize(folder, ids, results, [("e2", "candidate")], diagnostic_fields=(), source_metrics=())
    d = matrices["candidate"] - matrices["e2"]
    metrics = ("Qabf", "Nabf", "VIF")
    core = d[:, [METRICS.index(name) for name in metrics]]
    finite = np.isfinite(core).all(axis=1)
    winners = finite & (core[:, 0] > 0) & (core[:, 1] < 0) & (core[:, 2] > 0)
    counts = []
    for j, key in enumerate(METRICS):
        finite_col = np.isfinite(d[:, j])
        signed = -d[finite_col, j] if key == "Nabf" else d[finite_col, j]
        counts.append((key, len(signed), int((signed > 0).sum()), int((signed == 0).sum()), int((signed < 0).sum())))
    write_csv(folder / "paired_counts.csv", ("metric","common_count","better","equal","worse"), counts)
    details = dict(common_count=int(finite.sum()), joint_count=int(winners.sum()),
                   joint_ids=[scene for scene, win in zip(ids, winners) if win])
    write_json(folder / "joint.json", details)
    write_json(folder / "registry_metrics.json", means)


def process(epoch, action, manifest, final, hashes, pairs, device):
    folder = OUTPUT / f"epoch_{epoch:03d}"
    work = folder.with_suffix(".work")
    ids = manifest["splits"]["val"]
    paths = {k: run.run_dir / f"epoch_{epoch:03d}.pth" for k, run in RUNS.items()}
    meta = dict(scene_ids=ids, epoch=epoch, checkpoint_hashes={k: sha256_file(p) for k,p in paths.items()},
                checkpoints={k:p.as_posix() for k,p in paths.items()}, metric_profile=DEFAULT_PROFILE,
                output_policy="clamp_then_uint8", manifest_sha256=sha256_file(OUTPUT / "split_manifest.json"),
                sources={f:sha256_file(ROOT/f) for f in ("compare_local_exchange_epochs.py", "run_e2_v3_val936.py",
                    "run_e2_source_objective.py", "run_e2_aspr_pilot.py", "e2_local_correspondence.py", "e2_base_anchor.py",
                    "direct_fusion.py", "lwpd.py", "net.py", "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py")})
    if epoch == 16:
        meta.update(archived=FULL.result.as_posix(), archived_tables_sha256=hashes)
    else:
        meta.update(device=str(device), torch=str(torch.__version__))
    if folder.exists():
        verify_result(folder, meta, image_keys=() if epoch == 16 else RUNS)
    else:
        work.mkdir(exist_ok=True)
        bind(work, "run_metadata.json", meta)
        if epoch == 16:
            records = final
        else:
            models = {}
            for key, exp in RUNS.items():
                state = torch.load(paths[key], map_location="cpu", weights_only=False)
                if (state["epoch"], state["stage"], state["variant"]) != (epoch-1, "fusion", exp.variant):
                    raise ValueError(f"Unexpected checkpoint: {paths[key]}")
                model = exp.build_model()
                model.load_state_dict(state["model"], strict=True)
                models[key] = model.to(device).eval()
            records = {key: [] for key in RUNS}
            for i, pair in enumerate(pairs[:1] if action == "smoke" else pairs):
                data = evaluate_scene(work, pair, models, device)
                for key in RUNS:
                    records[key].append(data[key][:len(METRICS)])
                print(f"epoch {epoch}: {i+1}/{len(ids)} {pair[0].stem}", flush=True)
            if action == "smoke":
                return []
        compare(work, ids, records)
        seal_result(work)
        work.rename(folder)
    group = f"QXS-FULL-LOCAL-EXCHANGE-EPOCH{epoch:03d}-VAL64"
    entries = []
    for key, scores in read_json(folder / "registry_metrics.json").items():
        entries.append(dict(id=f"{group}-{key}", comparable_group=group, task="fusion", status="development_validation",
                            dataset="QXS-SAROPT", method=f"{RUNS[key].method}, epoch {epoch}", split="val", offset=0,
                            sample_count=len(ids), split_manifest=(OUTPUT/"split_manifest.json").as_posix(),
                            split_manifest_sha256=meta["manifest_sha256"], checkpoint=paths[key].as_posix(),
                            checkpoint_sha256=meta["checkpoint_hashes"][key], metric_profile=DEFAULT_PROFILE,
                            output_policy=meta["output_policy"], result_dir=(folder/key).as_posix(),
                            metadata_path=(folder/"run_metadata.json").as_posix(), metrics=scores,
                            metric_source=dict(type="csv_row", path=(folder/key/"metrics.csv").as_posix(),
                                               match={"image":"mean"}, columns={k:k for k in scores}),
                            notes="Full-train checkpoints on locked pilot val64; differences within the same epoch."))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "val"), default="smoke", nargs="?")
    action = parser.parse_args().action
    os.chdir(ROOT)
    manifest, final, hashes = prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT/".lock"), timeout=0):
        bind(OUTPUT, "split_manifest.json", manifest)
        pairs = manifest_pairs(DATA, OUTPUT/"split_manifest.json")[0]["val"]
        if [p[0].stem for p in pairs] != manifest["splits"]["val"]:
            raise ValueError("Validation order mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        entries = []
        for epoch in (16, 8, 12):
            entries.extend(process(epoch, action, manifest, final, hashes, pairs, device))
        if entries:
            register_many(entries)
        for epoch in EPOCHS:
            path = OUTPUT / f"epoch_{epoch:03d}/joint.json"
            if path.exists():
                joint = read_json(path)
                print(f"epoch {epoch}: {joint['joint_count']}/{joint['common_count']} jointly better")


if __name__ == "__main__":
    main()
