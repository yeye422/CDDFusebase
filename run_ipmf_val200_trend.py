"""Compare frozen IPMF epochs 1, 2 and 4 on the first 200 locked val scenes."""

import argparse
import csv
import math
import os
from pathlib import Path
from uuid import uuid4

import torch

import run_ipmf_frozen as ipmf
from run_e2_aspr_pilot import METRICS, summarize, write_csv
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import read_json, write_json
from train_direct_fusion import sha256_file


OUTPUT = ipmf.OUTPUT.with_name("val200_trend") / "first200_epochs_001_002_004"
SAMPLE_COUNT = 200
EPOCHS = (1, 2, 4)
ARMS = {epoch: f"IPMF-E{epoch:02d}" for epoch in EPOCHS}
METHODS = {"CF1": "CF1 第 4 轮", **{arm: f"IPMF-V1 第 {epoch} 轮" for epoch, arm in ARMS.items()}}
CONTRASTS = (*(("CF1", arm) for arm in ARMS.values()), (ARMS[1], ARMS[2]), (ARMS[2], ARMS[4]))
DIAGNOSTICS = (
    "epoch", "train_loss", "train_intensity", "train_gradient", "train_base_correction_rms_ratio",
    "train_detail_correction_rms_ratio", "train_cfdr_gain_o", "train_cfdr_gain_s", "train_cfdr_gain_null",
    "train_cfdr_out_of_range", "effective_updates", "skipped_updates",
)


def subset_archive(epoch, ids, training, config_hash):
    folder = ipmf.validation_dir(epoch)
    metadata = folder / "run_metadata.json"
    if not metadata.is_file():
        return None
    saved = read_json(metadata)
    expected = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=epoch,
                    metric_profile=training["metric_profile"], output_policy=training["output_policy"],
                    split_manifest_sha256=training["split_manifest_sha256"], training_config_sha256=config_hash)
    if (any(saved.get(name) != value for name, value in expected.items())
            or saved["checkpoint_hashes"][ipmf.CANDIDATE] != sha256_file(ipmf.checkpoint(epoch))
            or any(saved["source_sha256"].get(name) != digest
                   for name, digest in training["evaluator_sha256"].items())):
        raise ValueError(f"Existing IPMF epoch-{epoch} archive has a different protocol: {metadata}")
    raw = folder / ipmf.CANDIDATE / "evaluator_metrics.csv"
    return read_metric_rows(raw, ids)[:SAMPLE_COUNT], dict(
        metrics=raw.as_posix(), metrics_sha256=sha256_file(raw),
        metadata=metadata.as_posix(), metadata_sha256=sha256_file(metadata),
    )


def infer_or_reuse(path, folder, protocol, arm, device):
    ids = protocol["scene_ids"]
    if not folder.exists():
        attempt = folder.parent / f".{arm}-{uuid4().hex}"
        rows = ipmf.infer(path, attempt, ids, protocol["split_manifest_sha256"], device, protocol["metric_profile"])
        attempt.rename(folder)
        return rows
    metadata = read_json(folder / "run_metadata.json")
    expected = dict(split="val", offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids),
                    variant=ipmf.IPMFE2.VARIANT, checkpoint_sha256=protocol["checkpoint_hashes"][arm],
                    split_manifest_sha256=protocol["split_manifest_sha256"], metric_profile=protocol["metric_profile"],
                    output_range_policy="clamp")
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Unexpected completed subset inference: {folder}")
    return read_metric_rows(folder / "evaluator_metrics.csv", ids)


def write_overview(folder, scene_ids, results, summaries, history):
    write_csv(folder / "training_diagnostics.csv", DIAGNOSTICS, ([row[name] for name in DIAGNOSTICS] for row in history))
    with (folder / "paired_comparison.csv").open(newline="", encoding="utf-8-sig") as file:
        counts = {row["metric"]: int(row["common_count"]) for row in csv.DictReader(file) if row["control"] == "CF1"}
    write_csv(folder / "trend.csv", ("metric", "common_count", *METHODS), (
        (metric, counts[metric], *(summaries[arm].get(metric) for arm in METHODS)) for metric in METRICS
    ))
    # Keep the existing common exclusions and identify CF1-valid scenes that
    # each round loses so a change in coverage is visible next to the means.
    lost = {
        arm: {metric: [scene for scene, base, candidate in zip(scene_ids, results["CF1"], results[arm])
                       if math.isfinite(base[0][index]) and not math.isfinite(candidate[0][index])]
              for index, metric in enumerate(METRICS)}
        for arm in ARMS.values()
    }
    write_json(folder / "coverage.json", {
        "common_counts": counts, "lost_vs_cf1_scene_ids": lost,
    })


def evaluate(device):
    parent, _, training = ipmf.training_inputs()
    config = ipmf.ROOT / "config.json"
    if read_json(config) != training:
        raise ValueError(f"Unexpected IPMF training protocol: {config}")
    config_hash = sha256_file(config)
    for epoch in EPOCHS:
        ipmf.read_checkpoint(epoch, config_hash, parent["model"])
    all_ids = read_json(ipmf.MANIFEST)["splits"]["val"]
    if len(all_ids) != 1000 or len(set(all_ids)) != 1000:
        raise ValueError("Expected the locked val1000 split")
    ids = all_ids[:SAMPLE_COUNT]
    digest = training["split_manifest_sha256"]
    paths = {"CF1": ipmf.PARENT, **{arm: ipmf.checkpoint(epoch) for epoch, arm in ARMS.items()}}
    cf1, provenance = ipmf.published_rows(
        ipmf.CF1_E04_ARCHIVE, "CF1", "evaluator_metrics.csv", all_ids, digest, 4, paths["CF1"],
    )
    results, archives = {"CF1": cf1[:SAMPLE_COUNT]}, {"CF1": provenance}
    for epoch, arm in ARMS.items():
        archived = subset_archive(epoch, all_ids, training, config_hash)
        if archived is not None:
            results[arm], archives[arm] = archived
    history_path = ipmf.ROOT / "history.csv"
    with history_path.open(newline="", encoding="utf-8-sig") as file:
        history = [row for row in csv.DictReader(file) if 1 <= int(row["epoch"]) <= 4]
    if [int(row["epoch"]) for row in history] != list(range(1, 5)):
        raise ValueError(f"Expected four training log rows: {history_path}")
    protocol = dict(
        split="val", offset=0, scene_ids=ids, selection="first_200_in_locked_val_manifest_order", device=str(device),
        metric_profile=training["metric_profile"], output_policy=training["output_policy"],
        split_manifest=ipmf.MANIFEST.as_posix(), split_manifest_sha256=digest,
        checkpoints={arm: path.as_posix() for arm, path in paths.items()},
        checkpoint_hashes={arm: sha256_file(path) for arm, path in paths.items()},
        training_config=config.as_posix(), training_config_sha256=config_hash,
        training_history=history_path.as_posix(), training_history_sha256=sha256_file(history_path), cached=archives,
        source_sha256={**training["evaluator_sha256"], "run_ipmf_frozen.py": sha256_file(Path(ipmf.__file__)),
                       "run_e2_aspr_pilot.py": sha256_file(Path("run_e2_aspr_pilot.py")),
                       Path(__file__).name: sha256_file(Path(__file__))},
    )
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous IPMF val200 trend: {OUTPUT}")
    else:
        work = OUTPUT.with_name(f".{OUTPUT.name}-pending")
        manifest = work / "run_metadata.json"
        work.mkdir(parents=True, exist_ok=True)
        if manifest.is_file():
            if read_json(manifest) != protocol:
                raise FileExistsError(f"Preserve previous IPMF trend staging: {work}")
        else:
            write_json(manifest, protocol)
        for arm, path in paths.items():
            folder = work / arm
            if arm in archives:
                folder.mkdir(exist_ok=True)
            else:
                results[arm] = infer_or_reuse(path, folder, protocol, arm, device)
        summaries = summarize(work, ids, results, CONTRASTS, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_overview(work, ids, results, summaries, history)
        work.rename(OUTPUT)
    group = f"{ipmf.GROUP}-VAL200-FIRST200-TREND-E01-E02-E04-REF-FMI"
    register_many(ipmf.validation_entries(
        OUTPUT, group, protocol, METHODS, "固定 val200 趋势诊断。每项指标使用四组共同有限场景。",
    ))
    print(f"Validation trend: {OUTPUT / 'trend.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for IPMF val200 trend inference")
    evaluate(torch.device(args.device))


if __name__ == "__main__":
    main()
