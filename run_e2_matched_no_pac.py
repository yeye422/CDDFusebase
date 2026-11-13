"""Matched E2 continuation with the PAC projection fixed at zero."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import run_e2_pac_mamba as protocol
import train_pac_mamba as trainer


MODEL = Path("models/e2_matched_no_pac_v1")
RESULT = Path("test_result/e2_matched_no_pac_v1/val1000/seed42")
FINAL = MODEL / "seed42/epoch_012.pth"
ORIGINAL = protocol.BASELINE
CANDIDATE = protocol.RESULT
REGISTRY = Path("EXPERIMENT_REGISTRY.json")


def register(entry):
    register_many((entry,))


def register_many(entries):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in registry["experiments"]}
    changed = False
    for entry in entries:
        existing = by_id.get(entry["id"])
        if existing is None:
            registry["experiments"].append(entry)
            by_id[entry["id"]] = entry
            changed = True
        elif existing != entry:
            raise ValueError(f"Registration differs from the saved result: {entry['id']}")
    if changed:
        registry["updated_at"] = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        temporary = REGISTRY.with_suffix(".json.tmp")
        protocol.write_json(temporary, registry)
        temporary.replace(REGISTRY)
    protocol.run([sys.executable, "maintain_experiment_index.py", "--check", "--render"])


def train():
    if FINAL.exists():
        print(f"Completed checkpoint: {FINAL}")
        register_training()
        return
    args = protocol.train_args()
    args[args.index("--output-dir") + 1] = str(MODEL)
    configure = trainer.configure_stage

    def configure_without_pac(model, optimizer, stage, config):
        configure(model, optimizer, stage, config)
        projection = model.detail_fusion.update_projection[-1]
        if any(torch.count_nonzero(p).item() for p in projection.parameters()):
            raise ValueError("Expected an exactly zero PAC output projection")
        for param in trainer.pac_parameters(model):
            param.requires_grad_(False)

    # Keep the original model construction, RNG sequence and training loop.
    # The trainer does not persist sampler or augmentation RNG state for resume.
    with patch.object(trainer, "configure_stage", configure_without_pac), patch.object(sys, "argv", args):
        trainer.main()
    if FINAL.exists():
        register_training()


def register_training():
    history = FINAL.parent / "history.csv"
    config = FINAL.parent / "config.json"
    with history.open(encoding="utf-8-sig", newline="") as f:
        row = next(row for row in csv.DictReader(f) if row["epoch"] == "12")
    metrics = {key: float(row[key]) for key in ("train_loss", "val_loss")}
    register(dict(
        id="E2-MATCHED-NO-PAC-V1-SEED42-TRAIN8000", status="training_complete", task="training",
        method="E2 matched continuation; PAC frozen with exactly zero output",
        comparable_group="QXS-E2-MATCHED-NO-PAC-V1-TRAIN8000", dataset="QXS-SAROPT",
        split="train", offset=0, sample_count=8000, split_manifest=protocol.MANIFEST.as_posix(),
        split_manifest_sha256=protocol.sha256_file(protocol.MANIFEST), checkpoint=FINAL.as_posix(),
        checkpoint_sha256=protocol.sha256_file(FINAL), parent_checkpoint=protocol.INIT.as_posix(),
        parent_checkpoint_sha256=protocol.sha256_file(protocol.INIT), result_dir=FINAL.parent.as_posix(),
        metadata_path=config.as_posix(), metric_profile="training_loss", output_policy="float_loss",
        metrics=metrics, metric_source=dict(type="csv_row", path=history.as_posix(),
                                            match={"epoch": "12"}, columns={k: k for k in metrics}),
        notes="匹配E2 + PAC的12轮训练配置，PAC全程冻结且输出恒零。",
    ))


def validate():
    if not FINAL.is_file():
        raise FileNotFoundError(FINAL)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    folders = (ORIGINAL, CANDIDATE)
    for folder in folders:
        entry = next(e for e in registry["experiments"] if e["result_dir"] == folder.as_posix())
        if entry["comparable_group"] != protocol.GROUP:
            raise ValueError(f"Unexpected comparison group: {folder}")
    refs = [json.loads((folder / "run_metadata.json").read_text(encoding="utf-8")) for folder in folders]
    config = json.loads((FINAL.parent / "config.json").read_text(encoding="utf-8"))
    reference_config = json.loads((protocol.MODEL / "seed42/config.json").read_text(encoding="utf-8"))
    ignored = {"output_dir", "run_name", "resume", "saved_split_manifest"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in reference_config.items() if k not in ignored
    }:
        raise ValueError("Control training configuration differs from PAC training")
    checkpoint = torch.load(FINAL, map_location="cpu", weights_only=False)
    for suffix in ("weight", "bias"):
        if torch.count_nonzero(checkpoint["model"][f"detail_fusion.update_projection.2.{suffix}"]).item():
            raise ValueError("Control checkpoint has a nonzero PAC projection")
    if not (RESULT / "metrics.csv").is_file():
        if RESULT.exists():
            raise FileExistsError(f"Preserve incomplete result: {RESULT}")
        protocol.run([sys.executable, "-u", "test_OS.py", "--data-root", "../datasets/QXSLAB_SAROPT",
                      "--split-manifest", protocol.MANIFEST, "--split", "val", "--variant", "auto",
                      "--checkpoint", FINAL, "--output-dir", RESULT, "--offset", "0", "--limit", "1000",
                      "--metric-profile", "paper", "--output-range-policy", "clamp", "--timing-warmup", "10"])
    metadata = RESULT / "run_metadata.json"
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if meta["checkpoint_sha256"] != protocol.sha256_file(FINAL) or meta["variant"] != "pac_mamba_e2":
        raise ValueError("Unexpected control checkpoint")
    for ref in refs:
        for key in ("split_manifest_sha256", "split", "offset", "evaluated_pairs",
                    "metric_profile", "output_range_policy", "selected_scene_ids"):
            if meta[key] != ref[key]:
                raise ValueError(f"Validation metadata differs: {key}")
    metrics = RESULT / "metrics.csv"
    control, mean = protocol.read_metrics(metrics)
    if len(control) != 1000 or set(control) != set(meta["selected_scene_ids"]):
        raise ValueError("Unexpected validation IDs")
    if any(not math.isfinite(float(row[k])) for row in control.values() for k in protocol.METRICS):
        raise ValueError("Nonfinite validation metric")
    for label, folder in zip(("original_e2", "pac_mamba"), folders):
        baseline, base_mean = protocol.read_metrics(folder / "metrics.csv")
        records = [protocol.paired_record(k, baseline, control, base_mean, mean) for k in protocol.METRICS]
        with (RESULT / f"control_vs_{label}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    entry = dict(
        id="E2-MATCHED-NO-PAC-V1-SEED42-VAL1000", status="development_validation", task="fusion",
        method="E2 matched continuation; PAC frozen with exactly zero output",
        comparable_group=protocol.GROUP, dataset="QXS-SAROPT", split="val", offset=0, sample_count=1000,
        split_manifest=protocol.MANIFEST.as_posix(), split_manifest_sha256=meta["split_manifest_sha256"],
        checkpoint=FINAL.as_posix(), checkpoint_sha256=meta["checkpoint_sha256"],
        parent_checkpoint=protocol.INIT.as_posix(), parent_checkpoint_sha256=config["init_checkpoint_sha256"],
        result_dir=RESULT.as_posix(), metadata_path=metadata.as_posix(), metric_profile="paper",
        output_policy="clamp_then_uint8", metrics={k: float(mean[k]) for k in protocol.METRICS},
        metric_source=dict(type="csv_row", path=metrics.as_posix(), match={"image": "mean"},
                           columns={k: k for k in protocol.METRICS}),
        notes="匹配E2 + PAC的12轮训练配置。PAC全程冻结且输出恒零。仅在固定验证集比较。",
    )
    register(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "train-val"))
    args = parser.parse_args()
    os.chdir(protocol.ROOT)
    if not (protocol.MODEL / "workflow_protocol.json").is_file():
        raise FileNotFoundError("Missing original PAC training protocol lock")
    protocol.lock_protocol()
    if args.action in ("train", "train-val"):
        train()
    if args.action in ("val", "train-val"):
        validate()


if __name__ == "__main__":
    main()
