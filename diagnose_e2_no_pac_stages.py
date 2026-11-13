"""Locate metric drift across existing no-PAC checkpoints on a fixed val subset."""
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys

import run_e2_matched_no_pac as control
import run_e2_pac_mamba as protocol


ROOT = Path("test_result/e2_matched_no_pac_stage_v1")
OFFSET, COUNT = 450, 100
EPOCHS = (2, 5, 6)
METRICS = ("Nabf", "Qabf", "MS_SSIM", "SSIM", "MI", "VIF")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def subset(folder, expected, offset, count, checkpoint):
    meta = read_json(folder / "run_metadata.json")
    required = dict(split="val", offset=offset, evaluated_pairs=count, metric_profile="paper",
                    output_range_policy="clamp", split_manifest_sha256=protocol.sha256_file(protocol.MANIFEST),
                    checkpoint_sha256=protocol.sha256_file(checkpoint))
    if any(meta[key] != value for key, value in required.items()):
        raise ValueError(f"Metadata mismatch: {folder}")
    ids = meta["selected_scene_ids"]
    rows, _ = protocol.read_metrics(folder / "metrics.csv")
    if len(ids) != count or len(rows) != count or set(rows) != set(ids):
        raise ValueError(f"Unexpected scene IDs: {folder}")
    if expected != ids:
        raise ValueError(f"Scene order mismatch: {folder}")
    return rows


def main():
    os.chdir(protocol.ROOT)
    final_meta = read_json(control.RESULT / "run_metadata.json")
    full_ids = final_meta["selected_scene_ids"]
    ids = full_ids[OFFSET:OFFSET + COUNT]
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise ValueError("Incomplete validation subset")
    stages = {"original_e2": subset(control.ORIGINAL, full_ids, 0, 1000, protocol.INIT)}
    final = subset(control.RESULT, full_ids, 0, 1000, control.FINAL)
    entries = []
    for epoch in EPOCHS:
        name = f"epoch_{epoch:03d}"
        checkpoint = control.FINAL.parent / f"{name}.pth"
        folder = ROOT / name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not (folder / "metrics.csv").is_file():
            if folder.exists():
                raise FileExistsError(f"Preserve incomplete output: {folder}")
            protocol.run([sys.executable, "-u", "test_OS.py", "--data-root", "../datasets/QXSLAB_SAROPT",
                          "--split-manifest", protocol.MANIFEST, "--split", "val", "--variant", "auto",
                          "--checkpoint", checkpoint, "--output-dir", folder, "--offset", str(OFFSET),
                          "--limit", str(COUNT), "--metric-profile", "paper", "--output-range-policy", "clamp",
                          "--metrics-only", "--timing-warmup", "10"])
        stages[name] = subset(folder, ids, OFFSET, COUNT, checkpoint)
        meta = read_json(folder / "run_metadata.json")
        if meta["variant"] != "pac_mamba_e2" or not meta["metrics_only"]:
            raise ValueError(f"Unexpected diagnostic variant or output mode: {folder}")
        path = folder / "metrics.csv"
        _, means = protocol.read_metrics(path)
        entries.append(dict(
            id=f"E2-MATCHED-NO-PAC-STAGE-V1-E{epoch:02d}", status="diagnostic_fixed_subset", task="fusion",
            method=f"E2 no-PAC stage diagnostic, epoch {epoch}", comparable_group="QXS-E2-NO-PAC-STAGE-VAL100-V1",
            dataset="QXS-SAROPT", split="val", offset=OFFSET, sample_count=COUNT,
            split_manifest=protocol.MANIFEST.as_posix(), split_manifest_sha256=meta["split_manifest_sha256"],
            checkpoint=checkpoint.as_posix(), checkpoint_sha256=meta["checkpoint_sha256"],
            parent_checkpoint=protocol.INIT.as_posix(), parent_checkpoint_sha256=protocol.sha256_file(protocol.INIT),
            result_dir=folder.as_posix(), metadata_path=(folder / "run_metadata.json").as_posix(),
            metric_profile="paper", output_policy="clamp_then_uint8",
            metrics={k: float(means[k]) for k in protocol.METRICS},
            metric_source=dict(type="csv_row", path=path.as_posix(), match={"image": "mean"},
                               columns={k: k for k in protocol.METRICS}),
            notes="固定val位置450至549，定位阶段变化。复用原E2和第12轮的已有逐图数据。",
        ))
    stages["epoch_012"] = final
    summary = []
    for name, data in stages.items():
        for key in METRICS:
            original = stages["original_e2"]
            vals = [float(data[i][key]) for i in ids]
            diffs = [x - float(original[i][key]) for x, i in zip(vals, ids)]
            if not all(math.isfinite(x) for x in (*vals, *diffs)):
                raise ValueError(f"Non-finite metric: {name}/{key}")
            summary.append(dict(stage=name, metric=key, mean=statistics.mean(vals),
                                change_vs_original=statistics.mean(diffs),
                                improved_scenes=sum(d < 0 if key == "Nabf" else d > 0 for d in diffs)))
    with (ROOT / "stage_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    registry = read_json(control.REGISTRY)
    registered = {e["id"]: e for e in registry["experiments"]}
    for entry in entries:
        old = registered.get(entry["id"])
        if old is not None and old != entry:
            raise ValueError(f"Registration differs: {entry['id']}")
        if old is None:
            registry["experiments"].append(entry)
    if len(registry["experiments"]) != len(registered):
        registry["updated_at"] = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        protocol.write_json(control.REGISTRY, registry)
    protocol.run([sys.executable, "maintain_experiment_index.py", "--check", "--render"])
    print(ROOT / "stage_comparison.csv")


if __name__ == "__main__":
    main()
