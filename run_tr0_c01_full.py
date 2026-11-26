"""Train C01 on QXS-SAROPT train8000 and evaluate its fixed final checkpoint."""

import argparse
import csv
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
import sys
from unittest.mock import patch

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from run_tr0_factorial import checkpoint_path, read_json, sha256_file, train, write_json
from tr0_evaluation import METRICS, POLICY, provenance, register_many, register_trainings, verify_checkpoint
from utils.dataset_os import manifest_pairs, validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE


ARM = "C01"
RUN_ID = "train8000-seed42"
ROOT = Path("models/tr0_v1_train8000/seed42")
EXPECTED_COUNTS = {"train": 8000, "val": 1000, "test": 1000}
EVALUATION_ROOT = Path("test_result/tr0_v1_train8000/seed42")


def evaluate_full(config, manifest, split, device, *, output=None, limit=None):
    import test_OS

    path = checkpoint_path(config, ROOT, ARM)
    verify_checkpoint(path, config, ARM)
    ids = read_json(manifest)["splits"][split][:limit]
    output = output or EVALUATION_ROOT / f"{split}1000-reference-fmi-v1" / ARM
    metadata = output / "run_metadata.json"
    metrics_path = output / "metrics.csv"
    details = provenance(path, manifest)
    if not output.exists():
        argv = ["test_OS.py", "--data-root", config["data_root"], "--split-manifest", str(manifest),
                "--split", split, "--limit", str(len(ids)), "--checkpoint", str(path),
                "--variant", OpticalBaseAnchoredE2.VARIANT, "--metric-profile", DEFAULT_PROFILE,
                "--output-range-policy", "clamp", "--output-dir", str(output)]
        with patch.dict(test_OS.DIRECT_MODEL_TYPES, {OpticalBaseAnchoredE2.VARIANT: OpticalBaseAnchoredE2}), \
                patch.object(sys, "argv", argv):
            test_OS.main(device=str(device))
    if not metadata.is_file() or not metrics_path.is_file():
        raise FileExistsError(f"Incomplete evaluation is preserved at {output}; move it before retrying")
    meta = read_json(metadata)
    expected = dict(checkpoint_sha256=details["checkpoint_sha256"],
                    split_manifest_sha256=details["split_manifest_sha256"], selected_scene_ids=ids,
                    metric_profile=DEFAULT_PROFILE, variant=OpticalBaseAnchoredE2.VARIANT, split=split,
                    offset=0, evaluated_pairs=len(ids), metrics_only=False, output_range_policy="clamp")
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing evaluation differs from the protocol: {output}")
    for kind in ("gray", "rgb", "preview"):
        if {path.stem for path in (output / kind).glob("*.png")} != set(ids):
            raise ValueError(f"Missing or unexpected {kind} images in {output}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != ("image", *METRICS) or [row["image"] for row in rows] != [*ids, "mean", "std"]:
            raise ValueError(f"Unexpected metric columns or scenes in {metrics_path}")
    summary = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows[:-2]]
        finite = [value for value in values if math.isfinite(value)]
        summary[metric] = dict(mean=fmean(finite) if finite else None, std=pstdev(finite) if finite else None,
                               finite_count=len(finite))
    summary_path = output / "summary.json"
    payload = dict(details, split=split, sample_count=len(ids), metric_profile=DEFAULT_PROFILE,
                   output_policy=POLICY, metrics=summary, raw_metrics=metrics_path.as_posix(),
                   raw_metrics_sha256=sha256_file(metrics_path))
    if summary_path.exists() and read_json(summary_path) != payload:
        raise ValueError(f"Existing summary differs from the verified results: {summary_path}")
    if not summary_path.exists():
        write_json(summary_path, payload)
    print(f"Evaluation summary: {summary_path}", flush=True)
    if limit is not None:
        return
    group = f"QXS-TR0-V1-{RUN_ID}-{split.upper()}1000-C01-REF-FMI-V1"
    finite_means = {metric: stats["mean"] for metric, stats in summary.items() if stats["mean"] is not None}
    if split == "test":
        # Refresh the historical comparison under its own metric profile.
        from build_five_method_test1000_comparison import main as refresh_five_methods

        refresh_five_methods()
    register_many([dict(
        details, id=group, comparable_group=group, dataset="QXS-SAROPT", split=split, offset=0,
        sample_count=len(ids), task="fusion", status="formal_test" if split == "test" else "development_validation",
        method="TR0 C01 train8000 seed42", result_dir=output.as_posix(), metadata_path=metadata.as_posix(),
        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=finite_means,
        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                           paths={metric: ["metrics", metric, "mean"] for metric in finite_means}),
        notes="固定第 20 轮。逐图指标保留原值，汇总逐项使用有限值。",
    )])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--check", action="store_true", help="Verify the split and CUDA without model execution.")
    parser.add_argument("--evaluate", choices=("val", "test"), help="Evaluate the fixed epoch-20 checkpoint.")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)

    config = read_json("configs/tr0_v1.json")
    config.update(train_count=EXPECTED_COUNTS["train"], screen_count=EXPECTED_COUNTS["val"])
    source = Path(config["source_manifest"])
    manifest = read_json(source)
    validate_manifest_splits(manifest)
    if {key: len(values) for key, values in manifest["splits"].items()} != EXPECTED_COUNTS:
        raise ValueError(f"Expected the fixed 8000/1000/1000 split: {source}")
    if not torch.cuda.is_available() or (not args.evaluate and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("C01 requires CUDA and uses BF16 for training")

    locked = ROOT / "screen_manifest.json"
    digest = sha256_file(source)
    if locked.exists() and sha256_file(locked) != digest:
        raise ValueError(f"Existing run uses a different split: {locked}")
    mode = f"{args.evaluate}1000, CUDA FP32" if args.evaluate else "train8000/val1000, CUDA BF16"
    print(f"C01 {mode}, seed {config['seed']}; split SHA-256 {digest}", flush=True)
    print(f"Final checkpoint: {checkpoint_path(config, ROOT, ARM)}", flush=True)
    if args.check:
        pairs, _ = manifest_pairs(Path(config["data_root"]), source)
        print(f"Verified source files for {sum(len(split) for split in pairs.values())} paired scenes", flush=True)
        return
    if args.evaluate:
        evaluate_full(config, locked, args.evaluate, torch.device(args.device))
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    if not locked.exists():
        locked.write_bytes(source.read_bytes())
    final = train(config, ROOT, ARM, locked, torch.device(args.device), resume=True)
    register_trainings(config, RUN_ID, {ARM: final}, locked)


if __name__ == "__main__":
    main()
