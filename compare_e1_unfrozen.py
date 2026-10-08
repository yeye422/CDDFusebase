"""Paired val200 metric comparisons of C0 and the E1 unfrozen training variants."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("MI", "FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
EXPECTED_VARIANTS = {
    "e1_encoder_unfrozen": "e1_dual_stem",
    "c0_full_finetune": "frequency_minimax_baseline",
    "e1_end2end": "e1_dual_stem",
}


def read_metrics(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    values = {
        row["image"]: {metric: float(row[metric]) for metric in METRICS}
        for row in records if row["image"] not in ("mean", "std")
    }
    if not values:
        raise ValueError(f"No evaluation images found: {path}")
    return values


def read_metadata(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    baseline = read_metrics(root / "frequency_minimax_baseline" / "metrics.csv")
    baseline_meta = read_metadata(
        root / "frequency_minimax_baseline" / "run_metadata.json"
    )
    if baseline_meta.get("variant") != "frequency_minimax_baseline":
        raise ValueError("Baseline checkpoint is not C0")

    summary = []
    per_image = []
    for name, expected_variant in EXPECTED_VARIANTS.items():
        path = root / name / "metrics.csv"
        if not path.exists():
            if name == "e1_encoder_unfrozen":
                raise FileNotFoundError(f"Required E1 encoder-only result missing: {path}")
            print(f"optional evaluation not found: {name}")
            continue
        model = read_metrics(path)
        metadata = read_metadata(root / name / "run_metadata.json")
        if set(model) != set(baseline):
            raise ValueError(f"Unpaired scene IDs: {name}")
        for key in (
            "split", "split_manifest_sha256",
            "metric_profile", "output_range_policy",
            "selected_scene_ids",
        ):
            if baseline_meta.get(key) != metadata.get(key):
                raise ValueError(f"Unmatched evaluation metadata {key}: {name}")
        if metadata.get("variant") != expected_variant:
            raise ValueError(f"Checkpoint variant mismatch: {name}")
        if metadata.get("evaluated_pairs") != len(baseline):
            raise ValueError(f"Unexpected evaluation pair count: {name}")

        for metric in METRICS:
            delta = np.asarray([
                model[image][metric] - baseline[image][metric]
                for image in sorted(baseline)
            ], dtype=np.float64)
            upward = metric != "Nabf"
            improved = delta > 0 if upward else delta < 0
            summary.append({
                "model": name,
                "metric": metric,
                "baseline_mean": float(np.mean([row[metric] for row in baseline.values()])),
                "model_mean": float(np.mean([row[metric] for row in model.values()])),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "improved_fraction": float(improved.mean()),
            })
        for image in sorted(baseline):
            per_image.append({
                "model": name,
                "image": image,
                **{
                    f"delta_{metric}": model[image][metric] - baseline[image][metric]
                    for metric in METRICS
                },
            })
    write_rows(root / "e1_unfrozen_comparison_summary.csv", summary)
    write_rows(root / "e1_unfrozen_comparison_per_image.csv", per_image)
    print("=== Paired val200 comparison against original C0 ===")
    for row in summary:
        print(
            f"{row['model']} {row['metric']}: "
            f"C0={row['baseline_mean']:.6f}, "
            f"model={row['model_mean']:.6f}, "
            f"delta={row['mean_delta']:+.6f}, "
            f"improved_fraction={row['improved_fraction']:.3f}"
        )
    print(f"outputs: {root.resolve()}")


if __name__ == "__main__":
    main()
