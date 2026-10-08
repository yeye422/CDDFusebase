"""Compare two isolated C1 adapter ablations with the exact C0 val200."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("MI", "FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
HIGHER_BETTER = set(METRICS) - {"Nabf"}
VARIANTS = ("c1_base_adapter", "c1_detail_adapter")


def load_metrics(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["image"]: {metric: float(row[metric]) for metric in METRICS}
        for row in rows if row["image"] not in ("mean", "std")
    }


def load_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Empty comparison: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    baseline_dir = args.root / "frequency_minimax_baseline"
    c0 = load_metrics(baseline_dir / "metrics.csv")
    base_meta = load_metadata(baseline_dir / "run_metadata.json")
    if not c0:
        raise ValueError("No C0 evaluation samples")

    summary = []
    per_image = []
    for variant in VARIANTS:
        directory = args.root / variant
        observed = load_metrics(directory / "metrics.csv")
        meta = load_metadata(directory / "run_metadata.json")
        if set(observed) != set(c0):
            raise ValueError(f"Different paired validation images in {variant}")
        for field in ("split", "split_manifest_sha256", "metric_profile", "output_range_policy"):
            if meta.get(field) != base_meta.get(field):
                raise ValueError(f"{variant} evaluation differs in {field}")
        if meta.get("variant") != variant or len(observed) != meta.get("evaluated_pairs"):
            raise ValueError(f"{variant} metadata does not match evaluation")

        for metric in METRICS:
            values = np.asarray(
                [observed[image][metric] - c0[image][metric] for image in sorted(c0)],
                dtype=np.float64,
            )
            improved = values > 0 if metric in HIGHER_BETTER else values < 0
            summary.append({
                "variant": variant,
                "metric": metric,
                "c0_mean": float(np.mean([c0[name][metric] for name in c0])),
                "c1_mean": float(np.mean([observed[name][metric] for name in c0])),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "improved_fraction": float(improved.mean()),
                "worsened_fraction": float((values < 0 if metric in HIGHER_BETTER else values > 0).mean()),
            })
        for image in sorted(c0):
            per_image.append({
                "variant": variant,
                "image": image,
                **{
                    f"delta_{metric}": observed[image][metric] - c0[image][metric]
                    for metric in METRICS
                },
            })
    write_csv(args.root / "c1_vs_c0_summary.csv", summary)
    write_csv(args.root / "c1_vs_c0_per_image.csv", per_image)
    print("=== C1 vs frozen C0 (val200, paired) ===")
    for row in summary:
        print(
            f"{row['variant']} {row['metric']}: "
            f"c0={row['c0_mean']:.6f}, c1={row['c1_mean']:.6f}, "
            f"delta={row['mean_delta']:+.6f}, "
            f"improved_fraction={row['improved_fraction']:.3f}"
        )
    print(f"outputs: {args.root.resolve()}")


if __name__ == "__main__":
    main()
