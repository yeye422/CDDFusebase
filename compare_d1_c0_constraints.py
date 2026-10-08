"""Strict paired D1 A0-A3 versus untouched C0 on the same val200."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METRICS = ("MI", "FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
RUNS = ("A0", "A1", "A2", "A3")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    values = {
        row["image"]: {key: float(row[key]) for key in METRICS}
        for row in rows if row["image"] not in ("mean", "std")
    }
    if not values:
        raise RuntimeError(f"No rows in {path}")
    return values


def metadata(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    baseline = read_csv(root / "C0" / "metrics.csv")
    base_meta = metadata(root / "C0" / "run_metadata.json")
    if base_meta.get("variant") != "frequency_minimax_baseline":
        raise ValueError("Base checkpoint is not the original C0")
    summary, per_image = [], []
    for group in RUNS:
        observed = read_csv(root / group / "metrics.csv")
        meta = metadata(root / group / "run_metadata.json")
        if set(observed) != set(baseline):
            raise ValueError(f"D1-{group} does not contain identical validation image IDs")
        if meta.get("variant") != "frequency_minimax_baseline":
            raise ValueError(f"D1-{group} checkpoint is not a C0 variant")
        for field in ("split", "split_manifest_sha256", "metric_profile", "output_range_policy"):
            if meta.get(field) != base_meta.get(field):
                raise ValueError(f"Evaluation mismatch: {group}.{field}")
        if len(observed) != meta.get("evaluated_pairs"):
            raise ValueError(f"Evaluation count mismatch: D1-{group}")
        ordered = sorted(baseline)
        for metric in METRICS:
            c0 = np.asarray([baseline[n][metric] for n in ordered])
            new = np.asarray([observed[n][metric] for n in ordered])
            diff = new - c0
            favorable = (diff < 0) if metric == "Nabf" else (diff > 0)
            summary.append({
                "group": group, "metric": metric,
                "c0_mean": float(c0.mean()),
                "d1_mean": float(new.mean()),
                "mean_delta": float(diff.mean()),
                "median_delta": float(np.median(diff)),
                "improved_fraction": float(favorable.mean()),
            })
        for n in ordered:
            per_image.append({
                "group": group, "image": n,
                **{
                    "delta_" + metric: observed[n][metric] - baseline[n][metric]
                    for metric in METRICS
                },
            })
    write_csv(root / "d1_vs_c0_summary.csv", summary)
    write_csv(root / "d1_vs_c0_per_image.csv", per_image)
    print("Paired C0/D1 val200:")
    for row in summary:
        print(
            f'{row["group"]} {row["metric"]} '
            f'delta={row["mean_delta"]:+.6f} '
            f'improved_fraction={row["improved_fraction"]:.3f}'
        )


if __name__ == "__main__":
    main()
