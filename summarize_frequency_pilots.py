"""Summarize fixed-endpoint C0/C1/C2 validation metrics."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
REFERENCE = {
    "FMI": 0.243877,
    "Qabf": 0.286090,
    "Nabf": 0.031728,
    "MS_SSIM": 0.613404,
}


def mean_row(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    values = next(row for row in rows[1:] if row[0] == "mean")
    return {name: float(value) for name, value in zip(header[1:], values[1:])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("test_result/reliability_v5/validation_final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_result/reliability_v5/validation_final/summary.csv"),
    )
    args = parser.parse_args()

    records = []
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.csv"
        diagnostics_path = run_dir / "gate_stats.csv"
        metadata_path = run_dir / "run_metadata.json"
        if not metrics_path.exists() or not diagnostics_path.exists():
            continue
        metrics = mean_row(metrics_path)
        diagnostics = mean_row(diagnostics_path)
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        variant = metadata["variant"]
        passes = (
            metrics["Nabf"] <= 1.10 * REFERENCE["Nabf"]
            and metrics["Qabf"] >= REFERENCE["Qabf"]
            and metrics["VIF"] >= 0.161976
            and metrics["FMI"] >= REFERENCE["FMI"]
            and metrics["MS_SSIM"] >= 0.95 * REFERENCE["MS_SSIM"]
            and diagnostics["output_optical_mae"] >= 0.01
        )
        records.append(
            {
                "variant": variant,
                "passes": passes,
                **{name: metrics[name] for name in METRICS},
                "output_optical_mae": diagnostics["output_optical_mae"],
                "output_sar_mae": diagnostics["output_sar_mae"],
            }
        )

    if not records:
        raise RuntimeError(f"No completed validation runs found below {args.root}")
    records.sort(
        key=lambda row: (
            not row["passes"],
            row["variant"],
            row["Nabf"],
            -row["Qabf"],
            -row["VIF"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "variant",
        "passes",
        *METRICS,
        "output_optical_mae",
        "output_sar_mae",
    )
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"completed runs: {len(records)}")
    print(f"passing runs: {sum(row['passes'] for row in records)}")
    print(f"summary: {args.output.resolve()}")
    for variant in sorted({row["variant"] for row in records}):
        candidates = [row for row in records if row["variant"] == variant]
        best = candidates[0]
        print(
            f"{variant}: fixed_final=True, passes={best['passes']}, "
            f"Qabf={best['Qabf']:.6f}, Nabf={best['Nabf']:.6f}, "
            f"VIF={best['VIF']:.6f}, OpticalMAE={best['output_optical_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
