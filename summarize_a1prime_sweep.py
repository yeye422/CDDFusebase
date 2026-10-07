"""Summarize the A1-prime epoch/scale validation sweep."""

import argparse
import csv
import json
from pathlib import Path


REQUIRED_METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
DEFAULT_THRESHOLDS = {
    "Qabf": 0.272,
    "Nabf": 0.0317,
    "VIF": 0.154,
}


def read_means(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    mean_row = next(row for row in rows[1:] if row[0] == "mean")
    values = {name: float(value) for name, value in zip(header[1:], mean_row[1:])}
    missing = sorted(set(REQUIRED_METRICS) - set(values))
    if missing:
        raise ValueError(f"{path} is missing metrics: {', '.join(missing)}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("test_result/reliability_v4/sweep"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_result/reliability_v4/sweep/a1prime_pareto_summary.csv"),
    )
    args = parser.parse_args()

    records = []
    for metrics_path in sorted(args.sweep_root.glob("A1prime_e*_g*/metrics.csv")):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "run_metadata.json"
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        checkpoint = Path(metadata["checkpoint"])
        epoch = int(checkpoint.stem.split("_")[-1])
        scale = float(metadata["force_residual_scale"])
        metrics = read_means(metrics_path)
        passes = (
            metrics["Nabf"] <= DEFAULT_THRESHOLDS["Nabf"]
            and metrics["Qabf"] >= DEFAULT_THRESHOLDS["Qabf"]
            and metrics["VIF"] >= DEFAULT_THRESHOLDS["VIF"]
        )
        records.append({"epoch": epoch, "scale": scale, "passes": passes, **metrics})

    if not records:
        raise RuntimeError(f"No completed sweep runs found below {args.sweep_root}")

    records.sort(key=lambda row: (not row["passes"], row["Nabf"], -row["Qabf"], -row["VIF"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("epoch", "scale", "passes") + REQUIRED_METRICS
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    passing = sum(record["passes"] for record in records)
    print(f"completed runs: {len(records)}")
    print(f"passing runs: {passing}")
    print(f"summary: {args.output.resolve()}")
    best = records[0]
    print(
        "top candidate: "
        f"epoch={best['epoch']}, scale={best['scale']:.2f}, "
        f"Qabf={best['Qabf']:.6f}, Nabf={best['Nabf']:.6f}, "
        f"VIF={best['VIF']:.6f}, passes={best['passes']}"
    )


if __name__ == "__main__":
    main()
