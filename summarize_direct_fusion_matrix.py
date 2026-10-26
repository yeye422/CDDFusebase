"""Collect R0-R3 validation metrics without selecting on the held-out test set."""

import argparse
import csv
import re
from pathlib import Path


NAME_PATTERN = re.compile(r"^(r[0-3])_epoch_(\d{3})$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=20)
    return parser.parse_args()


def read_summary(metrics_path):
    with metrics_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["image"]: row for row in rows}
    if "mean" not in by_name or "std" not in by_name:
        raise ValueError(f"Missing mean/std rows in {metrics_path}")
    metric_names = [name for name in rows[0] if name != "image"]
    return metric_names, by_name["mean"], by_name["std"]


def main():
    args = parse_args()
    records = []
    expected_metrics = None
    for directory in sorted(args.eval_root.iterdir()):
        if not directory.is_dir():
            continue
        match = NAME_PATTERN.fullmatch(directory.name)
        if not match:
            continue
        metrics_path = directory / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation output: {metrics_path}")
        metric_names, means, stds = read_summary(metrics_path)
        if expected_metrics is None:
            expected_metrics = metric_names
        elif metric_names != expected_metrics:
            raise ValueError(f"Metric columns differ in {metrics_path}")
        record = {"variant": match.group(1), "epoch": int(match.group(2))}
        for name in metric_names:
            record[f"mean_{name}"] = means[name]
            record[f"std_{name}"] = stds[name]
        records.append(record)
    if len(records) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} R0-R3/T1 validation results, "
            f"found {len(records)}"
        )
    records.sort(key=lambda row: (row["variant"], row["epoch"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} validation rows: {args.output.resolve()}")


if __name__ == "__main__":
    main()
