"""Direct fixed-endpoint comparison of R1 spatial modulation against C0."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("MI", "FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
HIGHER_IS_BETTER = {"MI", "FMI", "Qabf", "SSIM", "MS_SSIM", "VIF"}


def mean_row(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    values = next(row for row in rows[1:] if row[0] == "mean")
    return {name: float(value) for name, value in zip(header[1:], values[1:])}


def percent_delta(value, reference):
    if reference == 0.0:
        return None
    return 100.0 * (value - reference) / abs(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--c0-dir", default="frequency_minimax_baseline"
    )
    parser.add_argument(
        "--r1-dir", default="r1_spatial_modulation"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    c0_dir = args.root / args.c0_dir
    r1_dir = args.root / args.r1_dir
    c0 = mean_row(c0_dir / "metrics.csv")
    r1 = mean_row(r1_dir / "metrics.csv")
    r1_diag = mean_row(r1_dir / "gate_stats.csv")

    rows = []
    strict_non_worse = True
    for name in METRICS:
        delta = r1[name] - c0[name]
        pct = percent_delta(r1[name], c0[name])
        improved = delta >= 0.0 if name in HIGHER_IS_BETTER else delta <= 0.0
        strict_non_worse = strict_non_worse and improved
        rows.append(
            {
                "metric": name,
                "c0": c0[name],
                "r1": r1[name],
                "delta_r1_minus_c0": delta,
                "percent_delta": pct,
                "direction": "higher" if name in HIGHER_IS_BETTER else "lower",
                "non_worse": improved,
            }
        )

    modulation_abs_mean = r1_diag.get("r1_modulation_abs_mean", 0.0)
    modulation_std = r1_diag.get("r1_modulation_std", 0.0)
    modulation_nonconstant = (
        modulation_abs_mean > 1e-6 and modulation_std > 1e-6
    )
    summary = {
        "strict_non_worse_on_all_reported_metrics": strict_non_worse,
        "modulation_nonconstant": modulation_nonconstant,
        "r1_modulation_abs_mean": modulation_abs_mean,
        "r1_modulation_std": modulation_std,
        "output_optical_mae": r1_diag.get("output_optical_mae"),
        "output_sar_mae": r1_diag.get("output_sar_mae"),
        "note": (
            "strict_non_worse is descriptive only: it requires every listed "
            "higher-is-better metric to be >= C0 and Nabf to be <= C0. "
            "No historical threshold is reused."
        ),
    }

    output = args.output or (args.root / "r1_vs_c0.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    for row in rows:
        pct = row["percent_delta"]
        pct_text = "n/a" if pct is None else f"{pct:+.3f}%"
        print(
            f"{row['metric']}: C0={row['c0']:.6f}, "
            f"R1={row['r1']:.6f}, delta={row['delta_r1_minus_c0']:+.6f}, "
            f"pct={pct_text}, non_worse={row['non_worse']}"
        )
    print(
        "R1 modulation: "
        f"abs_mean={modulation_abs_mean:.8f}, std={modulation_std:.8f}, "
        f"nonconstant={modulation_nonconstant}"
    )
    print(f"strict_non_worse_on_all_reported_metrics={strict_non_worse}")
    print(f"comparison: {output.resolve()}")


if __name__ == "__main__":
    main()
