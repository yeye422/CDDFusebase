"""Compare fixed-endpoint C3a/C3b pilots with C0-minimax pilot."""

import csv
from pathlib import Path


RUNS = {
    "C0_minimax": Path(
        "test_result/reliability_v5/validation_minimax_final/frequency_minimax_baseline"
    ),
    "C3a_RoA": Path("test_result/reliability_v5/c3_line_validation/fixed_line_roa"),
    "C3b_RoA_FA_bridge": Path(
        "test_result/reliability_v5/c3_line_validation/fixed_line_phase_bridge"
    ),
}
METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")


def read_mean(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    mean = next(row for row in rows[1:] if row[0] == "mean")
    return {name: float(value) for name, value in zip(header[1:], mean[1:])}


def main():
    results = {}
    diagnostics = {}
    for name, directory in RUNS.items():
        results[name] = read_mean(directory / "metrics.csv")
        diagnostics[name] = read_mean(directory / "gate_stats.csv")

    output = Path("test_result/reliability_v5/c3_line_validation/summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model", *METRICS, "output_optical_mae", "output_sar_mae")
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in RUNS:
            writer.writerow(
                {
                    "model": name,
                    **{metric: results[name][metric] for metric in METRICS},
                    "output_optical_mae": diagnostics[name]["output_optical_mae"],
                    "output_sar_mae": diagnostics[name]["output_sar_mae"],
                }
            )

    baseline = results["C0_minimax"]
    print(f"summary: {output.resolve()}")
    for name in ("C3a_RoA", "C3b_RoA_FA_bridge"):
        print(name)
        for metric in METRICS:
            delta = (results[name][metric] / baseline[metric] - 1.0) * 100.0
            if metric == "Nabf":
                delta = -delta
            print(f"  {metric}: {delta:+.2f}%")


if __name__ == "__main__":
    main()
