"""Compare B0, B1-v5 pilot, and fixed-endpoint C0-minimax on 200 val pairs."""

import csv
from pathlib import Path


ROOT = Path("test_result")
RUNS = {
    "B0": ROOT / "reliability_v5/validation_reference/B0_200",
    "B1_v5_pilot": ROOT / "reliability_v4/validation/B1_v5_pilot_200",
    "C0_minimax_final": (
        ROOT
        / "reliability_v5/validation_minimax_final/frequency_minimax_baseline"
    ),
}
METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")


def read_mean(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    mean = next(row for row in rows[1:] if row[0] == "mean")
    return {name: float(value) for name, value in zip(header[1:], mean[1:])}


def relative_change(value, reference, lower_is_better=False):
    change = (value / reference - 1.0) * 100.0
    return -change if lower_is_better else change


def main():
    results = {}
    diagnostics = {}
    for name, run_dir in RUNS.items():
        metrics_path = run_dir / "metrics.csv"
        diagnostics_path = run_dir / "gate_stats.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing completed result for {name}: {run_dir}")
        results[name] = read_mean(metrics_path)
        diagnostics[name] = (
            read_mean(diagnostics_path)
            if diagnostics_path.exists()
            else {"output_optical_mae": float("nan"), "output_sar_mae": float("nan")}
        )

    b0 = results["B0"]
    candidate = results["C0_minimax_final"]
    gates = {
        "nabf_reduction_15pct": candidate["Nabf"] <= 0.85 * b0["Nabf"],
        "qabf_retention_95pct": candidate["Qabf"] >= 0.95 * b0["Qabf"],
        "vif_retention_95pct": candidate["VIF"] >= 0.95 * b0["VIF"],
        "fmi_retention_95pct": candidate["FMI"] >= 0.95 * b0["FMI"],
        "not_optical_identity": (
            diagnostics["C0_minimax_final"]["output_optical_mae"] >= 0.01
        ),
    }

    output = ROOT / "reliability_v5/validation_reference/comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("model", *METRICS, "output_optical_mae", "output_sar_mae")
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    print(f"comparison: {output.resolve()}")
    print("C0-minimax relative to B0 (positive means improvement):")
    for metric in METRICS:
        print(
            f"  {metric}: "
            f"{relative_change(candidate[metric], b0[metric], metric == 'Nabf'):+.2f}%"
        )
    print("pre-registered B0 gates:")
    for name, passed in gates.items():
        print(f"  {name}: {passed}")
    print(f"all mandatory gates: {all(gates.values())}")


if __name__ == "__main__":
    main()
