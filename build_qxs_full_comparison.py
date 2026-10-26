"""Build strict QXS test1000 comparison and paired edge statistics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "EN", "MI", "FMI", "PSNR", "Qabf", "Nabf", "AG", "SF",
    "SD", "CC", "SCD", "SSIM", "MS_SSIM", "VIF",
)


def read_metrics_csv(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = row["image"]
            if image in ("mean", "std"):
                continue
            rows[str(image)] = {name: float(row[name]) for name in METRICS}
    return rows


def read_spd_comparison(path, method="e2_feature"):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["image"])] = {
                name: float(row[f"{method}_{name}"]) for name in METRICS
            }
    return rows


def paired_bootstrap(values, seed=42, iterations=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        means[index] = values[sample].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zero-shot-clamp", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    methods = {
        "Wavelet": read_metrics_csv(
            root.parent / "comparation/results/qxslab_paper_1000_manual/wavelet/metrics.csv"
        ),
        "U2Fusion": read_metrics_csv(
            root.parent / "comparation/results/qxslab_paper_1000_manual/u2fusion/metrics.csv"
        ),
        "Laplacian": read_metrics_csv(
            root.parent / "comparation/results/qxslab_paper_1000_manual/laplacian/metrics.csv"
        ),
        "B0-clamp": read_metrics_csv(
            root / "test_result/reliability_v4/B0_clamp/metrics.csv"
        ),
        "C0-minimax": read_metrics_csv(
            root / "test_result/reliability_v5/frequency_minimax_seed42/metrics.csv"
        ),
        "Detail-only-SPD": read_spd_comparison(
            root / "test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/per_image.csv"
        ),
    }
    if args.zero_shot_clamp:
        methods["CDDFuse-zero-shot-clamp"] = read_metrics_csv(
            args.zero_shot_clamp
        )

    id_sets = {name: set(rows) for name, rows in methods.items()}
    expected = next(iter(id_sets.values()))
    mismatches = {
        name: {"missing": len(expected - ids), "extra": len(ids - expected)}
        for name, ids in id_sets.items()
        if ids != expected
    }
    if mismatches:
        raise ValueError(f"Method image IDs differ: {mismatches}")
    ordered_ids = sorted(expected, key=lambda value: int(value))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method, rows in methods.items():
        record = {"method": method, "images": len(rows)}
        for metric in METRICS:
            values = np.asarray([rows[image][metric] for image in ordered_ids])
            record[metric] = float(values.mean())
            record[f"{metric}_std"] = float(values.std())
        summaries.append(record)

    summary_fields = ["method", "images"] + list(METRICS) + [
        f"{name}_std" for name in METRICS
    ]
    with (args.output_dir / "full_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    baseline = methods["B0-clamp"]
    edge_rows = []
    for method, rows in methods.items():
        q_delta = np.asarray([
            rows[image]["Qabf"] - baseline[image]["Qabf"]
            for image in ordered_ids
        ])
        n_delta = np.asarray([
            rows[image]["Nabf"] - baseline[image]["Nabf"]
            for image in ordered_ids
        ])
        edge_rows.append(
            {
                "method": method,
                "Qabf_mean": float(np.mean([rows[x]["Qabf"] for x in ordered_ids])),
                "Qabf_delta_vs_B0": float(q_delta.mean()),
                "Qabf_delta_ci95_low": paired_bootstrap(q_delta)[0],
                "Qabf_delta_ci95_high": paired_bootstrap(q_delta)[1],
                "Qabf_improved_fraction": float((q_delta > 0).mean()),
                "Nabf_mean": float(np.mean([rows[x]["Nabf"] for x in ordered_ids])),
                "Nabf_delta_vs_B0": float(n_delta.mean()),
                "Nabf_delta_ci95_low": paired_bootstrap(n_delta)[0],
                "Nabf_delta_ci95_high": paired_bootstrap(n_delta)[1],
                "Nabf_improved_fraction": float((n_delta < 0).mean()),
            }
        )
    with (args.output_dir / "edge_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)

    historical = {
        "warning": "Historical CDDFuse folders have no run_metadata; output normalization is unverified and they are excluded from the strict table.",
        "CDDFuse-zero-shot-historical": read_metrics_csv(
            root / "test_result/QXSLAB_SAROPT_1000_zero_shot/metrics.csv"
        ),
        "CDDFuse-trained-historical": read_metrics_csv(
            root / "test_result/QXSLAB_SAROPT_1000/metrics.csv"
        ),
    }
    historical_summary = {"warning": historical.pop("warning")}
    for method, rows in historical.items():
        historical_summary[method] = {
            metric: float(np.mean([rows[x][metric] for x in ordered_ids]))
            for metric in METRICS
        }

    payload = {
        "protocol": {
            "dataset": "QXS-SAROPT",
            "split": "test",
            "sample_count": len(ordered_ids),
            "metric_profile": "PaperEvaluator",
            "strict_output_policy_for_neural_models": "clamp_then_uint8",
            "source_protocol": str(
                (root.parent / "comparation/results/qxslab_paper_1000_manual/protocol.json").resolve()
            ),
        },
        "strict_methods": [row["method"] for row in summaries],
        "historical_unverified": historical_summary,
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote strict comparison for {len(methods)} methods and {len(ordered_ids)} images")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
