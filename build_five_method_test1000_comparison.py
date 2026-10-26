"""Build the canonical 14-metric comparison for five formal test1000 methods."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "test_result/v21_softmask/five_method_test1000_comparison"
METRICS = (
    "EN", "MI", "FMI", "PSNR", "Qabf", "Nabf", "AG", "SF",
    "SD", "CC", "SCD", "SSIM", "MS_SSIM", "VIF",
)
LOWER_IS_BETTER = {"Nabf"}
SOURCES = (
    {
        "method": "B0",
        "type": "csv",
        "path": ROOT / "test_result/reliability_v4/B0_clamp/metrics.csv",
    },
    {
        "method": "B0 + SPD",
        "type": "summary",
        "path": ROOT / "test_result/v21_softmask/spd_nomask_on_b0_test1000/summary.json",
        "key": "e2_feature",
    },
    {
        "method": "B0 + Detail-only SPD",
        "type": "summary",
        "path": ROOT / "test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/summary.json",
        "key": "e2_feature",
    },
    {
        "method": "C0-minimax",
        "type": "csv",
        "path": ROOT / "test_result/reliability_v5/frequency_minimax_seed42/metrics.csv",
    },
    {
        "method": "C0-minimax + SPD",
        "type": "summary",
        "path": ROOT / "test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/summary.json",
        "key": "e2_feature",
    },
)


def read_csv_source(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    mean = next(row for row in rows if row["image"] == "mean")
    std = next(row for row in rows if row["image"] == "std")
    return {
        metric: {"mean": float(mean[metric]), "std": float(std[metric])}
        for metric in METRICS
    }


def read_summary_source(path, key):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        metric: {
            "mean": float(payload["methods"][key][metric]["mean"]),
            "std": float(payload["methods"][key][metric]["std"]),
        }
        for metric in METRICS
    }


def main():
    records = []
    for source in SOURCES:
        if source["type"] == "csv":
            metrics = read_csv_source(source["path"])
        else:
            metrics = read_summary_source(source["path"], source["key"])
        records.append(
            {
                "method": source["method"],
                "source": str(source["path"].relative_to(ROOT)),
                "metrics": metrics,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_fields = ["method", "source"]
    for metric in METRICS:
        csv_fields.extend((f"{metric}_mean", f"{metric}_std"))
    with (OUTPUT / "comparison_14_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            row = {"method": record["method"], "source": record["source"]}
            for metric in METRICS:
                row[f"{metric}_mean"] = record["metrics"][metric]["mean"]
                row[f"{metric}_std"] = record["metrics"][metric]["std"]
            writer.writerow(row)

    best = {}
    for metric in METRICS:
        values = [record["metrics"][metric]["mean"] for record in records]
        best[metric] = min(values) if metric in LOWER_IS_BETTER else max(values)
    lines = [
        "# 五方法正式 test1000：14项融合指标",
        "",
        "协议：QXS-SAROPT test1000，PaperEvaluator，clamp→uint8。",
        "",
        "| 方法 | " + " | ".join(
            f"{metric}{'↓' if metric in LOWER_IS_BETTER else '↑'}"
            for metric in METRICS
        ) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for record in records:
        values = []
        for metric in METRICS:
            value = record["metrics"][metric]["mean"]
            text = f"{value:.4f}"
            if abs(value - best[metric]) < 5e-7:
                text = f"**{text}**"
            values.append(text)
        lines.append(f"| {record['method']} | " + " | ".join(values) + " |")
    lines.extend(["", "## 数据来源", ""])
    for record in records:
        lines.append(f"- {record['method']}：`{record['source']}`")
    (OUTPUT / "comparison_14_metrics.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    with (OUTPUT / "comparison_14_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "protocol": {
                    "dataset": "QXS-SAROPT",
                    "split": "test",
                    "sample_count": 1000,
                    "metric_profile": "paper",
                    "output_policy": "clamp_then_uint8",
                },
                "methods": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
