"""Build publication-style comparison tables with best/second-best marking."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "test_result/v21_softmask/extended_test1000_comparison"
LOWER_IS_BETTER = {"Nabf"}
GROUPS = (
    ("信息、边缘指标", ("EN", "MI", "FMI", "PSNR", "Qabf", "Nabf")),
    ("梯度、频率、相关性指标", ("AG", "SF", "SD", "CC", "SCD")),
    ("结构与视觉指标", ("SSIM", "MS_SSIM", "VIF")),
)


def csv_mean(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            if row["image"] == "mean":
                return {
                    key: float(value)
                    for key, value in row.items()
                    if key not in ("image", "time_ms") and value != ""
                }
    raise ValueError(f"No mean row in {path}")


def summary_method(path, key):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        name: float(values["mean"])
        for name, values in payload["methods"][key].items()
    }


def load_records():
    return [
        {
            "method": "Wavelet",
            "metrics": csv_mean(
                ROOT.parent / "comparation/results/qxslab_paper_1000/wavelet/metrics.csv"
            ),
            "source": "comparation/results/qxslab_paper_1000/wavelet/metrics.csv",
            "protocol": "paper1000",
        },
        {
            "method": "U2Fusion",
            "metrics": csv_mean(
                ROOT.parent / "comparation/results/qxslab_paper_1000/u2fusion/metrics.csv"
            ),
            "source": "comparation/results/qxslab_paper_1000/u2fusion/metrics.csv",
            "protocol": "paper1000",
        },
        {
            "method": "Laplacian",
            "metrics": csv_mean(
                ROOT.parent / "comparation/results/qxslab_paper_1000/laplacian/metrics.csv"
            ),
            "source": "comparation/results/qxslab_paper_1000/laplacian/metrics.csv",
            "protocol": "paper1000",
        },
        {
            "method": "CDDFuse零样本†",
            "metrics": csv_mean(
                ROOT / "test_result/QXSLAB_SAROPT_1000_zero_shot/metrics.csv"
            ),
            "source": "test_result/QXSLAB_SAROPT_1000_zero_shot/metrics.csv",
            "protocol": "historical_unknown_output_normalization",
        },
        {
            "method": "CDDFuse训练后†",
            "metrics": csv_mean(
                ROOT / "test_result/QXSLAB_SAROPT_1000/metrics.csv"
            ),
            "source": "test_result/QXSLAB_SAROPT_1000/metrics.csv",
            "protocol": "historical_unknown_output_normalization",
        },
        {
            "method": "B0",
            "metrics": csv_mean(
                ROOT / "test_result/reliability_v4/B0_clamp/metrics.csv"
            ),
            "source": "test_result/reliability_v4/B0_clamp/metrics.csv",
            "protocol": "paper_clamp",
        },
        {
            "method": "B0 + SPD",
            "metrics": summary_method(
                ROOT / "test_result/v21_softmask/spd_nomask_on_b0_test1000/summary.json",
                "e2_feature",
            ),
            "source": "test_result/v21_softmask/spd_nomask_on_b0_test1000/summary.json",
            "protocol": "paper_clamp",
        },
        {
            "method": "B0 + Detail-only SPD（本文）",
            "metrics": summary_method(
                ROOT / "test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/summary.json",
                "e2_feature",
            ),
            "source": "test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/summary.json",
            "protocol": "paper_clamp",
        },
        {
            "method": "C0-minimax",
            "metrics": csv_mean(
                ROOT / "test_result/reliability_v5/frequency_minimax_seed42/metrics.csv"
            ),
            "source": "test_result/reliability_v5/frequency_minimax_seed42/metrics.csv",
            "protocol": "paper_clamp",
        },
        {
            "method": "C0-minimax + SPD",
            "metrics": summary_method(
                ROOT / "test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/summary.json",
                "e2_feature",
            ),
            "source": "test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/summary.json",
            "protocol": "paper_clamp",
        },
    ]


def ranks(records, metric):
    ordered = sorted(
        ((record["metrics"][metric], index) for index, record in enumerate(records)),
        key=lambda item: item[0],
        reverse=metric not in LOWER_IS_BETTER,
    )
    return ordered[0][1], ordered[1][1]


def display_name(metric):
    names = {"Qabf": "QAB/F", "Nabf": "NAB/F", "MS_SSIM": "MS-SSIM"}
    return names.get(metric, metric)


def render(records):
    lines = [
        "# QXS-SAROPT test1000扩展对比",
        "",
        "所有行使用相同test1000图像 ID和PaperEvaluator。当前方法统一采用clamp→uint8。",
        "",
        "**粗体**表示该列最优，<u>下划线</u>表示该列第二名。NAB/F越低越好，其余越高越好。",
        "",
    ]
    for title, metrics in GROUPS:
        rank_map = {metric: ranks(records, metric) for metric in metrics}
        lines.extend(
            [
                f"## {title}",
                "",
                "| 方法 | "
                + " | ".join(
                    display_name(metric) + ("↓" if metric in LOWER_IS_BETTER else "↑")
                    for metric in metrics
                )
                + " |",
                "|---|" + "---:|" * len(metrics),
            ]
        )
        for index, record in enumerate(records):
            values = []
            for metric in metrics:
                text = f"{record['metrics'][metric]:.4f}"
                best, second = rank_map[metric]
                if index == best:
                    text = f"**{text}**"
                elif index == second:
                    text = f"<u>{text}</u>"
                values.append(text)
            lines.append(f"| {record['method']} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## 协议说明",
            "",
            "- Wavelet、U2Fusion、Laplacian来自`comparation/results/qxslab_paper_1000`，使用同一test1000清单与PaperEvaluator。",
            "- 带†的历史CDDFuse目录缺少`run_metadata.json`，其输出归一化策略无法完全追溯；数值可用于复现历史表格，但严格论文主表建议使用当前clamp重评的B0替代“CDDFuse训练后”。",
            "- B0、SPD、C0系列均来自可追溯的clamp→uint8正式结果目录。",
            "",
            "## 数据来源",
            "",
        ]
    )
    for record in records:
        lines.append(f"- {record['method']}：`{record['source']}`")
    return "\n".join(lines)


def main():
    records = load_records()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "extended_comparison_tables.md").write_text(
        render(records), encoding="utf-8"
    )
    with (OUTPUT / "extended_comparison_tables.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
