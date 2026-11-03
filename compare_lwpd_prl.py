"""Paired validation comparison for E2 and the PR-lifting candidate."""

import argparse
import csv
import math
from pathlib import Path


METRICS = (
    "EN",
    "MI",
    "FMI",
    "PSNR",
    "Qabf",
    "Nabf",
    "AG",
    "SF",
    "SD",
    "CC",
    "SCD",
    "SSIM",
    "MS_SSIM",
    "VIF",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    samples = {
        row["image"]: row
        for row in rows
        if row["image"] not in ("mean", "std")
    }
    means = next(row for row in rows if row["image"] == "mean")
    return samples, means


def paired_record(metric, baseline_rows, candidate_rows, baseline_mean, candidate_mean):
    common = sorted(set(baseline_rows) & set(candidate_rows))
    if len(common) != len(baseline_rows) or len(common) != len(candidate_rows):
        raise ValueError(f"Image sets differ for paired comparison: {metric}")
    direction = -1.0 if metric == "Nabf" else 1.0
    differences = [
        direction
        * (
            float(candidate_rows[image][metric])
            - float(baseline_rows[image][metric])
        )
        for image in common
    ]
    mean_difference = sum(differences) / len(differences)
    variance = sum(
        (value - mean_difference) ** 2 for value in differences
    ) / (len(differences) - 1)
    half_width = 1.96 * math.sqrt(variance / len(differences))
    base = float(baseline_mean[metric])
    candidate = float(candidate_mean[metric])
    return {
        "metric": metric,
        "direction": "down" if metric == "Nabf" else "up",
        "baseline": base,
        "candidate": candidate,
        "raw_change": candidate - base,
        "beneficial_change": direction * (candidate - base),
        "beneficial_relative_percent": (
            100.0 * direction * (candidate - base) / abs(base)
        ),
        "paired_ci_low": mean_difference - half_width,
        "paired_ci_high": mean_difference + half_width,
        "paired_win_percent": 100.0
        * sum(value > 0.0 for value in differences)
        / len(differences),
        "samples": len(differences),
    }


def screening(records):
    by_metric = {record["metric"]: record for record in records}
    mi_gain = by_metric["MI"]["beneficial_relative_percent"] >= 0.5
    fmi_gain = by_metric["FMI"]["beneficial_relative_percent"] >= 0.3
    checks = {
        "MI提高至少0.5%或FMI提高至少0.3%": mi_gain or fmi_gain,
        "QAB/F不低于0.7320": by_metric["Qabf"]["candidate"] >= 0.7320,
        "NAB/F不高于0.0660": by_metric["Nabf"]["candidate"] <= 0.0660,
        "VIF不低于E2": by_metric["VIF"]["candidate"] >= by_metric["VIF"]["baseline"],
        "SSIM不低于E2": by_metric["SSIM"]["candidate"] >= by_metric["SSIM"]["baseline"],
        "MS-SSIM不低于E2": by_metric["MS_SSIM"]["candidate"] >= by_metric["MS_SSIM"]["baseline"],
    }
    return checks


def main():
    args = parse_args()
    baseline_rows, baseline_mean = read_metrics(args.baseline)
    candidate_rows, candidate_mean = read_metrics(args.candidate)
    records = [
        paired_record(
            metric,
            baseline_rows,
            candidate_rows,
            baseline_mean,
            candidate_mean,
        )
        for metric in METRICS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    checks = screening(records)
    by_metric = {record["metric"]: record for record in records}
    lines = [
        "# E2 与 E2-PRL 配对验证摘要",
        "",
        "正向变化已经按指标方向统一；NAB/F为越低越好。95%区间为逐图配对均值差的正态近似区间。",
        "",
        "| 指标 | E2 | E2-PRL | 有利相对变化 | 95%区间 | 改善图像比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ("MI", "FMI", "Qabf", "Nabf", "VIF", "SSIM", "MS_SSIM"):
        record = by_metric[metric]
        lines.append(
            "| {metric} | {baseline:.6f} | {candidate:.6f} | {relative:+.3f}% | "
            "[{low:+.6f}, {high:+.6f}] | {win:.1f}% |".format(
                metric=metric,
                baseline=record["baseline"],
                candidate=record["candidate"],
                relative=record["beneficial_relative_percent"],
                low=record["paired_ci_low"],
                high=record["paired_ci_high"],
                win=record["paired_win_percent"],
            )
        )
    lines.extend(("", "## 单种子筛选门槛", ""))
    for label, passed in checks.items():
        lines.append(f"- {'通过' if passed else '未通过'}：{label}")
    lines.extend(
        (
            "",
            f"单种子筛选结论：{'通过' if all(checks.values()) else '未通过'}。",
            "该结论不替代多种子复核，也不授权根据test1000选择结构。",
        )
    )
    markdown_path = args.output_dir / "SUMMARY.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {csv_path.resolve()}")
    print(f"wrote: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()

