"""Validate EXPERIMENT_REGISTRY.json and render EXPERIMENT_INDEX.md."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "EXPERIMENT_REGISTRY.json"
INDEX_PATH = ROOT / "EXPERIMENT_INDEX.md"
OUTPUT_LOG_PATH = ROOT / "TEST_OUTPUT_LOG.md"
FUSION_METRICS = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
PAPER_METRICS = (
    "EN", "MI", "FMI", "PSNR", "Qabf", "Nabf", "AG", "SF",
    "SD", "CC", "SCD", "SSIM", "MS_SSIM", "VIF",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path):
    return (ROOT / path).resolve()


def csv_mean(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            first = next(iter(row.values()))
            if first == "mean":
                return {key: float(value) for key, value in row.items() if key != "image"}
    raise ValueError(f"No mean row in {path}")


def extract_metrics(entry):
    source = entry.get("metric_source")
    if not source:
        return {}
    path = resolve(source["path"])
    if source["type"] == "csv_mean":
        return csv_mean(path)
    if source["type"] == "csv_row":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            row = next(
                row for row in rows
                if all(str(row.get(key)) == str(value)
                       for key, value in source["match"].items())
            )
        return {
            name: float(row[column])
            for name, column in source["columns"].items()
        }
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if source["type"] == "json_paths":
        extracted = {}
        for name, keys in source["paths"].items():
            value = payload
            for key in keys:
                value = value[key]
            extracted[name] = float(value)
        return extracted
    if source["type"] == "summary_method":
        method = payload["methods"][source["method"]]
        return {name: values["mean"] for name, values in method.items()}
    if source["type"] == "summary_spd":
        return {
            "PSNR": payload["metrics"]["psnr"]["spd_mean"],
            "SSIM": payload["metrics"]["ssim"]["spd_mean"],
            "MAE": payload["metrics"]["mae"]["spd_mean"],
            "edge_MAE": payload["metrics"]["edge_mae"]["spd_mean"],
            "edge_correlation": payload["metrics"]["edge_correlation"]["spd_mean"],
        }
    if source["type"] == "summary_flat":
        return {
            name: values["mean"]
            for name, values in payload["summary"].items()
            if isinstance(values, dict) and "mean" in values
        }
    if source["type"] == "summary_scale":
        target = next(
            row for row in payload["summary"]
            if row["scale_id"] == source["scale_id"]
        )
        return {
            name: target[f"{name}_mean"]
            for name in PAPER_METRICS
            if f"{name}_mean" in target
        }
    raise ValueError(f"Unknown metric source type: {source['type']}")


def validate(registry):
    errors = []
    ids = set()
    for entry in registry["experiments"]:
        experiment_id = entry["id"]
        if experiment_id in ids:
            errors.append(f"duplicate id: {experiment_id}")
        ids.add(experiment_id)
        for key in ("result_dir", "metadata_path"):
            path = resolve(entry[key])
            if not path.exists():
                errors.append(f"{experiment_id}: missing {key} {path}")
        source = entry.get("metric_source")
        if source and not resolve(source["path"]).exists():
            errors.append(f"{experiment_id}: missing metric source {source['path']}")
        for prefix in ("", "parent_"):
            path_key = f"{prefix}checkpoint"
            hash_key = f"{prefix}checkpoint_sha256"
            if path_key not in entry:
                continue
            path = resolve(entry[path_key])
            if not path.exists():
                errors.append(f"{experiment_id}: missing {path_key} {path}")
            elif hash_key in entry:
                actual = sha256_file(path)
                if actual != entry[hash_key]:
                    errors.append(
                        f"{experiment_id}: {hash_key} mismatch "
                        f"expected={entry[hash_key]} actual={actual}"
                    )
        manifest = entry.get("split_manifest")
        expected_manifest_hash = entry.get("split_manifest_sha256")
        if manifest:
            path = resolve(manifest)
            if not path.exists():
                errors.append(f"{experiment_id}: missing split manifest {path}")
            elif expected_manifest_hash:
                actual = sha256_file(path)
                if actual != expected_manifest_hash:
                    errors.append(
                        f"{experiment_id}: split manifest hash mismatch "
                        f"expected={expected_manifest_hash} actual={actual}"
                    )
        try:
            extracted = extract_metrics(entry)
        except Exception as error:
            errors.append(f"{experiment_id}: metric extraction failed: {error}")
            continue
        for name, expected in entry.get("metrics", {}).items():
            if name not in extracted:
                errors.append(f"{experiment_id}: metric {name} absent from source")
            elif abs(float(extracted[name]) - float(expected)) > 5e-6:
                errors.append(
                    f"{experiment_id}: metric {name} mismatch "
                    f"registry={expected} source={extracted[name]}"
                )
    return errors


def format_value(value):
    return f"{value:.4f}" if value is not None else "—"


def render(registry):
    groups = defaultdict(list)
    for entry in registry["experiments"]:
        groups[entry["comparable_group"]].append(entry)
    lines = [
        "# 实验数据索引",
        "",
        f"更新时间：{registry['updated_at']}  ",
        f"Schema：v{registry['schema_version']}",
        "",
        "## 使用规则",
        "",
        "- 只有 `comparable_group` 完全相同的实验才能直接进入同一张定量对比表。",
        "- 新实验必须先写入 `EXPERIMENT_REGISTRY.json`，再执行校验和渲染。",
        "- `test`、`val`、offset、样本数、checkpoint、评价器或输出策略任一不同，都必须使用不同分组。",
        "- Markdown 仅用于阅读；JSON 注册表和原始结果文件才是事实来源。",
        "",
        "校验命令：",
        "",
        "```cmd",
        "cd /d D:\\#imageInfusion\\CDDFuse",
        ".venv\\Scripts\\python.exe maintain_experiment_index.py --check --render",
        "```",
        "",
    ]
    for group_name in sorted(groups):
        entries = groups[group_name]
        lines.extend([f"## {group_name}", ""])
        if all(entry["task"] == "fusion" for entry in entries):
            lines.extend(
                [
                    "| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for entry in entries:
                metrics = entry.get("metrics", {})
                split = f"{entry['split']}@{entry['offset']}"
                lines.append(
                    "| {id} | {method} | {status} | {split} | {n} | {values} |".format(
                        id=entry["id"],
                        method=entry["method"],
                        status=entry["status"],
                        split=split,
                        n=entry["sample_count"],
                        values=" | ".join(
                            format_value(metrics.get(name)) for name in FUSION_METRICS
                        ),
                    )
                )
        else:
            lines.extend(
                [
                    "| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |",
                    "|---|---|---|---|---:|---|",
                ]
            )
            for entry in entries:
                metrics = ", ".join(
                    f"{name}={format_value(value)}"
                    for name, value in entry.get("metrics", {}).items()
                )
                lines.append(
                    f"| {entry['id']} | {entry['method']} | {entry['status']} | "
                    f"{entry['split']}@{entry['offset']} | {entry['sample_count']} | {metrics} |"
                )
        lines.append("")
        lines.append("来源：")
        lines.append("")
        for entry in entries:
            lines.append(
                f"- `{entry['id']}`：`{entry['metric_source']['path']}`；"
                f"checkpoint `{entry['checkpoint']}`。"
            )
        lines.append("")
    lines.extend(
        [
            "## 已追溯的用户表格",
            "",
            "用户给出的 B0、B1-v5、C0-minimax 六项指标均来自 "
            "`QXS-TEST1000-PAPER-CLAMP-v1`，可以直接相互比较。",
            "当前 SPD 只有 val200 开发结果，不能直接追加到该正式 test1000 表中；"
            "必须在模型冻结后单独执行 test1000 并登记为同一组。",
            "",
        ]
    )
    return "\n".join(lines)


def render_output_log(registry):
    grouped = {}
    for entry in registry["experiments"]:
        grouped.setdefault(entry["result_dir"], []).append(entry)
    lines = [
        "# 测试输出目录记录",
        "",
        f"更新时间：{registry['updated_at']}  ",
        "来源：`EXPERIMENT_REGISTRY.json`（自动生成，请勿手工修改表格）",
        "",
        "## 维护规则",
        "",
        "- 每次模型、损失、推理路径、数据切分或评价协议发生变化，必须使用新的输出目录。",
        "- 定量结果生成后先登记 `EXPERIMENT_REGISTRY.json`，再运行维护命令。",
        "- 同一目录可包含多个同步评价路径，例如 `c0_raw/e1_image/e2_feature`。",
        "- Smoke test 使用 `tmp/`，不进入正式结果表；若用于研究结论，必须移入 `test_result/` 并登记。",
        "- 不允许覆盖已有正式结果目录。",
        "",
        "维护命令：",
        "",
        "```cmd",
        "cd /d D:\\#imageInfusion\\CDDFuse",
        ".venv\\Scripts\\python.exe maintain_experiment_index.py --check --render",
        "```",
        "",
        "## 已登记测试输出",
        "",
        "| 输出目录 | 实验/路径 | 状态 | 数据协议 | N | 可比组 |",
        "|---|---|---|---|---:|---|",
    ]
    for result_dir, entries in grouped.items():
        methods = "<br>".join(
            f"`{entry['id']}`：{entry['method']}" for entry in entries
        )
        statuses = "<br>".join(dict.fromkeys(entry["status"] for entry in entries))
        protocols = "<br>".join(
            dict.fromkeys(
                f"{entry['dataset']} / {entry['split']}@{entry['offset']}"
                for entry in entries
            )
        )
        counts = ", ".join(
            str(value) for value in dict.fromkeys(entry["sample_count"] for entry in entries)
        )
        groups = "<br>".join(
            dict.fromkeys(entry["comparable_group"] for entry in entries)
        )
        lines.append(
            f"| `{result_dir}` | {methods} | {statuses} | {protocols} | {counts} | {groups} |"
        )
    lines.extend(
        [
            "",
            "## 当前 SPD 主线目录",
            "",
            "| 阶段 | 目录 | 说明 |",
            "|---|---|---|",
            "| 原始 SPD 训练 | `models/v21_softmask/spd_nomask_seed42` | Base+Detail SPD |",
            "| 新主模型训练 | `models/v21_softmask/spd_detail_only_seed42` | 训练阶段删除 Base SPD |",
            "| 原始 SPD 去噪确认 | `test_result/v21_softmask/spd_nomask_seed42_internal_test50` | Toronto internal50 |",
            "| 新主模型去噪确认 | `test_result/v21_softmask/spd_detail_only_seed42_internal_test50` | Toronto internal50 |",
            "| B0+完整SPD | `test_result/v21_softmask/spd_nomask_on_b0_test1000` | 正式 test1000 |",
            "| 旧 Detail-only | `test_result/v21_softmask/spd_detail_only_on_b0_test1000` | 完整SPD训练后推理屏蔽Base |",
            "| 新 Detail-only val | `test_result/v21_softmask/spd_detail_only_retrained_on_b0_val1000` | 新模型 validation1000 |",
            "| 新 Detail-only test | `test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000` | 当前正式主结果 |",
            "| 强度扫描 | `test_result/v21_softmask/spd_scale_sweep_b0_val1000` | 选择 Detail-only |",
            "| 可视化比较 | `test_result/v21_softmask/visual_comparison` | 拼图和放大差分 |",
            "| 五方法14指标汇总 | `test_result/v21_softmask/five_method_test1000_comparison` | 自动聚合正式test1000 |",
            "",
            "## 未登记的工程输出",
            "",
            "`tmp/` 下的 `*_smoke`、`*_dry_run` 目录只验证接口和单样本前向，不作为研究结果，不得与正式目录合表。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    errors = validate(registry)
    if errors:
        print("Experiment registry validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Experiment registry valid: {len(registry['experiments'])} entries")
    if args.render:
        INDEX_PATH.write_text(render(registry), encoding="utf-8")
        print(f"Rendered {INDEX_PATH}")
        OUTPUT_LOG_PATH.write_text(render_output_log(registry), encoding="utf-8")
        print(f"Rendered {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    main()
