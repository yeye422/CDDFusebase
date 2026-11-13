"""Audit saved E2/interval images on the existing, fixed validation subset."""

import argparse
import csv
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from run_e2_matched_no_pac import register
from run_e2_source_interval_val import BASE_ANCHOR, DATA, INTERVAL
from run_e2_scratch_twostage import SOURCE_INTERVAL_PILOT
from run_qxs_reference_test1000 import read_image, read_json, write_json
from test_OS import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


OUTPUT = Path("test_result/e2_source_interval_tolerance_val64_v1")
GROUP = "QXS-E2-SOURCE-INTERVAL-V2-TOLERANCE-VAL64-v1"
BANDS = ("within_tolerance", "above_tolerance", "outside_support")
SUMMARY_METRICS = ("e2_nabf", "interval_nabf", "delta_nabf", *(f"delta_{band}" for band in BANDS))


def read_scores(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["image"] not in ("mean", "std")]
    if len({row["image"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate validation scenes: {path}")
    return {row["image"]: float(row["Nabf"]) for row in rows}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contribution_map(fused, optical_edges, sar_edges):
    gf, af = FusionEvaluator._kumar_edges(fused)
    numerator = np.zeros_like(gf)
    denominator = 0.0
    for magnitude, angle in (optical_edges, sar_edges):
        maximum = np.maximum(magnitude, gf)
        ratio = np.divide(np.minimum(magnitude, gf), maximum,
                          out=np.zeros_like(maximum), where=maximum > 0)
        angular = np.abs(np.abs(angle - af) - np.pi / 2) * 2 / np.pi
        qg = .9999 / (1 + np.exp(-19 * (ratio - .5)))
        qa = .9995 / (1 + np.exp(-22 * (angular - .5)))
        weight = np.where(magnitude >= 2, magnitude ** 1.5, .001)
        numerator += (1 - np.sqrt(qg * qa)) * weight
        denominator += weight.sum()
    cap = np.maximum(optical_edges[0], sar_edges[0])
    over = gf > cap
    return numerator * over / denominator, gf, cap, over


def analyze_scene(pair, image_paths, scores, quartile, spec, *, candidate_key="interval"):
    optical_path, sar_path = pair
    scene = optical_path.stem
    optical = cv2.cvtColor(read_image(optical_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sar = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
    if optical.shape != sar.shape:
        raise ValueError(f"Source shapes differ: {scene}")
    support = cv2.erode((sar > 0).astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    source_edges = tuple(FusionEvaluator._kumar_edges(image) for image in (optical, sar))
    row = dict(image=scene, baseline_nabf_quartile=quartile)
    hashes = {str(path): sha256_file(path) for path in pair}
    for method, directory in image_paths.items():
        path = directory / f"{scene}.png"
        image = read_image(path, cv2.IMREAD_UNCHANGED)
        if image.dtype != np.uint8 or image.shape != sar.shape:
            raise ValueError(f"Unexpected saved grayscale image: {path}")
        hashes[str(path)] = sha256_file(path)
        contributions, magnitude, cap, over = contribution_map(image, *source_edges)
        # Training uses unscaled Sobel on [0,1]. Kumar uses Sobel/8 on uint8.
        ceiling = spec["upper_ratio"] * cap + spec["upper_slack"] * 255 / 8
        masks = (over & support & (magnitude <= ceiling),
                 over & support & (magnitude > ceiling), over & ~support)
        value = float(contributions.sum())
        official = FusionEvaluator.Nabf(image, optical, sar)
        if not (np.isclose(value, official, rtol=1e-10, atol=1e-12)
                and np.isclose(value, scores[method][scene], rtol=1e-10, atol=1e-12)):
            raise ValueError(f"NAB/F differs from the evaluator or saved metrics: {method}/{scene}")
        if not np.array_equal(sum(mask.astype(np.uint8) for mask in masks), over.astype(np.uint8)):
            raise AssertionError("Overlapping or missing decomposition masks")
        row[f"{method}_nabf"] = value
        row[f"{method}_over_fraction"] = float(over.mean())
        for band, mask in zip(BANDS, masks):
            row[f"{method}_{band}"] = float(contributions[mask].sum())
            row[f"{method}_{band}_pixel_fraction"] = float(mask.mean())
    for name in ("nabf", *BANDS):
        row[f"delta_{name}"] = row[f"{candidate_key}_{name}"] - row[f"e2_{name}"]
    if not np.isclose(sum(row[f"delta_{band}"] for band in BANDS), row["delta_nabf"], atol=1e-12):
        raise AssertionError("Paired contribution changes do not sum to the NAB/F change")
    return row, hashes


def summarize(rows):
    groups = {"all64": rows, "low_half": [row for row in rows if row["baseline_nabf_quartile"] <= 2]}
    groups.update({f"quartile_{q}": [row for row in rows if row["baseline_nabf_quartile"] == q]
                   for q in range(1, 5)})
    summaries = []
    rng = np.random.default_rng(42)
    for name, selected in groups.items():
        if not selected:
            continue
        values = np.asarray([[row[key] for key in SUMMARY_METRICS] for row in selected])
        means = values.mean(axis=0)
        bootstrap = values[rng.integers(len(selected), size=(2000, len(selected)))].mean(axis=1)
        low, high = np.quantile(bootstrap, (.025, .975), axis=0)
        summary = dict(group=name, sample_count=len(selected),
                       worse_count=sum(row["delta_nabf"] > 0 for row in selected),
                       better_count=sum(row["delta_nabf"] < 0 for row in selected))
        for index, key in enumerate(SUMMARY_METRICS):
            summary[key] = float(means[index])
            summary[f"{key}_ci_low"] = float(low[index])
            summary[f"{key}_ci_high"] = float(high[index])
        summary["within_tolerance_delta_ratio"] = (
            summary["delta_within_tolerance"] / summary["delta_nabf"]
            if summary["delta_nabf"] > 0 else None)
        summaries.append(summary)
    return summaries


def write_report(path, summaries):
    overall = summaries[0]
    ratio = overall["within_tolerance_delta_ratio"]
    lines = ["固定 val64 的梯度过冲复核", "",
             "使用既有小试验证清单和已保存的 uint8 融合图，统计正式 Kumar NAB/F 的逐像素贡献。",
             "训练容差按原损失的 1.10 倍双源上限及 0.02 余量换算，3×3 SAR 有效邻域以外单独统计。",
             "贡献差使用带符号均值，其三部分之和等于 NAB/F 均值变化。正贡献占比可能超过 100%，表示其他部分抵消了部分增量。", "",
             "| 场景组 | 数量 | 变差数量 | E2 NAB/F | 区间 NAB/F | 差值 | 容差内贡献差 | 超容差贡献差 | 支持域外贡献差 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = dict(all64="固定64张", low_half="原E2低过冲半组",
                  **{f"quartile_{q}": f"原E2第{q}四分位组" for q in range(1, 5)})
    for row in summaries:
        values = [f"{row[key]:.6f}" for key in SUMMARY_METRICS]
        lines.append("| " + " | ".join((labels[row["group"]], str(row["sample_count"]),
                     str(row["worse_count"]), *values)) + " |")
    lines += ["", f"全部64张的配对差为 {overall['delta_nabf']:+.6f}，场景重采样95%区间为"
              f" [{overall['delta_nabf_ci_low']:+.6f}, {overall['delta_nabf_ci_high']:+.6f}]。"]
    if ratio is not None:
        lines.append(f"容差内贡献占总增量的 {ratio:.2%}。")
    if (ratio is not None and ratio > .5 and overall["delta_nabf_ci_low"] > 0
        and overall["delta_within_tolerance_ci_low"] > 0):
        lines.append("固定子集支持容差内过冲是退化的主要贡献来源，可以推进仅增加严格过冲惩罚的独立训练。")
    else:
        lines.append("固定子集对容差内过冲占主导的支持有限，应先检查其余贡献项。")
    lines += ["该判断采用预设的50%贡献阈值作研究筛选，置信区间描述固定子集的场景差异。",
              "结果基于已保存的量化输出，新的训练收益仍需独立验证。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Check only the first fixed scene")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    full_meta_path = INTERVAL.result / "run_metadata.json"
    full_meta = read_json(full_meta_path)
    subset = SOURCE_INTERVAL_PILOT.manifest
    subset_meta = read_json(subset)
    scene_ids = subset_meta["splits"]["val"]
    manifest_hash = sha256_file(INTERVAL.manifest)
    if (len(scene_ids) != 64 or len(set(scene_ids)) != 64
        or not set(scene_ids) <= set(full_meta["scene_ids"])
        or subset_meta["source_manifest_sha256"] != manifest_hash
        or full_meta["split_manifest_sha256"] != manifest_hash
        or full_meta["metric_profile"] != DEFAULT_PROFILE
        or full_meta["split"] != "val" or full_meta["offset"] != 0
        or full_meta["output_policy"] != "clamp_then_uint8"):
        raise ValueError("The fixed subset or saved evaluation protocol differs")
    checkpoints = dict(e2=BASE_ANCHOR.final, interval=INTERVAL.final)
    checkpoint_hashes = {key: sha256_file(path) for key, path in checkpoints.items()}
    if checkpoint_hashes != full_meta["checkpoint_hashes"]:
        raise ValueError("Checkpoints differ from the saved evaluation")
    paths = dict(e2=BASE_ANCHOR.result / "gray", interval=INTERVAL.result / "interval" / "gray")
    metric_paths = {key: INTERVAL.result / key / "metrics.csv" for key in paths}
    scores = {key: read_scores(path) for key, path in metric_paths.items()}
    if any(list(values) != full_meta["scene_ids"] for values in scores.values()):
        raise ValueError("Saved metric scene IDs differ from the metadata")
    ranked = sorted(scores["e2"], key=lambda scene: (scores["e2"][scene], scene))
    quartiles = {scene: min(4, 1 + index * 4 // len(ranked)) for index, scene in enumerate(ranked)}
    splits, _ = manifest_pairs(DATA, INTERVAL.manifest)
    pairs = {pair[0].stem: pair for pair in splits["val"]}
    config_path = INTERVAL.run_dir / "config.json"
    spec = read_json(config_path)["source_interval"]
    metadata = dict(scene_ids=scene_ids, split="val", offset=0, metric_profile=DEFAULT_PROFILE,
                    split_manifest=subset.as_posix(), split_manifest_sha256=sha256_file(subset),
                    source_manifest=INTERVAL.manifest.as_posix(), source_manifest_sha256=manifest_hash,
                    checkpoint_hashes=checkpoint_hashes, source_interval=spec, output_policy="saved_uint8",
                    diagnostic_source_sha256=sha256_file(Path(__file__)),
                    source_hashes={str(path): sha256_file(path) for path in
                                   (full_meta_path, config_path, *metric_paths.values())},
                    bootstrap_seed=42, bootstrap_samples=2000, majority_threshold=.5)
    if args.smoke:
        row, _ = analyze_scene(pairs[scene_ids[0]], paths, scores, quartiles[scene_ids[0]], spec)
        print(row)
        return
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != metadata:
            raise FileExistsError(f"Preserve output with a different protocol: {OUTPUT}")
        for path, digest in read_json(OUTPUT / "input_hashes.json").items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"Saved input changed: {path}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".interval-audit-", dir=OUTPUT.parent) as temporary:
            work = Path(temporary) / "results"
            work.mkdir()
            rows, hashes = [], {}
            for index, scene in enumerate(scene_ids, 1):
                row, input_hashes = analyze_scene(pairs[scene], paths, scores, quartiles[scene], spec)
                rows.append(row)
                hashes.update(input_hashes)
                print(f"saved-image audit {index}/64: {scene}", flush=True)
            summaries = summarize(rows)
            write_csv(work / "per_image.csv", rows)
            write_csv(work / "summary.csv", summaries)
            write_report(work / "REPORT.md", summaries)
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "input_hashes.json", hashes)
            work.rename(OUTPUT)
    with (OUTPUT / "summary.csv").open(encoding="utf-8-sig", newline="") as file:
        overall = next(row for row in csv.DictReader(file) if row["group"] == "all64")
    metrics = {key: float(overall[key]) for key in SUMMARY_METRICS}
    register(dict(id=GROUP, method="E2 梯度区间v2固定val64过冲分解", dataset="QXS-SAROPT", task="fusion",
                  status="diagnostic_fixed_subset", comparable_group=GROUP, split="val", offset=0, sample_count=64,
                  split_manifest=subset.as_posix(), split_manifest_sha256=metadata["split_manifest_sha256"],
                  checkpoint=INTERVAL.final.as_posix(), checkpoint_sha256=checkpoint_hashes["interval"],
                  parent_checkpoint=BASE_ANCHOR.final.as_posix(), parent_checkpoint_sha256=checkpoint_hashes["e2"],
                  metric_profile=DEFAULT_PROFILE, output_policy="saved_uint8", result_dir=OUTPUT.as_posix(),
                  metadata_path=(OUTPUT / "run_metadata.json").as_posix(), metrics=metrics,
                  metric_source=dict(type="csv_row", path=(OUTPUT / "summary.csv").as_posix(),
                                     match={"group": "all64"}, columns={key: key for key in metrics})))
    print((OUTPUT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
