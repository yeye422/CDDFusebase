"""Causal diagnostics for R1 SAR-detail spatial modulation.

This script performs no training. It reuses one trained R1 checkpoint and
evaluates fixed interventions on the same paired validation images:

- zero modulation (exact C0 path inside the frozen R1 checkpoint)
- scaled learned modulation: beta = 0.25, 0.50, 0.75, 1.00, 1.25
- inverted modulation: beta = -1.00
- same-mean constant modulation
- same-histogram spatial shuffle
- spatially smoothed modulation

It also writes per-image learned-vs-zero deltas so the R1 effect can be
distinguished from a mean-only change or a small number of outlier scenes.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from test_OS import (
    PAPER_METRIC_NAMES,
    evaluate,
    load_variant_models,
    normalize_fused_tensor,
)
from utils.dataset_os import manifest_pairs


PRIMARY_METRICS = ("MI", "FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")
HIGHER_IS_BETTER = {"MI", "FMI", "Qabf", "SSIM", "MS_SSIM", "VIF"}

SCALE_CONDITIONS = (
    ("zero", 0.0),
    ("beta_0.25", 0.25),
    ("beta_0.50", 0.50),
    ("beta_0.75", 0.75),
    ("learned", 1.0),
    ("beta_1.25", 1.25),
    ("invert", -1.0),
)
LAYOUT_CONDITIONS = ("mean", "shuffle", "smooth")
ALL_CONDITIONS = tuple(name for name, _ in SCALE_CONDITIONS) + LAYOUT_CONDITIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--smooth-kernel", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def tensor_stats(modulation):
    return {
        "modulation_mean": float(modulation.mean().cpu()),
        "modulation_std": float(modulation.std(unbiased=False).cpu()),
        "modulation_abs_mean": float(modulation.abs().mean().cpu()),
        "modulation_min": float(modulation.amin().cpu()),
        "modulation_max": float(modulation.amax().cpu()),
        "modulation_positive_fraction": float((modulation > 0).float().mean().cpu()),
        "modulation_negative_fraction": float((modulation < 0).float().mean().cpu()),
    }


def shuffled_like(modulation, seed):
    values = modulation.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values.reshape(-1)).reshape(values.shape)
    return torch.from_numpy(shuffled).to(
        device=modulation.device,
        dtype=modulation.dtype,
    )


def smooth_like(modulation, kernel_size):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("smooth-kernel must be a positive odd integer")
    if kernel_size == 1:
        return modulation
    pad = kernel_size // 2
    padded = F.pad(modulation, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def build_interventions(learned, seed, smooth_kernel):
    interventions = {
        name: learned * float(beta)
        for name, beta in SCALE_CONDITIONS
    }
    interventions["mean"] = torch.full_like(learned, learned.mean())
    interventions["shuffle"] = shuffled_like(learned, seed)
    interventions["smooth"] = smooth_like(learned, smooth_kernel)

    reference_sorted = torch.sort(learned.flatten().float()).values
    shuffled_sorted = torch.sort(interventions["shuffle"].flatten().float()).values
    if not torch.equal(reference_sorted, shuffled_sorted):
        if not torch.allclose(reference_sorted, shuffled_sorted, atol=0.0, rtol=0.0):
            raise AssertionError("Shuffle intervention did not preserve modulation histogram")
    return interventions


def load_pair(optical_path, sar_path, device):
    optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar_u8 is None:
        raise RuntimeError(f"Could not read pair {optical_path.stem}")
    if optical_bgr.shape[:2] != sar_u8.shape:
        raise RuntimeError(
            f"Shape mismatch for {optical_path.stem}: "
            f"optical={optical_bgr.shape[:2]}, SAR={sar_u8.shape}"
        )
    optical_y_u8 = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    optical = torch.from_numpy(
        optical_y_u8.astype(np.float32) / 255.0
    )[None, None].to(device)
    sar = torch.from_numpy(
        sar_u8.astype(np.float32) / 255.0
    )[None, None].to(device)
    return optical, sar, optical_y_u8, sar_u8


def forward_from_modulation(models, optical, sar, modulation):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    modulated_sar = (1.0 + modulation) * sar_detail
    fused_detail = models["detail_fuse"](optical_detail + modulated_sar)
    fused, _ = models["decoder"](optical, fused_base, fused_detail)
    return fused


def extract_features_and_modulation(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    _, diagnostics = models["detail_modulator"](
        optical_detail,
        sar_detail,
        return_maps=True,
    )
    return (
        optical_detail,
        sar_detail,
        fused_base,
        diagnostics["r1_modulation_map"],
    )


def forward_with_cached_features(
    models,
    optical,
    optical_detail,
    sar_detail,
    fused_base,
    modulation,
):
    modulated_sar = (1.0 + modulation) * sar_detail
    fused_detail = models["detail_fuse"](optical_detail + modulated_sar)
    fused, _ = models["decoder"](optical, fused_base, fused_detail)
    return fused


def metric_dict(fused_tensor, optical_y_u8, sar_u8):
    fused_tensor = normalize_fused_tensor(fused_tensor, "clamp")
    fused_u8 = np.clip(
        np.rint(fused_tensor.squeeze().cpu().numpy() * 255.0),
        0,
        255,
    ).astype(np.uint8)
    values = evaluate(
        fused_u8.astype(np.float32),
        optical_y_u8.astype(np.float32),
        sar_u8.astype(np.float32),
        "paper",
    )
    return {
        name: float(value)
        for name, value in zip(PAPER_METRIC_NAMES, values)
    }


def condition_summary(rows):
    grouped = {condition: [] for condition in ALL_CONDITIONS}
    for row in rows:
        grouped[row["condition"]].append(row)

    zero = grouped["zero"]
    zero_means = {
        metric: float(np.mean([row[metric] for row in zero]))
        for metric in PAPER_METRIC_NAMES
    }
    summary = []
    for condition in ALL_CONDITIONS:
        subset = grouped[condition]
        result = {
            "condition": condition,
            "images": len(subset),
        }
        for metric in PAPER_METRIC_NAMES:
            mean = float(np.mean([row[metric] for row in subset]))
            result[metric] = mean
            result[f"delta_{metric}_vs_zero"] = mean - zero_means[metric]
        summary.append(result)
    return summary


def paired_rows(all_rows):
    by_image = {}
    for row in all_rows:
        by_image.setdefault(row["image"], {})[row["condition"]] = row

    rows = []
    for image, conditions in by_image.items():
        if "zero" not in conditions or "learned" not in conditions:
            raise RuntimeError(f"Missing zero/learned condition for {image}")
        zero = conditions["zero"]
        learned = conditions["learned"]
        row = {
            "image": image,
            "modulation_mean": learned["learned_modulation_mean"],
            "modulation_std": learned["learned_modulation_std"],
            "modulation_abs_mean": learned["learned_modulation_abs_mean"],
            "modulation_min": learned["learned_modulation_min"],
            "modulation_max": learned["learned_modulation_max"],
            "modulation_positive_fraction": learned[
                "learned_modulation_positive_fraction"
            ],
            "modulation_negative_fraction": learned[
                "learned_modulation_negative_fraction"
            ],
        }
        for metric in PRIMARY_METRICS:
            row[f"c0_{metric}"] = zero[metric]
            row[f"r1_{metric}"] = learned[metric]
            row[f"delta_{metric}"] = learned[metric] - zero[metric]
        rows.append(row)
    return rows


def safe_corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def paired_summary(rows):
    result = {
        "images": len(rows),
        "primary_metrics": {},
    }
    abs_mean = [row["modulation_abs_mean"] for row in rows]
    mod_std = [row["modulation_std"] for row in rows]
    mod_mean = [row["modulation_mean"] for row in rows]
    for metric in PRIMARY_METRICS:
        deltas = np.asarray(
            [row[f"delta_{metric}"] for row in rows],
            dtype=np.float64,
        )
        if metric in HIGHER_IS_BETTER:
            improved = deltas > 0
            worsened = deltas < 0
        else:
            improved = deltas < 0
            worsened = deltas > 0
        result["primary_metrics"][metric] = {
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "q25_delta": float(np.quantile(deltas, 0.25)),
            "q75_delta": float(np.quantile(deltas, 0.75)),
            "improved_fraction": float(improved.mean()),
            "worsened_fraction": float(worsened.mean()),
            "unchanged_fraction": float((deltas == 0).mean()),
            "corr_delta_vs_mod_abs_mean": safe_corr(deltas, abs_mean),
            "corr_delta_vs_mod_std": safe_corr(deltas, mod_std),
            "corr_delta_vs_mod_mean": safe_corr(deltas, mod_mean),
        }
    return result


def intervention_vs_zero(all_rows):
    by_image = {}
    for row in all_rows:
        by_image.setdefault(row["image"], {})[row["condition"]] = row

    records = []
    for condition in ALL_CONDITIONS:
        if condition == "zero":
            continue
        for metric in PRIMARY_METRICS:
            deltas = []
            for conditions in by_image.values():
                deltas.append(
                    conditions[condition][metric] - conditions["zero"][metric]
                )
            values = np.asarray(deltas, dtype=np.float64)
            if metric in HIGHER_IS_BETTER:
                improved = values > 0
            else:
                improved = values < 0
            records.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean_delta_vs_zero": float(values.mean()),
                    "median_delta_vs_zero": float(np.median(values)),
                    "q25_delta_vs_zero": float(np.quantile(values, 0.25)),
                    "q75_delta_vs_zero": float(np.quantile(values, 0.75)),
                    "improved_fraction_vs_zero": float(improved.mean()),
                }
            )
    return records


def top_bottom_fmi(rows, count=20):
    ordered = sorted(rows, key=lambda row: row["delta_FMI"])
    selected = ordered[:count] + ordered[-count:]
    output = []
    for rank, row in enumerate(ordered[:count], start=1):
        output.append({"group": "bottom", "rank": rank, **row})
    for rank, row in enumerate(reversed(ordered[-count:]), start=1):
        output.append({"group": "top", "rank": rank, **row})
    return output


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, variant = load_variant_models(
        args.checkpoint,
        device,
        "r1_spatial_modulation",
    )
    if variant != "r1_spatial_modulation":
        raise ValueError(f"Checkpoint is not R1: {variant}")

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    selected = pairs[args.offset : args.offset + args.limit]
    if not selected:
        raise RuntimeError("No image pairs selected")

    rows = []
    with torch.no_grad():
        for index, (optical_path, sar_path) in enumerate(selected, start=1):
            optical, sar, optical_y_u8, sar_u8 = load_pair(
                optical_path,
                sar_path,
                device,
            )
            (
                optical_detail,
                sar_detail,
                fused_base,
                learned,
            ) = extract_features_and_modulation(models, optical, sar)
            interventions = build_interventions(
                learned,
                args.seed + index,
                args.smooth_kernel,
            )

            learned_stats = {
                f"learned_{name}": value
                for name, value in tensor_stats(learned).items()
            }
            for condition in ALL_CONDITIONS:
                modulation = interventions[condition]
                intervention_stats = {
                    f"intervention_{name}": value
                    for name, value in tensor_stats(modulation).items()
                }
                fused = forward_with_cached_features(
                    models,
                    optical,
                    optical_detail,
                    sar_detail,
                    fused_base,
                    modulation,
                )
                metrics = metric_dict(fused, optical_y_u8, sar_u8)
                row = {
                    "image": optical_path.stem,
                    "condition": condition,
                    **metrics,
                    **learned_stats,
                    **intervention_stats,
                }
                rows.append(row)

            if index == 1 or index % 25 == 0 or index == len(selected):
                print(
                    f"processed {index}/{len(selected)}: {optical_path.stem}",
                    flush=True,
                )

    summary = condition_summary(rows)
    paired = paired_rows(rows)
    pair_summary = paired_summary(paired)
    effects = intervention_vs_zero(rows)
    extremes = top_bottom_fmi(paired, count=min(20, len(paired) // 2))

    write_csv(args.output_dir / "intervention_metrics.csv", rows)
    write_csv(args.output_dir / "intervention_summary.csv", summary)
    write_csv(args.output_dir / "paired_learned_vs_zero.csv", paired)
    write_csv(args.output_dir / "intervention_effects_vs_zero.csv", effects)
    write_csv(args.output_dir / "top_bottom_fmi.csv", extremes)

    config = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split_manifest": str(args.split_manifest),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "seed": args.seed,
        "smooth_kernel": args.smooth_kernel,
        "conditions": list(ALL_CONDITIONS),
        "scale_conditions": {
            name: beta for name, beta in SCALE_CONDITIONS
        },
        "metric_profile": "paper",
        "output_range_policy": "clamp",
        "notes": [
            "zero is the exact unmodulated C0 detail path inside the frozen R1 checkpoint.",
            "shuffle preserves the learned modulation histogram per image.",
            "mean removes spatial layout while preserving the per-image modulation mean.",
            "smooth retains coarse layout while removing fine-scale spatial variation.",
            "invert tests whether the learned modulation direction is reversed.",
        ],
    }
    with (args.output_dir / "diagnosis_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "paired_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(pair_summary, handle, ensure_ascii=False, indent=2)

    print("")
    print("=== intervention means: primary metrics ===")
    for row in summary:
        values = ", ".join(
            f"{metric}={row[metric]:.6f}"
            for metric in PRIMARY_METRICS
        )
        print(f"{row['condition']}: {values}")
    print("")
    print("=== learned vs zero paired summary ===")
    for metric in PRIMARY_METRICS:
        item = pair_summary["primary_metrics"][metric]
        print(
            f"{metric}: mean_delta={item['mean_delta']:+.6f}, "
            f"median_delta={item['median_delta']:+.6f}, "
            f"improved_fraction={item['improved_fraction']:.3f}"
        )
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
