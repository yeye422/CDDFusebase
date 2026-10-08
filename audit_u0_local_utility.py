"""U0: no-training, within-image SAR-detail marginal-utility feasibility audit.

Each spatial tile receives a compact, tapered multiplicative intervention on SAR
detail only. All C0 weights and the optical/base paths are held fixed. Scores
are LOCAL reference-free fusion proxies, not ground-truth perceptual utility or
training labels. Discovery should use train split; replicate without selection
on held-out val before considering U1.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import forward_variant, load_variant_models, normalize_fused_tensor
from utils.Evaluator import PaperEvaluator
from utils.dataset_os import manifest_pairs, validate_manifest_splits


METRIC_FUNCTIONS = {
    "Qabf": PaperEvaluator.Qabf,
    "Nabf": PaperEvaluator.Nabf,
    "FMI": PaperEvaluator.FMI,
    "VIF": PaperEvaluator.VIFF,
    "SSIM": PaperEvaluator.SSIM,
}
HIGHER_IS_BETTER = ("Qabf", "FMI", "VIF", "SSIM")
METRICS = tuple(METRIC_FUNCTIONS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--ramp", type=int, default=8)
    parser.add_argument("--min-tile-size", type=int, default=48)
    parser.add_argument("--min-sar-valid", type=float, default=0.9)
    parser.add_argument(
        "--direction-tolerance",
        type=float,
        default=0.0002,
        help="Minimum Qabf plus/minus contrast before assigning a direction.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tiles(height, width, grid, min_tile_size):
    if grid < 2:
        raise ValueError("grid must be >= 2")
    ys = np.linspace(0, height, grid + 1, dtype=int)
    xs = np.linspace(0, width, grid + 1, dtype=int)
    result = []
    for row in range(grid):
        for col in range(grid):
            top, bottom = int(ys[row]), int(ys[row + 1])
            left, right = int(xs[col]), int(xs[col + 1])
            if min(bottom - top, right - left) < min_tile_size:
                raise ValueError(
                    "Tile smaller than minimum: "
                    f"image={height}x{width}, grid={grid}, "
                    f"tile={bottom-top}x{right-left}, minimum={min_tile_size}"
                )
            result.append((row, col, top, bottom, left, right))
    return result


def cosine_edge_weights(size, ramp):
    if ramp < 0 or ramp * 2 >= size:
        raise ValueError("ramp must be >= 0 and smaller than half the tile")
    weights = np.ones(size, dtype=np.float32)
    if ramp:
        # First and last pixel have zero weight, avoiding a hard tile boundary.
        transition = np.sin(
            np.linspace(0.0, np.pi / 2.0, ramp + 1, dtype=np.float32)
        )[:-1] ** 2
        weights[:ramp] = transition
        weights[-ramp:] = transition[::-1]
    return weights


def tile_mask(shape, box, ramp, device):
    height, width = shape
    _, _, top, bottom, left, right = box
    wy = cosine_edge_weights(bottom - top, ramp)
    wx = cosine_edge_weights(right - left, ramp)
    value = np.zeros((height, width), dtype=np.float32)
    value[top:bottom, left:right] = wy[:, None] * wx[None, :]
    return torch.from_numpy(value)[None, None].to(device)


def load_pair(optical_path, sar_path, device):
    optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar_u8 is None:
        raise RuntimeError(f"Cannot read pair {optical_path.stem}")
    if optical_bgr.shape[:2] != sar_u8.shape:
        raise RuntimeError(f"Optical/SAR size mismatch for {optical_path.stem}")
    optical_y = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    optical = torch.from_numpy(optical_y.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    sar = torch.from_numpy(sar_u8.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    return optical, sar, optical_y, sar_u8


def cached_c0_features(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    base = models["base_fuse"](optical_base + sar_base)
    return optical_detail, sar_detail, base


def fused_from_detail(models, optical, optical_detail, sar_detail, base, modulation):
    fused_detail = models["detail_fuse"](
        optical_detail + sar_detail * (1.0 + modulation)
    )
    fused, _ = models["decoder"](optical, base, fused_detail)
    return normalize_fused_tensor(fused, "clamp")


def u8_image(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0),
        0,
        255,
    ).astype(np.uint8)


def local_scores(fused_u8, optical_y, sar_u8, box):
    _, _, top, bottom, left, right = box
    area = np.s_[top:bottom, left:right]
    fused = fused_u8[area].astype(np.float32)
    optical = optical_y[area].astype(np.float32)
    sar = sar_u8[area].astype(np.float32)
    scores = {
        name: float(metric(fused, optical, sar))
        for name, metric in METRIC_FUNCTIONS.items()
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise RuntimeError(f"Nonfinite local metric at ROI {box}")
    return scores


def signed_contrast(plus, minus, metric):
    if metric == "Nabf":
        return minus[metric] - plus[metric]
    return plus[metric] - minus[metric]


def quality_gain(value, baseline, metric):
    return (
        baseline[metric] - value[metric]
        if metric == "Nabf"
        else value[metric] - baseline[metric]
    )


def direction(contrast, tolerance):
    if contrast > tolerance:
        return 1
    if contrast < -tolerance:
        return -1
    return 0


def beneficial_direction(plus, minus, baseline, tolerance):
    """Only call a direction beneficial if it beats the unmodified C0 output."""
    gain_plus = quality_gain(plus, baseline, "Qabf")
    gain_minus = quality_gain(minus, baseline, "Qabf")
    if max(gain_plus, gain_minus) <= tolerance:
        return 0
    return direction(gain_plus - gain_minus, tolerance)


def pareto_direction(plus, minus, baseline, tolerance):
    """Conservative proxy: improve local Qabf without worsening FMI/VIF/Nabf."""
    candidates = []
    for sign, scores in ((1, plus), (-1, minus)):
        gains = {
            metric: quality_gain(scores, baseline, metric)
            for metric in ("Qabf", "Nabf", "FMI", "VIF")
        }
        if gains["Qabf"] > tolerance and all(
            gains[name] >= -tolerance for name in ("Nabf", "FMI", "VIF")
        ):
            candidates.append((gains["Qabf"], sign))
    if not candidates:
        return 0
    candidates.sort(reverse=True)
    if len(candidates) > 1 and (
        candidates[0][0] - candidates[1][0] <= tolerance
    ):
        return 0
    return candidates[0][1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, records):
    if not records:
        raise ValueError(f"Empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(rows):
    # A tile is independent only across images; never count 16 tiles/image as
    # 16 independent validation images.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["eps"]].append(row)

    summary = []
    for eps, group in sorted(grouped.items()):
        valid = [row for row in group if row["q_direction"] != 0]
        plus = [row for row in valid if row["q_direction"] == 1]
        minus = [row for row in valid if row["q_direction"] == -1]
        same_direction_vif = [
            row for row in valid
            if row["vif_direction"] == row["q_direction"]
        ]
        # Per-image fraction of directional tiles is the statistical unit.
        by_image = defaultdict(list)
        for row in group:
            by_image[row["image"]].append(row)
        per_image = [
            {
                "image": image,
                "eps": eps,
                "tiles": len(items),
                "q_positive_tiles": sum(x["q_direction"] == 1 for x in items),
                "q_negative_tiles": sum(x["q_direction"] == -1 for x in items),
                "q_neutral_tiles": sum(x["q_direction"] == 0 for x in items),
                "q_positive_benefit_tiles": sum(
                    x["q_beneficial_direction"] == 1 for x in items
                ),
                "q_negative_benefit_tiles": sum(
                    x["q_beneficial_direction"] == -1 for x in items
                ),
                "pareto_positive_tiles": sum(
                    x["pareto_direction"] == 1 for x in items
                ),
                "pareto_negative_tiles": sum(
                    x["pareto_direction"] == -1 for x in items
                ),
                "both_benefit_signs_present": int(
                    any(x["q_beneficial_direction"] == 1 for x in items)
                    and any(x["q_beneficial_direction"] == -1 for x in items)
                ),
                "both_signs_present": int(
                    any(x["q_direction"] == 1 for x in items)
                    and any(x["q_direction"] == -1 for x in items)
                ),
                "mean_output_changed_fraction": float(
                    np.mean([x["output_changed_fraction"] for x in items])
                ),
                "mean_outside_change_fraction": float(
                    np.mean([x["outside_change_fraction"] for x in items])
                ),
            }
            for image, items in by_image.items()
        ]
        summary.append({
            "eps": eps,
            "images": len(by_image),
            "valid_tiles": len(group),
            "positive_direction_fraction": len(plus) / len(group),
            "negative_direction_fraction": len(minus) / len(group),
            "neutral_fraction": (len(group) - len(valid)) / len(group),
            "both_signs_image_fraction": float(np.mean(
                [r["both_signs_present"] for r in per_image]
            )),
            "q_positive_benefit_fraction": sum(
                r["q_beneficial_direction"] == 1 for r in group
            ) / len(group),
            "q_negative_benefit_fraction": sum(
                r["q_beneficial_direction"] == -1 for r in group
            ) / len(group),
            "both_benefit_signs_image_fraction": float(np.mean(
                [r["both_benefit_signs_present"] for r in per_image]
            )),
            "pareto_positive_fraction": sum(
                r["pareto_direction"] == 1 for r in group
            ) / len(group),
            "pareto_negative_fraction": sum(
                r["pareto_direction"] == -1 for r in group
            ) / len(group),
            "q_vif_direction_agreement_among_q_nonneutral": (
                len(same_direction_vif) / len(valid) if valid else None
            ),
            "mean_output_changed_fraction": float(np.mean(
                [r["output_changed_fraction"] for r in group]
            )),
            "mean_outside_change_fraction": float(np.mean(
                [r["outside_change_fraction"] for r in group]
            )),
        })
        yield summary[-1], per_image


def stability(rows):
    by_tile = defaultdict(dict)
    for row in rows:
        key = (row["image"], row["row"], row["col"])
        by_tile[key][row["eps"]] = row
    result = []
    for (image, row, col), values in by_tile.items():
        if len(values) != 2:
            raise ValueError("Expected two strengths per tile")
        strengths = sorted(values)
        low, high = values[strengths[0]], values[strengths[1]]
        low_dir = low["q_beneficial_direction"]
        high_dir = high["q_beneficial_direction"]
        result.append({
            "image": image,
            "row": row,
            "col": col,
            "eps_low": strengths[0],
            "eps_high": strengths[1],
            "q_beneficial_direction_low": low_dir,
            "q_beneficial_direction_high": high_dir,
            "direction_agrees": int(
                low_dir != 0 and high_dir != 0 and low_dir == high_dir
            ),
            "both_non_neutral": int(low_dir != 0 and high_dir != 0),
            "opposite_direction": int(low_dir * high_dir == -1),
            "q_contrast_low": low["contrast_Qabf"],
            "q_contrast_high": high["contrast_Qabf"],
        })
    return result


def main():
    args = parse_args()
    if args.limit < 1 or args.offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    if args.grid < 2 or args.min_tile_size < 32:
        raise ValueError("grid must be >= 2 and min-tile-size >= 32")
    if not 0.0 <= args.min_sar_valid <= 1.0:
        raise ValueError("min-sar-valid must be in [0,1]")
    if args.direction_tolerance < 0:
        raise ValueError("direction-tolerance must be nonnegative")
    strengths = sorted(set(float(eps) for eps in args.eps))
    if len(strengths) != 2 or any(not 0.0 < eps <= 0.5 for eps in strengths):
        raise ValueError("Specify exactly two distinct epsilon values in (0,0.5]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = checkpoint.get("args", {}).get("variant")
    if variant != "frequency_minimax_baseline":
        raise ValueError(
            "U0 requires a C0 frequency_minimax_baseline checkpoint, got "
            f"{variant!r}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, _ = load_variant_models(
        args.checkpoint, device, requested="frequency_minimax_baseline"
    )
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][args.offset:args.offset + args.limit]
    if not selected:
        raise RuntimeError("No image pairs selected")

    rows = []
    skipped = []
    max_zero_difference = 0.0
    with torch.no_grad():
        for image_index, (optical_path, sar_path) in enumerate(selected, start=1):
            optical, sar, optical_y, sar_u8 = load_pair(
                optical_path, sar_path, device
            )
            h, w = optical_y.shape
            positions = tiles(h, w, args.grid, args.min_tile_size)
            optical_detail, sar_detail, base = cached_c0_features(
                models, optical, sar
            )
            zero = torch.zeros((1, 1, h, w), dtype=sar_detail.dtype, device=device)
            baseline = fused_from_detail(
                models, optical, optical_detail, sar_detail, base, zero
            )
            baseline_u8 = u8_image(baseline)
            direct_c0, _ = forward_variant(
                models, "frequency_minimax_baseline", optical, sar
            )
            direct_c0 = normalize_fused_tensor(direct_c0, "clamp")
            zero_diff = float(torch.max(torch.abs(baseline-direct_c0)))
            max_zero_difference = max(max_zero_difference, zero_diff)
            if zero_diff > 1e-6:
                raise AssertionError(f"C0 zero repeat mismatch: {zero_diff}")

            for box in positions:
                row, col, top, bottom, left, right = box
                validity = float(
                    (sar_u8[top:bottom, left:right] > 0).mean()
                )
                if validity < args.min_sar_valid:
                    skipped.append({
                        "image": optical_path.stem,
                        "row": row,
                        "col": col,
                        "valid_fraction": validity,
                    })
                    continue
                reference = local_scores(baseline_u8, optical_y, sar_u8, box)
                mask = tile_mask((h, w), box, args.ramp, device)
                for eps in strengths:
                    plus_tensor = fused_from_detail(
                        models, optical, optical_detail, sar_detail, base, eps * mask
                    )
                    minus_tensor = fused_from_detail(
                        models, optical, optical_detail, sar_detail, base, -eps * mask
                    )
                    plus_u8 = u8_image(plus_tensor)
                    minus_u8 = u8_image(minus_tensor)
                    plus = local_scores(
                        plus_u8, optical_y, sar_u8, box
                    )
                    minus = local_scores(
                        minus_u8, optical_y, sar_u8, box
                    )
                    effect = torch.maximum(
                        (plus_tensor - baseline).abs(),
                        (minus_tensor - baseline).abs(),
                    )
                    inner = effect[..., top:bottom, left:right]
                    total = float(effect.sum().cpu())
                    inside = float(inner.sum().cpu())
                    changed = float(np.mean(
                        (plus_u8[top:bottom, left:right] !=
                         baseline_u8[top:bottom, left:right])
                        | (minus_u8[top:bottom, left:right] !=
                           baseline_u8[top:bottom, left:right])
                    ))
                    contrasts = {
                        name: signed_contrast(plus, minus, name)
                        for name in METRICS
                    }
                    directions = {
                        name: direction(
                            contrasts[name], args.direction_tolerance
                        )
                        for name in METRICS
                    }
                    record = {
                        "image": optical_path.stem,
                        "row": row,
                        "col": col,
                        "top": top,
                        "bottom": bottom,
                        "left": left,
                        "right": right,
                        "eps": eps,
                        "sar_valid_fraction": validity,
                        "mask_mean_in_tile": float(
                            mask[..., top:bottom, left:right].mean().cpu()
                        ),
                        "output_changed_fraction": changed,
                        "outside_change_fraction": (
                            (total-inside)/total if total > 0.0 else 0.0
                        ),
                        "q_direction": directions["Qabf"],
                        "vif_direction": directions["VIF"],
                        "q_beneficial_direction": beneficial_direction(
                            plus, minus, reference, args.direction_tolerance
                        ),
                        "pareto_direction": pareto_direction(
                            plus, minus, reference, args.direction_tolerance
                        ),
                    }
                    for name in METRICS:
                        record[f"c0_{name}"] = reference[name]
                        record[f"plus_{name}"] = plus[name]
                        record[f"minus_{name}"] = minus[name]
                        record[f"gain_plus_{name}"] = quality_gain(
                            plus, reference, name
                        )
                        record[f"gain_minus_{name}"] = quality_gain(
                            minus, reference, name
                        )
                        record[f"contrast_{name}"] = contrasts[name]
                        record[f"direction_{name}"] = directions[name]
                    rows.append(record)
            print(
                f"{image_index}/{len(selected)} image={optical_path.stem}, "
                f"valid_tile_evaluations={len(rows)}", flush=True
            )

    if not rows:
        raise RuntimeError("No eligible tiles: revisit min-sar-valid or grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tile_counterfactuals.csv", rows)
    if skipped:
        write_csv(args.output_dir / "skipped_tiles.csv", skipped)
    summaries = list(summarize(rows))
    summary_rows = [item[0] for item in summaries]
    write_csv(args.output_dir / "strength_summary.csv", summary_rows)
    per_image_rows = [r for _, records in summaries for r in records]
    write_csv(args.output_dir / "per_image_summary.csv", per_image_rows)

    stable = stability(rows)
    write_csv(args.output_dir / "tile_stability.csv", stable)
    comparable = [r for r in stable if r["both_non_neutral"]]
    stability_summary = {
        "tiles_with_both_strengths": len(stable),
        "nonneutral_both_strengths": len(comparable),
        "nonneutral_both_fraction": len(comparable) / len(stable),
        "direction_agreement_given_nonneutral_both": (
            sum(r["direction_agrees"] for r in comparable) / len(comparable)
            if comparable else None
        ),
        "opposite_direction_count": sum(r["opposite_direction"] for r in stable),
        "max_zero_repeat_abs_difference": max_zero_difference,
    }
    with (args.output_dir / "stability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(stability_summary, handle, indent=2)
    with (args.output_dir / "protocol.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "variant": variant,
            "split_manifest": str(args.split_manifest),
            "splits_sha256": validate_manifest_splits(manifest),
            "split": args.split,
            "offset": args.offset,
            "limit": len(selected),
            "grid": args.grid,
            "ramp": args.ramp,
            "eps": strengths,
            "min_sar_valid": args.min_sar_valid,
            "direction_tolerance": args.direction_tolerance,
            "metric_names": METRICS,
            "evaluation_policy": "local ROI metrics from fixed clamp->uint8 outputs",
            "primary_interpretation": (
                "q_direction compares +eps against -eps; "
                "q_beneficial_direction additionally requires improvement over C0; "
                "pareto_direction requires improved Qabf and non-worse "
                "FMI/VIF/Nabf within a descriptive tolerance. "
                "All are metric proxies, never ground-truth SAR utility."
            ),
            "selection": "manifest order, no cherry-picking",
            "not_ground_truth": True,
        }, handle, indent=2)
    print("=== U0 strength summary ===")
    for row in summary_rows:
        print(
            "eps={eps:.2f} tiles={valid_tiles} "
            "positive={positive_direction_fraction:.3f} "
            "negative={negative_direction_fraction:.3f} "
            "neutral={neutral_fraction:.3f} "
            "both_signs_images={both_signs_image_fraction:.3f} "
            "both_beneficial_signs_images={both_benefit_signs_image_fraction:.3f} "
            "pareto_plus={pareto_positive_fraction:.3f} "
            "pareto_minus={pareto_negative_fraction:.3f} "
            "q_vif_agreement={q_vif_direction_agreement_among_q_nonneutral}"
            .format(**row)
        )
    print("=== U0 sign stability ===")
    for name, value in stability_summary.items():
        print(f"{name}: {value}")
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
