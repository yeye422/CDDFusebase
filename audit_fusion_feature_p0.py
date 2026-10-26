"""P0 audit for frozen Optical-SAR feature separability.

This script does not train or modify a model.  It measures whether the frozen
B0 encoder retains enough provenance and perturbation-stability evidence for a
new feature-fusion module to separate shared structure, modality-private detail,
and SAR nuisance responses.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from net import Restormer_Encoder
from utils.dataset_os import OpticalSARDataset, manifest_pairs


FEATURE_NAMES = ("base", "detail")
REGION_NAMES = ("shared_edge", "optical_only", "sar_only", "flat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit frozen B0 Optical-SAR feature separability"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturb-sigma", type=float, default=0.03)
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--edge-quantile", type=float, default=0.80)
    parser.add_argument("--flat-quantile", type=float, default=0.50)
    parser.add_argument("--save-maps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def encoder_state(checkpoint):
    for name in ("CDDF_Encoder", "DIDF_Encoder"):
        if name in checkpoint:
            return canonical_state(checkpoint[name])
    raise KeyError("Checkpoint has neither CDDF_Encoder nor DIDF_Encoder")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_magnitude(image, epsilon=1e-8):
    dx = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + epsilon)


def masked_quantile(values, mask, quantile):
    selected = values[mask > 0.5]
    if not selected.numel():
        raise RuntimeError("Cannot compute a quantile on an empty valid mask")
    return torch.quantile(selected, float(quantile))


def structure_regions(optical, sar, valid, edge_quantile=0.8, flat_quantile=0.5):
    optical_gradient = gradient_magnitude(optical)
    sar_gradient = gradient_magnitude(sar)
    optical_high = optical_gradient >= masked_quantile(
        optical_gradient, valid, edge_quantile
    )
    sar_high = sar_gradient >= masked_quantile(sar_gradient, valid, edge_quantile)
    optical_low = optical_gradient <= masked_quantile(
        optical_gradient, valid, flat_quantile
    )
    sar_low = sar_gradient <= masked_quantile(sar_gradient, valid, flat_quantile)
    valid_bool = valid > 0.5
    regions = {
        "shared_edge": valid_bool & optical_high & sar_high,
        "optical_only": valid_bool & optical_high & sar_low,
        "sar_only": valid_bool & sar_high & optical_low,
        "flat": valid_bool & optical_low & sar_low,
    }
    return regions, optical_gradient, sar_gradient


def shift_without_wrap(features, dy, dx):
    height, width = features.shape[-2:]
    shifted = torch.zeros_like(features)
    y_src_start = max(-dy, 0)
    y_src_end = min(height - dy, height)
    x_src_start = max(-dx, 0)
    x_src_end = min(width - dx, width)
    y_dst_start = max(dy, 0)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(dx, 0)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)
    shifted[:, :, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = features[
        :, :, y_src_start:y_src_end, x_src_start:x_src_end
    ]
    valid = torch.zeros(
        (features.shape[0], 1, height, width),
        dtype=features.dtype,
        device=features.device,
    )
    valid[:, :, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = 1.0
    return shifted, valid


def cosine_map(first, second, epsilon=1e-6):
    first = F.normalize(first, dim=1, eps=epsilon)
    second = F.normalize(second, dim=1, eps=epsilon)
    return (first * second).sum(dim=1, keepdim=True)


def local_cosine_maps(optical, sar, radius):
    same = cosine_map(optical, sar)
    candidates = []
    validities = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted, valid = shift_without_wrap(sar, dy, dx)
            candidates.append(cosine_map(optical, shifted).masked_fill(valid < 0.5, -2.0))
            validities.append(valid)
    stacked = torch.stack(candidates, dim=0)
    best, best_shift = stacked.max(dim=0)
    any_valid = torch.stack(validities, dim=0).amax(dim=0)
    return same, best, best_shift, any_valid


def normalized_sensitivity(first, second, epsilon=1e-5):
    numerator = (first - second).abs().mean(dim=1, keepdim=True)
    denominator = 0.5 * (first.abs() + second.abs()).mean(dim=1, keepdim=True)
    return numerator / denominator.clamp_min(epsilon)


def feature_statistics(optical, sar, local_radius, valid):
    optical_rms = optical.square().mean().sqrt()
    sar_rms = sar.square().mean().sqrt()
    same, best, best_shift, local_valid = local_cosine_maps(
        optical, sar, local_radius
    )
    joint_valid = valid * local_valid
    denominator = (optical.square().sum(dim=1, keepdim=True).sqrt() + sar.square().sum(
        dim=1, keepdim=True
    ).sqrt()).clamp_min(1e-6)
    cancellation = (
        denominator
        - (optical + sar).square().sum(dim=1, keepdim=True).sqrt()
    ) / denominator
    selected_shifts = best_shift[joint_valid > 0.5]
    shift_count = (2 * int(local_radius) + 1) ** 2
    shift_histogram = torch.bincount(
        selected_shifts.reshape(-1), minlength=shift_count
    ).float()
    shift_probabilities = shift_histogram / shift_histogram.sum().clamp_min(1.0)
    nonzero = shift_probabilities > 0
    shift_entropy = -(
        shift_probabilities[nonzero]
        * torch.log(shift_probabilities[nonzero])
    ).sum()
    normalized_shift_entropy = shift_entropy / max(math.log(shift_count), 1.0)
    return {
        "optical_rms": float(optical_rms),
        "sar_rms": float(sar_rms),
        "sar_energy_share": float(sar_rms / (optical_rms + sar_rms).clamp_min(1e-6)),
        "same_cosine": masked_mean(same, joint_valid),
        "best_local_cosine": masked_mean(best, joint_valid),
        "local_alignment_gain": masked_mean(best - same, joint_valid),
        "negative_cosine_fraction": masked_mean((same < 0).float(), joint_valid),
        "cancellation": masked_mean(cancellation.clamp_min(0.0), joint_valid),
        "local_shift_mode_fraction": float(shift_probabilities.max()),
        "local_shift_entropy": float(normalized_shift_entropy),
        "same_map": same,
        "best_map": best,
        "shift_map": best_shift,
    }


def masked_mean(values, mask):
    expanded = mask.expand_as(values).to(values.dtype)
    denominator = expanded.sum()
    if float(denominator) <= 0.0:
        return float("nan")
    return float((values * expanded).sum() / denominator)


def masked_correlation(first, second, mask, epsilon=1e-8):
    expanded = mask.expand_as(first).to(first.dtype)
    if float(expanded.sum()) <= 1.0:
        return float("nan")
    first_mean = (first * expanded).sum() / expanded.sum()
    second_mean = (second * expanded).sum() / expanded.sum()
    first_centered = (first - first_mean) * expanded
    second_centered = (second - second_mean) * expanded
    denominator = torch.sqrt(
        first_centered.square().sum() * second_centered.square().sum()
    ).clamp_min(epsilon)
    return float((first_centered * second_centered).sum() / denominator)


def resize_mask(mask, size):
    return F.interpolate(mask.float(), size=size, mode="nearest")


def controlled_multiplicative_perturbation(sar, valid, sigma):
    noise = torch.randn_like(sar) * float(sigma)
    factor = torch.exp(noise - 0.5 * float(sigma) ** 2)
    perturbed = (sar * factor).clamp(0.0, 1.0)
    return torch.where(valid > 0.5, perturbed, sar)


def to_u8(values, lower=None, upper=None):
    values = values.detach().float().cpu().numpy().squeeze()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8)
    if lower is None:
        lower = float(np.quantile(finite, 0.02))
    if upper is None:
        upper = float(np.quantile(finite, 0.98))
    scaled = (values - lower) / max(upper - lower, 1e-8)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def save_diagnostic_maps(directory, stem, optical, sar, optical_gradient, sar_gradient,
                         detail_same, detail_best, detail_shift,
                         detail_sensitivity):
    panels = (
        ("optical", to_u8(optical, 0.0, 1.0)),
        ("sar", to_u8(sar, 0.0, 1.0)),
        ("optical_gradient", to_u8(optical_gradient)),
        ("sar_gradient", to_u8(sar_gradient)),
        ("detail_same_cosine", to_u8(detail_same, -1.0, 1.0)),
        ("detail_best_local_cosine", to_u8(detail_best, -1.0, 1.0)),
        ("detail_best_shift", to_u8(detail_shift)),
        ("detail_sensitivity", to_u8(detail_sensitivity)),
    )
    output = np.concatenate([image for _, image in panels], axis=1)
    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / f"{stem}.png"), output):
        raise RuntimeError(f"Could not save diagnostic map for {stem}")
    with (directory / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump({"panels": [name for name, _ in panels]}, handle, indent=2)


def summarize(rows):
    numeric_names = [
        name
        for name in rows[0]
        if name != "image" and isinstance(rows[0][name], (float, int))
    ]
    output = {}
    for name in numeric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "count": int(values.size),
        }
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def audit_pair(encoder, optical, sar, valid, args):
    optical_base, optical_detail, _ = encoder(optical)
    sar_base, sar_detail, _ = encoder(sar)
    sar_1 = controlled_multiplicative_perturbation(
        sar, valid, args.perturb_sigma
    )
    sar_2 = controlled_multiplicative_perturbation(
        sar, valid, args.perturb_sigma
    )
    sar_1_base, sar_1_detail, _ = encoder(sar_1)
    sar_2_base, sar_2_detail, _ = encoder(sar_2)
    regions, optical_gradient, sar_gradient = structure_regions(
        optical, sar, valid, args.edge_quantile, args.flat_quantile
    )
    features = {
        "base": (optical_base, sar_base, sar_1_base, sar_2_base),
        "detail": (optical_detail, sar_detail, sar_1_detail, sar_2_detail),
    }
    row = {}
    maps = {}
    for feature_name, (optical_f, sar_f, sar_1_f, sar_2_f) in features.items():
        feature_valid = F.interpolate(valid, size=optical_f.shape[-2:], mode="nearest")
        stats = feature_statistics(
            optical_f, sar_f, args.local_radius, feature_valid
        )
        for name, value in stats.items():
            if not name.endswith("_map"):
                row[f"{feature_name}_{name}"] = value
        sensitivity = normalized_sensitivity(sar_1_f, sar_2_f)
        row[f"{feature_name}_sensitivity_mean"] = masked_mean(
            sensitivity, feature_valid
        )
        sar_resized = F.interpolate(
            sar, size=optical_f.shape[-2:], mode="bilinear", align_corners=False
        )
        sar_gradient_resized = F.interpolate(
            sar_gradient,
            size=optical_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        row[f"{feature_name}_sensitivity_sar_intensity_corr"] = masked_correlation(
            sensitivity, sar_resized, feature_valid
        )
        row[f"{feature_name}_sensitivity_sar_gradient_corr"] = masked_correlation(
            sensitivity, sar_gradient_resized, feature_valid
        )
        for region_name, region in regions.items():
            feature_region = resize_mask(region, optical_f.shape[-2:])
            row[f"{feature_name}_sensitivity_{region_name}"] = masked_mean(
                sensitivity, feature_region
            )
            row[f"{feature_name}_cosine_{region_name}"] = masked_mean(
                stats["same_map"], feature_region
            )
        maps[f"{feature_name}_same"] = stats["same_map"]
        maps[f"{feature_name}_best"] = stats["best_map"]
        maps[f"{feature_name}_shift"] = stats["shift_map"]
        maps[f"{feature_name}_sensitivity"] = sensitivity
    valid_pixels = float((valid > 0.5).sum())
    for region_name, region in regions.items():
        row[f"region_fraction_{region_name}"] = float(region.sum()) / max(
            valid_pixels, 1.0
        )
    maps.update(
        {
            "optical_gradient": optical_gradient,
            "sar_gradient": sar_gradient,
        }
    )
    return row, maps


def main():
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("offset must be non-negative and limit must be positive")
    if args.local_radius < 0:
        raise ValueError("local-radius must be non-negative")
    if not 0.0 < args.edge_quantile < 1.0:
        raise ValueError("edge-quantile must be in (0, 1)")
    if not 0.0 < args.flat_quantile < args.edge_quantile:
        raise ValueError("flat-quantile must be in (0, edge-quantile)")
    set_seed(args.seed)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    selected = pairs[args.offset : args.offset + limit]
    if not selected:
        raise RuntimeError("P0 audit selected no Optical-SAR pairs")
    dataset = OpticalSARDataset(
        selected,
        patch_size=256,
        training=False,
        augment=False,
        invalid_threshold=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    encoder = Restormer_Encoder()
    encoder.load_state_dict(encoder_state(checkpoint))
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in range(len(dataset)):
        optical, sar, valid, stem = dataset[index]
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        valid = valid[None].to(device)
        row, maps = audit_pair(encoder, optical, sar, valid, args)
        row = {"image": stem, **row}
        rows.append(row)
        if index < args.save_maps:
            save_diagnostic_maps(
                args.output_dir / "maps",
                stem,
                optical,
                sar,
                maps["optical_gradient"],
                maps["sar_gradient"],
                maps["detail_same"],
                maps["detail_best"],
                maps["detail_shift"],
                maps["detail_sensitivity"],
            )
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)

    write_csv(args.output_dir / "per_image.csv", rows)
    payload = {
        "protocol": {
            "stage": "P0_frozen_feature_separability_audit",
            "dataset": manifest.get("dataset", "QXS-SAROPT"),
            "split": args.split,
            "offset": args.offset,
            "sample_count": len(rows),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "perturb_sigma": args.perturb_sigma,
            "local_radius": args.local_radius,
            "edge_quantile": args.edge_quantile,
            "flat_quantile": args.flat_quantile,
            "seed": args.seed,
            "device": str(device),
            "training_performed": False,
        },
        "summary": summarize(rows),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
