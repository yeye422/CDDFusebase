"""Quantitative evaluation for a trained V2.1 SPD checkpoint."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch

from net import Restormer_Encoder
from spd_modules import SPDModel, build_spd_from_checkpoint_args
from train_spd import (
    canonical_state,
    encoder_state,
    normalized_feature_distillation,
    split_official_validation,
)
from utils.dataset_spd import SARDenoisingDataset, paired_sar_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V2.1 SPD")
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--spd-checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument(
        "--split", choices=("val", "test", "all"), default="test"
    )
    parser.add_argument(
        "--official-val-test-fraction", type=float, default=0.50
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_starts(length, tile_size, overlap):
    if tile_size <= 0 or tile_size > length:
        raise ValueError(
            f"tile_size must be in [1, {length}], got {tile_size}"
        )
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_overlap must be in [0, tile_size)")
    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    final = length - tile_size
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def blending_window(tile_size, device, dtype):
    if tile_size == 1:
        return torch.ones(1, 1, 1, 1, device=device, dtype=dtype)
    window = torch.hann_window(
        tile_size, periodic=False, device=device, dtype=dtype
    )
    window = window.clamp_min(0.05)
    return (window[:, None] * window[None, :])[None, None]


@torch.no_grad()
def tiled_forward(model, noisy, reference, tile_size, overlap):
    _, _, height, width = noisy.shape
    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    weight = blending_window(tile_size, noisy.device, noisy.dtype)
    output_sum = torch.zeros_like(noisy)
    weight_sum = torch.zeros_like(noisy)
    feature_totals = {
        "raw_base": 0.0,
        "clean_base": 0.0,
        "raw_detail": 0.0,
        "clean_detail": 0.0,
        "base_correction_rms": 0.0,
        "detail_correction_rms": 0.0,
        "image_change": 0.0,
    }
    tiles = 0
    valid = torch.ones(
        noisy.shape[0], 1, tile_size, tile_size,
        device=noisy.device,
        dtype=noisy.dtype,
    )
    for top in y_starts:
        for left in x_starts:
            noisy_tile = noisy[:, :, top : top + tile_size, left : left + tile_size]
            reference_tile = reference[
                :, :, top : top + tile_size, left : left + tile_size
            ]
            raw_base, raw_detail, _ = model.encoder(noisy_tile)
            reference_base, reference_detail, _ = model.encoder(reference_tile)
            clean_base, clean_detail, diagnostics = model.spd(
                raw_base, raw_detail
            )
            denoised = model.denoising_head(
                noisy_tile, clean_base, clean_detail
            )
            output_sum[
                :, :, top : top + tile_size, left : left + tile_size
            ] += denoised * weight
            weight_sum[
                :, :, top : top + tile_size, left : left + tile_size
            ] += weight
            feature_totals["raw_base"] += float(
                normalized_feature_distillation(
                    raw_base, reference_base, valid
                )
            )
            feature_totals["clean_base"] += float(
                normalized_feature_distillation(
                    clean_base, reference_base, valid
                )
            )
            feature_totals["raw_detail"] += float(
                normalized_feature_distillation(
                    raw_detail, reference_detail, valid
                )
            )
            feature_totals["clean_detail"] += float(
                normalized_feature_distillation(
                    clean_detail, reference_detail, valid
                )
            )
            feature_totals["base_correction_rms"] += float(
                diagnostics["base_correction_rms"]
            )
            feature_totals["detail_correction_rms"] += float(
                diagnostics["detail_correction_rms"]
            )
            feature_totals["image_change"] += float(
                (denoised - noisy_tile).abs().mean()
            )
            tiles += 1
    denoised = output_sum / weight_sum.clamp_min(1e-6)
    return denoised, {
        name: value / tiles for name, value in feature_totals.items()
    }


def safe_correlation(first, second):
    first = first.float().flatten()
    second = second.float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return 1.0 if torch.allclose(first, second) else 0.0
    return float((first * second).sum() / denominator)


def image_metrics(image, reference):
    difference = image - reference
    mse = float(difference.square().mean())
    mae = float(difference.abs().mean())
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    dssim = kornia.losses.ssim(
        image, reference, window_size=11, reduction="mean"
    )
    ssim = float((1.0 - 2.0 * dssim).clamp(-1.0, 1.0))
    image_gradient = kornia.filters.spatial_gradient(image)
    reference_gradient = kornia.filters.spatial_gradient(reference)
    edge_mae = float((image_gradient - reference_gradient).abs().mean())
    image_magnitude = torch.sqrt(
        image_gradient.square().sum(dim=2) + 1e-12
    )
    reference_magnitude = torch.sqrt(
        reference_gradient.square().sum(dim=2) + 1e-12
    )
    edge_correlation = safe_correlation(image_magnitude, reference_magnitude)
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "edge_mae": edge_mae,
        "edge_correlation": edge_correlation,
    }


def summarize(records):
    baseline_names = (
        "mae", "mse", "psnr", "ssim", "edge_mae", "edge_correlation"
    )
    lower_is_better = {"mae", "mse", "edge_mae"}
    summary = {"images": len(records), "metrics": {}}
    for name in baseline_names:
        noisy_values = np.asarray([record[f"noisy_{name}"] for record in records])
        spd_values = np.asarray([record[f"spd_{name}"] for record in records])
        differences = spd_values - noisy_values
        improved = differences < 0 if name in lower_is_better else differences > 0
        summary["metrics"][name] = {
            "noisy_mean": float(noisy_values.mean()),
            "spd_mean": float(spd_values.mean()),
            "mean_delta_spd_minus_noisy": float(differences.mean()),
            "median_delta_spd_minus_noisy": float(np.median(differences)),
            "improved_fraction": float(improved.mean()),
        }
    for name in (
        "raw_base",
        "clean_base",
        "raw_detail",
        "clean_detail",
        "base_correction_rms",
        "detail_correction_rms",
        "image_change",
        "seconds",
    ):
        summary[name] = float(np.mean([record[name] for record in records]))
    summary["base_feature_relative_change"] = (
        summary["clean_base"] / max(summary["raw_base"], 1e-12) - 1.0
    )
    summary["detail_feature_relative_change"] = (
        summary["clean_detail"] / max(summary["raw_detail"], 1e-12) - 1.0
    )
    return summary


def save_gray(path, tensor):
    array = (
        tensor.detach().clamp(0.0, 1.0).mul(255.0).round()
        .byte().squeeze().cpu().numpy()
    )
    cv2.imwrite(str(path), array)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = paired_sar_paths(args.noisy_dir, args.reference_dir)
    validation, test, val_groups, test_groups = split_official_validation(
        pairs, args.official_val_test_fraction
    )
    selected = {
        "val": validation,
        "test": test,
        "all": pairs,
    }[args.split]
    if args.limit:
        selected = selected[: args.limit]
    dataset = SARDenoisingDataset(
        selected,
        patch_size=512,
        training=False,
        augment=False,
        invalid_threshold=-1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(
        args.encoder_checkpoint, map_location="cpu", weights_only=False
    )
    encoder = Restormer_Encoder()
    encoder.load_state_dict(encoder_state(source))
    checkpoint = torch.load(
        args.spd_checkpoint, map_location="cpu", weights_only=False
    )
    spd = build_spd_from_checkpoint_args(checkpoint.get("args", {}))
    model = SPDModel(encoder, spd=spd)
    expected_hash = checkpoint.get("source_encoder_checkpoint_sha256")
    actual_hash = sha256_file(args.encoder_checkpoint)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            "SPD checkpoint was trained with a different C0 encoder checkpoint"
        )
    model.spd.load_state_dict(canonical_state(checkpoint["SPD"]))
    model.denoising_head.load_state_dict(
        canonical_state(checkpoint["SPDDenoisingHead"])
    )
    model.to(device).eval()

    image_dir = output_dir / "images"
    if args.save_images:
        image_dir.mkdir(exist_ok=True)
    records = []
    for index in range(len(dataset)):
        batch = dataset[index]
        noisy = batch["noisy"][None].to(device)
        reference = batch["reference"][None].to(device)
        started = time.time()
        denoised, feature_stats = tiled_forward(
            model,
            noisy,
            reference,
            args.tile_size,
            args.tile_overlap,
        )
        seconds = time.time() - started
        record = {"stem": batch["stem"], "seconds": seconds}
        record.update(
            {f"noisy_{key}": value for key, value in image_metrics(noisy, reference).items()}
        )
        record.update(
            {f"spd_{key}": value for key, value in image_metrics(denoised, reference).items()}
        )
        record.update(feature_stats)
        records.append(record)
        if index < args.save_images:
            save_gray(image_dir / f"{batch['stem']}_noisy.png", noisy)
            save_gray(image_dir / f"{batch['stem']}_spd.png", denoised)
            save_gray(image_dir / f"{batch['stem']}_reference.png", reference)
        print(
            f"{index + 1}/{len(dataset)} {batch['stem']}: "
            f"PSNR {record['noisy_psnr']:.3f}->{record['spd_psnr']:.3f}, "
            f"SSIM {record['noisy_ssim']:.4f}->{record['spd_ssim']:.4f}",
            flush=True,
        )

    fieldnames = list(records[0])
    with (output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    summary.update(
        {
            "split": args.split,
            "validation_coordinate_groups": val_groups,
            "internal_test_coordinate_groups": test_groups,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "spd_checkpoint": str(Path(args.spd_checkpoint).resolve()),
            "spd_checkpoint_sha256": sha256_file(args.spd_checkpoint),
            "encoder_checkpoint": str(Path(args.encoder_checkpoint).resolve()),
            "encoder_checkpoint_sha256": actual_hash,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
