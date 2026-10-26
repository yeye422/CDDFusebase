"""Standalone Stage-I trainer for structure-preserving SAR denoising.

This trainer never updates the C0 encoder.  It requires paired noisy/reference
SAR images and optionally consumes precomputed line/point/reference-quality
soft masks.  Formal training should only start after the dataset audit in the
V6 protocol; --dry-run is provided for interface verification.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from net import Restormer_Encoder
from spd_modules import (
    SPDModel,
    build_spd_variant,
)
from utils.dataset_spd import SARDenoisingDataset, paired_sar_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Train V2.1 SPD Stage I")
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--val-noisy-dir")
    parser.add_argument("--val-reference-dir")
    parser.add_argument(
        "--official-val-test-fraction", type=float, default=0.50
    )
    parser.add_argument("--line-mask-dir")
    parser.add_argument("--val-line-mask-dir")
    parser.add_argument("--point-mask-dir")
    parser.add_argument("--val-point-mask-dir")
    parser.add_argument("--reference-mask-dir")
    parser.add_argument("--val-reference-mask-dir")
    parser.add_argument("--split-manifest")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="spd")
    parser.add_argument(
        "--spd-variant",
        choices=("base_detail", "detail_only", "selective_detail"),
        default="base_detail",
    )
    parser.add_argument(
        "--selective-scale-kernels",
        type=int,
        nargs="+",
        default=(3, 5, 9),
    )
    parser.add_argument(
        "--selective-correction-highpass-kernel", type=int, default=5
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--ssim-weight", type=float, default=0.5)
    parser.add_argument("--line-weight", type=float, default=0.0)
    parser.add_argument("--point-weight", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-sigma", type=float, default=0.03)
    parser.add_argument("--base-feature-weight", type=float, default=0.20)
    parser.add_argument("--detail-feature-weight", type=float, default=0.05)
    parser.add_argument("--max-relative-correction", type=float, default=0.10)
    parser.add_argument("--max-logit-residual", type=float, default=2.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--invalid-threshold", type=int, default=-1)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model state must be a dictionary")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def encoder_state(checkpoint):
    for key in ("CDDF_Encoder", "DIDF_Encoder"):
        if key in checkpoint:
            return canonical_state(checkpoint[key])
    raise KeyError("Checkpoint has neither CDDF_Encoder nor DIDF_Encoder")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_pairs(pairs, seed, val_fraction, test_fraction, manifest_path=None):
    by_stem = {noisy.stem: (noisy, reference) for noisy, reference in pairs}
    if manifest_path:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        splits = manifest.get("splits", manifest)
        result = {}
        seen = set()
        for name in ("train", "val", "test"):
            stems = [str(value) for value in splits[name]]
            overlap = seen.intersection(stems)
            if overlap:
                raise ValueError(f"SPD split overlap: {sorted(overlap)[:5]}")
            seen.update(stems)
            missing = [stem for stem in stems if stem not in by_stem]
            if missing:
                raise FileNotFoundError(
                    f"SPD split {name} contains missing stems: {missing[:5]}"
                )
            result[name] = [by_stem[stem] for stem in stems]
        return result, manifest

    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val/test fractions must be positive")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be below one")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    test_count = max(1, round(len(shuffled) * test_fraction))
    if len(shuffled) <= val_count + test_count:
        raise RuntimeError("Not enough pairs for train/val/test SPD splits")
    result = {
        "val": shuffled[:val_count],
        "test": shuffled[val_count : val_count + test_count],
        "train": shuffled[val_count + test_count :],
    }
    manifest = {
        "seed": seed,
        "warning": "Random image-level split; replace with scene-level split for formal experiments",
        "splits": {
            name: [noisy.stem for noisy, _ in values]
            for name, values in result.items()
        },
    }
    return result, manifest


def split_official_validation(pairs, test_fraction):
    """Split coordinate tiles by their first coordinate, never by image."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("official-val-test-fraction must be in [0, 1)")
    groups = {}
    for pair in pairs:
        stem = pair[0].stem
        prefix = stem.split("_", 1)[0]
        groups.setdefault(prefix, []).append(pair)
    ordered = sorted(
        groups,
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if test_fraction == 0.0:
        return list(pairs), [], ordered, []
    test_group_count = max(1, round(len(ordered) * test_fraction))
    if test_group_count >= len(ordered):
        raise RuntimeError(
            "Official validation needs at least two coordinate groups to "
            "create separate validation and internal-test subsets"
        )
    test_groups = ordered[-test_group_count:]
    val_groups = ordered[:-test_group_count]
    validation = [pair for key in val_groups for pair in groups[key]]
    test = [pair for key in test_groups for pair in groups[key]]
    return validation, test, val_groups, test_groups


def masked_mean(values, mask):
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(2)
    expanded = expanded.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class SPDLoss(nn.Module):
    def __init__(self, charbonnier_epsilon=1e-3):
        super().__init__()
        self.epsilon = float(charbonnier_epsilon)
        self.ssim = kornia.losses.SSIM(11, reduction="none")
        self.gradient = kornia.filters.SpatialGradient()

    def charbonnier(self, prediction, target, mask):
        difference = prediction - target
        penalty = torch.sqrt(difference.square() + self.epsilon ** 2)
        penalty = penalty - self.epsilon
        return masked_mean(penalty, mask)

    def forward(
        self,
        denoised,
        reference,
        valid,
        reference_quality,
        line_mask,
        point_mask,
        ssim_weight=0.5,
        line_weight=0.0,
        point_weight=0.0,
    ):
        reference_mask = valid * reference_quality.clamp(0.0, 1.0)
        reconstruction = self.charbonnier(
            denoised, reference, reference_mask
        )
        ssim = masked_mean(self.ssim(denoised, reference), reference_mask)

        line_weight_map = reference_mask * line_mask.clamp(0.0, 1.0)
        if line_weight > 0.0 and torch.any(line_weight_map > 0):
            line = masked_mean(
                torch.abs(
                    self.gradient(denoised) - self.gradient(reference)
                ),
                line_weight_map,
            )
        else:
            line = denoised.sum() * 0.0

        point_weight_map = reference_mask * point_mask.clamp(0.0, 1.0)
        if point_weight > 0.0 and torch.any(point_weight_map > 0):
            denoised_log = torch.abs(kornia.filters.laplacian(denoised, 5))
            reference_log = torch.abs(kornia.filters.laplacian(reference, 5))
            point = masked_mean(
                torch.abs(denoised_log - reference_log), point_weight_map
            )
        else:
            point = denoised.sum() * 0.0

        total = (
            reconstruction
            + float(ssim_weight) * ssim
            + float(line_weight) * line
            + float(point_weight) * point
        )
        return total, {
            "reconstruction": reconstruction,
            "ssim": ssim,
            "line": line,
            "point": point,
        }


def controlled_multiplicative_perturbation(sar, valid, sigma):
    if sigma <= 0.0:
        return sar
    noise = torch.randn_like(sar) * float(sigma)
    factor = torch.exp(noise - 0.5 * float(sigma) ** 2)
    perturbed = (sar * factor).clamp(0.0, 1.0)
    return torch.where(valid > 0.5, perturbed, sar)


def feature_consistency(outputs, perturbed_outputs, valid, feature_names=None):
    if feature_names is None:
        feature_names = ("base", "detail")
    losses = []
    for name in feature_names:
        losses.append(
            masked_mean(
                torch.abs(
                    perturbed_outputs[name] - outputs[name].detach()
                ),
                valid,
            )
        )
    return torch.stack(losses).mean()


def normalized_feature_distillation(prediction, target, mask, epsilon=1e-6):
    target_rms = target.detach().square().mean(
        dim=(1, 2, 3), keepdim=True
    ).sqrt().clamp_min(epsilon)
    return masked_mean(
        torch.abs(prediction - target.detach()) / target_rms,
        mask,
    )


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "ssim": 0.0,
        "line": 0.0,
        "point": 0.0,
        "consistency": 0.0,
        "base_feature": 0.0,
        "detail_feature": 0.0,
        "image_change": 0.0,
        "line_mask_mean": 0.0,
        "line_mask_support": 0.0,
        "line_active": 0.0,
    }
    batches = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    amp_enabled = bool(args.amp and device.type == "cuda")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            if maximum is not None and batches >= maximum:
                break
            tensors = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                outputs = model(tensors["noisy"])
                loss, components = criterion(
                    outputs["denoised"],
                    tensors["reference"],
                    tensors["valid"],
                    tensors["reference_quality"],
                    tensors["line_mask"],
                    tensors["point_mask"],
                    args.ssim_weight,
                    args.line_weight,
                    args.point_weight,
                )
                with torch.no_grad():
                    reference_base, reference_detail, _ = model.encoder(
                        tensors["reference"]
                    )
                if args.spd_variant in ("detail_only", "selective_detail"):
                    base_feature = loss * 0.0
                else:
                    base_feature = normalized_feature_distillation(
                        outputs["base"],
                        reference_base,
                        tensors["valid"] * tensors["reference_quality"],
                    )
                detail_feature = normalized_feature_distillation(
                    outputs["detail"],
                    reference_detail,
                    tensors["valid"] * tensors["reference_quality"],
                )
                loss = (
                    loss
                    + args.base_feature_weight * base_feature
                    + args.detail_feature_weight * detail_feature
                )
                consistency = loss * 0.0
                if training and args.consistency_weight > 0.0:
                    perturbed = controlled_multiplicative_perturbation(
                        tensors["noisy"],
                        tensors["valid"],
                        args.consistency_sigma,
                    )
                    perturbed_outputs = model(perturbed)
                    consistency = feature_consistency(
                        outputs,
                        perturbed_outputs,
                        tensors["valid"],
                        feature_names=(
                            ("detail",)
                            if args.spd_variant in (
                                "detail_only",
                                "selective_detail",
                            )
                            else ("base", "detail")
                        ),
                    )
                    loss = loss + args.consistency_weight * consistency
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.clip_grad,
                )
                scaler.step(optimizer)
                scaler.update()
            totals["loss"] += float(loss.detach())
            for name in ("reconstruction", "ssim", "line", "point"):
                totals[name] += float(components[name].detach())
            totals["consistency"] += float(consistency.detach())
            totals["base_feature"] += float(base_feature.detach())
            totals["detail_feature"] += float(detail_feature.detach())
            totals["image_change"] += float(
                outputs["diagnostics"]["image_change_abs_mean"].detach()
            )
            totals["line_mask_mean"] += float(
                masked_mean(tensors["line_mask"], tensors["valid"]).detach()
            )
            totals["line_mask_support"] += float(
                masked_mean(
                    (tensors["line_mask"] > 0).float(), tensors["valid"]
                ).detach()
            )
            totals["line_active"] += float(
                torch.any(
                    (tensors["line_mask"] > 0)
                    & (tensors["valid"] > 0.5)
                    & (tensors["reference_quality"] > 0)
                ).detach()
            )
            batches += 1
    if not batches:
        raise RuntimeError("SPD loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, model, optimizer, args, best_val, source_hash):
    torch.save(
        {
            "epoch": int(epoch),
            "best_val": float(best_val),
            "SPD": model.spd.state_dict(),
            "SPDDenoisingHead": model.denoising_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "source_encoder_checkpoint_sha256": source_hash,
        },
        path,
    )


def main():
    args = parse_args()
    if (
        args.spd_variant in ("detail_only", "selective_detail")
        and args.base_feature_weight != 0.0
    ):
        raise ValueError(
            f"{args.spd_variant} requires --base-feature-weight 0 because "
            "Base is frozen"
        )
    if args.line_weight > 0.0 and not args.line_mask_dir:
        raise ValueError("--line-mask-dir is required when --line-weight > 0")
    if (
        args.line_weight > 0.0
        and args.val_noisy_dir
        and not args.val_line_mask_dir
    ):
        raise ValueError(
            "--val-line-mask-dir is required for official validation when "
            "--line-weight > 0"
        )
    if args.point_weight > 0.0 and not args.point_mask_dir:
        raise ValueError("--point-mask-dir is required when --point-weight > 0")
    if (
        args.point_weight > 0.0
        and args.val_noisy_dir
        and not args.val_point_mask_dir
    ):
        raise ValueError(
            "--val-point-mask-dir is required for official validation when "
            "--point-weight > 0"
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pairs = paired_sar_paths(args.noisy_dir, args.reference_dir)
    if bool(args.val_noisy_dir) != bool(args.val_reference_dir):
        raise ValueError(
            "--val-noisy-dir and --val-reference-dir must be provided together"
        )
    if args.val_noisy_dir:
        if args.split_manifest:
            raise ValueError(
                "Do not combine official validation directories with "
                "--split-manifest"
            )
        official_validation = paired_sar_paths(
            args.val_noisy_dir, args.val_reference_dir
        )
        validation, test, val_groups, test_groups = split_official_validation(
            official_validation, args.official_val_test_fraction
        )
        train_stems = {pair[0].stem for pair in pairs}
        held_out_stems = {pair[0].stem for pair in official_validation}
        overlap = train_stems & held_out_stems
        if overlap:
            raise ValueError(
                f"Official SPD train/validation overlap: {sorted(overlap)[:5]}"
            )
        splits = {"train": pairs, "val": validation, "test": test}
        split_manifest = {
            "strategy": "official spatial train/validation folders; official validation split by first coordinate group",
            "train_noisy_dir": str(Path(args.noisy_dir).resolve()),
            "train_reference_dir": str(Path(args.reference_dir).resolve()),
            "official_val_noisy_dir": str(Path(args.val_noisy_dir).resolve()),
            "official_val_reference_dir": str(
                Path(args.val_reference_dir).resolve()
            ),
            "validation_coordinate_groups": val_groups,
            "internal_test_coordinate_groups": test_groups,
            "splits": {
                name: [pair[0].stem for pair in values]
                for name, values in splits.items()
            },
        }
    else:
        splits, split_manifest = split_pairs(
            pairs,
            args.seed,
            args.val_fraction,
            args.test_fraction,
            args.split_manifest,
        )
    with (run_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, ensure_ascii=False, indent=2)

    common_dataset_args = dict(
        patch_size=args.patch_size,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
    )
    train_dataset = SARDenoisingDataset(
        splits["train"],
        training=True,
        augment=True,
        line_mask_dir=args.line_mask_dir,
        point_mask_dir=args.point_mask_dir,
        reference_mask_dir=args.reference_mask_dir,
        limit=args.train_limit,
        **common_dataset_args,
    )
    if args.val_noisy_dir:
        val_line_mask_dir = args.val_line_mask_dir
        val_point_mask_dir = args.val_point_mask_dir
        val_reference_mask_dir = args.val_reference_mask_dir
    else:
        val_line_mask_dir = args.val_line_mask_dir or args.line_mask_dir
        val_point_mask_dir = args.val_point_mask_dir or args.point_mask_dir
        val_reference_mask_dir = (
            args.val_reference_mask_dir or args.reference_mask_dir
        )
    val_dataset = SARDenoisingDataset(
        splits["val"],
        training=False,
        augment=False,
        line_mask_dir=val_line_mask_dir,
        point_mask_dir=val_point_mask_dir,
        reference_mask_dir=val_reference_mask_dir,
        limit=args.val_limit,
        **common_dataset_args,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    source_checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    encoder = Restormer_Encoder()
    encoder.load_state_dict(encoder_state(source_checkpoint))
    spd = build_spd_variant(
        args.spd_variant,
        max_relative_correction=args.max_relative_correction,
        scale_kernels=tuple(args.selective_scale_kernels),
        correction_highpass_kernel=(
            args.selective_correction_highpass_kernel
        ),
    )
    model = SPDModel(encoder, spd=spd).to(device)
    if hasattr(model.spd, "base"):
        model.spd.base.max_relative_correction = args.max_relative_correction
    model.spd.detail.max_relative_correction = args.max_relative_correction
    model.denoising_head.max_logit_residual = args.max_logit_residual
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("SPD trainer must keep the C0 encoder frozen")

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    criterion = SPDLoss(args.charbonnier_epsilon).to(device)
    source_hash = sha256_file(args.init_checkpoint)
    config = dict(vars(args))
    config.update(
        {
            "device": str(device),
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
            "held_out_test_pairs": len(splits["test"]),
            "source_encoder_checkpoint_sha256": source_hash,
            "frozen_encoder_parameters": sum(
                parameter.numel() for parameter in model.encoder.parameters()
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(model, val_loader, criterion, device, args)
        print(
            f"SPD dry run complete: device={device}, val_loss={stats['loss']:.6f}, "
            f"line={stats['line']:.6f}, "
            f"line_mask_mean={stats['line_mask_mean']:.6f}, "
            f"line_mask_support={stats['line_mask_support']:.6f}, "
            f"line_active={stats['line_active']:.3f}, "
            f"train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test={len(splits['test'])}, run={run_dir.resolve()}",
            flush=True,
        )
        return

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_reconstruction",
                "val_reconstruction",
                "train_ssim",
                "val_ssim",
                "train_line",
                "val_line",
                "train_point",
                "val_point",
                "train_consistency",
                "train_base_feature",
                "val_base_feature",
                "train_detail_feature",
                "val_detail_feature",
                "train_line_mask_mean",
                "val_line_mask_mean",
                "train_line_mask_support",
                "val_line_mask_support",
                "train_line_active_fraction",
                "val_line_active_fraction",
                "seconds",
            ]
        )

    best_val = float("inf")
    for epoch in range(args.epochs):
        started = time.time()
        train_stats = run_epoch(
            model, train_loader, criterion, device, args, optimizer, scaler
        )
        val_stats = run_epoch(model, val_loader, criterion, device, args)
        seconds = time.time() - started
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth",
                epoch,
                model,
                optimizer,
                args,
                best_val,
                source_hash,
            )
        save_checkpoint(
            run_dir / "last.pth",
            epoch,
            model,
            optimizer,
            args,
            best_val,
            source_hash,
        )
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    train_stats["loss"],
                    val_stats["loss"],
                    train_stats["reconstruction"],
                    val_stats["reconstruction"],
                    train_stats["ssim"],
                    val_stats["ssim"],
                    train_stats["line"],
                    val_stats["line"],
                    train_stats["point"],
                    val_stats["point"],
                    train_stats["consistency"],
                    train_stats["base_feature"],
                    val_stats["base_feature"],
                    train_stats["detail_feature"],
                    val_stats["detail_feature"],
                    train_stats["line_mask_mean"],
                    val_stats["line_mask_mean"],
                    train_stats["line_mask_support"],
                    val_stats["line_mask_support"],
                    train_stats["line_active"],
                    val_stats["line_active"],
                    seconds,
                ]
            )
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"train={train_stats['loss']:.6f}, "
            f"val={val_stats['loss']:.6f}, {seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
