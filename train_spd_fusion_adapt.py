"""Matched C0-Adapt and E2-Adapt trainer for frozen SPD features."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from compare_spd_fusion import load_models, sha256_file
from train_OS import OpticalSARLosses, reference_normalized_smooth_minimax
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train matched SPD fusion adaptation")
    parser.add_argument(
        "--variant", choices=("c0_adapt", "e2_adapt"), required=True
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--spd-checkpoint", required=True)
    parser.add_argument("--frequency-reference-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument(
        "--structure-charbonnier-epsilon", type=float, default=1e-3
    )
    parser.add_argument("--minimax-temperature", type=float, default=0.1)
    parser.add_argument("--collapse-mae-threshold", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frequency_reference(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    reference = config.get("frequency_training_reference")
    if not reference:
        raise ValueError("frequency reference config has no training reference")
    for name in ("optical", "structure"):
        if float(reference[name]) <= 0.0:
            raise ValueError(f"invalid frequency reference {name}")
    return reference


def configure_trainability(models):
    for name, model in models.items():
        trainable = name in ("base_fuse", "detail_fuse")
        model.train(trainable)
        for parameter in model.parameters():
            parameter.requires_grad_(trainable)


def trainable_parameters(models):
    return [
        parameter
        for name in ("base_fuse", "detail_fuse")
        for parameter in models[name].parameters()
        if parameter.requires_grad
    ]


def forward_adapt(models, optical, sar, mask, variant, losses, args, reference):
    with torch.no_grad():
        optical_base, optical_detail, _ = models["encoder"](optical)
        sar_base, sar_detail, _ = models["encoder"](sar)
        if variant == "e2_adapt":
            sar_base, sar_detail, _ = models["spd"](sar_base, sar_detail)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    generated, _ = models["decoder"](optical, fused_base, fused_detail)
    (
        _,
        optical_loss,
        optical_l1,
        optical_ssim,
        optical_high,
        sar_mid,
        output_optical_mae,
        output_sar_mae,
    ) = losses.fusion_frequency_decoupled(
        optical,
        sar,
        generated,
        mask,
        args.optical_lowpass_sigma,
        args.frequency_highpass_sigma,
        args.frequency_midpass_sigma,
        args.optical_high_weight,
        args.structure_percentile,
        args.structure_charbonnier_epsilon,
    )
    total, relative_optical, relative_structure = (
        reference_normalized_smooth_minimax(
            optical_loss,
            sar_mid,
            reference["optical"],
            reference["structure"],
            args.minimax_temperature,
        )
    )
    return total, {
        "optical": optical_loss,
        "optical_l1": optical_l1,
        "optical_ssim": optical_ssim,
        "optical_high": optical_high,
        "sar_mid": sar_mid,
        "relative_optical": relative_optical,
        "relative_structure": relative_structure,
        "output_optical_mae": output_optical_mae,
        "output_sar_mae": output_sar_mae,
    }


def run_epoch(
    models,
    loader,
    losses,
    device,
    args,
    reference,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    models["base_fuse"].train(training)
    models["detail_fuse"].train(training)
    for name in ("encoder", "decoder", "spd", "denoising_head"):
        models[name].eval()
    totals = {
        "loss": 0.0,
        "optical": 0.0,
        "sar_mid": 0.0,
        "relative_optical": 0.0,
        "relative_structure": 0.0,
        "output_optical_mae": 0.0,
        "output_sar_mae": 0.0,
    }
    batches = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    amp_enabled = bool(args.amp and device.type == "cuda")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for optical, sar, mask, _ in loader:
            if maximum is not None and batches >= maximum:
                break
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                total, components = forward_adapt(
                    models,
                    optical,
                    sar,
                    mask,
                    args.variant,
                    losses,
                    args,
                    reference,
                )
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(models), args.clip_grad
                )
                scaler.step(optimizer)
                scaler.update()
            totals["loss"] += float(total.detach())
            for name in totals:
                if name != "loss":
                    totals[name] += float(components[name].detach())
            batches += 1
    if not batches:
        raise RuntimeError("adaptation loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(
    path,
    epoch,
    best_val,
    models,
    optimizer,
    args,
    reference,
    hashes,
):
    torch.save(
        {
            "epoch": int(epoch),
            "best_val": float(best_val),
            "CDDF_Encoder": models["encoder"].state_dict(),
            "CDDF_Decoder": models["decoder"].state_dict(),
            "BaseFuseLayer": models["base_fuse"].state_dict(),
            "DetailFuseLayer": models["detail_fuse"].state_dict(),
            "optimizer": optimizer.state_dict(),
            "frequency_training_reference": reference,
            "args": vars(args),
            **hashes,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    dataset_args = dict(
        patch_size=args.patch_size,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
    )
    train_dataset = OpticalSARDataset(
        splits["train"],
        training=True,
        augment=True,
        limit=args.train_limit,
        **dataset_args,
    )
    val_dataset = OpticalSARDataset(
        splits["val"],
        training=False,
        augment=False,
        limit=args.val_limit,
        **dataset_args,
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
    (
        models,
        c0_hash,
        spd_hash,
        encoder_equivalence,
        spd_source_checkpoint,
    ) = load_models(args.c0_checkpoint, args.spd_checkpoint, device)
    configure_trainability(models)
    parameters = trainable_parameters(models)
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    losses = OpticalSARLosses().to(device)
    reference = load_frequency_reference(args.frequency_reference_config)
    hashes = {
        "c0_checkpoint_sha256": c0_hash,
        "spd_checkpoint_sha256": spd_hash,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "frequency_reference_config_sha256": sha256_file(
            args.frequency_reference_config
        ),
    }
    config = dict(vars(args))
    config.update(
        {
            "device": str(device),
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
            "held_out_test_pairs": len(splits["test"]),
            "split_counts": {name: len(values) for name, values in splits.items()},
            "split_manifest_metadata": {
                key: value for key, value in manifest.items() if key != "splits"
            },
            "frequency_training_reference": reference,
            "encoder_equivalence": encoder_equivalence,
            "spd_training_source_checkpoint": spd_source_checkpoint,
            "trainable_modules": ["base_fuse", "detail_fuse"],
            "frozen_modules": ["encoder", "decoder", "spd", "denoising_head"],
            "trainable_parameters": sum(p.numel() for p in parameters),
            **hashes,
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(
            models, val_loader, losses, device, args, reference
        )
        print(
            f"adapt dry run complete: variant={args.variant}, "
            f"val_loss={stats['loss']:.6f}, "
            f"rel_opt={stats['relative_optical']:.6f}, "
            f"rel_sar={stats['relative_structure']:.6f}, "
            f"run={run_dir.resolve()}",
            flush=True,
        )
        return

    history_path = run_dir / "history.csv"
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_optical",
        "val_optical",
        "train_sar_mid",
        "val_sar_mid",
        "train_relative_optical",
        "val_relative_optical",
        "train_relative_structure",
        "val_relative_structure",
        "train_output_optical_mae",
        "val_output_optical_mae",
        "seconds",
    ]
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_val = float("inf")
    for epoch in range(args.epochs):
        started = time.time()
        train_stats = run_epoch(
            models,
            train_loader,
            losses,
            device,
            args,
            reference,
            optimizer,
            scaler,
        )
        val_stats = run_epoch(
            models, val_loader, losses, device, args, reference
        )
        seconds = time.time() - started
        collapsed = val_stats["output_optical_mae"] < args.collapse_mae_threshold
        if not collapsed and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth",
                epoch,
                best_val,
                models,
                optimizer,
                args,
                reference,
                hashes,
            )
        save_checkpoint(
            run_dir / "last.pth",
            epoch,
            best_val,
            models,
            optimizer,
            args,
            reference,
            hashes,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_optical": train_stats["optical"],
            "val_optical": val_stats["optical"],
            "train_sar_mid": train_stats["sar_mid"],
            "val_sar_mid": val_stats["sar_mid"],
            "train_relative_optical": train_stats["relative_optical"],
            "val_relative_optical": val_stats["relative_optical"],
            "train_relative_structure": train_stats["relative_structure"],
            "val_relative_structure": val_stats["relative_structure"],
            "train_output_optical_mae": train_stats["output_optical_mae"],
            "val_output_optical_mae": val_stats["output_optical_mae"],
            "seconds": seconds,
        }
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow(row)
        print(
            f"epoch {epoch + 1}/{args.epochs}: train={train_stats['loss']:.6f}, "
            f"val={val_stats['loss']:.6f}, "
            f"rel=({val_stats['relative_optical']:.4f},"
            f"{val_stats['relative_structure']:.4f}), "
            f"collapsed={collapsed}, {seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
