"""Phase-2 fusion-only trainer for MFFA v1 under the frozen C0 objective."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mffa_v1 import MFFAV1
from train_OS import OpticalSARLosses, reference_normalized_smooth_minimax
from train_mffa_v1_phase1 import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFFA v1 Phase 2 fusion")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--phase1-checkpoint", type=Path, required=True)
    parser.add_argument("--frequency-reference-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument("--structure-charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--minimax-temperature", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frequency_reference(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    reference = config.get("frequency_training_reference")
    if not reference:
        raise ValueError("frequency reference config has no training reference")
    if float(reference["optical"]) <= 0 or float(reference["structure"]) <= 0:
        raise ValueError("frequency training references must be positive")
    return reference


def configure_phase2(model):
    trainable = {"low_fusion", "mid_fusion", "high_fusion"}
    for name, module in model.named_children():
        active = name in trainable
        module.train(active)
        for parameter in module.parameters():
            parameter.requires_grad_(active)


def run_epoch(model, loader, losses, device, args, reference, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    for name in ("analysis", "low_encoder", "optical_detail_encoder", "sar_detail_encoder", "decoder"):
        getattr(model, name).eval()
    totals = {name: 0.0 for name in (
        "loss", "optical", "optical_l1", "optical_ssim", "optical_high",
        "sar_mid", "relative_optical", "relative_structure",
        "output_optical_mae", "output_sar_mae", "high_relative_sar",
        "fused_low_rms", "fused_mid_rms", "fused_high_rms",
    )}
    batches = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    amp_enabled = bool(args.amp and device.type == "cuda")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for optical, sar, valid, _ in loader:
            if maximum is not None and batches >= maximum:
                break
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                generated, outputs = model.fuse(
                    optical, sar, valid, return_diagnostics=True
                )
                (
                    _, optical_loss, optical_l1, optical_ssim,
                    optical_high, sar_mid, output_optical_mae, output_sar_mae,
                ) = losses.fusion_frequency_decoupled(
                    optical, sar, generated, valid,
                    args.optical_lowpass_sigma,
                    args.frequency_highpass_sigma,
                    args.frequency_midpass_sigma,
                    args.optical_high_weight,
                    args.structure_percentile,
                    args.structure_charbonnier_epsilon,
                )
                total, relative_optical, relative_structure = reference_normalized_smooth_minimax(
                    optical_loss, sar_mid, reference["optical"],
                    reference["structure"], args.minimax_temperature
                )
            if training:
                scaler.scale(total).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip_grad
                )
                scaler.step(optimizer); scaler.update()
            values = {
                "loss": total, "optical": optical_loss, "optical_l1": optical_l1,
                "optical_ssim": optical_ssim, "optical_high": optical_high,
                "sar_mid": sar_mid, "relative_optical": relative_optical,
                "relative_structure": relative_structure,
                "output_optical_mae": output_optical_mae,
                "output_sar_mae": output_sar_mae,
                "high_relative_sar": outputs["diagnostics"]["high_relative_sar"],
                "fused_low_rms": outputs["fused"]["low"].square().mean().sqrt(),
                "fused_mid_rms": outputs["fused"]["mid"].square().mean().sqrt(),
                "fused_high_rms": outputs["fused"]["high"].square().mean().sqrt(),
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
    if not batches:
        raise RuntimeError("MFFA Phase-2 loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, model, optimizer, args, best_val, phase1_hash, manifest_hash):
    torch.save({
        "epoch": int(epoch), "best_val": float(best_val),
        "variant": "mffa_v1_phase2", "MFFAV1": model.state_dict(),
        "optimizer": optimizer.state_dict(), "args": vars(args),
        "phase1_checkpoint_sha256": phase1_hash,
        "split_manifest_sha256": manifest_hash,
    }, path)


def metric_names():
    return (
        "loss", "optical", "optical_l1", "optical_ssim", "optical_high",
        "sar_mid", "relative_optical", "relative_structure",
        "output_optical_mae", "output_sar_mae", "high_relative_sar",
        "fused_low_rms", "fused_mid_rms", "fused_high_rms",
    )


def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    common = dict(patch_size=args.patch_size, invalid_threshold=0, min_valid_fraction=0.90)
    train_dataset = OpticalSARDataset(
        splits["train"], training=True, augment=True, limit=args.train_limit, **common
    )
    val_dataset = OpticalSARDataset(
        splits["val"], training=False, augment=False, limit=args.val_limit, **common
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    checkpoint = torch.load(args.phase1_checkpoint, map_location="cpu", weights_only=False)
    phase1_args = checkpoint.get("args", {})
    model = MFFAV1(max_relative_sar=float(phase1_args.get("max_relative_sar", 0.05)))
    model.load_state_dict(checkpoint["MFFAV1"]); model.to(device)
    configure_phase2(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    losses = OpticalSARLosses().to(device)
    reference = load_frequency_reference(args.frequency_reference_config)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    phase1_hash = sha256_file(args.phase1_checkpoint)
    manifest_hash = sha256_file(args.split_manifest)
    config = dict(vars(args)); config.update({
        "device": str(device), "train_pairs": len(train_dataset), "val_pairs": len(val_dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "phase1_checkpoint_sha256": phase1_hash,
        "split_manifest_sha256": manifest_hash,
        "split_manifest_metadata": manifest, "frequency_training_reference": reference,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(model, val_loader, losses, device, args, reference)
        print(json.dumps(stats, indent=2), flush=True); return
    fields = ("epoch",) + tuple(
        f"{prefix}_{name}" for name in metric_names() for prefix in ("train", "val")
    )
    history = run_dir / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, losses, device, args, reference, optimizer, scaler)
        val_stats = run_epoch(model, val_loader, losses, device, args, reference)
        row = {"epoch": epoch}
        for name in metric_names():
            row[f"train_{name}"] = train_stats[name]; row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(run_dir / "latest.pth", epoch, model, optimizer, args, best_val, phase1_hash, manifest_hash)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(run_dir / "best.pth", epoch, model, optimizer, args, best_val, phase1_hash, manifest_hash)
        print(
            f"epoch={epoch:03d} train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"rel_o={val_stats['relative_optical']:.4f} rel_s={val_stats['relative_structure']:.4f} "
            f"sar_high={val_stats['high_relative_sar']:.4f}", flush=True
        )


if __name__ == "__main__":
    main()
