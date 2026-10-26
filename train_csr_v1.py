"""Train SPD-guided Competitive Structure Routing on a frozen Detail-SPD anchor."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from compare_spd_fusion import load_models, sha256_file
from competitive_structure_routing import (
    CompetitiveStructureRouter,
    competitive_structure_losses,
)
from utils.dataset_os import OpticalSARDataset, manifest_pairs


LOSS_NAMES = (
    "loss", "optical_visibility", "sar_retention", "unsupported", "anchor",
    "low_frequency", "routing", "negative_scale", "mid_scale", "high_scale",
    "optical_gate", "correction_rms",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSR-v1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-mid-replacement", type=float, default=0.10)
    parser.add_argument("--max-high-replacement", type=float, default=0.03)
    parser.add_argument("--mid-sigma", type=float, default=3.0)
    parser.add_argument("--high-sigma", type=float, default=1.0)
    parser.add_argument("--optical-weight", type=float, default=1.0)
    parser.add_argument("--sar-weight", type=float, default=0.5)
    parser.add_argument("--unsupported-weight", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.3)
    parser.add_argument("--low-frequency-weight", type=float, default=0.5)
    parser.add_argument("--routing-weight", type=float, default=0.1)
    parser.add_argument("--negative-scale-weight", type=float, default=1.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def frozen_anchor(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    raw_base, raw_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, _ = models["spd"](raw_base, raw_detail)
    fused_base = models["base_fuse"](optical_base + clean_base)
    fused_detail = models["detail_fuse"](optical_detail + clean_detail)
    baseline, _ = models["decoder"](optical, fused_base, fused_detail)
    return baseline, raw_detail, clean_detail


def total_loss(parts, args):
    return (
        args.optical_weight * parts["optical_visibility"]
        + args.sar_weight * parts["sar_retention"]
        + args.unsupported_weight * parts["unsupported"]
        + args.anchor_weight * parts["anchor"]
        + args.low_frequency_weight * parts["low_frequency"]
        + args.routing_weight * parts["routing"]
        + args.negative_scale_weight * parts["negative_scale"]
    )


def run_epoch(models, router, loader, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    router.train(training)
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    totals = {name: 0.0 for name in LOSS_NAMES}
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
                baseline, raw_detail, clean_detail = frozen_anchor(models, optical, sar)
                output, diagnostics = router(
                    optical, sar, baseline, raw_detail, clean_detail,
                    valid=valid, return_maps=True,
                )
                # Structural losses remain float32 under AMP for stable Sobel/blur.
                parts = competitive_structure_losses(
                    output.float(), baseline.float(), optical.float(), valid.float(),
                    {key: value.float() for key, value in diagnostics.items()},
                    low_sigma=args.mid_sigma,
                )
                total = total_loss(parts, args)
            if training:
                if not total.requires_grad:
                    raise RuntimeError("CSR loss is detached before backward")
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(router.parameters(), args.clip_grad)
                scaler.step(optimizer); scaler.update()
            values = {
                "loss": total,
                **parts,
                "mid_scale": diagnostics["csr_mid_scale"],
                "high_scale": diagnostics["csr_high_scale"],
                "optical_gate": diagnostics["csr_optical_gate_mean"],
                "correction_rms": diagnostics["csr_correction_rms"],
            }
            for name in LOSS_NAMES:
                totals[name] += float(values[name].detach())
            batches += 1
    if not batches:
        raise RuntimeError("CSR loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, router, optimizer, args, best_val, hashes):
    torch.save({
        "epoch": int(epoch), "best_val": float(best_val), "variant": "csr_v1",
        "CompetitiveStructureRouter": router.state_dict(),
        "optimizer": optimizer.state_dict(), "args": vars(args), **hashes,
    }, path)


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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    router = CompetitiveStructureRouter(
        max_mid_replacement=args.max_mid_replacement,
        max_high_replacement=args.max_high_replacement,
        mid_sigma=args.mid_sigma, high_sigma=args.high_sigma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    hashes = {
        "b0_checkpoint_sha256": b0_hash,
        "spd_checkpoint_sha256": spd_hash,
        "split_manifest_sha256": sha256_file(args.split_manifest),
    }
    config = dict(vars(args)); config.update({
        "device": str(device), "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "trainable_parameters": sum(p.numel() for p in router.parameters()),
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
        "split_manifest_metadata": manifest, **hashes,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        print(json.dumps(
            run_epoch(models, router, val_loader, device, args), indent=2
        ), flush=True)
        return
    fields = ("epoch",) + tuple(
        f"{prefix}_{name}" for name in LOSS_NAMES for prefix in ("train", "val")
    )
    history = run_dir / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            models, router, train_loader, device, args, optimizer, scaler
        )
        val_stats = run_epoch(models, router, val_loader, device, args)
        row = {"epoch": epoch}
        for name in LOSS_NAMES:
            row[f"train_{name}"] = train_stats[name]
            row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(
            run_dir / "latest.pth", epoch, router, optimizer, args,
            best_val, hashes,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, router, optimizer, args,
                best_val, hashes,
            )
        print(
            f"epoch={epoch:03d} train={train_stats['loss']:.6f} "
            f"val={val_stats['loss']:.6f} mid={val_stats['mid_scale']:.5f} "
            f"high={val_stats['high_scale']:.5f} "
            f"gate={val_stats['optical_gate']:.5f}", flush=True,
        )


if __name__ == "__main__":
    main()
