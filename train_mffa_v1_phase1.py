"""Phase-1 autoencoding trainer for MFFA v1."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import kornia
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mffa_v1 import MFFAV1
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFFA v1 Phase 1 autoencoder")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--band-weight", type=float, default=0.25)
    parser.add_argument("--ssim-weight", type=float, default=0.5)
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--max-relative-sar", type=float, default=0.05)
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values, mask):
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(1)
    expanded = expanded.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


class Phase1Loss(nn.Module):
    def __init__(self, charbonnier_epsilon=1e-3):
        super().__init__()
        self.epsilon = float(charbonnier_epsilon)
        self.ssim = kornia.losses.SSIM(11, reduction="none")

    def charbonnier(self, difference):
        return torch.sqrt(difference.float().square() + self.epsilon ** 2).to(
            difference.dtype
        ) - self.epsilon

    def image_loss(self, prediction, target, valid, ssim_weight):
        # Kornia SSIM requires identical dtypes.  Under CUDA AMP the decoder
        # output is float16 while dataset targets remain float32, so compute
        # numerically sensitive image losses explicitly in float32.
        prediction_float = prediction.float()
        target_float = target.float()
        valid_float = valid.float()
        reconstruction = masked_mean(
            self.charbonnier(prediction_float - target_float), valid_float
        )
        with autocast(enabled=False):
            ssim_map = self.ssim(prediction_float, target_float)
        ssim = masked_mean(ssim_map, valid_float)
        return reconstruction + float(ssim_weight) * ssim, reconstruction, ssim

    @staticmethod
    def normalized_band_l1(prediction, target, valid, epsilon=1e-6):
        prediction_float = prediction.float()
        target_float = target.detach().float()
        target_rms = target_float.square().mean(
            dim=(1, 2, 3), keepdim=True
        ).sqrt().clamp_min(epsilon)
        return masked_mean(
            torch.abs(prediction_float - target_float) / target_rms,
            valid.float(),
        )

    def band_loss(self, decoded, bands):
        low = self.normalized_band_l1(
            decoded["low"], bands["low"], bands["low_valid"]
        )
        mid = self.normalized_band_l1(
            decoded["mid"], bands["mid"], bands["valid"]
        )
        high = self.normalized_band_l1(
            decoded["high"], bands["high"], bands["valid"]
        )
        return (low + mid + high) / 3.0, low, mid, high


def configure_phase1(model):
    frozen = {"low_fusion", "mid_fusion", "high_fusion"}
    for name, module in model.named_children():
        trainable = name not in frozen and name != "analysis"
        module.train(trainable)
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    model.low_fusion.eval(); model.mid_fusion.eval(); model.high_fusion.eval()
    totals = {name: 0.0 for name in (
        "loss", "optical_image", "sar_image", "optical_band", "sar_band",
        "optical_low", "optical_mid", "optical_high", "sar_low", "sar_mid", "sar_high"
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
            optical_valid = torch.ones_like(optical)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                optical_hat, optical_encoded, optical_decoded = model.reconstruct(
                    optical, "optical", optical_valid
                )
                sar_hat, sar_encoded, sar_decoded = model.reconstruct(
                    sar, "sar", valid
                )
                optical_image, _, _ = criterion.image_loss(
                    optical_hat, optical, optical_valid, args.ssim_weight
                )
                sar_image, _, _ = criterion.image_loss(
                    sar_hat, sar, valid, args.ssim_weight
                )
                optical_band, optical_low, optical_mid, optical_high = criterion.band_loss(
                    optical_decoded, optical_encoded["bands"]
                )
                sar_band, sar_low, sar_mid, sar_high = criterion.band_loss(
                    sar_decoded, sar_encoded["bands"]
                )
                total = (
                    optical_image + sar_image
                    + args.band_weight * (optical_band + sar_band)
                )
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip_grad
                )
                scaler.step(optimizer); scaler.update()
            values = {
                "loss": total, "optical_image": optical_image, "sar_image": sar_image,
                "optical_band": optical_band, "sar_band": sar_band,
                "optical_low": optical_low, "optical_mid": optical_mid,
                "optical_high": optical_high, "sar_low": sar_low,
                "sar_mid": sar_mid, "sar_high": sar_high,
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
    if not batches:
        raise RuntimeError("MFFA Phase-1 loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, model, optimizer, args, best_val, manifest_hash):
    torch.save(
        {
            "epoch": int(epoch), "best_val": float(best_val),
            "variant": "mffa_v1_phase1", "MFFAV1": model.state_dict(),
            "optimizer": optimizer.state_dict(), "args": vars(args),
            "split_manifest_sha256": manifest_hash,
        },
        path,
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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = MFFAV1(max_relative_sar=args.max_relative_sar).to(device)
    configure_phase1(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    criterion = Phase1Loss(args.charbonnier_epsilon).to(device)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args)); config.update({
        "device": str(device), "train_pairs": len(train_dataset), "val_pairs": len(val_dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_manifest_metadata": manifest,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(model, val_loader, criterion, device, args)
        print(json.dumps(stats, indent=2), flush=True); return
    fields = ("epoch",) + tuple(
        f"{prefix}_{name}" for name in totals_names() for prefix in ("train", "val")
    )
    history = run_dir / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_val = float("inf"); manifest_hash = sha256_file(args.split_manifest)
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, args, optimizer, scaler)
        val_stats = run_epoch(model, val_loader, criterion, device, args)
        row = {"epoch": epoch}
        for name in totals_names():
            row[f"train_{name}"] = train_stats[name]; row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(run_dir / "latest.pth", epoch, model, optimizer, args, best_val, manifest_hash)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(run_dir / "best.pth", epoch, model, optimizer, args, best_val, manifest_hash)
        print(f"epoch={epoch:03d} train={train_stats['loss']:.6f} val={val_stats['loss']:.6f}", flush=True)


def totals_names():
    return (
        "loss", "optical_image", "sar_image", "optical_band", "sar_band",
        "optical_low", "optical_mid", "optical_high", "sar_low", "sar_mid", "sar_high"
    )


if __name__ == "__main__":
    main()
