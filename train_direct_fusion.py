"""T0/T1 trainer for the source-preserving CDDFuse R0-R3 controls.

This is intentionally separate from train_OS.py so historical variants and
checkpoint semantics remain unchanged.  T0 adapts the Decoder to the selected
image-skip policy; T1 trains the fusion modules and Decoder while the shared
Encoder stays frozen.
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

from direct_fusion import DirectFusionCDDFuse
from utils.dataset_os import (
    OpticalSARDataset,
    digest_splits,
    manifest_pairs,
    split_pairs,
)
from utils.loss import Sobelxy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=DirectFusionCDDFuse.VARIANTS, default="r3")
    parser.add_argument("--data-root", type=Path, default=Path("../datasets/Dongying"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--optical-dir", default="GF2")
    parser.add_argument("--sar-dir", default="GF3")
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-epochs", type=int, default=1)
    parser.add_argument("--t1-epochs", type=int, default=5)
    parser.add_argument("--t0-decoder-lr", type=float, default=1e-4)
    parser.add_argument("--t1-fusion-lr", type=float, default=1e-4)
    parser.add_argument("--t1-decoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=10.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/direct_fusion"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-incomplete-restart",
        action="store_true",
        help=(
            "Restart a run directory that contains setup metadata but no epoch "
            "checkpoint. Intended for recovery from a first-epoch failure."
        ),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def deterministic_subset(pairs, limit, seed):
    pairs = list(pairs)
    if not limit or limit >= len(pairs):
        return pairs
    indices = random.Random(seed).sample(range(len(pairs)), limit)
    return [pairs[index] for index in sorted(indices)]


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class DirectFusionLosses(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = kornia.losses.SSIM(11, reduction="none")
        self.sobel = Sobelxy()

    def reconstruction(self, source, reconstructed, mask=None):
        # R0 becomes float32 when the AMP decoder output is added to the
        # float32 image skip, whereas direct R1-R3 outputs remain float16.
        # Kornia SSIM rejects mixed dtypes, so all loss arithmetic is kept in
        # float32 without changing the model forward dtype.
        with autocast(enabled=False):
            source = source.float()
            reconstructed = reconstructed.float()
            dssim = self.ssim(source, reconstructed)
            squared = (source - reconstructed).square()
        if mask is None:
            return 5.0 * dssim.mean() + squared.mean()
        return 5.0 * masked_mean(dssim, mask) + masked_mean(squared, mask)

    def fusion(self, optical, sar, fused, valid, gradient_weight):
        with autocast(enabled=False):
            optical = optical.float()
            sar = sar.float()
            fused = fused.float()
            valid = valid.float()
            intensity_target = torch.where(
                valid > 0.5, torch.maximum(optical, sar), optical
            )
            optical_gradient = self.sobel(optical)
            sar_gradient = self.sobel(sar)
            gradient_target = torch.where(
                valid > 0.5,
                torch.maximum(optical_gradient, sar_gradient),
                optical_gradient,
            )
            intensity = F.l1_loss(fused, intensity_target)
            gradient = F.l1_loss(self.sobel(fused), gradient_target)
        return intensity + float(gradient_weight) * gradient, intensity, gradient


def encode_frozen(model, optical, sar):
    with torch.no_grad():
        return model.encode_pair(optical, sar)


def forward_losses(model, losses, optical, sar, valid, stage, args):
    optical_base, optical_detail, sar_base, sar_detail = encode_frozen(
        model, optical, sar
    )
    optical_hat = model.reconstruct(optical, optical_base, optical_detail)
    sar_hat = model.reconstruct(sar, sar_base, sar_detail)
    optical_reconstruction = losses.reconstruction(optical, optical_hat)
    sar_reconstruction = losses.reconstruction(sar, sar_hat, valid)
    reconstruction = 0.5 * (optical_reconstruction + sar_reconstruction)
    components = {
        "reconstruction": reconstruction,
        "optical_reconstruction": optical_reconstruction,
        "sar_reconstruction": sar_reconstruction,
    }
    if stage == "t0":
        return reconstruction, components

    fused_base, fused_detail = model.fuse_features(
        optical_base, optical_detail, sar_base, sar_detail
    )
    decoder_input = optical if model.uses_image_skip else None
    fused = model.decoder(decoder_input, fused_base, fused_detail)[0]
    fusion, intensity, gradient = losses.fusion(
        optical, sar, fused, valid, args.gradient_weight
    )
    total = fusion + float(args.reconstruction_weight) * reconstruction
    components.update({"fusion": fusion, "intensity": intensity, "gradient": gradient})
    return total, components


def set_stage_learning_rates(optimizer, stage, args):
    rates = {
        "decoder": args.t0_decoder_lr if stage == "t0" else args.t1_decoder_lr,
        "base_fusion": 0.0 if stage == "t0" else args.t1_fusion_lr,
        "detail_fusion": 0.0 if stage == "t0" else args.t1_fusion_lr,
    }
    for group in optimizer.param_groups:
        group["lr"] = rates[group["group_name"]]


def make_optimizer(model, args):
    return torch.optim.AdamW(
        [
            {
                "params": model.decoder.parameters(),
                "lr": args.t0_decoder_lr,
                "group_name": "decoder",
            },
            {
                "params": model.base_fusion.parameters(),
                "lr": 0.0,
                "group_name": "base_fusion",
            },
            {
                "params": model.detail_fusion.parameters(),
                "lr": 0.0,
                "group_name": "detail_fusion",
            },
        ],
        weight_decay=args.weight_decay,
    )


def run_epoch(model, losses, loader, device, stage, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.encoder.eval()
    model.decoder.train(training)
    model.base_fusion.train(training and stage == "t1")
    model.detail_fusion.train(training and stage == "t1")
    totals = {}
    sample_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and not args.no_amp
    for batch_index, (optical, sar, valid, _) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast(enabled=amp_enabled):
                total, components = forward_losses(
                    model, losses, optical, sar, valid, stage, args
                )
                scaled_total = total / args.accumulation_steps
            if training:
                scaler.scale(scaled_total).backward()
                should_step = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
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
                    optimizer.zero_grad(set_to_none=True)
        batch_size = optical.shape[0]
        sample_count += batch_size
        values = {"loss": total, **components}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    return {name: value / sample_count for name, value in totals.items()}


def save_checkpoint(
    path, epoch, model, optimizer, scaler, args, lowest_val_loss
):
    torch.save(
        {
            "epoch": epoch,
            "variant": args.variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            # Diagnostic only. Architecture selection must use the frozen
            # image-fusion metrics, not this training objective.
            "lowest_val_loss": lowest_val_loss,
            "args": vars(args),
            "CDDF_Encoder": model.encoder.state_dict(),
            "CDDF_Decoder": model.decoder.state_dict(),
            "DirectBaseFusion": model.base_fusion.state_dict(),
            "DirectDetailFusion": model.detail_fusion.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    if args.t0_epochs < 0 or args.t1_epochs < 1:
        raise ValueError("t0-epochs must be nonnegative and t1-epochs must be positive")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps must be positive")
    set_seed(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = requested_device

    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        epoch_checkpoints = list(run_dir.glob("epoch_*.pth"))
        if not args.allow_incomplete_restart or epoch_checkpoints:
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
        print(f"restarting incomplete run with no epoch checkpoint: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.split_manifest:
        splits, split_metadata = manifest_pairs(args.data_root, args.split_manifest)
    else:
        splits = split_pairs(
            args.data_root,
            args.test_count,
            args.val_count,
            args.optical_dir,
            args.sar_dir,
        )
        split_metadata = None
    split_stems = {
        name: [pair[0].stem for pair in pairs]
        for name, pairs in splits.items()
    }
    saved_split_manifest = {
        "optical_dir": (
            split_metadata.get("optical_dir", args.optical_dir)
            if split_metadata
            else args.optical_dir
        ),
        "sar_dir": (
            split_metadata.get("sar_dir", args.sar_dir)
            if split_metadata
            else args.sar_dir
        ),
        "splits": split_stems,
        "splits_sha256": digest_splits(split_stems),
    }
    saved_split_path = run_dir / "split_manifest.json"
    saved_split_path.write_text(
        json.dumps(saved_split_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    train_pairs = deterministic_subset(splits["train"], args.train_limit, args.seed)
    val_pairs = list(splits["val"][: args.val_limit or None])
    subset_manifest = {
        "seed": args.seed,
        "source_split_sha256": saved_split_manifest["splits_sha256"],
        "train_stems": [pair[0].stem for pair in train_pairs],
        "val_stems": [pair[0].stem for pair in val_pairs],
    }
    subset_path = run_dir / "pilot_subset.json"
    subset_path.write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_dataset = OpticalSARDataset(
        train_pairs,
        patch_size=args.patch_size,
        training=True,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
    )
    val_dataset = OpticalSARDataset(
        val_pairs,
        patch_size=args.patch_size,
        training=False,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        persistent_workers=args.num_workers > 0,
    )

    model = DirectFusionCDDFuse(args.variant).to(device)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_backbone_checkpoint(checkpoint)
    model.freeze_encoder()
    losses = DirectFusionLosses().to(device)
    optimizer = make_optimizer(model, args)
    scaler = GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    start_epoch = 0
    lowest_val_loss = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        if resume.get("variant") != args.variant:
            raise ValueError("Resume checkpoint variant mismatch")
        model.load_state_dict(resume["model"])
        model.freeze_encoder()
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        lowest_val_loss = float(
            resume.get("lowest_val_loss", resume.get("best_val", lowest_val_loss))
        )

    parameter_counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in (
            ("encoder", model.encoder),
            ("decoder", model.decoder),
            ("base_fusion", model.base_fusion),
            ("detail_fusion", model.detail_fusion),
        )
    }
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "init_checkpoint": str(args.init_checkpoint),
        "resume": str(args.resume) if args.resume else None,
        "init_checkpoint_sha256": sha256_file(args.init_checkpoint),
        "split_manifest_sha256": (
            sha256_file(args.split_manifest) if args.split_manifest else None
        ),
        "saved_split_manifest": str(saved_split_path),
        "saved_split_manifest_sha256": sha256_file(saved_split_path),
        "pilot_subset_sha256": sha256_file(subset_path),
        "split_metadata": (
            {key: value for key, value in split_metadata.items() if key != "splits"}
            if split_metadata
            else None
        ),
        "parameter_counts": parameter_counts,
        "total_parameters": sum(parameter_counts.values()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"variant={args.variant}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"parameters={config['total_parameters']}, trainable={config['trainable_parameters']}"
    )
    if args.dry_run:
        optical, sar, _, _ = next(iter(val_loader))
        with torch.no_grad():
            output = model(optical.to(device), sar.to(device))
        print(f"dry run output={tuple(output.shape)}")
        return

    history_path = run_dir / "history.csv"
    fieldnames = (
        "epoch",
        "stage",
        "seconds",
        "train_loss",
        "train_fusion",
        "train_intensity",
        "train_gradient",
        "train_reconstruction",
        "val_loss",
        "val_fusion",
        "val_intensity",
        "val_gradient",
        "val_reconstruction",
    )
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    total_epochs = args.t0_epochs + args.t1_epochs
    for epoch in range(start_epoch, total_epochs):
        stage = "t0" if epoch < args.t0_epochs else "t1"
        set_stage_learning_rates(optimizer, stage, args)
        started = time.time()
        train_stats = run_epoch(
            model, losses, train_loader, device, stage, args, optimizer, scaler
        )
        with torch.no_grad():
            val_stats = run_epoch(model, losses, val_loader, device, stage, args)
        seconds = time.time() - started
        row = {"epoch": epoch + 1, "stage": stage, "seconds": seconds}
        for prefix, stats in (("train", train_stats), ("val", val_stats)):
            for name in ("loss", "fusion", "intensity", "gradient", "reconstruction"):
                row[f"{prefix}_{name}"] = stats.get(name, "")
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        is_lowest_loss = (
            stage == "t1" and val_stats["loss"] < lowest_val_loss
        )
        if is_lowest_loss:
            lowest_val_loss = val_stats["loss"]
        save_checkpoint(
            run_dir / f"epoch_{epoch + 1:03d}.pth",
            epoch,
            model,
            optimizer,
            scaler,
            args,
            lowest_val_loss,
        )
        if is_lowest_loss:
            save_checkpoint(
                run_dir / "lowest_val_loss_diagnostic.pth",
                epoch,
                model,
                optimizer,
                scaler,
                args,
                lowest_val_loss,
            )
        print(
            f"epoch={epoch + 1}/{total_epochs} stage={stage} "
            f"train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"seconds={seconds:.1f}"
        )


if __name__ == "__main__":
    main()
