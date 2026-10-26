"""Train the P1 stable-SAR Detail router with a frozen B0 backbone."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from p1_stable_detail_fusion import StableSARDetailRouter, stability_teacher
from train_OS import OpticalSARLosses, checkpoint_model_state
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train P1 stable Detail fusion")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--perturb-sigma", type=float, default=0.03)
    parser.add_argument("--stability-low-quantile", type=float, default=0.20)
    parser.add_argument("--stability-high-quantile", type=float, default=0.80)
    parser.add_argument("--stability-smooth-kernel", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.05)
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument("--structure-charbonnier-epsilon", type=float, default=1e-3)
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


def build_frozen_models(checkpoint, device):
    models = {
        "encoder": Restormer_Encoder().to(device),
        "decoder": Restormer_Decoder().to(device),
        "base_fuse": BaseFeatureExtraction(dim=64, num_heads=8).to(device),
        "detail_fuse": DetailFeatureExtraction(num_layers=1).to(device),
        "detail_router": StableSARDetailRouter().to(device),
    }
    models["encoder"].load_state_dict(
        checkpoint_model_state(checkpoint, "CDDF_Encoder", "DIDF_Encoder")
    )
    models["decoder"].load_state_dict(
        checkpoint_model_state(checkpoint, "CDDF_Decoder", "DIDF_Decoder")
    )
    models["base_fuse"].load_state_dict(
        checkpoint_model_state(checkpoint, "BaseFuseLayer")
    )
    models["detail_fuse"].load_state_dict(
        checkpoint_model_state(checkpoint, "DetailFuseLayer")
    )
    for name, model in models.items():
        trainable = name == "detail_router"
        model.train(trainable)
        for parameter in model.parameters():
            parameter.requires_grad_(trainable)
    return models


def perturb(sar, valid, sigma):
    noise = torch.randn_like(sar) * float(sigma)
    factor = torch.exp(noise - 0.5 * float(sigma) ** 2)
    changed = (sar * factor).clamp(0.0, 1.0)
    return torch.where(valid > 0.5, changed, sar)


def frozen_features(models, optical, sar):
    with torch.no_grad():
        optical_base, optical_detail, _ = models["encoder"](optical)
        sar_base, sar_detail, _ = models["encoder"](sar)
        fused_base = models["base_fuse"](optical_base + sar_base)
        optical_expert = models["detail_fuse"](optical_detail)
        joint_expert = models["detail_fuse"](optical_detail + sar_detail)
    return optical_detail, sar_detail, fused_base, optical_expert, joint_expert


def routed_output(models, optical, optical_detail, sar_detail, fused_base,
                  optical_expert, joint_expert, return_maps=False):
    fused_detail, diagnostics = models["detail_router"](
        optical_detail,
        sar_detail,
        optical_expert,
        joint_expert,
        return_maps=return_maps,
    )
    generated, _ = models["decoder"](optical, fused_base, fused_detail)
    return generated, diagnostics


def run_epoch(models, loader, losses, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    models["detail_router"].train(training)
    for name in ("encoder", "decoder", "base_fuse", "detail_fuse"):
        models[name].eval()
    totals = {
        "loss": 0.0,
        "teacher": 0.0,
        "fusion": 0.0,
        "consistency": 0.0,
        "anchor": 0.0,
        "gate": 0.0,
        "teacher_mean": 0.0,
        "optical": 0.0,
        "sar_mid": 0.0,
    }
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
                (
                    optical_detail,
                    sar_detail,
                    fused_base,
                    optical_expert,
                    joint_expert,
                ) = frozen_features(models, optical, sar)
                generated, diagnostics = routed_output(
                    models,
                    optical,
                    optical_detail,
                    sar_detail,
                    fused_base,
                    optical_expert,
                    joint_expert,
                    return_maps=True,
                )
                with torch.no_grad():
                    sar_1 = perturb(sar, valid, args.perturb_sigma)
                    sar_2 = perturb(sar, valid, args.perturb_sigma)
                    _, detail_1, _ = models["encoder"](sar_1)
                    _, detail_2, _ = models["encoder"](sar_2)
                    teacher = stability_teacher(
                        detail_1,
                        detail_2,
                        valid,
                        args.stability_low_quantile,
                        args.stability_high_quantile,
                        args.stability_smooth_kernel,
                    )
                    optical_expert_1 = optical_expert
                    optical_expert_2 = optical_expert
                    joint_expert_1 = models["detail_fuse"](
                        optical_detail + detail_1
                    )
                    joint_expert_2 = models["detail_fuse"](
                        optical_detail + detail_2
                    )
                reliability_map = diagnostics["p1_reliability_map"].mean(
                    dim=1, keepdim=True
                )
                teacher_loss = F.l1_loss(reliability_map, teacher)
                (
                    fusion_loss,
                    optical_loss,
                    _,
                    _,
                    _,
                    sar_mid_loss,
                    _,
                    _,
                ) = losses.fusion_frequency_decoupled(
                    optical,
                    sar,
                    generated,
                    valid,
                    args.optical_lowpass_sigma,
                    args.frequency_highpass_sigma,
                    args.frequency_midpass_sigma,
                    args.optical_high_weight,
                    args.structure_percentile,
                    args.structure_charbonnier_epsilon,
                )
                anchor_loss = F.l1_loss(generated, (
                    models["decoder"](optical, fused_base, joint_expert)[0]
                ).detach())
                generated_1, _ = routed_output(
                    models, optical, optical_detail, detail_1, fused_base,
                    optical_expert_1, joint_expert_1
                )
                generated_2, _ = routed_output(
                    models, optical, optical_detail, detail_2, fused_base,
                    optical_expert_2, joint_expert_2
                )
                consistency_loss = F.l1_loss(generated_1, generated_2)
                total = (
                    args.teacher_weight * teacher_loss
                    + args.fusion_weight * fusion_loss
                    + args.consistency_weight * consistency_loss
                    + args.baseline_anchor_weight * anchor_loss
                )
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    models["detail_router"].parameters(), args.clip_grad
                )
                scaler.step(optimizer)
                scaler.update()
            values = {
                "loss": total,
                "teacher": teacher_loss,
                "fusion": fusion_loss,
                "consistency": consistency_loss,
                "anchor": anchor_loss,
                "gate": diagnostics["p1_gate_mean"],
                "teacher_mean": teacher.mean(),
                "optical": optical_loss,
                "sar_mid": sar_mid_loss,
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
    if not batches:
        raise RuntimeError("P1 loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, models, optimizer, args, source_hash, best_val):
    torch.save(
        {
            "epoch": int(epoch),
            "best_val": float(best_val),
            "variant": "p1_stable_detail",
            "P1StableDetailRouter": models["detail_router"].state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "source_checkpoint_sha256": source_hash,
            "source_checkpoint": str(args.init_checkpoint.resolve()),
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    common = dict(
        patch_size=args.patch_size,
        invalid_threshold=0,
        min_valid_fraction=0.90,
    )
    train_dataset = OpticalSARDataset(
        splits["train"], training=True, augment=True,
        limit=args.train_limit, **common
    )
    val_dataset = OpticalSARDataset(
        splits["val"], training=False, augment=False,
        limit=args.val_limit, **common
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    models = build_frozen_models(source, device)
    trainable = list(models["detail_router"].parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    losses = OpticalSARLosses().to(device)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(args.init_checkpoint)
    config = dict(vars(args))
    config.update(
        {
            "device": str(device),
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "source_checkpoint_sha256": source_hash,
            "split_manifest_metadata": manifest,
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(models, val_loader, losses, device, args)
        print(json.dumps(stats, indent=2), flush=True)
        return
    history_path = run_dir / "history.csv"
    fieldnames = (
        "epoch", "train_loss", "val_loss", "train_teacher", "val_teacher",
        "train_fusion", "val_fusion", "train_consistency", "val_consistency",
        "train_anchor", "val_anchor", "train_gate", "val_gate",
        "train_teacher_mean", "val_teacher_mean",
    )
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            models, train_loader, losses, device, args, optimizer, scaler
        )
        val_stats = run_epoch(models, val_loader, losses, device, args)
        row = {"epoch": epoch}
        for name in ("loss", "teacher", "fusion", "consistency", "anchor", "gate", "teacher_mean"):
            row[f"train_{name}"] = train_stats[name]
            row[f"val_{name}"] = val_stats[name]
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        save_checkpoint(
            run_dir / "latest.pth", epoch, models, optimizer, args,
            source_hash, best_val
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, models, optimizer, args,
                source_hash, best_val
            )
        print(
            f"epoch={epoch:03d} train={train_stats['loss']:.6f} "
            f"val={val_stats['loss']:.6f} gate={val_stats['gate']:.4f} "
            f"teacher={val_stats['teacher_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
