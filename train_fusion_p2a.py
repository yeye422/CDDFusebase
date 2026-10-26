"""Two-stage trainer for P2-A source-preserving Detail residual fusion."""

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

from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
from p1_stable_detail_fusion import stability_teacher
from p2a_source_preserving_fusion import (
    PseudoSensingHeads,
    SourcePreservingDetailAdapter,
    weighted_normalized_l1,
)
from train_OS import OpticalSARLosses, checkpoint_model_state
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train P2-A source-preserving fusion")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--perturb-sigma", type=float, default=0.03)
    parser.add_argument("--max-relative-correction", type=float, default=0.03)
    parser.add_argument("--optical-feature-weight", type=float, default=0.25)
    parser.add_argument("--sar-feature-weight", type=float, default=0.25)
    parser.add_argument("--fusion-weight", type=float, default=0.25)
    parser.add_argument("--invariance-weight", type=float, default=0.05)
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.01)
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


def build_models(checkpoint, device, max_relative_correction=0.03):
    models = {
        "encoder": Restormer_Encoder().to(device),
        "decoder": Restormer_Decoder().to(device),
        "base_fuse": BaseFeatureExtraction(dim=64, num_heads=8).to(device),
        "detail_fuse": DetailFeatureExtraction(num_layers=1).to(device),
        "adapter": SourcePreservingDetailAdapter(
            max_relative_correction=max_relative_correction
        ).to(device),
        "pseudo_heads": PseudoSensingHeads().to(device),
    }
    models["encoder"].load_state_dict(checkpoint_model_state(checkpoint, "CDDF_Encoder", "DIDF_Encoder"))
    models["decoder"].load_state_dict(checkpoint_model_state(checkpoint, "CDDF_Decoder", "DIDF_Decoder"))
    models["base_fuse"].load_state_dict(checkpoint_model_state(checkpoint, "BaseFuseLayer"))
    models["detail_fuse"].load_state_dict(checkpoint_model_state(checkpoint, "DetailFuseLayer"))
    for name, model in models.items():
        if name not in ("adapter", "pseudo_heads"):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
    return models


def set_active(models, stage):
    for name in ("adapter", "pseudo_heads"):
        active = (stage == "heads" and name == "pseudo_heads") or (
            stage == "adapter" and name == "adapter"
        )
        models[name].train(active)
        for parameter in models[name].parameters():
            parameter.requires_grad_(active)


def perturb(sar, valid, sigma):
    noise = torch.randn_like(sar) * float(sigma)
    factor = torch.exp(noise - 0.5 * float(sigma) ** 2)
    changed = (sar * factor).clamp(0.0, 1.0)
    return torch.where(valid > 0.5, changed, sar)


@torch.no_grad()
def frozen_features(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    b0_detail = models["detail_fuse"](optical_detail + sar_detail)
    return optical_detail, sar_detail, fused_base, b0_detail


def forward_adapter(models, optical_detail, sar_detail, b0_detail):
    return models["adapter"](optical_detail, sar_detail, b0_detail)


def run_epoch(models, loader, losses, device, args, stage, optimizer=None, scaler=None):
    training = optimizer is not None
    set_active(models, stage if training else stage)
    if not training:
        models["adapter"].eval()
        models["pseudo_heads"].eval()
    totals = {name: 0.0 for name in (
        "loss", "optical_feature", "sar_feature", "fusion", "invariance",
        "anchor", "residual", "relative_correction"
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
                optical_detail, sar_detail, fused_base, b0_detail = frozen_features(
                    models, optical, sar
                )
                with torch.no_grad():
                    sar_1 = perturb(sar, valid, args.perturb_sigma)
                    sar_2 = perturb(sar, valid, args.perturb_sigma)
                    _, detail_1, _ = models["encoder"](sar_1)
                    _, detail_2, _ = models["encoder"](sar_2)
                    stable_weight = stability_teacher(detail_1, detail_2, valid)
                    stable_target = 0.5 * (detail_1 + detail_2)
                    b0_detail_1 = models["detail_fuse"](optical_detail + detail_1)
                    b0_detail_2 = models["detail_fuse"](optical_detail + detail_2)
                fused_detail, diagnostics = forward_adapter(
                    models, optical_detail, sar_detail, b0_detail
                )
                predicted_optical, predicted_sar = models["pseudo_heads"](fused_detail)
                feature_valid = F.interpolate(valid, size=fused_detail.shape[-2:], mode="nearest")
                optical_feature = weighted_normalized_l1(
                    predicted_optical, optical_detail, feature_valid
                )
                sar_feature = weighted_normalized_l1(
                    predicted_sar, stable_target, feature_valid * stable_weight
                )
                if stage == "heads":
                    fusion = fused_detail.sum() * 0.0
                    invariance = fusion
                    anchor = fusion
                    residual = fusion
                    total = optical_feature + sar_feature
                else:
                    generated, _ = models["decoder"](
                        optical, fused_base, fused_detail
                    )
                    with torch.no_grad():
                        b0_output, _ = models["decoder"](
                            optical, fused_base, b0_detail
                        )
                    fused_detail_1, _ = forward_adapter(
                        models, optical_detail, detail_1, b0_detail_1
                    )
                    fused_detail_2, _ = forward_adapter(
                        models, optical_detail, detail_2, b0_detail_2
                    )
                    invariance = F.l1_loss(fused_detail_1, fused_detail_2)
                    anchor = F.l1_loss(generated, b0_output)
                    residual = diagnostics["p2a_correction_abs_mean"]
                    (
                        fusion, _, _, _, _, _, _, _
                    ) = losses.fusion_frequency_decoupled(
                        optical, sar, generated, valid,
                        args.optical_lowpass_sigma,
                        args.frequency_highpass_sigma,
                        args.frequency_midpass_sigma,
                        args.optical_high_weight,
                        args.structure_percentile,
                        args.structure_charbonnier_epsilon,
                    )
                    total = (
                        args.optical_feature_weight * optical_feature
                        + args.sar_feature_weight * sar_feature
                        + args.fusion_weight * fusion
                        + args.invariance_weight * invariance
                        + args.baseline_anchor_weight * anchor
                        + args.residual_weight * residual
                    )
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                active_model = models["pseudo_heads"] if stage == "heads" else models["adapter"]
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            values = {
                "loss": total,
                "optical_feature": optical_feature,
                "sar_feature": sar_feature,
                "fusion": fusion,
                "invariance": invariance,
                "anchor": anchor,
                "residual": residual,
                "relative_correction": diagnostics["p2a_relative_correction"],
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
    if not batches:
        raise RuntimeError("P2-A loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, models, optimizers, args, source_hash, best_val):
    torch.save(
        {
            "epoch": int(epoch),
            "best_val": float(best_val),
            "variant": "p2a_source_preserving",
            "P2ASourceAdapter": models["adapter"].state_dict(),
            "P2APseudoHeads": models["pseudo_heads"].state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "args": vars(args),
            "source_checkpoint": str(args.init_checkpoint.resolve()),
            "source_checkpoint_sha256": source_hash,
        },
        path,
    )


def main():
    args = parse_args()
    if not 0 <= args.head_warmup_epochs < args.epochs:
        raise ValueError("head-warmup-epochs must be in [0, epochs)")
    set_seed(args.seed)
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
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    models = build_models(source, device, args.max_relative_correction)
    optimizers = {
        "heads": torch.optim.AdamW(
            models["pseudo_heads"].parameters(), lr=args.head_lr,
            weight_decay=args.weight_decay
        ),
        "adapter": torch.optim.AdamW(
            models["adapter"].parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        ),
    }
    scalers = {
        name: GradScaler(enabled=bool(args.amp and device.type == "cuda"))
        for name in optimizers
    }
    losses = OpticalSARLosses().to(device)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(args.init_checkpoint)
    config = dict(vars(args))
    config.update({
        "device": str(device), "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "adapter_parameters": sum(p.numel() for p in models["adapter"].parameters()),
        "pseudo_head_parameters": sum(p.numel() for p in models["pseudo_heads"].parameters()),
        "source_checkpoint_sha256": source_hash,
        "split_manifest_metadata": manifest,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        set_active(models, "adapter")
        stats = run_epoch(models, val_loader, losses, device, args, "adapter")
        print(json.dumps(stats, indent=2), flush=True)
        return
    names = (
        "epoch", "stage", "train_loss", "val_loss", "train_optical_feature",
        "val_optical_feature", "train_sar_feature", "val_sar_feature",
        "train_fusion", "val_fusion", "train_invariance", "val_invariance",
        "train_anchor", "val_anchor", "train_residual", "val_residual",
        "train_relative_correction", "val_relative_correction",
    )
    history = run_dir / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=names).writeheader()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        stage = "heads" if epoch <= args.head_warmup_epochs else "adapter"
        set_active(models, stage)
        train_stats = run_epoch(
            models, train_loader, losses, device, args, stage,
            optimizers[stage], scalers[stage]
        )
        val_stats = run_epoch(models, val_loader, losses, device, args, stage)
        row = {"epoch": epoch, "stage": stage}
        for name in (
            "loss", "optical_feature", "sar_feature", "fusion", "invariance",
            "anchor", "residual", "relative_correction"
        ):
            row[f"train_{name}"] = train_stats[name]
            row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=names).writerow(row)
        save_checkpoint(
            run_dir / "latest.pth", epoch, models, optimizers, args, source_hash, best_val
        )
        if stage == "adapter" and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, models, optimizers, args, source_hash, best_val
            )
        print(
            f"epoch={epoch:03d} stage={stage} train={train_stats['loss']:.6f} "
            f"val={val_stats['loss']:.6f} correction={val_stats['relative_correction']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
