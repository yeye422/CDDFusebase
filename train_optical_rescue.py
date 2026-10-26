"""Train Optical Rescue on top of the frozen retrained Detail-only SPD path."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from compare_spd_fusion import load_models, sha256_file
from optical_rescue import OpticalRescueAdapter, RescueProbeHeads, normalized_l1
from train_OS import OpticalSARLosses
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train Optical Rescue")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-relative-correction", type=float, default=0.005)
    parser.add_argument("--optical-feature-weight", type=float, default=1.0)
    parser.add_argument("--sar-feature-weight", type=float, default=0.5)
    parser.add_argument("--fusion-weight", type=float, default=0.1)
    parser.add_argument("--output-anchor-weight", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument("--structure-charbonnier-epsilon", type=float, default=1e-3)
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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def set_active(adapter, heads, stage, training):
    if stage not in {"heads", "adapter"}:
        raise ValueError(f"Unknown Optical Rescue stage: {stage}")
    # Trainability belongs to the optimization stage, not to whether this is a
    # train or validation pass. Validation is already protected by no_grad().
    # Keeping these two concerns separate prevents a validation pass from
    # leaving the next training pass with a completely detached graph.
    adapter_active = stage == "adapter"
    heads_active = stage == "heads"
    adapter.train(adapter_active and training)
    heads.train(heads_active and training)
    for parameter in adapter.parameters(): parameter.requires_grad_(adapter_active)
    for parameter in heads.parameters(): parameter.requires_grad_(heads_active)


@torch.no_grad()
def frozen_path(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, _ = models["spd"](sar_base, sar_detail)
    fused_base = models["base_fuse"](optical_base + clean_base)
    fused_detail = models["detail_fuse"](optical_detail + clean_detail)
    baseline, _ = models["decoder"](optical, fused_base, fused_detail)
    return optical_detail, clean_detail, fused_base, fused_detail, baseline


def run_epoch(models, adapter, heads, loader, losses, device, args, stage,
              optimizer=None, scaler=None):
    training = optimizer is not None
    set_active(adapter, heads, stage, training)
    totals = {name: 0.0 for name in (
        "loss", "optical_feature", "sar_feature", "fusion", "output_anchor",
        "residual", "relative_correction"
    )}
    batches = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    amp_enabled = bool(args.amp and device.type == "cuda")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for optical, sar, valid, _ in loader:
            if maximum is not None and batches >= maximum: break
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                optical_detail, clean_detail, fused_base, fused_detail, baseline = frozen_path(
                    models, optical, sar
                )
                if stage == "heads":
                    # Probe warm-up must have a direct graph from the loss to
                    # the probe parameters. The Rescue adapter is intentionally
                    # absent from this stage.
                    rescued_detail = fused_detail.detach()
                    predicted_optical, predicted_sar = heads(rescued_detail)
                    optical_feature = normalized_l1(predicted_optical, optical_detail)
                    sar_feature = normalized_l1(predicted_sar, clean_detail)
                    zero = fused_detail.new_zeros(())
                    fusion = zero; output_anchor = zero; residual = zero
                    diagnostics = {"rescue_relative_rms": zero}
                    total = optical_feature + sar_feature
                else:
                    with torch.no_grad():
                        optical_from_baseline, _ = heads(fused_detail)
                        optical_error = optical_detail - optical_from_baseline
                    rescued_detail, diagnostics = adapter(
                        optical_detail, clean_detail, fused_detail, optical_error
                    )
                    predicted_optical, predicted_sar = heads(rescued_detail)
                    optical_feature = normalized_l1(predicted_optical, optical_detail)
                    sar_feature = normalized_l1(predicted_sar, clean_detail)
                    generated, _ = models["decoder"](
                        optical, fused_base, rescued_detail
                    )
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
                    output_anchor = F.l1_loss(generated, baseline.detach())
                    residual = diagnostics["rescue_abs_mean"]
                    total = (
                        args.optical_feature_weight * optical_feature
                        + args.sar_feature_weight * sar_feature
                        + args.fusion_weight * fusion
                        + args.output_anchor_weight * output_anchor
                        + args.residual_weight * residual
                    )
            if training:
                if not total.requires_grad or total.grad_fn is None:
                    active = heads if stage == "heads" else adapter
                    trainable = sum(p.requires_grad for p in active.parameters())
                    raise RuntimeError(
                        f"{stage} loss is detached before backward; "
                        f"active_trainable_tensors={trainable}, "
                        f"grad_enabled={torch.is_grad_enabled()}"
                    )
                scaler.scale(total).backward(); scaler.unscale_(optimizer)
                active = heads if stage == "heads" else adapter
                torch.nn.utils.clip_grad_norm_(active.parameters(), args.clip_grad)
                scaler.step(optimizer); scaler.update()
            values = {
                "loss": total, "optical_feature": optical_feature,
                "sar_feature": sar_feature, "fusion": fusion,
                "output_anchor": output_anchor, "residual": residual,
                "relative_correction": diagnostics["rescue_relative_rms"],
            }
            for name, value in values.items(): totals[name] += float(value.detach())
            batches += 1
    if not batches: raise RuntimeError("Optical Rescue loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, adapter, heads, optimizers, args, best_val,
                    b0_hash, spd_hash, manifest_hash):
    torch.save({
        "epoch": int(epoch), "best_val": float(best_val),
        "variant": "optical_rescue_detail_spd",
        "OpticalRescueAdapter": adapter.state_dict(),
        "OpticalRescueHeads": heads.state_dict(),
        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
        "args": vars(args), "b0_checkpoint_sha256": b0_hash,
        "spd_checkpoint_sha256": spd_hash,
        "split_manifest_sha256": manifest_hash,
    }, path)


def names():
    return ("loss", "optical_feature", "sar_feature", "fusion", "output_anchor",
            "residual", "relative_correction")


def main():
    args = parse_args()
    if not 0 <= args.head_warmup_epochs < args.epochs:
        raise ValueError("head-warmup-epochs must be in [0, epochs)")
    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    common = dict(patch_size=args.patch_size, invalid_threshold=0, min_valid_fraction=0.90)
    train_dataset = OpticalSARDataset(splits["train"], training=True, augment=True,
                                      limit=args.train_limit, **common)
    val_dataset = OpticalSARDataset(splits["val"], training=False, augment=False,
                                    limit=args.val_limit, **common)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    adapter = OpticalRescueAdapter(max_relative_correction=args.max_relative_correction).to(device)
    heads = RescueProbeHeads().to(device)
    optimizers = {
        "heads": torch.optim.AdamW(heads.parameters(), lr=args.head_lr, weight_decay=args.weight_decay),
        "adapter": torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    }
    scalers = {name: GradScaler(enabled=bool(args.amp and device.type == "cuda")) for name in optimizers}
    losses = OpticalSARLosses().to(device)
    run_dir = args.output_dir / args.run_name; run_dir.mkdir(parents=True, exist_ok=False)
    manifest_hash = sha256_file(args.split_manifest)
    config = dict(vars(args)); config.update({
        "device": str(device), "train_pairs": len(train_dataset), "val_pairs": len(val_dataset),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "probe_parameters": sum(p.numel() for p in heads.parameters()),
        "b0_checkpoint_sha256": b0_hash, "spd_checkpoint_sha256": spd_hash,
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
        "split_manifest_sha256": manifest_hash, "split_manifest_metadata": manifest,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        stats = run_epoch(models, adapter, heads, val_loader, losses, device, args, "adapter")
        print(json.dumps(stats, indent=2), flush=True); return
    fields = ("epoch", "stage") + tuple(
        f"{prefix}_{name}" for name in names() for prefix in ("train", "val")
    )
    history = run_dir / "history.csv"
    with history.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        stage = "heads" if epoch <= args.head_warmup_epochs else "adapter"
        train_stats = run_epoch(models, adapter, heads, train_loader, losses, device, args,
                                stage, optimizers[stage], scalers[stage])
        val_stats = run_epoch(models, adapter, heads, val_loader, losses, device, args, stage)
        row = {"epoch": epoch, "stage": stage}
        for name in names():
            row[f"train_{name}"] = train_stats[name]; row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(run_dir / "latest.pth", epoch, adapter, heads, optimizers, args,
                        best_val, b0_hash, spd_hash, manifest_hash)
        if stage == "adapter" and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(run_dir / "best.pth", epoch, adapter, heads, optimizers, args,
                            best_val, b0_hash, spd_hash, manifest_hash)
        print(f"epoch={epoch:03d} stage={stage} train={train_stats['loss']:.6f} "
              f"val={val_stats['loss']:.6f} correction={val_stats['relative_correction']:.6f}", flush=True)


if __name__ == "__main__": main()
