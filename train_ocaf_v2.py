"""Train OCAF-v2 independent additive SAR gates."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from compare_spd_fusion import load_models, sha256_file
from ocaf_v2 import OpticalCanvasFusionV2, ocaf_v2_losses
from train_ocaf_v1 import frozen_paths, perturb_sar, set_seed
from utils.dataset_os import OpticalSARDataset, manifest_pairs


LOSS_NAMES = (
    "loss", "low_identity", "optical_preservation", "common_injection",
    "sar_injection", "gate_alignment", "common_budget", "exclusive_budget",
    "removed_suppression", "unsupported", "negative_amplitude",
    "common_amplitude", "exclusive_mid_amplitude", "exclusive_high_amplitude",
    "common_gate", "sar_gate", "common_gate_coverage_050",
    "sar_gate_coverage_050", "common_score", "exclusive_score", "novelty",
    "stability", "common_delta_rms", "exclusive_delta_rms", "total_delta_rms",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train OCAF-v2")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-common-amplitude", type=float, default=2.0 / 255.0)
    parser.add_argument("--max-exclusive-mid-amplitude", type=float, default=1.5 / 255.0)
    parser.add_argument("--max-exclusive-high-amplitude", type=float, default=1.0 / 255.0)
    parser.add_argument("--target-common-rms", type=float, default=0.5 / 255.0)
    parser.add_argument("--target-exclusive-rms", type=float, default=0.25 / 255.0)
    parser.add_argument("--high-sigma", type=float, default=1.0)
    parser.add_argument("--low-sigma", type=float, default=3.0)
    parser.add_argument("--perturb-sigma", type=float, default=0.03)
    parser.add_argument("--low-weight", type=float, default=1.0)
    parser.add_argument("--optical-weight", type=float, default=0.5)
    parser.add_argument("--common-weight", type=float, default=0.8)
    parser.add_argument("--sar-weight", type=float, default=1.0)
    parser.add_argument("--gate-alignment-weight", type=float, default=0.01)
    parser.add_argument("--common-budget-weight", type=float, default=1.0)
    parser.add_argument("--exclusive-budget-weight", type=float, default=2.0)
    parser.add_argument("--removed-weight", type=float, default=0.1)
    parser.add_argument("--unsupported-weight", type=float, default=0.5)
    parser.add_argument("--negative-amplitude-weight", type=float, default=1.0)
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


def total_loss(parts, args):
    return (
        args.low_weight * parts["low_identity"]
        + args.optical_weight * parts["optical_preservation"]
        + args.common_weight * parts["common_injection"]
        + args.sar_weight * parts["sar_injection"]
        + args.gate_alignment_weight * parts["gate_alignment"]
        + args.common_budget_weight * parts["common_budget"]
        + args.exclusive_budget_weight * parts["exclusive_budget"]
        + args.removed_weight * parts["removed_suppression"]
        + args.unsupported_weight * parts["unsupported"]
        + args.negative_amplitude_weight * parts["negative_amplitude"]
    )


def run_epoch(models, ocaf, loader, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    ocaf.train(training)
    totals = {name: 0.0 for name in LOSS_NAMES}
    batches = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    amp_enabled = bool(args.amp and device.type == "cuda")
    perturb_generator = torch.Generator(device=device)
    perturb_generator.manual_seed(int(args.seed) + (29 if training else 1000029))
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
                baseline, clean_sar, raw_detail, clean_detail = frozen_paths(
                    models, optical, sar
                )
                perturbed = perturb_sar(
                    sar, valid, args.perturb_sigma, perturb_generator
                )
                _, counterfactual_clean_sar, _, _ = frozen_paths(
                    models, optical, perturbed
                )
                output, diagnostics = ocaf(
                    optical, clean_sar, counterfactual_clean_sar, baseline,
                    raw_detail, clean_detail, valid=valid, return_maps=True,
                )
                float_diagnostics = {
                    key: value.float() for key, value in diagnostics.items()
                }
                parts = ocaf_v2_losses(
                    output.float(), optical.float(), valid.float(),
                    float_diagnostics,
                    target_common_rms=args.target_common_rms,
                    target_exclusive_rms=args.target_exclusive_rms,
                    high_sigma=args.high_sigma, low_sigma=args.low_sigma,
                )
                total = total_loss(parts, args)
            if training:
                if not total.requires_grad:
                    raise RuntimeError("OCAF-v2 loss is detached before backward")
                scaler.scale(total).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ocaf.parameters(), args.clip_grad)
                scaler.step(optimizer); scaler.update()
            values = {
                "loss": total, **parts,
                "common_amplitude": diagnostics["ocaf_common_amplitude"],
                "exclusive_mid_amplitude": diagnostics["ocaf_exclusive_mid_amplitude"],
                "exclusive_high_amplitude": diagnostics["ocaf_exclusive_high_amplitude"],
                "common_gate": diagnostics["ocaf_common_gate_mean"],
                "sar_gate": diagnostics["ocaf_sar_gate_mean"],
                "common_gate_coverage_050": diagnostics["ocaf_common_gate_coverage_050"],
                "sar_gate_coverage_050": diagnostics["ocaf_sar_gate_coverage_050"],
                "common_score": diagnostics["ocaf_common_score_mean"],
                "exclusive_score": diagnostics["ocaf_exclusive_score_mean"],
                "novelty": diagnostics["ocaf_novelty_mean"],
                "stability": diagnostics["ocaf_stability_mean"],
                "common_delta_rms": diagnostics["ocaf_common_delta_rms"],
                "exclusive_delta_rms": diagnostics["ocaf_exclusive_delta_rms"],
                "total_delta_rms": diagnostics["ocaf_total_delta_rms"],
            }
            for name in LOSS_NAMES:
                totals[name] += float(values[name].detach())
            batches += 1
    if not batches:
        raise RuntimeError("OCAF-v2 loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(path, epoch, ocaf, optimizer, args, best_val, hashes):
    torch.save({
        "epoch": int(epoch), "best_val": float(best_val), "variant": "ocaf_v2",
        "OpticalCanvasFusionV2": ocaf.state_dict(),
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
    ocaf = OpticalCanvasFusionV2(
        max_common_amplitude=args.max_common_amplitude,
        max_exclusive_mid_amplitude=args.max_exclusive_mid_amplitude,
        max_exclusive_high_amplitude=args.max_exclusive_high_amplitude,
        high_sigma=args.high_sigma, low_sigma=args.low_sigma,
    ).to(device)
    optimizer = torch.optim.AdamW(
        ocaf.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
        "trainable_parameters": sum(p.numel() for p in ocaf.parameters()),
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
        "split_manifest_metadata": manifest, **hashes,
    })
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)
    if args.dry_run:
        args.max_val_batches = 1
        print(json.dumps(
            run_epoch(models, ocaf, val_loader, device, args), indent=2
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
            models, ocaf, train_loader, device, args, optimizer, scaler
        )
        val_stats = run_epoch(models, ocaf, val_loader, device, args)
        row = {"epoch": epoch}
        for name in LOSS_NAMES:
            row[f"train_{name}"] = train_stats[name]
            row[f"val_{name}"] = val_stats[name]
        with history.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(
            run_dir / "latest.pth", epoch, ocaf, optimizer, args,
            best_val, hashes,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, ocaf, optimizer, args,
                best_val, hashes,
            )
        print(
            f"epoch={epoch:03d} train={train_stats['loss']:.6f} "
            f"val={val_stats['loss']:.6f} common_gate={val_stats['common_gate']:.4f} "
            f"sar_gate={val_stats['sar_gate']:.4f} "
            f"delta={val_stats['total_delta_rms']:.6f}", flush=True,
        )


if __name__ == "__main__":
    main()
