"""Train the PDCF P2 minimal single-path fusion or raw-feature control."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pdcf_model import PDCFModel
from pdcf_predictability import BidirectionalPredictability
from pdcf_single_path import PDCFSinglePathModel, SinglePathFusionAdapter
from train_OS import OpticalSARLosses, reference_normalized_smooth_minimax
from train_pdcf_predictability import sha256_file, worker_seed
from utils.dataset_os import OpticalSARDataset, manifest_pairs


HISTORY_METRICS = (
    "loss", "minimax", "optical", "structure", "relative_optical",
    "relative_structure", "anchor", "correction", "relative_base_correction",
    "relative_detail_correction", "output_deviation_rms",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PDCF P2 single-path fusion")
    parser.add_argument(
        "--input-mode", choices=SinglePathFusionAdapter.INPUT_MODES, required=True
    )
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--max-relative-correction", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-weight", type=float, default=0.0)
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument("--structure-charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--minimax-temperature", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_predictor(checkpoint_path, c0_hash, split_hash, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("source_checkpoint_sha256") != c0_hash:
        raise ValueError("P1 predictor source does not match the requested C0")
    if checkpoint.get("split_manifest_sha256") != split_hash:
        raise ValueError("P1 predictor split does not match the requested manifest")
    saved_args = checkpoint["args"]
    predictor = BidirectionalPredictability(
        channels=64,
        hidden_channels=int(saved_args["hidden_channels"]),
        blocks=int(saved_args["predictor_blocks"]),
    ).to(device)
    predictor.load_state_dict(checkpoint["PDCFPredictability"])
    predictor.eval()
    return predictor, checkpoint


def make_loaders(args, device):
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    common = dict(
        patch_size=args.patch_size,
        invalid_threshold=0,
        min_valid_fraction=0.9,
    )
    train_dataset = OpticalSARDataset(
        splits["train"], training=True, augment=True,
        limit=args.train_limit, **common
    )
    val_dataset = OpticalSARDataset(
        splits["val"], training=False, augment=False,
        limit=args.val_limit, **common
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        persistent_workers=args.num_workers > 0,
    )
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
        manifest,
    )


def run_epoch(model, loader, losses, reference, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in HISTORY_METRICS}
    samples = 0
    maximum = args.max_train_batches if training else args.max_val_batches
    for batch_index, (optical, sar, valid, _) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            generated, _, diagnostics = model(
                optical, sar, return_diagnostics=True
            )
            (
                _, optical_loss, _, _, _, structure_loss, _, _
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
            minimax, relative_optical, relative_structure = (
                reference_normalized_smooth_minimax(
                    optical_loss,
                    structure_loss,
                    reference["optical"],
                    reference["structure"],
                    args.minimax_temperature,
                )
            )
            anchor = F.l1_loss(generated, diagnostics["c0_output"])
            correction = 0.5 * (
                diagnostics["relative_base_correction"]
                + diagnostics["relative_detail_correction"]
            )
            total = minimax
            # Do not attach disabled regularizers to the autograd graph.  In
            # particular, norm-like diagnostics may have delicate zero points.
            if args.anchor_weight:
                total = total + args.anchor_weight * anchor
            if args.correction_weight:
                total = total + args.correction_weight * correction
            finite_values = {
                "total": total,
                "minimax": minimax,
                "optical": optical_loss,
                "structure": structure_loss,
                "relative_optical": relative_optical,
                "relative_structure": relative_structure,
            }
            nonfinite = [
                name for name, value in finite_values.items()
                if not torch.isfinite(value).all()
            ]
            if nonfinite:
                raise FloatingPointError(
                    f"Non-finite P2 values before backward: {nonfinite}"
                )
            if training:
                total.backward()
                bad_gradients = [
                    name for name, parameter in model.adapter.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                if bad_gradients:
                    raise FloatingPointError(
                        "Non-finite P2 gradients: " + ", ".join(bad_gradients)
                    )
                torch.nn.utils.clip_grad_norm_(
                    model.adapter.parameters(), args.clip_grad
                )
                optimizer.step()
                bad_parameters = [
                    name for name, parameter in model.adapter.named_parameters()
                    if not torch.isfinite(parameter).all()
                ]
                if bad_parameters:
                    raise FloatingPointError(
                        "Non-finite P2 parameters after optimizer step: "
                        + ", ".join(bad_parameters)
                    )
        values = {
            "loss": total,
            "minimax": minimax,
            "optical": optical_loss,
            "structure": structure_loss,
            "relative_optical": relative_optical,
            "relative_structure": relative_structure,
            "anchor": anchor,
            "correction": correction,
            "relative_base_correction": diagnostics["relative_base_correction"],
            "relative_detail_correction": diagnostics["relative_detail_correction"],
            "output_deviation_rms": diagnostics["output_deviation_rms"],
        }
        batch_size = optical.shape[0]
        samples += batch_size
        for name, value in values.items():
            totals[name] += float(value.detach()) * batch_size
        if maximum and batch_index + 1 >= maximum:
            break
    if not samples:
        raise RuntimeError("P2 loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


def save_checkpoint(path, epoch, model, optimizer, args, hashes, best):
    torch.save(
        {
            "epoch": epoch,
            "best_val": best,
            "variant": "pdcf_single_path",
            "input_mode": args.input_mode,
            "PDCFSinglePathAdapter": model.adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            **hashes,
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c0_hash = sha256_file(args.c0_checkpoint)
    p1_hash = sha256_file(args.p1_checkpoint)
    split_hash = sha256_file(args.split_manifest)
    c0_checkpoint = torch.load(
        args.c0_checkpoint, map_location="cpu", weights_only=False
    )
    reference = c0_checkpoint.get("frequency_training_reference")
    if not reference:
        raise ValueError("C0 checkpoint has no frequency_training_reference")
    c0 = PDCFModel.from_c0_checkpoint(
        args.c0_checkpoint, map_location=device
    ).to(device)
    predictor, _ = load_predictor(
        args.p1_checkpoint, c0_hash, split_hash, device
    )
    model = PDCFSinglePathModel(
        c0,
        predictor,
        input_mode=args.input_mode,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        max_relative_correction=args.max_relative_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    losses = OpticalSARLosses().to(device)
    train_loader, val_loader, manifest = make_loaders(args, device)
    if args.dry_run:
        before_update = run_epoch(
            model, train_loader, losses, reference, device, args,
            optimizer=optimizer,
        )
        with torch.no_grad():
            after_update = run_epoch(
                model, val_loader, losses, reference, device, args
            )
        print(json.dumps({
            "dry_run": True,
            "device": str(device),
            "before_update": before_update,
            "after_update": after_update,
        }, indent=2))
        return

    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "source_checkpoint_sha256": c0_hash,
        "predictor_checkpoint_sha256": p1_hash,
        "split_manifest_sha256": split_hash,
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        **hashes,
        "protocol": "PDCF_P2_single_path_v1",
        "device": str(device),
        "split_strategy": manifest.get("split_strategy"),
        "frequency_training_reference": reference,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.adapter.parameters()
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["epoch", "seconds"] + [
        f"{split}_{name}"
        for split in ("train", "val")
        for name in HISTORY_METRICS
    ]
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_stats = run_epoch(
            model, train_loader, losses, reference, device, args, optimizer
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model, val_loader, losses, reference, device, args
            )
        seconds = time.time() - started
        row = {"epoch": epoch, "seconds": f"{seconds:.3f}"}
        for split, stats in (("train", train_stats), ("val", val_stats)):
            row.update(
                {f"{split}_{name}": f"{stats[name]:.9g}" for name in HISTORY_METRICS}
            )
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, model, optimizer,
                args, hashes, best,
            )
        save_checkpoint(
            run_dir / "last.pth", epoch, model, optimizer,
            args, hashes, best,
        )
        print(
            f"epoch {epoch}/{args.epochs}: val={val_stats['loss']:.6f}, "
            f"base_delta={val_stats['relative_base_correction']:.4f}, "
            f"detail_delta={val_stats['relative_detail_correction']:.4f}, "
            f"{seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
