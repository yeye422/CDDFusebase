"""P1 training entry for frozen-feature bidirectional predictability.

This script trains only the nonlinear predictors and a matched 1x1 linear
baseline.  It never updates the C0 Encoder or any fusion/decoder component.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pdcf_model import PDCFModel
from pdcf_predictability import (
    BidirectionalLinearBaseline,
    BidirectionalPredictability,
    FeatureStatisticsAccumulator,
    balanced_prediction_loss,
    prediction_metrics,
)
from utils.dataset_os import OpticalSARDataset, manifest_pairs


METRIC_NAMES = (
    "loss",
    "o2s_base",
    "o2s_detail",
    "s2o_base",
    "s2o_detail",
    "o2s_base_nrmse",
    "o2s_detail_nrmse",
    "s2o_base_nrmse",
    "s2o_detail_nrmse",
    "o2s_mean_base_nrmse",
    "o2s_mean_detail_nrmse",
    "s2o_mean_base_nrmse",
    "s2o_mean_detail_nrmse",
    "residual_optical_rms",
    "residual_sar_rms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PDCF P1 bidirectional frozen-feature predictors"
    )
    parser.add_argument(
        "--c0-checkpoint",
        type=Path,
        default=Path("models/reliability_v5/frequency_minimax_seed42/best.pth"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("../datasets/QXSLAB_SAROPT")
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "../datasets/QXSLAB_SAROPT/splits/"
            "cddfuse_8000_1000_1000_seed20260924.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/pdcf_p1"))
    parser.add_argument("--run-name", default="predictability_seed42")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--predictor-blocks", type=int, default=2)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument(
        "--statistics-batches",
        type=int,
        default=0,
        help="0 uses the complete deterministic training split.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_loaders(args, device):
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    common = dict(
        patch_size=args.patch_size,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
    )
    train_dataset = OpticalSARDataset(
        splits["train"], training=True, limit=args.train_limit, **common
    )
    reference_dataset = OpticalSARDataset(
        splits["train"], training=False, augment=False,
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
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_args
    )
    reference_loader = DataLoader(reference_dataset, shuffle=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    return train_loader, reference_loader, val_loader, manifest


@torch.no_grad()
def encode(encoder, optical, sar):
    optical_base, optical_detail, _ = encoder(optical)
    sar_base, sar_detail, _ = encoder(sar)
    return optical_base, optical_detail, sar_base, sar_detail


@torch.no_grad()
def compute_statistics(encoder, loader, device, max_batches=0):
    accumulator = FeatureStatisticsAccumulator(channels=64)
    processed_batches = 0
    for optical, sar, mask, _ in loader:
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        accumulator.update(*encode(encoder, optical, sar), mask)
        processed_batches += 1
        if max_batches and processed_batches >= max_batches:
            break
    statistics = accumulator.finalize()
    statistics["batches"] = torch.tensor(processed_batches)
    return statistics


def linear_outputs(linear, nonlinear_outputs):
    predictions = linear(
        nonlinear_outputs["optical"].detach(),
        nonlinear_outputs["sar"].detach(),
    )
    return {
        "optical": nonlinear_outputs["optical"].detach(),
        "sar": nonlinear_outputs["sar"].detach(),
        **predictions,
    }


def accumulate(target, values, batch_size):
    for name, value in values.items():
        target[name] = target.get(name, 0.0) + float(value.detach()) * batch_size


def finalize(accumulator, samples):
    return {name: value / max(samples, 1) for name, value in accumulator.items()}


def run_epoch(
    encoder,
    predictor,
    linear,
    loader,
    device,
    beta,
    max_batches=0,
    predictor_optimizer=None,
    linear_optimizer=None,
    clip_grad=1.0,
):
    training = predictor_optimizer is not None
    predictor.train(training)
    linear.train(training)
    totals = {"predictor": {}, "linear": {}}
    samples = 0
    for batch_index, (optical, sar, mask, _) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = encode(encoder, optical, sar)

        with torch.set_grad_enabled(training):
            outputs = predictor(*features)
            predictor_losses = balanced_prediction_loss(
                outputs, mask, channels=64, beta=beta
            )
            linear_result = linear_outputs(linear, outputs)
            linear_losses = balanced_prediction_loss(
                linear_result, mask, channels=64, beta=beta
            )
            if training:
                predictor_optimizer.zero_grad(set_to_none=True)
                predictor_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                predictor_optimizer.step()

                linear_optimizer.zero_grad(set_to_none=True)
                linear_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(linear.parameters(), clip_grad)
                linear_optimizer.step()

        batch_size = optical.shape[0]
        samples += batch_size
        predictor_values = {
            "loss": predictor_losses["total"],
            **{name: value for name, value in predictor_losses.items() if name != "total"},
            **prediction_metrics(outputs, mask, channels=64),
        }
        linear_values = {
            "loss": linear_losses["total"],
            **{name: value for name, value in linear_losses.items() if name != "total"},
            **prediction_metrics(linear_result, mask, channels=64),
        }
        accumulate(totals["predictor"], predictor_values, batch_size)
        accumulate(totals["linear"], linear_values, batch_size)
        if max_batches and batch_index + 1 >= max_batches:
            break
    return {
        "predictor": finalize(totals["predictor"], samples),
        "linear": finalize(totals["linear"], samples),
        "samples": samples,
    }


def save_checkpoint(path, epoch, predictor, linear, optimizers, statistics, args):
    torch.save(
        {
            "epoch": epoch,
            "PDCFPredictability": predictor.state_dict(),
            "PDCFLinearBaseline": linear.state_dict(),
            "predictor_optimizer": optimizers[0].state_dict(),
            "linear_optimizer": optimizers[1].state_dict(),
            "feature_statistics": statistics,
            "source_checkpoint_sha256": sha256_file(args.c0_checkpoint),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patch_size <= 0:
        raise ValueError("epochs, batch-size, and patch-size must be positive")
    if args.lr <= 0 or args.clip_grad <= 0 or args.smooth_l1_beta <= 0:
        raise ValueError("lr, clip-grad, and smooth-l1-beta must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, reference_loader, val_loader, manifest = make_loaders(args, device)

    c0 = PDCFModel.from_c0_checkpoint(
        args.c0_checkpoint, map_location=device
    ).to(device).eval()
    encoder = c0.encoder
    statistics_batches = 1 if args.dry_run else args.statistics_batches
    statistics = compute_statistics(
        encoder, reference_loader, device, statistics_batches
    )
    predictor = BidirectionalPredictability(
        channels=64,
        hidden_channels=args.hidden_channels,
        blocks=args.predictor_blocks,
    ).to(device)
    predictor.set_statistics(statistics)
    linear = BidirectionalLinearBaseline(channels=128).to(device)
    predictor_optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    linear_optimizer = torch.optim.AdamW(
        linear.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.dry_run:
        result = run_epoch(
            encoder,
            predictor,
            linear,
            train_loader,
            device,
            args.smooth_l1_beta,
            max_batches=1,
            predictor_optimizer=predictor_optimizer,
            linear_optimizer=linear_optimizer,
            clip_grad=args.clip_grad,
        )
        print(json.dumps({"dry_run": True, "device": str(device), **result}, indent=2))
        return

    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": "PDCF_P1_predictability_v1",
        "device": str(device),
        "c0_checkpoint_sha256": sha256_file(args.c0_checkpoint),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_strategy": manifest.get("split_strategy"),
        "feature_statistics_samples": float(statistics["count"]),
        "feature_statistics_batches": int(statistics["batches"]),
        "trainable_predictor_parameters": sum(p.numel() for p in predictor.parameters()),
        "trainable_linear_parameters": sum(p.numel() for p in linear.parameters()),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    history_path = run_dir / "history.csv"
    fields = ["epoch", "seconds"] + [
        f"{split}_{model}_{metric}"
        for split in ("train", "val")
        for model in ("predictor", "linear")
        for metric in METRIC_NAMES
    ]
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best = float("inf")
    optimizers = (predictor_optimizer, linear_optimizer)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_result = run_epoch(
            encoder, predictor, linear, train_loader, device,
            args.smooth_l1_beta, args.max_train_batches,
            predictor_optimizer, linear_optimizer, args.clip_grad,
        )
        with torch.no_grad():
            val_result = run_epoch(
                encoder, predictor, linear, val_loader, device,
                args.smooth_l1_beta, args.max_val_batches,
            )
        seconds = time.time() - started
        row = {"epoch": epoch, "seconds": f"{seconds:.3f}"}
        for split, result in (("train", train_result), ("val", val_result)):
            for model_name in ("predictor", "linear"):
                for metric in METRIC_NAMES:
                    row[f"{split}_{model_name}_{metric}"] = (
                        f"{result[model_name][metric]:.9g}"
                    )
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        save_checkpoint(
            run_dir / "last.pth", epoch, predictor, linear,
            optimizers, statistics, args,
        )
        if val_result["predictor"]["loss"] < best:
            best = val_result["predictor"]["loss"]
            save_checkpoint(
                run_dir / "best.pth", epoch, predictor, linear,
                optimizers, statistics, args,
            )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"predictor val={val_result['predictor']['loss']:.6f}, "
            f"linear val={val_result['linear']['loss']:.6f}, "
            f"{seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
