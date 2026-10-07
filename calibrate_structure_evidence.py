"""Estimate global SAR evidence thresholds from deterministic training crops."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from structure_evidence import CALIBRATION_VERSION, FixedSARStructureEvidence
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pixels-per-image", type=int, default=2048)
    parser.add_argument("--quantile", type=float, default=99.0)
    parser.add_argument("--temperature-ratio", type=float, default=0.15)
    parser.add_argument("--scales", type=int, nargs="+", default=(3, 5, 9))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_subsample(values, valid, count):
    samples = []
    for sample, sample_valid in zip(values, valid):
        selected = sample[sample_valid.expand_as(sample) > 0.5].flatten()
        if selected.numel() > count:
            indices = torch.linspace(
                0, selected.numel() - 1, steps=count, device=selected.device
            ).long()
            selected = selected[indices]
        samples.append(selected.detach().float().cpu())
    return samples


def main():
    args = parse_args()
    if args.limit <= 0 or args.pixels_per_image <= 0:
        raise ValueError("limit and pixels-per-image must be positive")
    if not 50.0 < args.quantile < 100.0:
        raise ValueError("quantile must be in (50, 100)")
    if args.temperature_ratio <= 0.0:
        raise ValueError("temperature-ratio must be positive")

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    dataset = OpticalSARDataset(
        splits["train"],
        patch_size=args.patch_size,
        training=False,
        augment=False,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedSARStructureEvidence(scales=args.scales).to(device).eval()
    names = [
        *(f"roa_s{scale}" for scale in args.scales),
        *(f"point_s{scale}" for scale in args.scales),
        "tensor_energy",
    ]
    collected = {name: [] for name in names}

    with torch.no_grad():
        for batch_index, (_, sar, valid, _) in enumerate(loader, start=1):
            sar = sar.to(device)
            valid = valid.to(device)
            raw = model.raw_evidence(sar, valid)
            for name in names:
                collected[name].extend(
                    deterministic_subsample(
                        raw[name], valid, args.pixels_per_image
                    )
                )
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                print(f"calibration batch {batch_index}/{len(loader)}", flush=True)

    thresholds = {}
    temperatures = {}
    summaries = {}
    quantile = args.quantile / 100.0
    for name in names:
        values = torch.cat(collected[name])
        threshold = max(float(torch.quantile(values, quantile)), 1e-6)
        thresholds[name] = threshold
        temperatures[name] = max(threshold * args.temperature_ratio, 1e-4)
        summaries[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "median": float(values.median()),
            "threshold_quantile": args.quantile,
            "sampled_pixels": int(values.numel()),
        }

    payload = {
        "version": CALIBRATION_VERSION,
        "scales": list(args.scales),
        "thresholds": thresholds,
        "temperatures": temperatures,
        "metadata": {
            "data_root": str(args.data_root.resolve()),
            "split": "train",
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "images": len(dataset),
            "patch_size": args.patch_size,
            "pixels_per_image": args.pixels_per_image,
            "quantile": args.quantile,
            "temperature_ratio": args.temperature_ratio,
            "device": str(device),
            "summaries": summaries,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"calibration: {args.output.resolve()}")


if __name__ == "__main__":
    main()
