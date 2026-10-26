"""Evaluate MFFA v1 Phase-2 fusion against exact B0 on a fixed split."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mffa_v1 import MFFAV1
from test_OS import PAPER_METRIC_NAMES, evaluate, load_models
from train_mffa_v1_phase1 import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MFFA v1 Phase 2")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--mffa-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.detach().squeeze().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


def signed_u8(tensor):
    values = tensor.squeeze().float().cpu().numpy()
    scale = max(float(np.quantile(np.abs(values), 0.99)), 1e-8)
    return np.clip(np.rint(127.5 + 127.5 * values / scale), 0, 255).astype(np.uint8)


@torch.no_grad()
def forward_b0(models, optical, sar):
    encoder, decoder, base_fuse, detail_fuse = models
    optical_base, optical_detail, _ = encoder(optical)
    sar_base, sar_detail, _ = encoder(sar)
    fused_base = base_fuse(optical_base + sar_base)
    fused_detail = detail_fuse(optical_detail + sar_detail)
    fused, _ = decoder(optical, fused_base, fused_detail)
    return fused.clamp(0.0, 1.0)


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparison": {}}
    matrices = {}
    for method in ("b0", "mffa"):
        matrix = np.asarray([[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows])
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    delta = matrices["mffa"] - matrices["b0"]
    for i, name in enumerate(PAPER_METRIC_NAMES):
        improved = delta[:, i] < 0 if name == "Nabf" else delta[:, i] > 0
        output["comparison"][name] = {
            "mean_delta": float(delta[:, i].mean()),
            "median_delta": float(np.median(delta[:, i])),
            "improved_fraction": float(improved.mean()),
        }
    for name in ("high_relative_sar", "fused_low_rms", "fused_mid_rms", "fused_high_rms"):
        output[name] = float(np.mean([row[name] for row in rows]))
    return output


@torch.no_grad()
def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    dataset = OpticalSARDataset(
        pairs[args.offset:args.offset + limit], patch_size=256,
        training=False, augment=False, invalid_threshold=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.mffa_checkpoint, map_location="cpu", weights_only=False)
    # max_relative_sar is stored in the Phase-1 checkpoint that initialized this model;
    # current v1 protocol fixes it at 0.05.
    model = MFFAV1(max_relative_sar=0.05)
    model.load_state_dict(checkpoint["MFFAV1"]); model.to(device).eval()
    b0_models = load_models(args.b0_checkpoint, str(device))
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, _, stem = dataset[index]
        optical, sar = optical[None].to(device), sar[None].to(device)
        b0 = forward_b0(b0_models, optical, sar)
        mffa, outputs = model.fuse(optical, sar, return_diagnostics=True)
        optical_u8, sar_u8, b0_u8, mffa_u8 = map(as_u8, (optical, sar, b0, mffa))
        row = {"image": stem}
        for method, image in (("b0", b0_u8), ("mffa", mffa_u8)):
            values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
            row.update({f"{method}_{name}": float(value) for name, value in zip(PAPER_METRIC_NAMES, values)})
        row.update({
            "high_relative_sar": float(outputs["diagnostics"]["high_relative_sar"]),
            "fused_low_rms": float(outputs["fused"]["low"].square().mean().sqrt()),
            "fused_mid_rms": float(outputs["fused"]["mid"].square().mean().sqrt()),
            "fused_high_rms": float(outputs["fused"]["high"].square().mean().sqrt()),
        })
        rows.append(row)
        if index < args.save_images:
            preview = np.concatenate((optical_u8, sar_u8, b0_u8, mffa_u8), axis=1)
            bands = np.concatenate((
                as_u8(torch.sigmoid(torch.nn.functional.interpolate(outputs["decoded"]["low"], size=optical.shape[-2:], mode="bilinear", align_corners=False))),
                signed_u8(outputs["decoded"]["mid"]),
                signed_u8(outputs["decoded"]["high"]),
            ), axis=1)
            cv2.imwrite(str(image_dir / f"{stem}_preview.png"), preview)
            cv2.imwrite(str(image_dir / f"{stem}_bands.png"), bands)
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary["protocol"] = {
        "dataset": manifest.get("dataset", "QXS-SAROPT"), "split": args.split,
        "offset": args.offset, "limit": len(rows), "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "b0_checkpoint": str(args.b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": sha256_file(args.b0_checkpoint),
        "mffa_checkpoint": str(args.mffa_checkpoint.resolve()),
        "mffa_checkpoint_sha256": sha256_file(args.mffa_checkpoint),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
