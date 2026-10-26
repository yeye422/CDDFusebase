"""Phase-1 reconstruction evaluation for a trained MFFA v1 checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from mffa_v1 import MFFAV1
from train_mffa_v1_phase1 import Phase1Loss, sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MFFA v1 Phase 1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(np.rint(tensor.squeeze().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


def image_metrics(prediction, target, valid):
    prediction = prediction.astype(np.float64) / 255.0
    target = target.astype(np.float64) / 255.0
    valid = valid.astype(bool)
    difference = prediction[valid] - target[valid]
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    # SSIM is reported on the full crop; invalid SAR pixels are copied from target.
    prediction_for_ssim = prediction.copy()
    prediction_for_ssim[~valid] = target[~valid]
    ssim = float(structural_similarity(target, prediction_for_ssim, data_range=1.0))
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def summarize(rows):
    output = {}
    for name in rows[0]:
        if name == "image":
            continue
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        output[name] = {"mean": float(values.mean()), "std": float(values.std())}
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    model = MFFAV1(max_relative_sar=float(model_args.get("max_relative_sar", 0.05)))
    model.load_state_dict(checkpoint["MFFAV1"])
    model.to(device).eval()
    criterion = Phase1Loss(float(model_args.get("charbonnier_epsilon", 1e-3))).to(device)
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, valid, stem = dataset[index]
        optical, sar, valid = optical[None].to(device), sar[None].to(device), valid[None].to(device)
        optical_hat, optical_encoded, optical_decoded = model.reconstruct(optical, "optical")
        sar_hat, sar_encoded, sar_decoded = model.reconstruct(sar, "sar", valid)
        optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
        optical_hat_u8, sar_hat_u8 = as_u8(optical_hat), as_u8(sar_hat)
        valid_np = valid.squeeze().cpu().numpy() > 0.5
        row = {"image": stem}
        row.update({f"optical_{k}": v for k, v in image_metrics(optical_hat_u8, optical_u8, np.ones_like(valid_np)).items()})
        row.update({f"sar_{k}": v for k, v in image_metrics(sar_hat_u8, sar_u8, valid_np).items()})
        optical_band = criterion.band_loss(optical_decoded, optical_encoded["bands"])
        sar_band = criterion.band_loss(sar_decoded, sar_encoded["bands"])
        for prefix, values in (("optical", optical_band), ("sar", sar_band)):
            row[f"{prefix}_band"] = float(values[0])
            row[f"{prefix}_low"] = float(values[1])
            row[f"{prefix}_mid"] = float(values[2])
            row[f"{prefix}_high"] = float(values[3])
        rows.append(row)
        if index < args.save_images:
            optical_diff = np.clip(cv2.absdiff(optical_hat_u8, optical_u8).astype(np.float32) * 4, 0, 255).astype(np.uint8)
            sar_diff = np.clip(cv2.absdiff(sar_hat_u8, sar_u8).astype(np.float32) * 4, 0, 255).astype(np.uint8)
            preview = np.concatenate((optical_u8, optical_hat_u8, optical_diff, sar_u8, sar_hat_u8, sar_diff), axis=1)
            cv2.imwrite(str(image_dir / f"{stem}_preview.png"), preview)
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {
        "images": len(rows), "summary": summarize(rows),
        "protocol": {
            "stage": "MFFA_v1_phase1_reconstruction", "dataset": manifest.get("dataset", "QXS-SAROPT"),
            "split": args.split, "offset": args.offset, "limit": len(rows),
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
