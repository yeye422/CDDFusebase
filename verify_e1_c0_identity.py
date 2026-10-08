"""Check that E1 with the real C0 checkpoint is numerically identical before fitting."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import forward_variant, load_variant_models
from train_OS import build_models, load_model_weights
from utils.dataset_os import manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def load_pair(optical_path, sar_path, device):
    optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar_gray = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar_gray is None:
        raise RuntimeError(f"Missing optical/SAR pair: {optical_path.stem}")
    optical_gray = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if optical_gray.shape != sar_gray.shape:
        raise ValueError(f"Unmatched optical/SAR shapes for {optical_path.stem}")
    optical = torch.from_numpy(optical_gray.astype(np.float32) / 255.0)[None, None].to(device)
    sar = torch.from_numpy(sar_gray.astype(np.float32) / 255.0)[None, None].to(device)
    return optical, sar


def main():
    args = parse_args()
    if args.limit < 1 or args.tolerance < 0.0:
        raise ValueError("Invalid identity-check parameters")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("args", {}).get("variant") != "frequency_minimax_baseline":
        raise ValueError("E1 identity check requires a trained C0 checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c0, _ = load_variant_models(
        args.checkpoint, str(device), "frequency_minimax_baseline"
    )
    e1 = build_models(device, "e1_dual_stem")
    load_model_weights(e1, checkpoint)
    for model in e1.values():
        model.eval()

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits["val"][:args.limit]
    if not pairs:
        raise RuntimeError("No pairs available for identity check")

    max_error = 0.0
    with torch.no_grad():
        for optical_path, sar_path in pairs:
            optical, sar = load_pair(optical_path, sar_path, device)
            expected, _ = forward_variant(c0, "frequency_minimax_baseline", optical, sar)
            actual, _ = forward_variant(e1, "e1_dual_stem", optical, sar)
            max_error = max(max_error, float((expected - actual).abs().max()))
            print(f"image={optical_path.stem}, max_abs_difference={max_error:.8g}", flush=True)
    if max_error > args.tolerance:
        raise AssertionError(f"E1 initialization not equal to C0: {max_error} > {args.tolerance}")
    print("E1/C0 real checkpoint identity passed.")


if __name__ == "__main__":
    main()
