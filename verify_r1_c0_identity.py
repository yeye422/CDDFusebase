"""Verify that zero-initialized R1 is numerically identical to C0 before training."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import forward_variant
from train_OS import build_models, load_model_weights
from utils.dataset_os import manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def load_pair(optical_path, sar_path, device):
    optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar_u8 is None:
        raise RuntimeError(f"Could not read pair {optical_path.stem}")
    optical_y = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    optical = torch.from_numpy(optical_y.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    sar = torch.from_numpy(sar_u8.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    return optical, sar


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    c0 = build_models(device, "frequency_minimax_baseline")
    r1 = build_models(device, "r1_spatial_modulation")
    load_model_weights(c0, checkpoint)
    load_model_weights(r1, checkpoint)
    for model in c0.values():
        model.eval()
    for model in r1.values():
        model.eval()

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][: args.limit]
    if not selected:
        raise RuntimeError("No image pairs selected")

    max_diff = 0.0
    mean_diffs = []
    modulation_max = 0.0
    with torch.no_grad():
        for optical_path, sar_path in selected:
            optical, sar = load_pair(optical_path, sar_path, device)
            c0_output, _ = forward_variant(
                c0,
                "frequency_minimax_baseline",
                optical,
                sar,
            )
            r1_output, diagnostics = forward_variant(
                r1,
                "r1_spatial_modulation",
                optical,
                sar,
                return_maps=True,
            )
            diff = torch.abs(c0_output - r1_output)
            sample_max = float(diff.max().cpu())
            sample_mean = float(diff.mean().cpu())
            sample_mod = float(
                diagnostics["r1_modulation_map"].abs().max().cpu()
            )
            max_diff = max(max_diff, sample_max)
            mean_diffs.append(sample_mean)
            modulation_max = max(modulation_max, sample_mod)
            print(
                f"{optical_path.stem}: max_abs_diff={sample_max:.10g}, "
                f"mean_abs_diff={sample_mean:.10g}, "
                f"max_abs_modulation={sample_mod:.10g}",
                flush=True,
            )

    mean_diff = float(np.mean(mean_diffs))
    print(
        f"summary: samples={len(selected)}, device={device}, "
        f"max_abs_diff={max_diff:.10g}, mean_abs_diff={mean_diff:.10g}, "
        f"max_abs_modulation={modulation_max:.10g}",
        flush=True,
    )
    if modulation_max != 0.0:
        raise AssertionError(
            f"R1 zero initialization produced nonzero modulation: {modulation_max}"
        )
    if max_diff > args.tolerance:
        raise AssertionError(
            f"R1/C0 identity check failed: {max_diff} > {args.tolerance}"
        )
    print("R1/C0 checkpoint identity check passed.")


if __name__ == "__main__":
    main()
