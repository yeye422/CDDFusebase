"""Verify C1's zero-initialized adapters reproduce the same frozen C0."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import forward_variant, load_variant_models
from train_OS import build_models, load_model_weights
from utils.dataset_os import manifest_pairs


C1_VARIANTS = ("c1_base_adapter", "c1_detail_adapter")


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
    sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if optical_bgr is None or sar is None:
        raise RuntimeError(f"Image missing: {optical_path}, {sar_path}")
    optical = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if optical.shape != sar.shape:
        raise RuntimeError("Pair shape mismatch")
    o = torch.from_numpy(optical.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    s = torch.from_numpy(sar.astype(np.float32) / 255.0)[
        None, None
    ].to(device)
    return o, s


def main():
    args = parse_args()
    if args.limit <= 0 or args.tolerance < 0:
        raise ValueError("Invalid comparison arguments")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("args", {}).get("variant") != "frequency_minimax_baseline":
        raise ValueError("Identity validation requires a trained C0 checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c0, _ = load_variant_models(
        args.checkpoint, str(device), "frequency_minimax_baseline"
    )
    variants = {}
    for variant in C1_VARIANTS:
        models = build_models(device, variant)
        load_model_weights(models, checkpoint)
        for module in models.values():
            module.eval()
        variants[variant] = models

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits["val"][: args.limit]
    if not pairs:
        raise RuntimeError("No validation pairs found")

    max_diff = {variant: 0.0 for variant in C1_VARIANTS}
    with torch.no_grad():
        for image_path, sar_path in pairs:
            optical, sar = load_pair(image_path, sar_path, device)
            reference, _ = forward_variant(
                c0, "frequency_minimax_baseline", optical, sar
            )
            for variant, models in variants.items():
                current, diagnostics = forward_variant(
                    models, variant, optical, sar
                )
                diff = (current - reference).abs()
                max_diff[variant] = max(max_diff[variant], float(diff.max()))
                if float(diagnostics["c1_delta_abs_mean"]) != 0.0:
                    raise AssertionError(f"{variant} was not initialized to identity")
            print(
                f"sample={image_path.stem}, "
                + ", ".join(
                    f"{name} max_abs_diff={value:.8g}"
                    for name, value in max_diff.items()
                ),
                flush=True,
            )
    for variant, value in max_diff.items():
        if value > args.tolerance:
            raise AssertionError(
                f"{variant} differs from C0 before training: {value} > {args.tolerance}"
            )
    print("C1/C0 checkpoint identity verified for both paths.")


if __name__ == "__main__":
    main()
