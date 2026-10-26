"""Validation-only scale sweep for a frozen trained P2-A correction."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from test_OS import PAPER_METRIC_NAMES, evaluate
from train_fusion_p2a import build_models, sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep frozen P2-A residual scales")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--p2a-checkpoint", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scale_id(value):
    return f"a{value:.3f}".replace(".", "p")


def as_u8(tensor):
    return np.clip(np.rint(tensor.squeeze().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


def scaled_detail(b0_detail, correction, scale):
    return b0_detail + float(scale) * correction


@torch.no_grad()
def forward_scales(models, optical, sar, scales):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    b0_detail = models["detail_fuse"](optical_detail + sar_detail)
    _, diagnostics = models["adapter"](
        optical_detail, sar_detail, b0_detail, return_map=True
    )
    correction = diagnostics["p2a_correction_map"]
    outputs = {}
    for scale in scales:
        detail = scaled_detail(b0_detail, correction, scale)
        fused, _ = models["decoder"](optical, fused_base, detail)
        outputs[scale_id(scale)] = fused.clamp(0.0, 1.0)
    return outputs, diagnostics


def summarize(rows, method_ids, baseline):
    output = {"images": len(rows), "methods": {}, "comparisons_to_identity": {}}
    matrices = {}
    for method in method_ids:
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows],
            dtype=np.float64,
        )
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    for method in method_ids:
        delta = matrices[method] - matrices[baseline]
        output["comparisons_to_identity"][method] = {}
        for i, name in enumerate(PAPER_METRIC_NAMES):
            improved = delta[:, i] < 0 if name == "Nabf" else delta[:, i] > 0
            output["comparisons_to_identity"][method][name] = {
                "mean_delta": float(delta[:, i].mean()),
                "median_delta": float(np.median(delta[:, i])),
                "improved_fraction": float(improved.mean()),
            }
    return output


def main():
    args = parse_args()
    if len(set(args.scales)) != len(args.scales):
        raise ValueError("scales must be unique")
    if any(scale < 0.0 or scale > 1.0 for scale in args.scales):
        raise ValueError("P2-A validation scales must be in [0, 1]")
    if 0.0 not in args.scales:
        raise ValueError("scales must include 0 for the exact B0 identity")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    dataset = OpticalSARDataset(
        pairs[args.offset:args.offset + limit], patch_size=256,
        training=False, augment=False, invalid_threshold=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b0_checkpoint = torch.load(args.b0_checkpoint, map_location="cpu", weights_only=False)
    p2a_checkpoint = torch.load(args.p2a_checkpoint, map_location="cpu", weights_only=False)
    b0_hash = sha256_file(args.b0_checkpoint)
    if p2a_checkpoint.get("source_checkpoint_sha256") not in (None, b0_hash):
        raise ValueError("P2-A checkpoint source does not match B0")
    model_args = p2a_checkpoint.get("args", {})
    models = build_models(
        b0_checkpoint, device, float(model_args.get("max_relative_correction", 0.03))
    )
    models["adapter"].load_state_dict(p2a_checkpoint["P2ASourceAdapter"])
    models["pseudo_heads"].load_state_dict(p2a_checkpoint["P2APseudoHeads"])
    for model in models.values():
        model.eval()
    method_ids = [scale_id(scale) for scale in args.scales]
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir()
    rows = []
    correction_stats = []
    for index in range(len(dataset)):
        optical, sar, _, stem = dataset[index]
        optical, sar = optical[None].to(device), sar[None].to(device)
        outputs, diagnostics = forward_scales(models, optical, sar, args.scales)
        optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
        row = {"image": stem}
        images = []
        for method in method_ids:
            fused_u8 = as_u8(outputs[method])
            values = evaluate(fused_u8, optical_u8, sar_u8, metric_profile="paper")
            row.update({f"{method}_{name}": float(value) for name, value in zip(PAPER_METRIC_NAMES, values)})
            images.append(fused_u8)
        rows.append(row)
        correction_stats.append(float(diagnostics["p2a_relative_correction"]))
        if index < args.save_images:
            cv2.imwrite(
                str(image_dir / f"{stem}_scales.png"),
                np.concatenate((optical_u8, sar_u8, *images), axis=1),
            )
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows, method_ids, scale_id(0.0))
    summary["mean_full_relative_correction"] = float(np.mean(correction_stats))
    summary["protocol"] = {
        "dataset": manifest.get("dataset", "QXS-SAROPT"), "split": args.split,
        "offset": args.offset, "limit": len(rows), "metric_profile": "paper",
        "output_policy": "clamp_then_uint8", "scales": args.scales,
        "selection_warning": "Validation-only P2-A residual scale selection; do not tune on test.",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "b0_checkpoint": str(args.b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": b0_hash,
        "p2a_checkpoint": str(args.p2a_checkpoint.resolve()),
        "p2a_checkpoint_sha256": sha256_file(args.p2a_checkpoint),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
