"""Evaluate P2-A source-preserving fusion against exact B0."""

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
    parser = argparse.ArgumentParser(description="Evaluate P2-A fusion")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--p2a-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(np.rint(tensor.squeeze().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


@torch.no_grad()
def forward_two(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    b0_detail = models["detail_fuse"](optical_detail + sar_detail)
    p2a_detail, diagnostics = models["adapter"](optical_detail, sar_detail, b0_detail)
    b0, _ = models["decoder"](optical, fused_base, b0_detail)
    p2a, _ = models["decoder"](optical, fused_base, p2a_detail)
    return b0.clamp(0, 1), p2a.clamp(0, 1), diagnostics


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparison": {}}
    matrices = {}
    for method in ("b0", "p2a"):
        matrix = np.asarray([[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows])
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    delta = matrices["p2a"] - matrices["b0"]
    for i, name in enumerate(PAPER_METRIC_NAMES):
        improved = delta[:, i] < 0 if name == "Nabf" else delta[:, i] > 0
        output["comparison"][name] = {
            "mean_delta": float(delta[:, i].mean()),
            "median_delta": float(np.median(delta[:, i])),
            "improved_fraction": float(improved.mean()),
        }
    for name in (
        "p2a_correction_abs_mean", "p2a_correction_rms",
        "p2a_relative_correction", "p2a_output_deviation_rms",
    ):
        output[name] = float(np.mean([row[name] for row in rows]))
    return output


def main():
    args = parse_args()
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
    actual_hash = sha256_file(args.b0_checkpoint)
    if p2a_checkpoint.get("source_checkpoint_sha256") not in (None, actual_hash):
        raise ValueError("P2-A checkpoint was trained from a different B0")
    model_args = p2a_checkpoint.get("args", {})
    models = build_models(
        b0_checkpoint, device, float(model_args.get("max_relative_correction", 0.03))
    )
    models["adapter"].load_state_dict(p2a_checkpoint["P2ASourceAdapter"])
    models["pseudo_heads"].load_state_dict(p2a_checkpoint["P2APseudoHeads"])
    for model in models.values():
        model.eval()
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, _, stem = dataset[index]
        optical, sar = optical[None].to(device), sar[None].to(device)
        b0, p2a, diagnostics = forward_two(models, optical, sar)
        optical_u8, sar_u8, b0_u8, p2a_u8 = map(as_u8, (optical, sar, b0, p2a))
        row = {"image": stem}
        for method, image in (("b0", b0_u8), ("p2a", p2a_u8)):
            values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
            row.update({f"{method}_{name}": float(value) for name, value in zip(PAPER_METRIC_NAMES, values)})
        row.update({name: float(value) for name, value in diagnostics.items() if not name.endswith("_map")})
        rows.append(row)
        if index < args.save_images:
            cv2.imwrite(str(image_dir / f"{stem}_preview.png"), np.concatenate((optical_u8, sar_u8, b0_u8, p2a_u8), axis=1))
            cv2.imwrite(str(image_dir / f"{stem}_p2a.png"), p2a_u8)
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    summary["protocol"] = {
        "dataset": manifest.get("dataset", "QXS-SAROPT"), "split": args.split,
        "offset": args.offset, "limit": len(rows), "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "b0_checkpoint": str(args.b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": actual_hash,
        "p2a_checkpoint": str(args.p2a_checkpoint.resolve()),
        "p2a_checkpoint_sha256": sha256_file(args.p2a_checkpoint),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
