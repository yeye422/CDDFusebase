"""Evaluate a trained P1 router against its exact frozen B0 source."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from p1_stable_detail_fusion import StableSARDetailRouter
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_fusion_p1 import build_frozen_models, sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate P1 stable Detail fusion")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


@torch.no_grad()
def forward_two(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    fused_base = models["base_fuse"](optical_base + sar_base)
    optical_expert = models["detail_fuse"](optical_detail)
    joint_expert = models["detail_fuse"](optical_detail + sar_detail)
    routed_detail, diagnostics = models["detail_router"](
        optical_detail,
        sar_detail,
        optical_expert,
        joint_expert,
        return_maps=False,
    )
    b0, _ = models["decoder"](optical, fused_base, joint_expert)
    p1, _ = models["decoder"](optical, fused_base, routed_detail)
    return b0.clamp(0.0, 1.0), p1.clamp(0.0, 1.0), diagnostics


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparison": {}}
    for method in ("b0", "p1"):
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows]
        )
        output["methods"][method] = {
            name: {"mean": float(matrix[:, index].mean()), "std": float(matrix[:, index].std())}
            for index, name in enumerate(PAPER_METRIC_NAMES)
        }
    b0 = np.asarray([[row[f"b0_{name}"] for name in PAPER_METRIC_NAMES] for row in rows])
    p1 = np.asarray([[row[f"p1_{name}"] for name in PAPER_METRIC_NAMES] for row in rows])
    delta = p1 - b0
    for index, name in enumerate(PAPER_METRIC_NAMES):
        improved = delta[:, index] < 0 if name == "Nabf" else delta[:, index] > 0
        output["comparison"][name] = {
            "mean_delta": float(delta[:, index].mean()),
            "median_delta": float(np.median(delta[:, index])),
            "improved_fraction": float(improved.mean()),
        }
    for name in (
        "p1_gate_mean", "p1_gate_std", "p1_sar_contribution_rms",
        "p1_injected_sar_rms", "p1_baseline_deviation_rms",
        "p1_reliability_mean", "p1_reliability_std", "p1_suppression_cap",
        "p1_spatial_gate_mean", "p1_channel_gate_mean",
    ):
        output[name] = float(np.mean([row[name] for row in rows]))
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    selected = pairs[args.offset : args.offset + limit]
    dataset = OpticalSARDataset(
        selected, patch_size=256, training=False, augment=False,
        invalid_threshold=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b0_checkpoint = torch.load(
        args.b0_checkpoint, map_location="cpu", weights_only=False
    )
    p1_checkpoint = torch.load(
        args.p1_checkpoint, map_location="cpu", weights_only=False
    )
    expected_hash = p1_checkpoint.get("source_checkpoint_sha256")
    actual_hash = sha256_file(args.b0_checkpoint)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("P1 router was trained from a different B0 checkpoint")
    models = build_frozen_models(b0_checkpoint, device)
    models["detail_router"].load_state_dict(
        p1_checkpoint["P1StableDetailRouter"]
    )
    for model in models.values():
        model.eval()
    image_dir = args.output_dir / "images"
    if args.save_images:
        image_dir.mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, _, stem = dataset[index]
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        b0, p1, diagnostics = forward_two(models, optical, sar)
        optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
        b0_u8, p1_u8 = as_u8(b0), as_u8(p1)
        row = {"image": stem}
        for method, image in (("b0", b0_u8), ("p1", p1_u8)):
            values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
            row.update(
                {f"{method}_{name}": float(value) for name, value in zip(PAPER_METRIC_NAMES, values)}
            )
        for name, value in diagnostics.items():
            if not name.endswith("_map"):
                row[name] = float(value)
        rows.append(row)
        if index < args.save_images:
            preview = np.concatenate((optical_u8, sar_u8, b0_u8, p1_u8), axis=1)
            cv2.imwrite(str(image_dir / f"{stem}_preview.png"), preview)
            cv2.imwrite(str(image_dir / f"{stem}_p1.png"), p1_u8)
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["protocol"] = {
        "dataset": manifest.get("dataset", "QXS-SAROPT"),
        "split": args.split,
        "offset": args.offset,
        "limit": len(rows),
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "b0_checkpoint": str(args.b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": actual_hash,
        "p1_checkpoint": str(args.p1_checkpoint.resolve()),
        "p1_checkpoint_sha256": sha256_file(args.p1_checkpoint),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
