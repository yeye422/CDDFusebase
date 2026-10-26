"""Evaluate OCAF-v1 against Optical and exact Detail-only SPD outputs."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_spd_fusion import load_models, sha256_file
from evaluate_csr_v1 import as_u8, fused_rgb
from ocaf_v1 import OpticalCanvasFusion, gaussian_blur, gradient_magnitude
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_ocaf_v1 import frozen_paths
from utils.dataset_os import OpticalSARDataset, manifest_pairs


SCALAR_DIAGNOSTICS = (
    "ocaf_common_scale", "ocaf_exclusive_mid_scale",
    "ocaf_exclusive_high_scale", "ocaf_common_gate_mean",
    "ocaf_sar_gate_mean", "ocaf_optical_probability",
    "ocaf_common_probability", "ocaf_sar_probability",
    "ocaf_common_prior_mean", "ocaf_exclusive_prior_mean",
    "ocaf_removed_mean", "ocaf_reliability_mean",
    "ocaf_common_delta_rms", "ocaf_exclusive_delta_rms",
    "ocaf_total_delta_rms", "ocaf_optical_low_mae",
    "ocaf_optical_gradient_mae", "ocaf_stable_sar_gradient_gain",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OCAF-v1")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--ocaf-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparisons": {}}
    matrices = {}
    for method in ("optical", "detail_spd", "ocaf_v1"):
        matrix = np.asarray([
            [row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES]
            for row in rows
        ])
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    for reference in ("optical", "detail_spd"):
        delta = matrices["ocaf_v1"] - matrices[reference]
        output["comparisons"][f"ocaf_v1_minus_{reference}"] = {
            name: {
                "mean_delta": float(delta[:, i].mean()),
                "median_delta": float(np.median(delta[:, i])),
                "improved_fraction": float(
                    (delta[:, i] < 0).mean() if name == "Nabf"
                    else (delta[:, i] > 0).mean()
                ),
            }
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    output["diagnostics"] = {
        name: float(np.mean([row[name] for row in rows]))
        for name in SCALAR_DIAGNOSTICS
    }
    return output


@torch.no_grad()
def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    selected = pairs[args.offset:args.offset + limit]
    dataset = OpticalSARDataset(
        selected, patch_size=256, training=False, augment=False, invalid_threshold=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    checkpoint = torch.load(args.ocaf_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("b0_checkpoint_sha256") != b0_hash:
        raise ValueError("OCAF-v1 B0 checkpoint mismatch")
    if checkpoint.get("spd_checkpoint_sha256") != spd_hash:
        raise ValueError("OCAF-v1 SPD checkpoint mismatch")
    model_args = checkpoint.get("args", {})
    ocaf = OpticalCanvasFusion(
        max_common_mid=float(model_args.get("max_common_mid", 0.10)),
        max_exclusive_mid=float(model_args.get("max_exclusive_mid", 0.05)),
        max_exclusive_high=float(model_args.get("max_exclusive_high", 0.02)),
        high_sigma=float(model_args.get("high_sigma", 1.0)),
        low_sigma=float(model_args.get("low_sigma", 3.0)),
    ).to(device)
    ocaf.load_state_dict(checkpoint["OpticalCanvasFusion"])
    ocaf.eval()
    if args.save_images:
        for folder in ("gray", "rgb", "preview", "triptych", "maps"):
            (args.output_dir / folder).mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, valid, stem = dataset[index]
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        valid = valid[None].to(device)
        baseline, clean_sar, raw_detail, clean_detail = frozen_paths(
            models, optical, sar
        )
        output, diagnostics = ocaf(
            optical, clean_sar, baseline, raw_detail, clean_detail,
            valid=valid, return_maps=True,
        )
        optical_low = gaussian_blur(optical, ocaf.low_sigma)
        output_low = gaussian_blur(output, ocaf.low_sigma)
        diagnostics["ocaf_optical_low_mae"] = (
            output_low - optical_low
        ).abs().mean()
        diagnostics["ocaf_optical_gradient_mae"] = (
            gradient_magnitude(output) - gradient_magnitude(optical)
        ).abs().mean()
        exclusive = diagnostics["ocaf_exclusive_prior_map"]
        diagnostics["ocaf_stable_sar_gradient_gain"] = (
            (
                gradient_magnitude(output) - gradient_magnitude(optical)
            ).clamp_min(0.0) * exclusive
        ).sum() / exclusive.sum().clamp_min(1e-6)
        baseline = baseline.clamp(0.0, 1.0)
        clean_sar = clean_sar.clamp(0.0, 1.0)
        output = output.clamp(0.0, 1.0)
        optical_u8, sar_u8, clean_u8, baseline_u8, output_u8 = map(
            as_u8, (optical, sar, clean_sar, baseline, output)
        )
        row = {"image": stem}
        for method, image in (
            ("optical", optical_u8), ("detail_spd", baseline_u8),
            ("ocaf_v1", output_u8),
        ):
            values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
            row.update({
                f"{method}_{name}": float(value)
                for name, value in zip(PAPER_METRIC_NAMES, values)
            })
        row.update({name: float(diagnostics[name]) for name in SCALAR_DIAGNOSTICS})
        rows.append(row)
        if index < args.save_images:
            cv2.imwrite(str(args.output_dir / "gray" / f"{stem}.png"), output_u8)
            cv2.imwrite(
                str(args.output_dir / "rgb" / f"{stem}.png"),
                fused_rgb(selected[index][0], output_u8),
            )
            cv2.imwrite(
                str(args.output_dir / "preview" / f"{stem}.png"),
                np.concatenate(
                    (optical_u8, sar_u8, clean_u8, baseline_u8, output_u8), axis=1
                ),
            )
            cv2.imwrite(
                str(args.output_dir / "triptych" / f"{stem}.png"),
                np.concatenate((optical_u8, sar_u8, output_u8), axis=1),
            )
            maps = np.concatenate(tuple(as_u8(diagnostics[name]) for name in (
                "ocaf_removed_map", "ocaf_reliability_map",
                "ocaf_common_prior_map", "ocaf_exclusive_prior_map",
                "ocaf_common_gate_map", "ocaf_sar_gate_map",
            )), axis=1)
            cv2.imwrite(str(args.output_dir / "maps" / f"{stem}.png"), maps)
        print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
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
        "b0_checkpoint_sha256": b0_hash,
        "spd_checkpoint": str(args.spd_checkpoint.resolve()),
        "spd_checkpoint_sha256": spd_hash,
        "ocaf_checkpoint": str(args.ocaf_checkpoint.resolve()),
        "ocaf_checkpoint_sha256": sha256_file(args.ocaf_checkpoint),
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
