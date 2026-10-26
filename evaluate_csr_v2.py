"""Evaluate CSR-v2 against the exact retrained Detail-only SPD anchor."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_spd_fusion import load_models, sha256_file
from competitive_structure_routing_v2 import CompetitiveStructureRouterV2
from evaluate_csr_v1 import as_u8, fused_rgb
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_csr_v1 import frozen_anchor
from utils.dataset_os import OpticalSARDataset, manifest_pairs


SCALAR_DIAGNOSTICS = (
    "csr_mid_scale", "csr_high_scale", "csr_optical_gate_mean",
    "csr_optical_gate_coverage_010", "csr_missing_recall",
    "csr_sar_false_activation", "csr_anchor_probability",
    "csr_sar_probability", "csr_missing_mean", "csr_removed_mean",
    "csr_removed_spatial_std", "csr_sar_reliability_mean",
    "csr_sar_reliability_spatial_std", "csr_correction_rms",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSR-v2")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--csr-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparison": {}}
    matrices = {}
    for method in ("detail_spd", "csr_v2"):
        matrix = np.asarray([
            [row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES]
            for row in rows
        ])
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    delta = matrices["csr_v2"] - matrices["detail_spd"]
    for i, name in enumerate(PAPER_METRIC_NAMES):
        improved = delta[:, i] < 0 if name == "Nabf" else delta[:, i] > 0
        output["comparison"][name] = {
            "mean_delta": float(delta[:, i].mean()),
            "median_delta": float(np.median(delta[:, i])),
            "improved_fraction": float(improved.mean()),
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
    checkpoint = torch.load(args.csr_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("b0_checkpoint_sha256") != b0_hash:
        raise ValueError("CSR-v2 B0 checkpoint mismatch")
    if checkpoint.get("spd_checkpoint_sha256") != spd_hash:
        raise ValueError("CSR-v2 SPD checkpoint mismatch")
    model_args = checkpoint.get("args", {})
    router = CompetitiveStructureRouterV2(
        max_mid_replacement=float(model_args.get("max_mid_replacement", 0.10)),
        max_high_replacement=float(model_args.get("max_high_replacement", 0.03)),
        mid_sigma=float(model_args.get("mid_sigma", 3.0)),
        high_sigma=float(model_args.get("high_sigma", 1.0)),
    ).to(device)
    router.load_state_dict(checkpoint["CompetitiveStructureRouterV2"])
    router.eval()
    if args.save_images:
        for folder in ("gray", "rgb", "preview", "triptych", "maps"):
            (args.output_dir / folder).mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, valid, stem = dataset[index]
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        valid = valid[None].to(device)
        baseline, raw_detail, clean_detail = frozen_anchor(models, optical, sar)
        output, diagnostics = router(
            optical, sar, baseline, raw_detail, clean_detail,
            valid=valid, return_maps=True,
        )
        baseline = baseline.clamp(0.0, 1.0)
        output = output.clamp(0.0, 1.0)
        optical_u8, sar_u8, baseline_u8, output_u8 = map(
            as_u8, (optical, sar, baseline, output)
        )
        row = {"image": stem}
        for method, image in (("detail_spd", baseline_u8), ("csr_v2", output_u8)):
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
                np.concatenate((optical_u8, sar_u8, baseline_u8, output_u8), axis=1),
            )
            cv2.imwrite(
                str(args.output_dir / "triptych" / f"{stem}.png"),
                np.concatenate((optical_u8, sar_u8, output_u8), axis=1),
            )
            maps = np.concatenate(tuple(as_u8(diagnostics[name]) for name in (
                "csr_missing_map", "csr_removed_map",
                "csr_sar_reliability_map", "csr_optical_gate_map",
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
        "csr_checkpoint": str(args.csr_checkpoint.resolve()),
        "csr_checkpoint_sha256": sha256_file(args.csr_checkpoint),
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
