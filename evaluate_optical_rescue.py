"""Evaluate Optical Rescue against the exact retrained Detail-only SPD path."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_spd_fusion import load_models, sha256_file
from optical_rescue import OpticalRescueAdapter, RescueProbeHeads
from test_OS import PAPER_METRIC_NAMES, evaluate
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Optical Rescue")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--rescue-checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(np.rint(tensor.detach().squeeze().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)


@torch.no_grad()
def forward_two(models, adapter, heads, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    sar_base, sar_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, _ = models["spd"](sar_base, sar_detail)
    fused_base = models["base_fuse"](optical_base + clean_base)
    fused_detail = models["detail_fuse"](optical_detail + clean_detail)
    baseline, _ = models["decoder"](optical, fused_base, fused_detail)
    optical_from_baseline, _ = heads(fused_detail)
    optical_error = optical_detail - optical_from_baseline
    rescued_detail, diagnostics = adapter(
        optical_detail, clean_detail, fused_detail, optical_error
    )
    rescued, _ = models["decoder"](optical, fused_base, rescued_detail)
    return baseline.clamp(0, 1), rescued.clamp(0, 1), diagnostics


def summarize(rows):
    output = {"images": len(rows), "methods": {}, "comparison": {}}
    matrices = {}
    for method in ("detail_spd", "optical_rescue"):
        matrix = np.asarray([[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows])
        matrices[method] = matrix
        output["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(PAPER_METRIC_NAMES)
        }
    delta = matrices["optical_rescue"] - matrices["detail_spd"]
    for i, name in enumerate(PAPER_METRIC_NAMES):
        improved = delta[:, i] < 0 if name == "Nabf" else delta[:, i] > 0
        output["comparison"][name] = {
            "mean_delta": float(delta[:, i].mean()),
            "median_delta": float(np.median(delta[:, i])),
            "improved_fraction": float(improved.mean()),
        }
    for name in ("rescue_abs_mean", "rescue_rms", "rescue_relative_rms", "rescue_output_deviation_rms"):
        output[name] = float(np.mean([row[name] for row in rows]))
    return output


def fused_rgb(optical_path, fused_y):
    bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = fused_y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


@torch.no_grad()
def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs) - args.offset
    selected = pairs[args.offset:args.offset + limit]
    dataset = OpticalSARDataset(selected, patch_size=256, training=False, augment=False, invalid_threshold=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    checkpoint = torch.load(args.rescue_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("b0_checkpoint_sha256") not in (None, b0_hash):
        raise ValueError("Optical Rescue B0 checkpoint mismatch")
    if checkpoint.get("spd_checkpoint_sha256") not in (None, spd_hash):
        raise ValueError("Optical Rescue SPD checkpoint mismatch")
    model_args = checkpoint.get("args", {})
    adapter = OpticalRescueAdapter(
        max_relative_correction=float(model_args.get("max_relative_correction", 0.005))
    ).to(device)
    heads = RescueProbeHeads().to(device)
    adapter.load_state_dict(checkpoint["OpticalRescueAdapter"])
    heads.load_state_dict(checkpoint["OpticalRescueHeads"])
    adapter.eval(); heads.eval()
    if args.save_images:
        for folder in ("gray", "rgb", "preview", "triptych"):
            (args.output_dir / folder).mkdir()
    rows = []
    for index in range(len(dataset)):
        optical, sar, _, stem = dataset[index]
        optical, sar = optical[None].to(device), sar[None].to(device)
        baseline, rescued, diagnostics = forward_two(models, adapter, heads, optical, sar)
        optical_u8, sar_u8, baseline_u8, rescued_u8 = map(as_u8, (optical, sar, baseline, rescued))
        row = {"image": stem}
        for method, image in (("detail_spd", baseline_u8), ("optical_rescue", rescued_u8)):
            values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
            row.update({f"{method}_{name}": float(value) for name, value in zip(PAPER_METRIC_NAMES, values)})
        row.update({name: float(value) for name, value in diagnostics.items() if not name.endswith("_map")})
        rows.append(row)
        if index < args.save_images:
            cv2.imwrite(str(args.output_dir / "gray" / f"{stem}.png"), rescued_u8)
            cv2.imwrite(str(args.output_dir / "rgb" / f"{stem}.png"), fused_rgb(selected[index][0], rescued_u8))
            cv2.imwrite(str(args.output_dir / "preview" / f"{stem}.png"), np.concatenate((optical_u8, sar_u8, baseline_u8, rescued_u8), axis=1))
            cv2.imwrite(str(args.output_dir / "triptych" / f"{stem}.png"), np.concatenate((optical_u8, sar_u8, rescued_u8), axis=1))
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
        "b0_checkpoint": str(args.b0_checkpoint.resolve()), "b0_checkpoint_sha256": b0_hash,
        "spd_checkpoint": str(args.spd_checkpoint.resolve()), "spd_checkpoint_sha256": spd_hash,
        "rescue_checkpoint": str(args.rescue_checkpoint.resolve()),
        "rescue_checkpoint_sha256": sha256_file(args.rescue_checkpoint),
        "encoder_equivalence": equivalence, "spd_source_checkpoint": source,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__": main()
