"""Validation-only sweep of continuity-filtered line-guided Detail preservation."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from compare_spd_fusion import as_u8, fuse, load_models, sha256_file
from prepare_spd_line_masks import high_confidence_line_mask, joint_line_mask
from structure_evidence_v2 import FixedSARLineEvidenceV2, load_calibration
from utils.Evaluator import PaperEvaluator
from utils.dataset_os import manifest_pairs


LOWER_IS_BETTER = {"Nabf"}
PRIMARY = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep line-guided preservation of raw SAR Detail"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--line-calibration", type=Path, required=True)
    parser.add_argument("--confidence-floor", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--min-extent", type=int, default=9)
    parser.add_argument("--min-elongation", type=float, default=1.5)
    parser.add_argument("--large-area", type=int, default=48)
    parser.add_argument(
        "--preserve-strengths", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def strength_id(value):
    return f"line_p{float(value):.2f}".replace(".", "p")


def continuity_filter(
    mask,
    min_area=16,
    min_extent=9,
    min_elongation=1.5,
    large_area=48,
):
    """Keep soft-mask components with sufficient spatial continuity."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("line mask must be Bx1xHxW")
    filtered = []
    counts = []
    for sample in mask.detach().cpu().numpy():
        soft = sample[0]
        binary = (soft > 0.0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        keep = np.zeros_like(binary)
        kept = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            extent = max(width, height)
            elongation = extent / max(min(width, height), 1)
            accepted = (
                area >= int(min_area)
                and extent >= int(min_extent)
                and (
                    elongation >= float(min_elongation)
                    or area >= int(large_area)
                )
            )
            if accepted:
                keep[labels == label] = 1
                kept += 1
        filtered.append(soft * keep)
        counts.append(kept)
    result = torch.from_numpy(np.stack(filtered)[:, None]).to(
        device=mask.device, dtype=mask.dtype
    )
    return result, counts


@torch.no_grad()
def line_gated_outputs(models, line_model, optical, sar, args):
    raw_base, raw_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, diagnostics = models["spd"](
        raw_base, raw_detail
    )
    denoised = models["denoising_head"](
        sar, clean_base, clean_detail
    )
    valid = torch.ones_like(sar)
    raw_line = line_model(sar, valid)["r_line"]
    clean_line = line_model(denoised, valid)["r_line"]
    joint = joint_line_mask(raw_line, clean_line)
    high = high_confidence_line_mask(joint, args.confidence_floor)
    filtered, kept_counts = continuity_filter(
        high,
        min_area=args.min_area,
        min_extent=args.min_extent,
        min_elongation=args.min_elongation,
        large_area=args.large_area,
    )
    if filtered.shape[-2:] != clean_detail.shape[-2:]:
        detail_mask = F.interpolate(
            filtered, size=clean_detail.shape[-2:], mode="bilinear", align_corners=False
        )
    else:
        detail_mask = filtered
    outputs = {"b0_raw": fuse(models, optical, raw_base, raw_detail)}
    delta_to_raw = raw_detail - clean_detail
    for strength in args.preserve_strengths:
        candidate_detail = clean_detail + float(strength) * detail_mask * delta_to_raw
        outputs[strength_id(strength)] = fuse(
            models, optical, clean_base, candidate_detail
        )
    high_mass = float(high.sum())
    filtered_mass = float(filtered.sum())
    diagnostics = {
        "detail_correction_rms": float(diagnostics["detail_correction_rms"]),
        "line_joint_mean": float(joint.mean()),
        "line_high_mean": float(high.mean()),
        "line_high_support": float((high > 0).float().mean()),
        "line_filtered_mean": float(filtered.mean()),
        "line_filtered_support": float((filtered > 0).float().mean()),
        "line_retained_mass_fraction": filtered_mass / max(high_mass, 1e-12),
        "line_kept_components": float(np.mean(kept_counts)),
    }
    return outputs, filtered, denoised, diagnostics


def summarize(rows, methods, metric_names, baseline_id):
    payload = {"images": len(rows), "methods": {}, "comparisons": {}}
    matrices = {}
    for method in methods:
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in metric_names] for row in rows],
            dtype=np.float64,
        )
        matrices[method] = matrix
        payload["methods"][method] = {
            name: {"mean": float(matrix[:, i].mean()), "std": float(matrix[:, i].std())}
            for i, name in enumerate(metric_names)
        }
    baseline = matrices[baseline_id]
    for method in methods:
        if method == baseline_id:
            continue
        delta = matrices[method] - baseline
        payload["comparisons"][f"{method}_minus_{baseline_id}"] = {
            name: {
                "mean_delta": float(delta[:, i].mean()),
                "median_delta": float(np.median(delta[:, i])),
                "improved_fraction": float(
                    (delta[:, i] < 0).mean()
                    if name in LOWER_IS_BETTER
                    else (delta[:, i] > 0).mean()
                ),
            }
            for i, name in enumerate(metric_names)
        }
    diagnostic_names = (
        "detail_correction_rms",
        "line_joint_mean",
        "line_high_mean",
        "line_high_support",
        "line_filtered_mean",
        "line_filtered_support",
        "line_retained_mass_fraction",
        "line_kept_components",
    )
    payload["diagnostics"] = {
        name: float(np.mean([row[name] for row in rows])) for name in diagnostic_names
    }
    return payload


def main():
    args = parse_args()
    if args.split != "val":
        raise ValueError("Line-gate selection is validation-only")
    if not 0.0 <= args.confidence_floor < 1.0:
        raise ValueError("confidence floor must be in [0,1)")
    if any(value < 0.0 or value > 1.0 for value in args.preserve_strengths):
        raise ValueError("preserve strengths must be in [0,1]")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][args.offset : args.offset + args.limit]
    if not selected:
        raise RuntimeError("No Optical-SAR pairs selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    calibration = load_calibration(args.line_calibration)
    line_model = FixedSARLineEvidenceV2(
        calibration=calibration, scales=tuple(calibration["scales"])
    ).to(device).eval()
    methods = ["b0_raw"] + [strength_id(value) for value in args.preserve_strengths]
    baseline_id = strength_id(0.0)
    if len(set(methods)) != len(methods):
        raise ValueError("preserve strengths must be unique and include 0 only once")
    if baseline_id not in methods:
        raise ValueError("preserve strengths must include 0.0 as the SPD baseline")
    image_root = args.output_dir / "images"
    if args.save_images:
        for method in methods:
            (image_root / method).mkdir(parents=True, exist_ok=True)
        (image_root / "line_mask").mkdir(parents=True, exist_ok=True)
        (image_root / "spd_sar").mkdir(parents=True, exist_ok=True)
    rows = []
    metric_names = PaperEvaluator.METRIC_NAMES
    for index, (optical_path, sar_path) in enumerate(selected):
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        if optical_bgr is None or sar_u8 is None:
            raise RuntimeError(f"Could not read pair {optical_path.stem}")
        optical_y_u8 = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        optical = torch.from_numpy(optical_y_u8.astype(np.float32) / 255.0)[
            None, None
        ].to(device)
        sar = torch.from_numpy(sar_u8.astype(np.float32) / 255.0)[None, None].to(device)
        outputs, line_mask, denoised, diagnostics = line_gated_outputs(
            models, line_model, optical, sar, args
        )
        row = {"image": optical_path.stem, **diagnostics}
        for method, output in outputs.items():
            output_u8 = as_u8(output)
            metrics = PaperEvaluator.metric_vector(
                output_u8.astype(np.float32),
                optical_y_u8.astype(np.float32),
                sar_u8.astype(np.float32),
            )
            for name, value in zip(metric_names, metrics):
                row[f"{method}_{name}"] = float(value)
            if index < args.save_images:
                cv2.imwrite(str(image_root / method / f"{optical_path.stem}.png"), output_u8)
        if index < args.save_images:
            cv2.imwrite(str(image_root / "line_mask" / f"{optical_path.stem}.png"), as_u8(line_mask))
            cv2.imwrite(str(image_root / "spd_sar" / f"{optical_path.stem}.png"), as_u8(denoised))
        rows.append(row)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(selected):
            print(f"processed {index + 1}/{len(selected)}: {optical_path.stem}", flush=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, methods, metric_names, baseline_id)
    summary["protocol"] = {
        "dataset": "QXS-SAROPT",
        "split": args.split,
        "offset": args.offset,
        "sample_count": len(selected),
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "b0_checkpoint": str(args.b0_checkpoint.resolve()),
        "b0_checkpoint_sha256": b0_hash,
        "spd_checkpoint": str(args.spd_checkpoint.resolve()),
        "spd_checkpoint_sha256": spd_hash,
        "encoder_equivalence": equivalence,
        "spd_training_source_checkpoint": source,
        "line_calibration": str(args.line_calibration.resolve()),
        "line_calibration_file_sha256": sha256_file(args.line_calibration),
        "line_calibration_payload_sha256": calibration.get("sha256"),
        "confidence_floor": args.confidence_floor,
        "min_area": args.min_area,
        "min_extent": args.min_extent,
        "min_elongation": args.min_elongation,
        "large_area": args.large_area,
        "preserve_strengths": list(args.preserve_strengths),
        "baseline_method": baseline_id,
        "selection_warning": "Validation-only; do not select or tune on test.",
        "primary_metrics": list(PRIMARY),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
