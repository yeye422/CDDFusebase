"""No-training scale diagnostic for failed PDCF P2 corrections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_pdcf_single_path import load_p2, load_predictor
from pdcf_model import PDCFModel
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_OS import OpticalSARLosses, reference_normalized_smooth_minimax
from train_pdcf_predictability import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


DEFAULT_SCALES = (0.05, 0.1, 0.25, 0.5, 1.0)
PROXY_ARGUMENTS = (
    "optical_lowpass_sigma",
    "frequency_highpass_sigma",
    "frequency_midpass_sigma",
    "optical_high_weight",
    "structure_percentile",
    "structure_charbonnier_epsilon",
    "minimax_temperature",
)


def scale_id(value):
    magnitude = f"{abs(float(value)):.3f}".replace(".", "p")
    return f"n{magnitude}" if value < 0.0 else magnitude


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep trained PDCF P2 correction scales without retraining"
    )
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--predictable-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--save-previews", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    parser.add_argument("--local-block-size", type=int, default=32)
    parser.add_argument("--local-gradient-budget", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


@torch.no_grad()
def correction_tensors(model, optical, sar):
    c0_base, c0_detail, predictor_outputs = model.frozen_features(optical, sar)
    fused_base, fused_detail, _ = model.adapter(
        predictor_outputs, c0_base, c0_detail, model.input_mode
    )
    return c0_base, c0_detail, fused_base - c0_base, fused_detail - c0_detail


@torch.no_grad()
def decode_scaled(c0, optical, c0_base, c0_detail, delta_base, delta_detail, scale):
    output, _ = c0.decoder(
        optical,
        c0_base + float(scale) * delta_base,
        c0_detail + float(scale) * delta_detail,
    )
    return output


def paired_bootstrap_mean_ci(values, repetitions, rng, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "confidence": float(confidence),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "repetitions": int(repetitions),
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_tolerance = np.finfo(np.float64).eps * max(1.0, float(np.abs(left).max())) * 8
    right_tolerance = np.finfo(np.float64).eps * max(1.0, float(np.abs(right).max())) * 8
    if (
        left.size < 2
        or np.ptp(left) <= left_tolerance
        or np.ptp(right) <= right_tolerance
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def local_change_diagnostics(candidate, baseline, block_size, gradient_budget):
    if block_size <= 0:
        raise ValueError("local block size must be positive")
    difference = candidate.float() - baseline.float()
    size = min(block_size, difference.shape[-2], difference.shape[-1])
    tile_rms = F.avg_pool2d(
        difference.square(), kernel_size=size, stride=size
    ).sqrt().flatten()

    sobel_x = candidate.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    candidate_gradient = torch.cat(
        (
            F.conv2d(candidate.float(), sobel_x.float(), padding=1),
            F.conv2d(candidate.float(), sobel_y.float(), padding=1),
        ),
        dim=1,
    )
    baseline_gradient = torch.cat(
        (
            F.conv2d(baseline.float(), sobel_x.float(), padding=1),
            F.conv2d(baseline.float(), sobel_y.float(), padding=1),
        ),
        dim=1,
    )
    gradient_error = (candidate_gradient - baseline_gradient).square().sum(
        dim=1, keepdim=True
    ).sqrt()
    baseline_magnitude = baseline_gradient.square().sum(dim=1, keepdim=True).sqrt()
    tile_error = F.avg_pool2d(gradient_error, kernel_size=size, stride=size)
    tile_reference = F.avg_pool2d(
        baseline_magnitude, kernel_size=size, stride=size
    ).clamp_min(1.0 / 255.0)
    tile_ratio = (tile_error / tile_reference).flatten()
    return {
        "local_output_rms_p95": float(torch.quantile(tile_rms, 0.95)),
        "local_output_rms_max": float(tile_rms.max()),
        "local_gradient_ratio_mean": float(tile_ratio.mean()),
        "local_gradient_ratio_p95": float(torch.quantile(tile_ratio, 0.95)),
        "local_gradient_ratio_max": float(tile_ratio.max()),
        "local_gradient_budget_exceeded_fraction": float(
            (tile_ratio > gradient_budget).float().mean()
        ),
    }


def proxy_losses(losses, reference, proxy_args, optical, sar, generated, valid):
    (_, optical_loss, _, _, _, structure_loss, _, _) = (
        losses.fusion_frequency_decoupled(
            optical,
            sar,
            generated,
            valid,
            proxy_args["optical_lowpass_sigma"],
            proxy_args["frequency_highpass_sigma"],
            proxy_args["frequency_midpass_sigma"],
            proxy_args["optical_high_weight"],
            proxy_args["structure_percentile"],
            proxy_args["structure_charbonnier_epsilon"],
        )
    )
    minimax, _, _ = reference_normalized_smooth_minimax(
        optical_loss,
        structure_loss,
        reference["optical"],
        reference["structure"],
        proxy_args["minimax_temperature"],
    )
    return {
        "proxy_minimax": float(minimax),
        "proxy_optical": float(optical_loss),
        "proxy_structure": float(structure_loss),
    }


def summarize(rows, methods, bootstrap_repetitions, bootstrap_seed):
    baseline = np.asarray(
        [[row[f"c0_{name}"] for name in PAPER_METRIC_NAMES] for row in rows]
    )
    summary = {"images": len(rows), "methods": {}, "comparison_to_c0": {}}
    rng = np.random.default_rng(bootstrap_seed)
    for method in methods:
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows]
        )
        summary["methods"][method] = {
            name: {
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std()),
            }
            for index, name in enumerate(PAPER_METRIC_NAMES)
        }
        summary["methods"][method]["output_deviation_rms"] = {
            "mean": float(np.mean([row[f"{method}_output_deviation_rms"] for row in rows])),
            "std": float(np.std([row[f"{method}_output_deviation_rms"] for row in rows])),
        }
        for diagnostic in (
            "proxy_minimax", "proxy_optical", "proxy_structure",
            "local_output_rms_p95", "local_output_rms_max",
            "local_gradient_ratio_mean", "local_gradient_ratio_p95",
            "local_gradient_ratio_max", "local_gradient_budget_exceeded_fraction",
        ):
            values = [row[f"{method}_{diagnostic}"] for row in rows]
            summary["methods"][method][diagnostic] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        if method != "c0":
            delta = matrix - baseline
            proxy_delta = np.asarray(
                [row[f"{method}_proxy_minimax"] - row["c0_proxy_minimax"] for row in rows]
            )
            comparisons = {
                name: {
                    "mean_delta": float(delta[:, index].mean()),
                    "median_delta": float(np.median(delta[:, index])),
                    "paired_bootstrap_mean_delta_ci": paired_bootstrap_mean_ci(
                        delta[:, index], bootstrap_repetitions, rng
                    ),
                    "improved_fraction": float(
                        (delta[:, index] < 0).mean()
                        if name == "Nabf"
                        else (delta[:, index] > 0).mean()
                    ),
                    "proxy_delta_pearson": safe_correlation(
                        proxy_delta, delta[:, index]
                    ),
                    "proxy_delta_spearman": safe_correlation(
                        average_ranks(proxy_delta), average_ranks(delta[:, index])
                    ),
                }
                for index, name in enumerate(PAPER_METRIC_NAMES)
            }
            comparisons["proxy_minimax"] = {
                "mean_delta": float(proxy_delta.mean()),
                "median_delta": float(np.median(proxy_delta)),
                "paired_bootstrap_mean_delta_ci": paired_bootstrap_mean_ci(
                    proxy_delta, bootstrap_repetitions, rng
                ),
            }
            summary["comparison_to_c0"][method] = comparisons
    return summary


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if any(scale < -1.0 or scale > 1.0 for scale in args.scales):
        raise ValueError("All diagnostic scales must be in [-1,1]")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap-repetitions must be positive")
    if args.local_gradient_budget < 0.0:
        raise ValueError("local-gradient-budget must be non-negative")
    scales = tuple(dict.fromkeys(float(value) for value in args.scales))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c0_hash = sha256_file(args.c0_checkpoint)
    p1_hash = sha256_file(args.p1_checkpoint)
    split_hash = sha256_file(args.split_manifest)
    hashes = {
        "source_checkpoint_sha256": c0_hash,
        "predictor_checkpoint_sha256": p1_hash,
        "split_manifest_sha256": split_hash,
    }
    c0_checkpoint = torch.load(
        args.c0_checkpoint, map_location="cpu", weights_only=False
    )
    reference = c0_checkpoint.get("frequency_training_reference")
    if not reference:
        raise ValueError("C0 checkpoint has no frequency_training_reference")
    c0 = PDCFModel.from_c0_checkpoint(
        args.c0_checkpoint, map_location=device
    ).to(device).eval()
    predictor = load_predictor(args.p1_checkpoint, device)
    predictable, predictable_checkpoint = load_p2(
        args.predictable_checkpoint, c0, predictor,
        "predictable_residual", hashes, device,
    )
    raw, raw_checkpoint = load_p2(
        args.raw_checkpoint, c0, predictor,
        "raw_evidence", hashes, device,
    )
    models = {"predictable": predictable, "raw": raw}
    proxy_args = {
        name: predictable_checkpoint["args"][name] for name in PROXY_ARGUMENTS
    }
    for name in PROXY_ARGUMENTS:
        if raw_checkpoint["args"][name] != proxy_args[name]:
            raise ValueError(f"P2 checkpoints disagree on proxy argument {name}")
    losses = OpticalSARLosses().to(device)
    methods = ["c0"] + [
        f"{name}_s{scale_id(scale)}"
        for name in models
        for scale in scales
    ]

    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][args.offset:args.offset + args.limit]
    dataset = OpticalSARDataset(
        selected, patch_size=args.patch_size, training=False,
        augment=False, invalid_threshold=0,
    )
    args.output_dir.mkdir(parents=True)
    preview_dir = args.output_dir / "previews"
    if args.save_previews:
        preview_dir.mkdir()
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            optical, sar, valid, stem = dataset[index]
            optical = optical[None].to(device)
            sar = sar[None].to(device)
            valid = valid[None].to(device)
            c0_output, _ = c0.forward_c0(optical, sar)
            c0_clamped = c0_output.clamp(0.0, 1.0)
            outputs = {"c0": c0_output}
            for name, model in models.items():
                c0_base, c0_detail, delta_base, delta_detail = correction_tensors(
                    model, optical, sar
                )
                for scale in scales:
                    method = f"{name}_s{scale_id(scale)}"
                    outputs[method] = decode_scaled(
                        c0, optical, c0_base, c0_detail,
                        delta_base, delta_detail, scale,
                    )
            optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
            row = {"image": stem}
            preview = [optical_u8, sar_u8]
            for method in methods:
                tensor = outputs[method].clamp(0.0, 1.0)
                image = as_u8(tensor)
                values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
                row.update(
                    {f"{method}_{metric}": float(value)
                     for metric, value in zip(PAPER_METRIC_NAMES, values)}
                )
                row[f"{method}_output_deviation_rms"] = float(
                    (tensor - c0_clamped).square().mean().sqrt()
                )
                row.update({
                    f"{method}_{name}": value
                    for name, value in proxy_losses(
                        losses, reference, proxy_args,
                        optical, sar, outputs[method], valid,
                    ).items()
                })
                row.update({
                    f"{method}_{name}": value
                    for name, value in local_change_diagnostics(
                        tensor, c0_clamped,
                        args.local_block_size, args.local_gradient_budget,
                    ).items()
                })
                preview.append(image)
            rows.append(row)
            if index < args.save_previews:
                cv2.imwrite(
                    str(preview_dir / f"{stem}.png"),
                    np.concatenate(preview, axis=1),
                )
            print(f"[{index + 1}/{len(dataset)}] {stem}", flush=True)
    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(
        rows, methods, args.bootstrap_repetitions, args.bootstrap_seed
    )
    summary["protocol"] = {
        "name": "PDCF_P2_no_training_scale_diagnostic_v1",
        "dataset": manifest.get("dataset", "QXS-SAROPT"),
        "split": args.split,
        "offset": args.offset,
        "limit": len(rows),
        "patch_size": args.patch_size,
        "scales": list(scales),
        "metric_delta_convention": "candidate_minus_c0",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "local_block_size": args.local_block_size,
        "local_gradient_budget": args.local_gradient_budget,
        "proxy_loss_arguments": proxy_args,
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "c0_checkpoint_sha256": c0_hash,
        "p1_checkpoint_sha256": p1_hash,
        "predictable_checkpoint_sha256": sha256_file(args.predictable_checkpoint),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        "split_manifest_sha256": split_hash,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
