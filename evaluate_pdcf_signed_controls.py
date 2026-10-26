"""Confirm signed PDCF corrections against RMS-matched non-learned controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_pdcf_single_path import load_p2, load_predictor
from pdcf_model import PDCFModel
from sweep_pdcf_p2_scales import (
    PROXY_ARGUMENTS,
    as_u8,
    correction_tensors,
    decode_scaled,
    local_change_diagnostics,
    paired_bootstrap_mean_ci,
    proxy_losses,
    summarize,
)
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_OS import OpticalSARLosses
from train_pdcf_predictability import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


METHODS = (
    "c0",
    "predictable_neg005",
    "raw_neg005",
    "unsharp_match_predictable",
    "detail_gain_match_predictable",
    "unsharp_match_raw",
    "detail_gain_match_raw",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate non-learned controls on dev images and evaluate all fixed "
            "candidates once on a disjoint holdout"
        )
    )
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--predictable-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int, default=200)
    parser.add_argument("--evaluation-offset", type=int, default=200)
    parser.add_argument("--evaluation-limit", type=int, default=800)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--signed-scale", type=float, default=-0.05)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--calibration-initial-beta", type=float, default=0.01)
    parser.add_argument("--calibration-iterations", type=int, default=4)
    parser.add_argument("--calibration-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    parser.add_argument("--local-block-size", type=int, default=32)
    parser.add_argument("--local-gradient-budget", type=float, default=0.05)
    parser.add_argument("--save-previews", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def c0_state(c0, optical, sar):
    features = c0.extract_features(optical, sar)
    c0_base, c0_detail = c0.c0_features(*features)
    c0_output, _ = c0.decoder(optical, c0_base, c0_detail)
    return c0_base, c0_detail, c0_output


@torch.no_grad()
def unsharp_output(losses, c0_output, beta, sigma):
    highpass = c0_output - losses.gaussian_blur(c0_output, sigma)
    return c0_output + float(beta) * highpass


@torch.no_grad()
def detail_gain_output(c0, optical, c0_base, c0_detail, beta):
    output, _ = c0.decoder(
        optical, c0_base, (1.0 + float(beta)) * c0_detail
    )
    return output


def output_rms(candidate, baseline):
    return float(
        (
            candidate.clamp(0.0, 1.0) - baseline.clamp(0.0, 1.0)
        ).square().mean().sqrt()
    )


def make_dataset(pairs, args):
    return OpticalSARDataset(
        pairs,
        patch_size=args.patch_size,
        training=False,
        augment=False,
        invalid_threshold=0,
    )


@torch.no_grad()
def measure_target_rms(dataset, model, c0, device, signed_scale):
    values = []
    for optical, sar, _, _ in dataset:
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        c0_base, c0_detail, delta_base, delta_detail = correction_tensors(
            model, optical, sar
        )
        c0_output, _ = c0.decoder(optical, c0_base, c0_detail)
        candidate = decode_scaled(
            c0, optical, c0_base, c0_detail,
            delta_base, delta_detail, signed_scale,
        )
        values.append(output_rms(candidate, c0_output))
    if not values:
        raise RuntimeError("RMS calibration dataset is empty")
    return float(np.mean(values))


@torch.no_grad()
def measure_control_rms(dataset, control, beta, c0, losses, device, args):
    values = []
    for optical, sar, _, _ in dataset:
        optical = optical[None].to(device)
        sar = sar[None].to(device)
        c0_base, c0_detail, c0_output = c0_state(c0, optical, sar)
        if control == "unsharp":
            candidate = unsharp_output(
                losses, c0_output, beta, args.unsharp_sigma
            )
        elif control == "detail_gain":
            candidate = detail_gain_output(
                c0, optical, c0_base, c0_detail, beta
            )
        else:
            raise ValueError(f"Unknown control: {control}")
        values.append(output_rms(candidate, c0_output))
    if not values:
        raise RuntimeError("RMS calibration dataset is empty")
    return float(np.mean(values))


def next_calibration_beta(beta, target_rms, measured_rms):
    if beta <= 0.0 or target_rms <= 0.0 or measured_rms <= 0.0:
        raise ValueError("Calibration beta and RMS values must be positive")
    return float(beta * target_rms / measured_rms)


def calibrate_control(dataset, control, target_rms, c0, losses, device, args):
    beta = float(args.calibration_initial_beta)
    history = []
    for iteration in range(1, args.calibration_iterations + 1):
        measured = measure_control_rms(
            dataset, control, beta, c0, losses, device, args
        )
        relative_error = abs(measured - target_rms) / target_rms
        history.append({
            "iteration": iteration,
            "beta": beta,
            "mean_output_rms": measured,
            "relative_error": relative_error,
        })
        if relative_error <= args.calibration_tolerance:
            break
        beta = next_calibration_beta(beta, target_rms, measured)
        if not np.isfinite(beta) or beta > 100.0:
            raise RuntimeError(
                f"{control} RMS calibration produced invalid beta {beta}"
            )
    if history[-1]["relative_error"] > args.calibration_tolerance:
        raise RuntimeError(
            f"{control} RMS calibration did not reach tolerance: {history[-1]}"
        )
    return beta, history


def compare_methods(rows, candidate, reference, repetitions, seed):
    rng = np.random.default_rng(seed)
    output = {}
    for metric in PAPER_METRIC_NAMES:
        delta = np.asarray([
            row[f"{candidate}_{metric}"] - row[f"{reference}_{metric}"]
            for row in rows
        ])
        output[metric] = {
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "improved_fraction": float(
                (delta < 0.0).mean() if metric == "Nabf" else (delta > 0.0).mean()
            ),
            "paired_bootstrap_mean_delta_ci": paired_bootstrap_mean_ci(
                delta, repetitions, rng
            ),
        }
    return output


def validate_args(args, split_size):
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not np.isclose(args.signed_scale, -0.05):
        raise ValueError("This frozen confirmation protocol requires signed-scale=-0.05")
    if args.unsharp_sigma <= 0.0:
        raise ValueError("unsharp-sigma must be positive")
    if args.calibration_initial_beta <= 0.0:
        raise ValueError("calibration-initial-beta must be positive")
    if args.calibration_iterations <= 0:
        raise ValueError("calibration-iterations must be positive")
    if not 0.0 < args.calibration_tolerance < 1.0:
        raise ValueError("calibration-tolerance must be in (0,1)")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap-repetitions must be positive")
    if args.local_block_size <= 0:
        raise ValueError("local-block-size must be positive")
    if args.local_gradient_budget < 0.0:
        raise ValueError("local-gradient-budget must be non-negative")
    calibration_end = args.calibration_offset + args.calibration_limit
    evaluation_end = args.evaluation_offset + args.evaluation_limit
    if min(
        args.calibration_offset, args.evaluation_offset,
        args.calibration_limit, args.evaluation_limit,
    ) < 0:
        raise ValueError("split offsets and limits must be non-negative")
    if args.calibration_limit == 0 or args.evaluation_limit == 0:
        raise ValueError("calibration and evaluation limits must be positive")
    calibration_indices = set(range(args.calibration_offset, calibration_end))
    evaluation_indices = set(range(args.evaluation_offset, evaluation_end))
    if calibration_indices & evaluation_indices:
        raise ValueError("calibration and evaluation subsets must be disjoint")
    if max(calibration_end, evaluation_end) > split_size:
        raise ValueError("requested subset exceeds split size")


def main():
    args = parse_args()
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    validate_args(args, len(splits[args.split]))
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
    proxy_args = {
        name: predictable_checkpoint["args"][name] for name in PROXY_ARGUMENTS
    }
    for name in PROXY_ARGUMENTS:
        if raw_checkpoint["args"][name] != proxy_args[name]:
            raise ValueError(f"P2 checkpoints disagree on proxy argument {name}")
    losses = OpticalSARLosses().to(device)

    calibration_pairs = splits[args.split][
        args.calibration_offset:args.calibration_offset + args.calibration_limit
    ]
    evaluation_pairs = splits[args.split][
        args.evaluation_offset:args.evaluation_offset + args.evaluation_limit
    ]
    calibration_dataset = make_dataset(calibration_pairs, args)
    predictable_target_rms = measure_target_rms(
        calibration_dataset, predictable, c0, device, args.signed_scale
    )
    raw_target_rms = measure_target_rms(
        calibration_dataset, raw, c0, device, args.signed_scale
    )
    predictable_unsharp_beta, predictable_unsharp_history = calibrate_control(
        calibration_dataset, "unsharp", predictable_target_rms,
        c0, losses, device, args,
    )
    predictable_detail_beta, predictable_detail_history = calibrate_control(
        calibration_dataset, "detail_gain", predictable_target_rms,
        c0, losses, device, args,
    )
    raw_unsharp_beta, raw_unsharp_history = calibrate_control(
        calibration_dataset, "unsharp", raw_target_rms,
        c0, losses, device, args,
    )
    raw_detail_beta, raw_detail_history = calibrate_control(
        calibration_dataset, "detail_gain", raw_target_rms,
        c0, losses, device, args,
    )

    calibration_record = {
        "subset": {
            "split": args.split,
            "offset": args.calibration_offset,
            "limit": len(calibration_dataset),
        },
        "target_signed_scale": args.signed_scale,
        "criterion": "mean clamped output RMS only; no fusion metric tuning",
        "tolerance": args.calibration_tolerance,
        "predictable_residual": {
            "target_mean_output_rms": predictable_target_rms,
            "unsharp": {
                "sigma": args.unsharp_sigma,
                "beta": predictable_unsharp_beta,
                "history": predictable_unsharp_history,
            },
            "detail_gain": {
                "beta": predictable_detail_beta,
                "history": predictable_detail_history,
            },
        },
        "raw_evidence": {
            "target_mean_output_rms": raw_target_rms,
            "unsharp": {
                "sigma": args.unsharp_sigma,
                "beta": raw_unsharp_beta,
                "history": raw_unsharp_history,
            },
            "detail_gain": {
                "beta": raw_detail_beta,
                "history": raw_detail_history,
            },
        },
        "c0_checkpoint_sha256": c0_hash,
        "p1_checkpoint_sha256": p1_hash,
        "predictable_checkpoint_sha256": sha256_file(args.predictable_checkpoint),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        "split_manifest_sha256": split_hash,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preview_dir = args.output_dir / "previews"
    if args.save_previews:
        preview_dir.mkdir()
    evaluation_dataset = make_dataset(evaluation_pairs, args)
    rows = []
    with torch.inference_mode():
        for index in range(len(evaluation_dataset)):
            optical, sar, valid, stem = evaluation_dataset[index]
            optical = optical[None].to(device)
            sar = sar[None].to(device)
            valid = valid[None].to(device)
            c0_base, c0_detail, c0_output = c0_state(c0, optical, sar)
            c0_clamped = c0_output.clamp(0.0, 1.0)
            outputs = {"c0": c0_output}
            for name, model in (
                ("predictable_neg005", predictable),
                ("raw_neg005", raw),
            ):
                model_base, model_detail, delta_base, delta_detail = (
                    correction_tensors(model, optical, sar)
                )
                outputs[name] = decode_scaled(
                    c0, optical, model_base, model_detail,
                    delta_base, delta_detail, args.signed_scale,
                )
            outputs["unsharp_match_predictable"] = unsharp_output(
                losses, c0_output,
                predictable_unsharp_beta, args.unsharp_sigma,
            )
            outputs["detail_gain_match_predictable"] = detail_gain_output(
                c0, optical, c0_base, c0_detail, predictable_detail_beta
            )
            outputs["unsharp_match_raw"] = unsharp_output(
                losses, c0_output, raw_unsharp_beta, args.unsharp_sigma
            )
            outputs["detail_gain_match_raw"] = detail_gain_output(
                c0, optical, c0_base, c0_detail, raw_detail_beta
            )

            optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
            row = {"image": stem}
            preview = [optical_u8, sar_u8]
            for method in METHODS:
                tensor = outputs[method].clamp(0.0, 1.0)
                image = as_u8(tensor)
                values = evaluate(
                    image, optical_u8, sar_u8, metric_profile="paper"
                )
                row.update({
                    f"{method}_{metric}": float(value)
                    for metric, value in zip(PAPER_METRIC_NAMES, values)
                })
                row[f"{method}_output_deviation_rms"] = output_rms(
                    tensor, c0_clamped
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
            print(f"[{index + 1}/{len(evaluation_dataset)}] {stem}", flush=True)

    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(
        rows, METHODS, args.bootstrap_repetitions, args.bootstrap_seed
    )
    summary["pairwise_comparison"] = {
        "predictable_vs_raw": compare_methods(
            rows, "predictable_neg005", "raw_neg005",
            args.bootstrap_repetitions, args.bootstrap_seed + 1,
        ),
        "predictable_vs_unsharp": compare_methods(
            rows, "predictable_neg005", "unsharp_match_predictable",
            args.bootstrap_repetitions, args.bootstrap_seed + 2,
        ),
        "predictable_vs_detail_gain": compare_methods(
            rows, "predictable_neg005", "detail_gain_match_predictable",
            args.bootstrap_repetitions, args.bootstrap_seed + 3,
        ),
        "raw_vs_unsharp": compare_methods(
            rows, "raw_neg005", "unsharp_match_raw",
            args.bootstrap_repetitions, args.bootstrap_seed + 4,
        ),
        "raw_vs_detail_gain": compare_methods(
            rows, "raw_neg005", "detail_gain_match_raw",
            args.bootstrap_repetitions, args.bootstrap_seed + 5,
        ),
    }
    summary["calibration"] = calibration_record
    summary["protocol"] = {
        "name": "PDCF_P2_signed_controls_disjoint_holdout_v1",
        "dataset": manifest.get("dataset", "QXS-SAROPT"),
        "split": args.split,
        "evaluation_offset": args.evaluation_offset,
        "evaluation_limit": len(evaluation_dataset),
        "patch_size": args.patch_size,
        "methods": list(METHODS),
        "metric_delta_convention": "candidate_minus_reference",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "local_block_size": args.local_block_size,
        "local_gradient_budget": args.local_gradient_budget,
        "proxy_loss_arguments": proxy_args,
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "preview_columns": ["optical", "sar", *METHODS],
        "c0_checkpoint_sha256": c0_hash,
        "p1_checkpoint_sha256": p1_hash,
        "predictable_checkpoint_sha256": sha256_file(args.predictable_checkpoint),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        "split_manifest_sha256": split_hash,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
