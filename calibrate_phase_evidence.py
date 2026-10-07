"""Calibrate phase-antisymmetry and RoA+FA evidence on training flat pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from phase_antisymmetry import MonogenicPhaseAntisymmetry, local_phase_bridge
from structure_evidence_v2 import FixedSARLineEvidenceV2, load_calibration
from utils.dataset_os import manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fse-calibration", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--pixels-per-image", type=int, default=2048)
    parser.add_argument("--flat-quantile", type=float, default=0.40)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--transition-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def calibrated_entry(values, transition_quantile, threshold_quantile):
    q_transition = float(np.quantile(values, transition_quantile))
    q_threshold = float(np.quantile(values, threshold_quantile))
    return {
        "threshold": q_threshold,
        "transition": max(q_threshold - q_transition, 1e-4),
        "q95": q_transition,
        "q995": q_threshold,
        "sampled_pixels": int(values.size),
    }


def sigmoid_probability(values, entry):
    return 1.0 / (
        1.0
        + np.exp(-(values - entry["threshold"]) / entry["transition"])
    )


def main():
    args = parse_args()
    fse_calibration = load_calibration(args.fse_calibration)
    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    train_pairs = splits["train"]
    rng = np.random.default_rng(args.seed)
    count = min(args.sample_count, len(train_pairs))
    indices = np.sort(rng.choice(len(train_pairs), size=count, replace=False))
    selected = [train_pairs[int(index)] for index in indices]
    selected_ids = [optical.stem for optical, _ in selected]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fse = FixedSARLineEvidenceV2(
        calibration=fse_calibration,
        scales=tuple(fse_calibration["scales"]),
    ).to(device).eval()
    phase = MonogenicPhaseAntisymmetry().to(device).eval()
    fa_samples = []
    persistence_samples = []
    roa_line_samples = []
    roa_support_samples = []

    with torch.no_grad():
        for index, (_, sar_path) in enumerate(selected, start=1):
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            if sar_u8 is None:
                raise RuntimeError(f"Could not read {sar_path}")
            sar = torch.from_numpy(sar_u8)[None, None].float().to(device) / 255.0
            valid = (sar > 0.0).float()
            raw = fse.raw_features(sar, valid)
            fse_result = fse(sar, valid)
            phase_result = phase(sar, valid)
            energy = raw["tensor_energy"].squeeze()
            keep = raw["valid"].squeeze() > 0.5
            threshold = torch.quantile(energy[keep].float(), args.flat_quantile)
            flat_indices = torch.nonzero(
                (keep & (energy <= threshold)).flatten(), as_tuple=False
            ).squeeze(1)
            if flat_indices.numel() > args.pixels_per_image:
                generator = torch.Generator(device=flat_indices.device)
                generator.manual_seed(args.seed * 1000003 + index)
                order = torch.randperm(
                    flat_indices.numel(), generator=generator, device=flat_indices.device
                )[: args.pixels_per_image]
                flat_indices = flat_indices[order]
            fa_samples.append(
                phase_result["fa_mean"].flatten()[flat_indices].cpu().numpy()
            )
            persistence_samples.append(
                fse_result["scale_persistence"].flatten()[flat_indices].cpu().numpy()
            )
            roa_line_samples.append(
                fse_result["r_line"].flatten()[flat_indices].cpu().numpy()
            )
            roa_support = torch.nn.functional.max_pool2d(
                (fse_result["r_line"] > 0.5).to(
                    fse_result["r_line"].dtype
                ),
                5,
                stride=1,
                padding=2,
            )
            roa_support_samples.append(
                roa_support.flatten()[flat_indices].cpu().numpy()
            )
            if index == 1 or index % 50 == 0 or index == len(selected):
                print(f"calibrated {index}/{len(selected)}", flush=True)

    fa_values = np.concatenate(fa_samples)
    persistence_values = np.concatenate(persistence_samples)
    roa_line_values = np.concatenate(roa_line_samples)
    roa_support_values = np.concatenate(roa_support_samples)
    fa_entry = calibrated_entry(
        fa_values, args.transition_quantile, args.threshold_quantile
    )
    fa_probability = sigmoid_probability(fa_values, fa_entry)
    combined_score = np.sqrt(np.clip(fa_probability * persistence_values, 0.0, None))
    combined_entry = calibrated_entry(
        combined_score, args.transition_quantile, args.threshold_quantile
    )
    bridge_score = roa_line_values + (
        (1.0 - roa_line_values) * fa_probability * roa_support_values
    )
    bridge_entry = calibrated_entry(
        bridge_score, args.transition_quantile, args.threshold_quantile
    )
    payload = {
        "version": "phase_antisymmetry_roa_v2_local_bridge",
        "dataset": "QXS-SAROPT",
        "split": "train",
        "split_manifest": str(args.split_manifest.resolve()),
        "sample_count": len(selected),
        "pixels_per_image": args.pixels_per_image,
        "flat_quantile": args.flat_quantile,
        "threshold_quantile": args.threshold_quantile,
        "transition_quantile": args.transition_quantile,
        "seed": args.seed,
        "wavelengths": list(phase.wavelengths),
        "sigma_on_f": phase.sigma_on_f,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "fse_calibration_sha256": fse_calibration["sha256"],
        "responses": {
            "fa_mean": fa_entry,
            "roa_fa_score": combined_entry,
            "bridge_score": bridge_entry,
        },
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"device: {device}")
    print(f"calibration: {args.output.resolve()}")


if __name__ == "__main__":
    main()
