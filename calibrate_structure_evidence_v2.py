"""Calibrate shared FSE-v2a RoA thresholds from the training split only."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from structure_evidence_v2 import FixedSARLineEvidenceV2
from utils.dataset_os import manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--pixels-per-image", type=int, default=2048)
    parser.add_argument("--flat-quantile", type=float, default=0.40)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--transition-quantile", type=float, default=0.95)
    parser.add_argument("--scales", type=int, nargs="+", default=(3, 5, 9))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_count <= 0 or args.pixels_per_image <= 0:
        raise ValueError("sample-count and pixels-per-image must be positive")
    if not 0.0 < args.flat_quantile < 1.0:
        raise ValueError("flat-quantile must be in (0,1)")
    if not 0.0 < args.transition_quantile < args.threshold_quantile < 1.0:
        raise ValueError("require 0 < transition-quantile < threshold-quantile < 1")

    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    train_pairs = splits["train"]
    rng = np.random.default_rng(args.seed)
    count = min(args.sample_count, len(train_pairs))
    indices = np.sort(rng.choice(len(train_pairs), size=count, replace=False))
    selected = [train_pairs[int(index)] for index in indices]
    selected_ids = [optical.stem for optical, _ in selected]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedSARLineEvidenceV2(
        calibration=None, scales=args.scales
    ).to(device).eval()
    samples = {f"roa_raw_s{scale}": [] for scale in model.scales}
    samples["coherence"] = []
    samples["direction_consistency"] = []

    with torch.no_grad():
        for index, (_, sar_path) in enumerate(selected, start=1):
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            if sar_u8 is None:
                raise RuntimeError(f"Could not read {sar_path}")
            sar = torch.from_numpy(sar_u8)[None, None].float().to(device) / 255.0
            valid = (sar > 0.0).float()
            raw = model.raw_features(sar, valid)
            energy = raw["tensor_energy"].squeeze()
            keep = raw["valid"].squeeze() > 0.5
            if not torch.any(keep):
                continue
            energy_threshold = torch.quantile(
                energy[keep].float(), args.flat_quantile
            )
            flat = keep & (energy <= energy_threshold)
            flat_indices = torch.nonzero(flat.flatten(), as_tuple=False).squeeze(1)
            if flat_indices.numel() > args.pixels_per_image:
                generator = torch.Generator(device=flat_indices.device)
                generator.manual_seed(args.seed * 1000003 + index)
                order = torch.randperm(
                    flat_indices.numel(), generator=generator, device=flat_indices.device
                )[: args.pixels_per_image]
                flat_indices = flat_indices[order]
            for name in samples:
                values = raw[name].flatten()[flat_indices].float().cpu().numpy()
                samples[name].append(values)
            if index == 1 or index % 50 == 0 or index == len(selected):
                print(f"calibrated {index}/{len(selected)}", flush=True)

    arrays = {}
    for name, chunks in samples.items():
        if not chunks:
            raise RuntimeError(f"No calibration samples collected for {name}")
        arrays[name] = np.concatenate(chunks)

    responses = {}
    for name in (f"roa_raw_s{scale}" for scale in model.scales):
        values = arrays[name]
        q_transition = float(np.quantile(values, args.transition_quantile))
        q_threshold = float(np.quantile(values, args.threshold_quantile))
        responses[name] = {
            "threshold": q_threshold,
            "transition": max(q_threshold - q_transition, 1e-4),
            "q95": q_transition,
            "q995": q_threshold,
            "sampled_pixels": int(values.size),
        }

    probabilities = []
    for scale in model.scales:
        name = f"roa_raw_s{scale}"
        entry = responses[name]
        probabilities.append(
            1.0
            / (
                1.0
                + np.exp(
                    -(arrays[name] - entry["threshold"])
                    / entry["transition"]
                )
            )
        )
    sorted_probabilities = np.sort(np.stack(probabilities, axis=0), axis=0)
    scale_persistence = sorted_probabilities[-2]
    line_score = np.cbrt(
        np.clip(
            scale_persistence
            * arrays["coherence"]
            * arrays["direction_consistency"],
            0.0,
            None,
        )
    )
    line_q_transition = float(
        np.quantile(line_score, args.transition_quantile)
    )
    line_q_threshold = float(
        np.quantile(line_score, args.threshold_quantile)
    )
    responses["line_score"] = {
        "threshold": line_q_threshold,
        "transition": max(line_q_threshold - line_q_transition, 1e-4),
        "q95": line_q_transition,
        "q995": line_q_threshold,
        "sampled_pixels": int(line_score.size),
    }

    payload = {
        "version": "fse_v2a_two_stage_global_line",
        "dataset": "QXS-SAROPT",
        "split": "train",
        "split_manifest": str(args.split_manifest.resolve()),
        "sample_count": len(selected),
        "pixels_per_image": args.pixels_per_image,
        "flat_quantile": args.flat_quantile,
        "threshold_quantile": args.threshold_quantile,
        "transition_quantile": args.transition_quantile,
        "scales": list(model.scales),
        "seed": args.seed,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "responses": responses,
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
