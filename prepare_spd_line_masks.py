"""Calibrate, generate, and audit SPD line-structure soft masks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from structure_evidence_v2 import FixedSARLineEvidenceV2, load_calibration
from utils.dataset_spd import paired_sar_paths


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    if image.ndim == 3:
        first = image[:, :, 0]
        if not np.all(image == first[:, :, None]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = first
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}: {path}")
    return image


def as_tensor(image, device):
    return torch.from_numpy(image)[None, None].float().to(device) / 255.0


def joint_line_mask(noisy_probability, reference_probability):
    return torch.sqrt(
        (noisy_probability.clamp(0.0, 1.0)
         * reference_probability.clamp(0.0, 1.0)).clamp_min(0.0)
    )


def high_confidence_line_mask(mask, floor=0.5):
    if not 0.0 <= floor < 1.0:
        raise ValueError("confidence floor must be in [0,1)")
    return ((mask - float(floor)) / (1.0 - float(floor))).clamp(0.0, 1.0)


def save_mask(path, mask):
    array = np.clip(
        np.rint(mask.detach().squeeze().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)
    cv2.imwrite(str(path), array)


def pearson(first, second):
    first = first.detach().float().flatten()
    second = second.detach().float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator) <= 1e-12:
        return 1.0 if torch.allclose(first, second) else 0.0
    return float((first * second).sum() / denominator)


def connected_statistics(mask, threshold=0.5):
    binary = (mask.detach().squeeze().cpu().numpy() >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
    return {
        "component_count": int(max(count - 1, 0)),
        "component_area_mean": float(areas.mean()) if areas.size else 0.0,
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "component_area_max": int(areas.max()) if areas.size else 0,
    }


def perturb(image, sigma, seed):
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        image.shape, generator=generator, device=image.device, dtype=image.dtype
    ) * sigma
    factor = torch.exp(noise - 0.5 * sigma ** 2)
    return (image * factor).clamp(0.0, 1.0)


def calibration_command(args):
    pairs = paired_sar_paths(args.noisy_dir, args.reference_dir)
    rng = np.random.default_rng(args.seed)
    count = min(args.sample_count, len(pairs))
    indices = np.sort(rng.choice(len(pairs), size=count, replace=False))
    selected = [pairs[int(index)] for index in indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedSARLineEvidenceV2(
        calibration=None, scales=tuple(args.scales)
    ).to(device).eval()
    names = [f"roa_raw_s{scale}" for scale in model.scales]
    names += ["coherence", "direction_consistency"]
    samples = {name: [] for name in names}

    with torch.no_grad():
        for pair_index, (noisy_path, reference_path) in enumerate(selected):
            for branch_index, path in enumerate((noisy_path, reference_path)):
                image = as_tensor(read_gray(path), device)
                valid = torch.ones_like(image)
                raw = model.raw_features(image, valid)
                energy = raw["tensor_energy"].squeeze()
                threshold = torch.quantile(
                    energy.flatten().float(), args.flat_quantile
                )
                flat_indices = torch.nonzero(
                    (energy <= threshold).flatten(), as_tuple=False
                ).squeeze(1)
                if flat_indices.numel() > args.pixels_per_image:
                    generator = torch.Generator(device=flat_indices.device)
                    generator.manual_seed(
                        args.seed * 1000003 + pair_index * 2 + branch_index
                    )
                    order = torch.randperm(
                        flat_indices.numel(),
                        generator=generator,
                        device=flat_indices.device,
                    )[: args.pixels_per_image]
                    flat_indices = flat_indices[order]
                for name in names:
                    samples[name].append(
                        raw[name].flatten()[flat_indices].float().cpu().numpy()
                    )
            if pair_index == 0 or (pair_index + 1) % 50 == 0:
                print(f"calibrated {pair_index + 1}/{len(selected)}", flush=True)

    arrays = {name: np.concatenate(chunks) for name, chunks in samples.items()}
    responses = {}
    for scale in model.scales:
        name = f"roa_raw_s{scale}"
        values = arrays[name]
        transition = float(np.quantile(values, args.transition_quantile))
        threshold = float(np.quantile(values, args.threshold_quantile))
        responses[name] = {
            "threshold": threshold,
            "transition": max(threshold - transition, 1e-4),
            "sampled_pixels": int(values.size),
        }
    probabilities = []
    for scale in model.scales:
        name = f"roa_raw_s{scale}"
        entry = responses[name]
        probabilities.append(
            1.0 / (
                1.0 + np.exp(
                    -(arrays[name] - entry["threshold"])
                    / entry["transition"]
                )
            )
        )
    persistence = np.sort(np.stack(probabilities), axis=0)[-2]
    line_score = np.cbrt(np.clip(
        persistence * arrays["coherence"] * arrays["direction_consistency"],
        0.0,
        None,
    ))
    transition = float(np.quantile(line_score, args.transition_quantile))
    threshold = float(np.quantile(line_score, args.threshold_quantile))
    responses["line_score"] = {
        "threshold": threshold,
        "transition": max(threshold - transition, 1e-4),
        "sampled_pixels": int(line_score.size),
    }
    payload = {
        "version": "spd_joint_line_v1",
        "dataset": "SAR despeckling filters dataset",
        "calibration_split": "Noisy+GTruth training pairs",
        "noisy_dir": str(Path(args.noisy_dir).resolve()),
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "sample_count": len(selected),
        "pixels_per_image_per_branch": args.pixels_per_image,
        "flat_quantile": args.flat_quantile,
        "threshold_quantile": args.threshold_quantile,
        "transition_quantile": args.transition_quantile,
        "scales": list(model.scales),
        "seed": args.seed,
        "selected_stems_sha256": hashlib.sha256(
            "\n".join(pair[0].stem for pair in selected).encode("utf-8")
        ).hexdigest(),
        "responses": responses,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"calibration: {args.output.resolve()}")


def overlay(image, mask):
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(
        np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    return cv2.addWeighted(color, 0.55, heat, 0.45, 0.0)


def generation_command(args):
    calibration = load_calibration(args.calibration)
    pairs = paired_sar_paths(args.noisy_dir, args.reference_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedSARLineEvidenceV2(
        calibration=calibration,
        scales=tuple(calibration["scales"]),
    ).to(device).eval()
    folders = {
        name: args.output_dir / name
        for name in (
            "line_noisy",
            "line_reference",
            "line_joint",
            "line_joint_high",
            "previews",
        )
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for index, (noisy_path, reference_path) in enumerate(pairs):
            noisy_u8 = read_gray(noisy_path)
            reference_u8 = read_gray(reference_path)
            noisy = as_tensor(noisy_u8, device)
            reference = as_tensor(reference_u8, device)
            valid = torch.ones_like(noisy)
            noisy_result = model(noisy, valid)
            reference_result = model(reference, valid)
            noisy_line = noisy_result["r_line"]
            reference_line = reference_result["r_line"]
            joint = joint_line_mask(noisy_line, reference_line)
            joint_high = high_confidence_line_mask(
                joint, args.confidence_floor
            )
            perturbed_line = model(
                perturb(noisy, args.perturb_sigma, args.seed + index), valid
            )["r_line"]
            energy = noisy_result["tensor_energy"]
            flat_threshold = torch.quantile(
                energy.flatten().float(), args.flat_quantile
            )
            flat = (energy <= flat_threshold).to(joint.dtype)
            stats = connected_statistics(joint)
            high_stats = connected_statistics(joint_high, threshold=0.01)
            row = {
                "stem": noisy_path.stem,
                "noisy_mean": float(noisy_line.mean()),
                "reference_mean": float(reference_line.mean()),
                "joint_mean": float(joint.mean()),
                "joint_std": float(joint.std(unbiased=False)),
                "joint_gt_0p5": float((joint > 0.5).float().mean()),
                "joint_gt_0p7": float((joint > 0.7).float().mean()),
                "high_mean": float(joint_high.mean()),
                "high_std": float(joint_high.std(unbiased=False)),
                "high_nonzero_fraction": float((joint_high > 0).float().mean()),
                "high_gt_0p5": float((joint_high > 0.5).float().mean()),
                "noisy_reference_correlation": pearson(noisy_line, reference_line),
                "perturbation_correlation": pearson(noisy_line, perturbed_line),
                "perturbation_mae": float((noisy_line - perturbed_line).abs().mean()),
                "flat_joint_mean": float((joint * flat).sum() / flat.sum().clamp_min(1.0)),
                "flat_joint_gt_0p5": float(
                    ((joint > 0.5).float() * flat).sum() / flat.sum().clamp_min(1.0)
                ),
                "flat_high_mean": float(
                    (joint_high * flat).sum() / flat.sum().clamp_min(1.0)
                ),
                "flat_high_nonzero": float(
                    ((joint_high > 0).float() * flat).sum()
                    / flat.sum().clamp_min(1.0)
                ),
                "flat_high_gt_0p5": float(
                    ((joint_high > 0.5).float() * flat).sum()
                    / flat.sum().clamp_min(1.0)
                ),
                **stats,
                **{f"high_{name}": value for name, value in high_stats.items()},
            }
            rows.append(row)
            save_mask(folders["line_noisy"] / f"{noisy_path.stem}.png", noisy_line)
            save_mask(
                folders["line_reference"] / f"{noisy_path.stem}.png",
                reference_line,
            )
            save_mask(folders["line_joint"] / f"{noisy_path.stem}.png", joint)
            save_mask(
                folders["line_joint_high"] / f"{noisy_path.stem}.png",
                joint_high,
            )
            if index < args.preview_count:
                joint_np = joint.squeeze().cpu().numpy()
                joint_high_np = joint_high.squeeze().cpu().numpy()
                panel = np.concatenate(
                    (
                        cv2.cvtColor(noisy_u8, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(reference_u8, cv2.COLOR_GRAY2BGR),
                        overlay(noisy_u8, joint_np),
                        overlay(noisy_u8, joint_high_np),
                    ),
                    axis=1,
                )
                cv2.imwrite(
                    str(folders["previews"] / f"{noisy_path.stem}.png"), panel
                )
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(pairs):
                print(f"generated {index + 1}/{len(pairs)}", flush=True)

    fields = list(rows[0])
    with (args.output_dir / "per_image_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    numeric = [name for name in fields if name != "stem"]
    summary = {
        "images": len(rows),
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": calibration["sha256"],
        "perturb_sigma": args.perturb_sigma,
        "confidence_floor": args.confidence_floor,
        "statistics": {
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "std": float(np.std([row[name] for row in rows])),
                "median": float(np.median([row[name] for row in rows])),
            }
            for name in numeric
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--noisy-dir", required=True)
    calibrate.add_argument("--reference-dir", required=True)
    calibrate.add_argument("--sample-count", type=int, default=512)
    calibrate.add_argument("--pixels-per-image", type=int, default=2048)
    calibrate.add_argument("--flat-quantile", type=float, default=0.40)
    calibrate.add_argument("--threshold-quantile", type=float, default=0.995)
    calibrate.add_argument("--transition-quantile", type=float, default=0.95)
    calibrate.add_argument("--scales", type=int, nargs="+", default=(3, 5, 9))
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--output", type=Path, required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--noisy-dir", required=True)
    generate.add_argument("--reference-dir", required=True)
    generate.add_argument("--calibration", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--preview-count", type=int, default=24)
    generate.add_argument("--perturb-sigma", type=float, default=0.03)
    generate.add_argument("--flat-quantile", type=float, default=0.40)
    generate.add_argument("--confidence-floor", type=float, default=0.50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "calibrate":
        calibration_command(args)
    else:
        generation_command(args)


if __name__ == "__main__":
    main()
