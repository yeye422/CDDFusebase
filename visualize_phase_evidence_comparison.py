"""Compare globally calibrated RoA, phase antisymmetry, and their fusion."""

import argparse
import csv
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
    parser.add_argument("--phase-calibration", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def probability(values, entry):
    return torch.sigmoid(
        (values - float(entry["threshold"]))
        / max(float(entry["transition"]), 1e-6)
    )


def to_color(values):
    gray = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def labeled(image, label):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(
        result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def add_stats(row, name, values, valid):
    selected = values[valid]
    row[f"{name}_mean"] = float(selected.mean())
    row[f"{name}_std"] = float(selected.std())
    row[f"{name}_detect_fraction_05"] = float((selected > 0.5).mean())
    row[f"{name}_strong_fraction_07"] = float((selected > 0.7).mean())


def main():
    args = parse_args()
    fse_calibration = load_calibration(args.fse_calibration)
    with args.phase_calibration.open(encoding="utf-8") as handle:
        phase_calibration = json.load(handle)
    if phase_calibration["fse_calibration_sha256"] != fse_calibration["sha256"]:
        raise ValueError("Phase calibration was built from another FSE calibration")
    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split][args.offset : args.offset + args.limit]
    if not pairs:
        raise RuntimeError("No image pairs selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fse = FixedSARLineEvidenceV2(
        calibration=fse_calibration,
        scales=tuple(fse_calibration["scales"]),
    ).to(device).eval()
    phase = MonogenicPhaseAntisymmetry(
        wavelengths=phase_calibration["wavelengths"],
        sigma_on_f=phase_calibration["sigma_on_f"],
    ).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overview").mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        for index, (optical_path, sar_path) in enumerate(pairs, start=1):
            optical = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            if optical is None or sar_u8 is None:
                raise RuntimeError(f"Could not read pair {optical_path.stem}")
            sar = torch.from_numpy(sar_u8)[None, None].float().to(device) / 255.0
            valid = (sar > 0.0).float()
            fse_result = fse(sar, valid)
            phase_result = phase(sar, valid)
            fa_probability = probability(
                phase_result["fa_mean"],
                phase_calibration["responses"]["fa_mean"],
            ) * valid
            combined_score = torch.sqrt(
                (fa_probability * fse_result["scale_persistence"]).clamp_min(0.0)
            ) * valid
            combined_probability = probability(
                combined_score,
                phase_calibration["responses"]["roa_fa_score"],
            ) * valid
            bridge_score, bridge_support = local_phase_bridge(
                fse_result["r_line"], fa_probability, kernel_size=5
            )
            bridge_probability = probability(
                bridge_score,
                phase_calibration["responses"]["bridge_score"],
            ) * valid

            maps = {
                "roa_line": fse_result["r_line"],
                "fa_raw": phase_result["fa_mean"],
                "fa_prob": fa_probability,
                "roa_fa_score": combined_score,
                "roa_fa_prob": combined_probability,
                "bridge_support": bridge_support,
                "bridge_score": bridge_score,
                "bridge_prob": bridge_probability,
            }
            panels = [
                labeled(optical, "Optical"),
                labeled(cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR), "SAR"),
            ]
            valid_np = valid.squeeze().cpu().numpy() > 0.5
            row = {"image": optical_path.stem}
            for name, tensor in maps.items():
                values = tensor.squeeze().cpu().numpy()
                panels.append(labeled(to_color(values), name))
                add_stats(row, name, values, valid_np)
            for name, tensor in (
                ("roa_mask", fse_result["r_line"] > 0.5),
                ("fa_mask", fa_probability > 0.5),
                ("combined_mask", combined_probability > 0.5),
                ("bridge_mask", bridge_probability > 0.5),
            ):
                mask = tensor.squeeze().cpu().numpy().astype(np.float32)
                panels.append(labeled(to_color(mask), name))
            rows.append(row)
            cv2.imwrite(
                str(args.output_dir / "overview" / f"{optical_path.stem}.png"),
                np.concatenate(panels, axis=1),
            )
            print(f"processed {index}/{len(pairs)}: {optical_path.stem}", flush=True)

    with (args.output_dir / "comparison_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"device: {device}")
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
