"""Render fixed SAR structure evidence maps before C3 training is allowed."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from structure_evidence import FixedSARStructureEvidence
from utils.dataset_os import manifest_pairs


DISPLAY_MAPS = (
    "roa_peak",
    "coherence",
    "line_persistence",
    "point_persistence",
    "point_support",
    "r_line",
    "r_point",
    "r_fixed",
    "r_union",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--scales", type=int, nargs="+", default=(3, 5, 9))
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Global training calibration JSON. Omit only for pre-calibration diagnostics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def to_u8(values):
    return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)


def labeled(image, label):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def main():
    args = parse_args()
    splits, _ = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    if args.ids:
        wanted = set(args.ids)
        pairs = [pair for pair in pairs if pair[0].stem in wanted]
        missing = sorted(wanted - {pair[0].stem for pair in pairs})
        if missing:
            raise ValueError("IDs not found in split: " + ", ".join(missing))
    else:
        pairs = pairs[args.offset : args.offset + args.limit]
    if not pairs:
        raise RuntimeError("No image pairs selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = None
    if args.thresholds:
        with args.thresholds.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        thresholds = payload.get("thresholds", payload)
    evidence_model = FixedSARStructureEvidence(
        scales=args.scales,
        percentile=args.percentile,
        thresholds=thresholds,
    ).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in DISPLAY_MAPS:
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overview").mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for index, (optical_path, sar_path) in enumerate(pairs, start=1):
            optical = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            if optical is None or sar_u8 is None:
                raise RuntimeError(f"Could not read pair {optical_path.stem}")
            if optical.shape[:2] != sar_u8.shape:
                raise RuntimeError(f"Shape mismatch for {optical_path.stem}")
            sar = torch.from_numpy(sar_u8)[None, None].float().to(device) / 255.0
            valid = (sar > 0.0).float()
            maps = evidence_model(sar, valid)
            valid_np = valid.squeeze().cpu().numpy() > 0.5
            panels = [
                labeled(optical, "Optical"),
                labeled(cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR), "SAR"),
            ]
            row = {"image": optical_path.stem}
            for name in DISPLAY_MAPS:
                values = maps[name].squeeze().cpu().numpy()
                gray = to_u8(values)
                color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                cv2.imwrite(
                    str(args.output_dir / name / f"{optical_path.stem}.png"),
                    color,
                )
                panels.append(labeled(color, name))
                selected = values[valid_np]
                row[f"{name}_mean"] = float(selected.mean())
                row[f"{name}_std"] = float(selected.std())
                row[f"{name}_high_fraction"] = float((selected > 0.7).mean())
                row[f"{name}_low_fraction"] = float((selected < 0.2).mean())
            rows.append(row)
            cv2.imwrite(
                str(args.output_dir / "overview" / f"{optical_path.stem}.png"),
                np.concatenate(panels, axis=1),
            )
            print(f"processed {index}/{len(pairs)}: {optical_path.stem}", flush=True)

    fieldnames = tuple(rows[0])
    with (args.output_dir / "evidence_stats.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"device: {device}")
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
