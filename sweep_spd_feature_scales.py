"""Validation-only sweep of independent SPD Base/Detail correction scales."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_spd_fusion import as_u8, fuse, load_models
from utils.Evaluator import PaperEvaluator
from utils.dataset_os import manifest_pairs


DEFAULT_SCALES = (
    (0.0, 0.0),
    (0.25, 0.25),
    (0.50, 0.50),
    (0.75, 0.75),
    (1.00, 1.00),
    (1.00, 0.00),
    (0.00, 1.00),
    (0.50, 1.00),
    (1.00, 0.50),
)
LOWER_IS_BETTER = {"Nabf"}
PRIMARY = ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep SPD feature scales")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scale_id(base_scale, detail_scale):
    def encode(value):
        return f"{value:.2f}".replace(".", "p")

    return f"b{encode(base_scale)}_d{encode(detail_scale)}"


@torch.no_grad()
def scaled_outputs(models, optical, sar):
    optical_base, optical_detail, _ = models["encoder"](optical)
    raw_base, raw_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, diagnostics = models["spd"](
        raw_base, raw_detail
    )
    delta_base = clean_base - raw_base
    delta_detail = clean_detail - raw_detail
    outputs = {}
    for base_scale, detail_scale in DEFAULT_SCALES:
        scaled_base = raw_base + float(base_scale) * delta_base
        scaled_detail = raw_detail + float(detail_scale) * delta_detail
        fused_base = models["base_fuse"](optical_base + scaled_base)
        fused_detail = models["detail_fuse"](optical_detail + scaled_detail)
        fused, _ = models["decoder"](optical, fused_base, fused_detail)
        outputs[scale_id(base_scale, detail_scale)] = fused.clamp(0.0, 1.0)
    return outputs, {
        "base_correction_rms": float(diagnostics["base_correction_rms"]),
        "detail_correction_rms": float(diagnostics["detail_correction_rms"]),
    }


def summarize(rows, metric_names):
    baseline_id = scale_id(0.0, 0.0)
    baseline = {
        name: np.asarray(
            [row[f"{baseline_id}_{name}"] for row in rows], dtype=np.float64
        )
        for name in metric_names
    }
    summaries = []
    for base_scale, detail_scale in DEFAULT_SCALES:
        identifier = scale_id(base_scale, detail_scale)
        record = {
            "scale_id": identifier,
            "base_scale": base_scale,
            "detail_scale": detail_scale,
        }
        for name in metric_names:
            values = np.asarray(
                [row[f"{identifier}_{name}"] for row in rows],
                dtype=np.float64,
            )
            delta = values - baseline[name]
            improved = delta < 0 if name in LOWER_IS_BETTER else delta > 0
            record[f"{name}_mean"] = float(values.mean())
            record[f"{name}_delta"] = float(delta.mean())
            record[f"{name}_median_delta"] = float(np.median(delta))
            record[f"{name}_improved_fraction"] = float(improved.mean())
            baseline_mean = float(baseline[name].mean())
            if name in LOWER_IS_BETTER:
                record[f"{name}_relative_change"] = (
                    float(values.mean()) / max(baseline_mean, 1e-12) - 1.0
                )
            else:
                record[f"{name}_retention"] = (
                    float(values.mean()) / max(baseline_mean, 1e-12)
                )
        summaries.append(record)
    return summaries


def pareto_flags(summaries):
    """Mark candidates not dominated across six primary metrics."""
    for candidate in summaries:
        candidate["pareto"] = True
        for other in summaries:
            if other is candidate:
                continue
            no_worse = True
            strictly_better = False
            for name in PRIMARY:
                left = other[f"{name}_mean"]
                right = candidate[f"{name}_mean"]
                if name in LOWER_IS_BETTER:
                    no_worse &= left <= right
                    strictly_better |= left < right
                else:
                    no_worse &= left >= right
                    strictly_better |= left > right
            if no_worse and strictly_better:
                candidate["pareto"] = False
                break


def main():
    args = parse_args()
    if args.split == "test":
        raise ValueError("Scale selection must not use the test split")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][args.offset : args.offset + args.limit]
    if not selected:
        raise RuntimeError("No Optical-SAR pairs selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, b0_hash, spd_hash, equivalence, source = load_models(
        args.b0_checkpoint, args.spd_checkpoint, device
    )
    metric_names = PaperEvaluator.METRIC_NAMES
    rows = []
    for index, (optical_path, sar_path) in enumerate(selected, start=1):
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        if optical_bgr is None or sar_u8 is None:
            raise RuntimeError(f"Could not read pair {optical_path.stem}")
        optical_y_u8 = cv2.cvtColor(
            optical_bgr, cv2.COLOR_BGR2YCrCb
        )[:, :, 0]
        optical = torch.from_numpy(
            optical_y_u8.astype(np.float32) / 255.0
        )[None, None].to(device)
        sar = torch.from_numpy(
            sar_u8.astype(np.float32) / 255.0
        )[None, None].to(device)
        outputs, diagnostics = scaled_outputs(models, optical, sar)
        row = {"image": optical_path.stem, **diagnostics}
        for identifier, output in outputs.items():
            metrics = PaperEvaluator.metric_vector(
                as_u8(output).astype(np.float32),
                optical_y_u8.astype(np.float32),
                sar_u8.astype(np.float32),
            )
            for name, value in zip(metric_names, metrics):
                row[f"{identifier}_{name}"] = float(value)
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"processed {index}/{len(selected)}: {optical_path.stem}")

    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = summarize(rows, metric_names)
    pareto_flags(summaries)
    fields = list(summaries[0])
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "protocol": {
            "dataset": "QXS-SAROPT",
            "split": args.split,
            "offset": args.offset,
            "sample_count": len(selected),
            "metric_profile": "paper",
            "output_policy": "clamp_then_uint8",
            "selection_warning": "Validation-only scale selection; do not tune on test.",
            "split_manifest": str(args.split_manifest.resolve()),
            "split_counts": {
                name: len(values) for name, values in splits.items()
            },
            "b0_checkpoint": str(args.b0_checkpoint.resolve()),
            "b0_checkpoint_sha256": b0_hash,
            "spd_checkpoint": str(args.spd_checkpoint.resolve()),
            "spd_checkpoint_sha256": spd_hash,
            "encoder_equivalence": equivalence,
            "spd_training_source_checkpoint": source,
            "scales": [
                {"base": base, "detail": detail}
                for base, detail in DEFAULT_SCALES
            ],
        },
        "summary": summaries,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
