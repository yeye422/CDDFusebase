"""P1 validation and residual-intervention audit for trained predictors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pdcf_model import PDCFModel
from pdcf_predictability import (
    BidirectionalLinearBaseline,
    BidirectionalPredictability,
    balanced_prediction_loss,
    prediction_metrics,
    residual_intervention,
)
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_pdcf_predictability import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


INTERVENTIONS = (
    "c0",
    "original",
    "zero_both",
    "zero_optical",
    "zero_sar",
    "spatial_shuffle",
    "batch_swap",
    "rms_noise",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PDCF P1 prediction and residual interventions"
    )
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-previews", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


def predictor_from_checkpoint(checkpoint, device):
    saved_args = checkpoint["args"]
    predictor = BidirectionalPredictability(
        channels=64,
        hidden_channels=int(saved_args["hidden_channels"]),
        blocks=int(saved_args["predictor_blocks"]),
    ).to(device)
    predictor.load_state_dict(checkpoint["PDCFPredictability"])
    linear = BidirectionalLinearBaseline(channels=128).to(device)
    linear.load_state_dict(checkpoint["PDCFLinearBaseline"])
    predictor.eval()
    linear.eval()
    return predictor, linear


def reconstruct_features(predictor, outputs, optical_residual, sar_residual):
    optical = outputs["predictable_optical"] + optical_residual
    sar = outputs["predictable_sar"] + sar_residual
    channels = predictor.channels
    optical_parts = predictor.normalizer.inverse(
        optical[:, :channels], optical[:, channels:]
    )
    sar_parts = predictor.normalizer.inverse(
        sar[:, :channels], sar[:, channels:]
    )
    return optical_parts[0], optical_parts[1], sar_parts[0], sar_parts[1]


@torch.no_grad()
def intervention_outputs(c0, predictor, optical, features, generator):
    outputs = predictor(*features)
    residual_o = outputs["residual_optical"]
    residual_s = outputs["residual_sar"]
    results = {}
    for name in INTERVENTIONS[1:]:
        if name == "original":
            changed_o, changed_s = residual_o, residual_s
        elif name == "zero_both":
            changed_o = residual_intervention(residual_o, "zero")
            changed_s = residual_intervention(residual_s, "zero")
        elif name == "zero_optical":
            changed_o = residual_intervention(residual_o, "zero")
            changed_s = residual_s
        elif name == "zero_sar":
            changed_o = residual_o
            changed_s = residual_intervention(residual_s, "zero")
        else:
            changed_o = residual_intervention(residual_o, name, generator)
            changed_s = residual_intervention(residual_s, name, generator)
        reconstructed = reconstruct_features(
            predictor, outputs, changed_o, changed_s
        )
        fused_base, fused_detail = c0.c0_features(*reconstructed)
        results[name], _ = c0.decoder(optical, fused_base, fused_detail)
    return outputs, results


def summarize(rows, prediction_summary):
    summary = {
        "images": len(rows),
        "prediction": prediction_summary,
        "interventions": {},
    }
    for method in INTERVENTIONS:
        matrix = np.asarray(
            [
                [row[f"{method}_{metric}"] for metric in PAPER_METRIC_NAMES]
                for row in rows
            ],
            dtype=np.float64,
        )
        summary["interventions"][method] = {
            metric: {
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std()),
            }
            for index, metric in enumerate(PAPER_METRIC_NAMES)
        }
        summary["interventions"][method]["output_deviation_rms"] = {
            "mean": float(np.mean([row[f"{method}_output_deviation_rms"] for row in rows])),
            "std": float(np.std([row[f"{method}_output_deviation_rms"] for row in rows])),
        }
    baseline = np.asarray(
        [[row[f"c0_{metric}"] for metric in PAPER_METRIC_NAMES] for row in rows]
    )
    summary["comparison_to_c0"] = {}
    for method in INTERVENTIONS[1:]:
        matrix = np.asarray(
            [[row[f"{method}_{metric}"] for metric in PAPER_METRIC_NAMES] for row in rows]
        )
        delta = matrix - baseline
        summary["comparison_to_c0"][method] = {
            metric: {
                "mean_delta": float(delta[:, index].mean()),
                "median_delta": float(np.median(delta[:, index])),
                "improved_fraction": float(
                    (delta[:, index] < 0).mean()
                    if metric == "Nabf"
                    else (delta[:, index] > 0).mean()
                ),
            }
            for index, metric in enumerate(PAPER_METRIC_NAMES)
        }
    return summary


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 for batch_swap")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    selected = splits[args.split][args.offset: args.offset + args.limit]
    if len(selected) < 2 or len(selected) % args.batch_size == 1:
        raise ValueError(
            "Selected sample count must avoid a final one-sample batch for batch_swap"
        )
    dataset = OpticalSARDataset(
        selected,
        patch_size=args.patch_size,
        training=False,
        augment=False,
        invalid_threshold=0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.p1_checkpoint, map_location="cpu", weights_only=False)
    actual_c0_hash = sha256_file(args.c0_checkpoint)
    if checkpoint.get("source_checkpoint_sha256") != actual_c0_hash:
        raise ValueError("P1 predictor was trained from a different C0 checkpoint")
    if checkpoint.get("split_manifest_sha256") != sha256_file(args.split_manifest):
        raise ValueError("P1 predictor used a different split manifest")
    c0 = PDCFModel.from_c0_checkpoint(
        args.c0_checkpoint, map_location=device
    ).to(device).eval()
    predictor, linear = predictor_from_checkpoint(checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    prediction_totals = {"predictor": {}, "linear": {}}
    prediction_samples = 0
    args.output_dir.mkdir(parents=True)
    preview_dir = args.output_dir / "previews"
    if args.save_previews:
        preview_dir.mkdir()

    for optical, sar, mask, stems in loader:
        optical = optical.to(device)
        sar = sar.to(device)
        mask = mask.to(device)
        features = c0.extract_features(optical, sar)
        c0_base, c0_detail = c0.c0_features(*features)
        c0_output, _ = c0.decoder(optical, c0_base, c0_detail)
        outputs, variants = intervention_outputs(
            c0, predictor, optical, features, generator
        )
        variants = {"c0": c0_output, **variants}

        predictor_losses = balanced_prediction_loss(outputs, mask, channels=64)
        predictor_values = {
            "loss": predictor_losses["total"],
            **prediction_metrics(outputs, mask, channels=64),
        }
        linear_predictions = linear(
            outputs["optical"], outputs["sar"]
        )
        linear_outputs = {
            "optical": outputs["optical"],
            "sar": outputs["sar"],
            **linear_predictions,
        }
        linear_losses = balanced_prediction_loss(linear_outputs, mask, channels=64)
        linear_values = {
            "loss": linear_losses["total"],
            **prediction_metrics(linear_outputs, mask, channels=64),
        }
        for model_name, values in (
            ("predictor", predictor_values), ("linear", linear_values)
        ):
            for name, value in values.items():
                prediction_totals[model_name][name] = (
                    prediction_totals[model_name].get(name, 0.0)
                    + float(value) * optical.shape[0]
                )
        prediction_samples += optical.shape[0]

        for index, stem in enumerate(stems):
            optical_u8 = as_u8(optical[index])
            sar_u8 = as_u8(sar[index])
            c0_tensor = variants["c0"][index]
            row = {"image": stem}
            preview_images = [optical_u8, sar_u8]
            for method in INTERVENTIONS:
                tensor = variants[method][index].clamp(0.0, 1.0)
                image = as_u8(tensor)
                values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
                row.update(
                    {
                        f"{method}_{name}": float(value)
                        for name, value in zip(PAPER_METRIC_NAMES, values)
                    }
                )
                row[f"{method}_output_deviation_rms"] = float(
                    (tensor - c0_tensor).square().mean().sqrt()
                )
                preview_images.append(image)
            rows.append(row)
            if len(rows) <= args.save_previews:
                cv2.imwrite(
                    str(preview_dir / f"{stem}.png"),
                    np.concatenate(preview_images, axis=1),
                )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)

    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prediction_summary = {
        model_name: {
            name: value / prediction_samples for name, value in values.items()
        }
        for model_name, values in prediction_totals.items()
    }
    summary = summarize(rows, prediction_summary)
    summary["protocol"] = {
        "name": "PDCF_P1_predictability_and_residual_intervention_v1",
        "dataset": manifest.get("dataset", "QXS-SAROPT"),
        "split": args.split,
        "offset": args.offset,
        "limit": len(rows),
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "c0_checkpoint": str(args.c0_checkpoint.resolve()),
        "c0_checkpoint_sha256": actual_c0_hash,
        "p1_checkpoint": str(args.p1_checkpoint.resolve()),
        "p1_checkpoint_sha256": sha256_file(args.p1_checkpoint),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
