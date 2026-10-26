"""Evaluate matched PDCF P2 single-path pilots against exact C0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pdcf_model import PDCFModel
from pdcf_predictability import BidirectionalPredictability
from pdcf_single_path import PDCFSinglePathModel
from test_OS import PAPER_METRIC_NAMES, evaluate
from train_pdcf_predictability import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PDCF P2 matched pilots")
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
    parser.add_argument("--save-previews", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


def load_predictor(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    predictor = BidirectionalPredictability(
        channels=64,
        hidden_channels=int(saved_args["hidden_channels"]),
        blocks=int(saved_args["predictor_blocks"]),
    ).to(device)
    predictor.load_state_dict(checkpoint["PDCFPredictability"])
    return predictor.eval()


def load_p2(path, c0, predictor, expected_mode, hashes, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("input_mode") != expected_mode:
        raise ValueError(f"Expected {expected_mode}, got {checkpoint.get('input_mode')}")
    for key, expected in hashes.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"P2 checkpoint {path} has mismatched {key}")
    saved_args = checkpoint["args"]
    model = PDCFSinglePathModel(
        c0,
        predictor,
        input_mode=expected_mode,
        hidden_channels=int(saved_args["hidden_channels"]),
        blocks=int(saved_args["blocks"]),
        max_relative_correction=float(saved_args["max_relative_correction"]),
    ).to(device)
    model.adapter.load_state_dict(checkpoint["PDCFSinglePathAdapter"])
    return model.eval(), checkpoint


def summarize(rows, methods):
    output = {"images": len(rows), "methods": {}, "comparison_to_c0": {}}
    baseline = np.asarray(
        [[row[f"c0_{name}"] for name in PAPER_METRIC_NAMES] for row in rows]
    )
    for method in methods:
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in PAPER_METRIC_NAMES] for row in rows]
        )
        output["methods"][method] = {
            name: {
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std()),
            }
            for index, name in enumerate(PAPER_METRIC_NAMES)
        }
        output["methods"][method]["output_deviation_rms"] = {
            "mean": float(np.mean([row[f"{method}_output_deviation_rms"] for row in rows])),
            "std": float(np.std([row[f"{method}_output_deviation_rms"] for row in rows])),
        }
        output["methods"][method]["relative_base_correction"] = {
            "mean": float(np.mean([row[f"{method}_relative_base_correction"] for row in rows])),
            "std": float(np.std([row[f"{method}_relative_base_correction"] for row in rows])),
        }
        output["methods"][method]["relative_detail_correction"] = {
            "mean": float(np.mean([row[f"{method}_relative_detail_correction"] for row in rows])),
            "std": float(np.std([row[f"{method}_relative_detail_correction"] for row in rows])),
        }
        if method != "c0":
            delta = matrix - baseline
            output["comparison_to_c0"][method] = {
                name: {
                    "mean_delta": float(delta[:, index].mean()),
                    "median_delta": float(np.median(delta[:, index])),
                    "improved_fraction": float(
                        (delta[:, index] < 0).mean()
                        if name == "Nabf"
                        else (delta[:, index] > 0).mean()
                    ),
                }
                for index, name in enumerate(PAPER_METRIC_NAMES)
            }
    return output


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c0_hash = sha256_file(args.c0_checkpoint)
    p1_hash = sha256_file(args.p1_checkpoint)
    split_hash = sha256_file(args.split_manifest)
    hashes = {
        "source_checkpoint_sha256": c0_hash,
        "predictor_checkpoint_sha256": p1_hash,
        "split_manifest_sha256": split_hash,
    }
    c0 = PDCFModel.from_c0_checkpoint(
        args.c0_checkpoint, map_location=device
    ).to(device).eval()
    predictor = load_predictor(args.p1_checkpoint, device)
    predictable, _ = load_p2(
        args.predictable_checkpoint, c0, predictor,
        "predictable_residual", hashes, device,
    )
    raw, _ = load_p2(
        args.raw_checkpoint, c0, predictor,
        "raw_evidence", hashes, device,
    )
    models = {"predictable_residual": predictable, "raw_evidence": raw}
    counts = {
        name: sum(p.numel() for p in model.adapter.parameters())
        for name, model in models.items()
    }
    if len(set(counts.values())) != 1:
        raise AssertionError(f"P2 controls are not capacity matched: {counts}")

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
    methods = ("c0", "predictable_residual", "raw_evidence")
    with torch.inference_mode():
        for index in range(len(dataset)):
            optical, sar, _, stem = dataset[index]
            optical = optical[None].to(device)
            sar = sar[None].to(device)
            c0_output, _ = c0.forward_c0(optical, sar)
            outputs = {"c0": c0_output}
            diagnostics = {
                "c0": {
                    "output_deviation_rms": torch.zeros((), device=device),
                    "relative_base_correction": torch.zeros((), device=device),
                    "relative_detail_correction": torch.zeros((), device=device),
                }
            }
            for name, model in models.items():
                output, _, diagnostic = model(
                    optical, sar, return_diagnostics=True
                )
                outputs[name] = output
                diagnostics[name] = diagnostic
            optical_u8, sar_u8 = as_u8(optical), as_u8(sar)
            row = {"image": stem}
            preview = [optical_u8, sar_u8]
            for name in methods:
                image = as_u8(outputs[name].clamp(0.0, 1.0))
                values = evaluate(image, optical_u8, sar_u8, metric_profile="paper")
                row.update(
                    {f"{name}_{metric}": float(value)
                     for metric, value in zip(PAPER_METRIC_NAMES, values)}
                )
                for diagnostic_name in (
                    "output_deviation_rms", "relative_base_correction",
                    "relative_detail_correction",
                ):
                    row[f"{name}_{diagnostic_name}"] = float(
                        diagnostics[name][diagnostic_name]
                    )
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
    summary = summarize(rows, methods)
    summary["protocol"] = {
        "name": "PDCF_P2_matched_single_path_val_v1",
        "dataset": manifest.get("dataset", "QXS-SAROPT"),
        "split": args.split,
        "offset": args.offset,
        "limit": len(rows),
        "patch_size": args.patch_size,
        "metric_profile": "paper",
        "output_policy": "clamp_then_uint8",
        "trainable_parameter_counts": counts,
        "c0_checkpoint": str(args.c0_checkpoint.resolve()),
        "c0_checkpoint_sha256": c0_hash,
        "p1_checkpoint": str(args.p1_checkpoint.resolve()),
        "p1_checkpoint_sha256": p1_hash,
        "predictable_checkpoint": str(args.predictable_checkpoint.resolve()),
        "predictable_checkpoint_sha256": sha256_file(args.predictable_checkpoint),
        "raw_checkpoint": str(args.raw_checkpoint.resolve()),
        "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split_hash,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

