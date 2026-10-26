"""Compare C0, image-domain SPD preprocessing, and feature-domain SPD fusion."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from spd_modules import SPDModel, build_spd_from_checkpoint_args
from train_spd import canonical_state, encoder_state
from utils.Evaluator import PaperEvaluator
from utils.dataset_os import manifest_pairs


METHODS = ("c0_raw", "e1_image", "e2_feature")
LOWER_IS_BETTER = {"Nabf"}
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="SPD E1/E2 fusion comparison")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="val"
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-checkpoint", type=Path, required=True)
    parser.add_argument("--spd-base-scale", type=float, default=1.0)
    parser.add_argument("--spd-detail-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument(
        "--save-methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Fusion paths whose gray/RGB/preview images are saved.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(checkpoint, *keys):
    for key in keys:
        if key in checkpoint:
            return canonical_state(checkpoint[key])
    raise KeyError(f"Checkpoint is missing all keys: {keys}")


def state_dicts_exactly_equal(first, second):
    first = canonical_state(first)
    second = canonical_state(second)
    if first.keys() != second.keys():
        return False
    return all(torch.equal(first[key], second[key]) for key in first)


def load_models(c0_path, spd_path, device):
    c0 = torch.load(c0_path, map_location="cpu", weights_only=False)
    encoder = Restormer_Encoder()
    decoder = Restormer_Decoder()
    base_fuse = BaseFeatureExtraction(dim=64, num_heads=8)
    detail_fuse = DetailFeatureExtraction(num_layers=1)
    encoder.load_state_dict(encoder_state(c0))
    decoder.load_state_dict(
        checkpoint_state(c0, "CDDF_Decoder", "DIDF_Decoder")
    )
    base_fuse.load_state_dict(checkpoint_state(c0, "BaseFuseLayer"))
    detail_fuse.load_state_dict(checkpoint_state(c0, "DetailFuseLayer"))

    spd_checkpoint = torch.load(
        spd_path, map_location="cpu", weights_only=False
    )
    expected_hash = spd_checkpoint.get("source_encoder_checkpoint_sha256")
    actual_hash = sha256_file(c0_path)
    encoder_equivalence = "same_checkpoint"
    spd_source_path = c0_path
    if expected_hash and expected_hash != actual_hash:
        source_value = spd_checkpoint.get("args", {}).get("init_checkpoint")
        if not source_value:
            raise ValueError(
                "SPD checkpoint source differs from C0 and has no source path"
            )
        spd_source_path = Path(source_value)
        if not spd_source_path.is_absolute():
            spd_source_path = (ROOT / spd_source_path).resolve()
        if not spd_source_path.exists():
            raise FileNotFoundError(
                f"SPD source checkpoint is missing: {spd_source_path}"
            )
        source_hash = sha256_file(spd_source_path)
        if source_hash != expected_hash:
            raise ValueError(
                "SPD source checkpoint SHA does not match its training metadata"
            )
        source_checkpoint = torch.load(
            spd_source_path, map_location="cpu", weights_only=False
        )
        if not state_dicts_exactly_equal(
            encoder_state(source_checkpoint), encoder_state(c0)
        ):
            raise ValueError(
                "Requested C0 has a different Encoder from the SPD training source"
            )
        encoder_equivalence = "exact_state_dict_match"
    spd_module = build_spd_from_checkpoint_args(
        spd_checkpoint.get("args", {})
    )
    spd_model = SPDModel(encoder, spd=spd_module)
    spd_model.spd.load_state_dict(canonical_state(spd_checkpoint["SPD"]))
    spd_model.denoising_head.load_state_dict(
        canonical_state(spd_checkpoint["SPDDenoisingHead"])
    )

    models = {
        "encoder": encoder,
        "decoder": decoder,
        "base_fuse": base_fuse,
        "detail_fuse": detail_fuse,
        "spd": spd_model.spd,
        "denoising_head": spd_model.denoising_head,
    }
    for model in models.values():
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return (
        models,
        actual_hash,
        sha256_file(spd_path),
        encoder_equivalence,
        str(Path(spd_source_path).resolve()),
    )


def fuse(models, optical, sar_base, sar_detail):
    optical_base, optical_detail, _ = models["encoder"](optical)
    fused_base = models["base_fuse"](optical_base + sar_base)
    fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    fused, _ = models["decoder"](optical, fused_base, fused_detail)
    return fused.clamp(0.0, 1.0)


@torch.no_grad()
def forward_three_paths(
    models, optical, sar, spd_base_scale=1.0, spd_detail_scale=1.0
):
    raw_base, raw_detail, _ = models["encoder"](sar)
    clean_base, clean_detail, diagnostics = models["spd"](
        raw_base, raw_detail
    )
    clean_base = raw_base + float(spd_base_scale) * (clean_base - raw_base)
    clean_detail = raw_detail + float(spd_detail_scale) * (
        clean_detail - raw_detail
    )
    denoised = models["denoising_head"](
        sar, clean_base, clean_detail
    )
    denoised_base, denoised_detail, _ = models["encoder"](denoised)
    outputs = {
        "c0_raw": fuse(models, optical, raw_base, raw_detail),
        "e1_image": fuse(
            models, optical, denoised_base, denoised_detail
        ),
        "e2_feature": fuse(models, optical, clean_base, clean_detail),
    }
    output_diagnostics = {
        "sar_denoised_mae": float((denoised - sar).abs().mean()),
        "base_correction_rms": float(
            diagnostics["base_correction_rms"]
        ),
        "detail_correction_rms": float(
            diagnostics["detail_correction_rms"]
        ),
        "e1_c0_mae": float(
            (outputs["e1_image"] - outputs["c0_raw"]).abs().mean()
        ),
        "e2_c0_mae": float(
            (outputs["e2_feature"] - outputs["c0_raw"]).abs().mean()
        ),
        "e2_e1_mae": float(
            (outputs["e2_feature"] - outputs["e1_image"]).abs().mean()
        ),
        "e1_optical_mae": float(
            (outputs["e1_image"] - optical).abs().mean()
        ),
        "e2_optical_mae": float(
            (outputs["e2_feature"] - optical).abs().mean()
        ),
    }
    return outputs, denoised, output_diagnostics


def as_u8(tensor):
    return np.clip(
        np.rint(tensor.squeeze().detach().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)


def summarize(rows, metric_names):
    summary = {"images": len(rows), "methods": {}, "comparisons": {}}
    matrices = {}
    for method in METHODS:
        matrix = np.asarray(
            [[row[f"{method}_{name}"] for name in metric_names] for row in rows],
            dtype=np.float64,
        )
        matrices[method] = matrix
        summary["methods"][method] = {
            name: {
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std()),
            }
            for index, name in enumerate(metric_names)
        }
    for candidate, reference in (
        ("e1_image", "c0_raw"),
        ("e2_feature", "c0_raw"),
        ("e2_feature", "e1_image"),
    ):
        key = f"{candidate}_minus_{reference}"
        delta = matrices[candidate] - matrices[reference]
        summary["comparisons"][key] = {}
        for index, name in enumerate(metric_names):
            improved = (
                delta[:, index] < 0
                if name in LOWER_IS_BETTER
                else delta[:, index] > 0
            )
            summary["comparisons"][key][name] = {
                "mean_delta": float(delta[:, index].mean()),
                "median_delta": float(np.median(delta[:, index])),
                "improved_fraction": float(improved.mean()),
            }
    diagnostic_names = (
        "sar_denoised_mae",
        "base_correction_rms",
        "detail_correction_rms",
        "e1_c0_mae",
        "e2_c0_mae",
        "e2_e1_mae",
        "e1_optical_mae",
        "e2_optical_mae",
    )
    summary["diagnostics"] = {
        name: float(np.mean([row[name] for row in rows]))
        for name in diagnostic_names
    }
    return summary


def main():
    args = parse_args()
    if args.spd_base_scale < 0.0 or args.spd_detail_scale < 0.0:
        raise ValueError("SPD feature scales must be non-negative")
    splits, manifest = manifest_pairs(args.data_root, args.split_manifest)
    pairs = splits[args.split]
    limit = args.limit if args.limit is not None else len(pairs)
    selected = pairs[args.offset : args.offset + limit]
    if not selected:
        raise RuntimeError("No Optical-SAR pairs selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.output_dir / "images"
    if args.save_images:
        for method in args.save_methods:
            (image_root / method).mkdir(parents=True, exist_ok=True)
            (args.output_dir / "rgb" / method).mkdir(parents=True, exist_ok=True)
            (args.output_dir / "preview" / method).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        models,
        c0_hash,
        spd_hash,
        encoder_equivalence,
        spd_source_checkpoint,
    ) = load_models(
        args.c0_checkpoint, args.spd_checkpoint, device
    )
    metric_names = PaperEvaluator.METRIC_NAMES
    rows = []
    for index, (optical_path, sar_path) in enumerate(selected):
        optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        if optical_bgr is None or sar_u8 is None:
            raise RuntimeError(f"Could not read {optical_path.stem}")
        optical_ycrcb = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
        optical_y_u8 = optical_ycrcb[:, :, 0]
        optical = torch.from_numpy(
            optical_y_u8.astype(np.float32) / 255.0
        )[None, None].to(device)
        sar = torch.from_numpy(sar_u8.astype(np.float32) / 255.0)[
            None, None
        ].to(device)
        outputs, denoised, diagnostics = forward_three_paths(
            models,
            optical,
            sar,
            spd_base_scale=args.spd_base_scale,
            spd_detail_scale=args.spd_detail_scale,
        )
        row = {"image": optical_path.stem, **diagnostics}
        for method, output in outputs.items():
            output_u8 = as_u8(output)
            metrics = PaperEvaluator.metric_vector(
                output_u8.astype(np.float32),
                optical_y_u8.astype(np.float32),
                sar_u8.astype(np.float32),
            )
            for name, value in zip(metric_names, metrics):
                row[f"{method}_{name}"] = float(value)
            if index < args.save_images and method in args.save_methods:
                cv2.imwrite(
                    str(image_root / method / f"{optical_path.stem}.png"),
                    output_u8,
                )
                fused_ycrcb = optical_ycrcb.copy()
                fused_ycrcb[:, :, 0] = output_u8
                fused_bgr = cv2.cvtColor(
                    fused_ycrcb, cv2.COLOR_YCrCb2BGR
                )
                cv2.imwrite(
                    str(
                        args.output_dir
                        / "rgb"
                        / method
                        / f"{optical_path.stem}.png"
                    ),
                    fused_bgr,
                )
                sar_bgr = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)
                preview = np.concatenate(
                    (optical_bgr, sar_bgr, fused_bgr), axis=1
                )
                cv2.imwrite(
                    str(
                        args.output_dir
                        / "preview"
                        / method
                        / f"{optical_path.stem}.png"
                    ),
                    preview,
                )
        rows.append(row)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(selected):
            print(f"processed {index + 1}/{len(selected)}: {optical_path.stem}")

    with (args.output_dir / "per_image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, metric_names)
    compact_manifest = {
        key: value for key, value in manifest.items() if key != "splits"
    }
    compact_manifest["split_counts"] = {
        name: len(values) for name, values in manifest["splits"].items()
    }
    summary.update(
        {
            "split": args.split,
            "offset": args.offset,
            "limit": limit,
            "metric_names": list(metric_names),
            "metric_profile": "paper",
            "output_policy": "clamp_then_uint8",
            "data_root": str(args.data_root.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_metadata": compact_manifest,
            "c0_checkpoint": str(args.c0_checkpoint.resolve()),
            "c0_checkpoint_sha256": c0_hash,
            "spd_checkpoint": str(args.spd_checkpoint.resolve()),
            "spd_checkpoint_sha256": spd_hash,
            "spd_base_scale": args.spd_base_scale,
            "spd_detail_scale": args.spd_detail_scale,
            "saved_visual_methods": list(args.save_methods),
            "spd_training_source_checkpoint": spd_source_checkpoint,
            "spd_vs_evaluation_encoder": encoder_equivalence,
            "spd_inference_scale": "full_input_resolution",
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
