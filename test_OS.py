"""Optical-SAR fusion inference and paper-aligned metric evaluation."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from train_OS import build_models, load_model_weights
from direct_fusion import DirectFusionCDDFuse
from pac_mamba import PACMambaCDDFuse
from fixed_line_evidence import FixedLineEvidence
from utils.Evaluator import Evaluator, PaperEvaluator
from utils.dataset_os import manifest_pairs, paired_paths


LEGACY_METRIC_NAMES = ("EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM")
PAPER_METRIC_NAMES = PaperEvaluator.METRIC_NAMES


def natural_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../datasets/Dongying"),
        help="Folder containing GF2 and GF3 subdirectories.",
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Manifest split to evaluate; formal reporting uses test.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "auto",
            "original",
            "demand_only",
            "residual_only",
            "frequency_matched_baseline",
            "frequency_minimax_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
            "fixed_line_roa",
            "fixed_line_phase_bridge",
            "srr_detail",
            "r1_spatial_modulation",
            "c1_base_adapter",
            "c1_detail_adapter",
            "e1_dual_stem",
            *DirectFusionCDDFuse.VARIANTS,
            PACMambaCDDFuse.VARIANT,
        ),
        default="auto",
    )
    parser.add_argument("--optical-dir", default="GF2")
    parser.add_argument("--sar-dir", default="GF3")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/CDDFuse_IVF.pth")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("test_result/Dongying_IVF")
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-demand-value", type=float)
    parser.add_argument(
        "--force-residual-scale",
        type=float,
        help=(
            "Override the residual-only injection scale during inference. "
            "This is an evaluation ablation and does not modify the checkpoint."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics without saving gray, RGB, or preview images.",
    )
    parser.add_argument("--save-gate-maps", action="store_true")
    parser.add_argument(
        "--metric-profile",
        choices=("paper", "legacy"),
        default="paper",
        help=(
            "paper: 14 metrics aligned with thesis Section 4.3.2; "
            "legacy: the original eight CDDFuse metrics."
        ),
    )
    parser.add_argument(
        "--output-range-policy",
        choices=("clamp", "legacy_minmax"),
        default="clamp",
        help=(
            "clamp fixes the network output range to [0,1] for comparable experiments; "
            "legacy_minmax reproduces the previous per-image normalization."
        ),
    )
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=10,
        help="Untimed forward passes on the first image before latency measurement.",
    )
    parser.add_argument(
        "--typical-scene-ids",
        nargs="*",
        default=(),
        help="Pre-registered image stems to preserve in the evaluation metadata.",
    )
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_fused_tensor(fused_tensor, policy):
    if policy == "clamp":
        return fused_tensor.clamp(0.0, 1.0)
    if policy == "legacy_minmax":
        minimum, maximum = fused_tensor.min(), fused_tensor.max()
        return (fused_tensor - minimum) / (maximum - minimum).clamp_min(1e-8)
    raise ValueError(f"Unknown output range policy: {policy}")


def synchronize_if_cuda(device):
    if device == "cuda":
        torch.cuda.synchronize()


def load_models(checkpoint_path: Path, device: str):
    encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    base_fuse = nn.DataParallel(
        BaseFeatureExtraction(dim=64, num_heads=8)
    ).to(device)
    detail_fuse = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    def load_compatible(model, state):
        first_key = next(iter(state))
        if first_key.startswith("module."):
            model.load_state_dict(state)
        else:
            model.module.load_state_dict(state)

    load_compatible(
        encoder, checkpoint.get("CDDF_Encoder", checkpoint.get("DIDF_Encoder"))
    )
    load_compatible(
        decoder, checkpoint.get("CDDF_Decoder", checkpoint.get("DIDF_Decoder"))
    )
    load_compatible(base_fuse, checkpoint["BaseFuseLayer"])
    load_compatible(detail_fuse, checkpoint["DetailFuseLayer"])
    for model in (encoder, decoder, base_fuse, detail_fuse):
        model.eval()
    return encoder, decoder, base_fuse, detail_fuse


def resolve_variant(checkpoint, requested):
    checkpoint_variant = checkpoint.get("args", {}).get("variant")
    if checkpoint_variant in (
        *DirectFusionCDDFuse.VARIANTS,
        PACMambaCDDFuse.VARIANT,
    ) or checkpoint_variant in (
        "frequency_matched_baseline",
        "frequency_minimax_baseline",
        "base_optical_detail",
        "base_optical_detail_refine",
        "fixed_line_roa",
        "fixed_line_phase_bridge",
        "srr_detail",
        "r1_spatial_modulation",
        "c1_base_adapter",
        "c1_detail_adapter",
        "e1_dual_stem",
    ):
        detected = checkpoint_variant
    elif "ResidualDetailInjector" in checkpoint or checkpoint_variant == "residual_only":
        detected = "residual_only"
    elif "DemandDetailInjector" in checkpoint or checkpoint_variant == "demand_only":
        detected = "demand_only"
    else:
        detected = "original"
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"Requested variant {requested}, but checkpoint contains {detected}"
        )
    return detected


def load_variant_models(checkpoint_path: Path, device: str, requested="auto"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    variant = resolve_variant(checkpoint, requested)
    if variant == PACMambaCDDFuse.VARIANT:
        model = PACMambaCDDFuse().to(device)
        state = checkpoint.get("model")
        if state is None:
            raise KeyError("PAC-Mamba checkpoint is missing the model state")
        model.load_state_dict(state)
        model.eval()
        return {"pac_mamba": model}, variant
    if variant in DirectFusionCDDFuse.VARIANTS:
        model = DirectFusionCDDFuse(variant).to(device)
        state = checkpoint.get("model")
        if state is None:
            raise KeyError("Direct-fusion checkpoint is missing the model state")
        model.load_state_dict(state)
        model.eval()
        return {"direct": model}, variant
    build_variant = (
        variant
        if variant in (
            "demand_only",
            "residual_only",
            "frequency_matched_baseline",
            "frequency_minimax_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
            "fixed_line_roa",
            "fixed_line_phase_bridge",
            "srr_detail",
            "r1_spatial_modulation",
            "c1_base_adapter",
            "c1_detail_adapter",
            "e1_dual_stem",
        )
        else "old_baseline"
    )
    models = build_models(torch.device(device), build_variant)
    if variant == "srr_detail":
        structure_calibration = checkpoint.get("args", {}).get(
            "structure_calibration"
        )
        if not structure_calibration:
            raise ValueError("srr_detail checkpoint has no structure_calibration")
        models["line_evidence"] = FixedLineEvidence.from_paths(
            structure_calibration
        ).to(device)
    load_model_weights(models, checkpoint)
    for model in models.values():
        model.eval()
    return models, variant


def forward_variant(
    models,
    variant,
    optical_tensor,
    sar_tensor,
    force_demand_value=None,
    force_residual_scale=None,
    return_maps=False,
):
    if variant == PACMambaCDDFuse.VARIANT:
        if return_maps:
            output, features = models["pac_mamba"](
                optical_tensor, sar_tensor, return_features=True
            )
            diagnostics = {
                "interaction_map": features["pac"]["interaction"],
                "phase_match_map": features["pac"]["phase_match"],
                "reliability_map": features["pac"]["reliability"],
                "update_map": torch.sqrt(
                    features["pac"]["update"].square().mean(dim=1, keepdim=True)
                    + 1e-12
                ),
                "interaction_mean": features["pac"]["interaction"].mean(),
                "phase_match_mean": features["pac"]["phase_match"].mean(),
                "reliability_mean": features["pac"]["reliability"].mean(),
                "update_rms": torch.sqrt(
                    features["pac"]["update"].square().mean() + 1e-12
                ),
            }
            return output, diagnostics
        return models["pac_mamba"](optical_tensor, sar_tensor), {}
    if variant in DirectFusionCDDFuse.VARIANTS:
        return models["direct"](optical_tensor, sar_tensor), {}
    if variant == "e1_dual_stem":
        optical_base, optical_detail, _ = models["encoder"](
            optical_tensor, "optical"
        )
        sar_base, sar_detail, _ = models["encoder"](
            sar_tensor, "sar"
        )
    else:
        optical_base, optical_detail, _ = models["encoder"](optical_tensor)
        sar_base, sar_detail, _ = models["encoder"](sar_tensor)
    diagnostics = {}
    if variant == "c1_base_adapter":
        sar_base, diagnostics = models["c1_adapter"](optical_base, sar_base)
    fused_base = models["base_fuse"](optical_base + sar_base)
    if variant == "demand_only":
        fused_detail, diagnostics = models["detail_injector"](
            optical_detail,
            sar_detail,
            force_demand=force_demand_value,
            return_maps=return_maps,
        )
    elif variant == "residual_only":
        fused_detail, diagnostics = models["detail_injector"](
            optical_detail,
            sar_detail,
            force_scale=force_residual_scale,
        )
    elif variant == "base_optical_detail":
        fused_detail = optical_detail
    elif variant == "base_optical_detail_refine":
        fused_detail = models["detail_refiner"](optical_detail)
    elif variant == "srr_detail":
        valid = (sar_tensor > 0.0).to(sar_tensor.dtype)
        structure_prior = models["line_evidence"](sar_tensor, valid)[
            "reliability"
        ]
        fused_detail, diagnostics = models["detail_router"](
            optical_detail,
            sar_detail,
            structure_prior,
            valid=valid,
            return_maps=return_maps,
        )
    elif variant == "r1_spatial_modulation":
        modulated_sar, diagnostics = models["detail_modulator"](
            optical_detail,
            sar_detail,
            return_maps=return_maps,
        )
        fused_detail = models["detail_fuse"](optical_detail + modulated_sar)
    elif variant == "c1_detail_adapter":
        adapted, diagnostics = models["c1_adapter"](
            optical_detail, sar_detail
        )
        fused_detail = models["detail_fuse"](optical_detail + adapted)
    else:
        fused_detail = models["detail_fuse"](optical_detail + sar_detail)
    fused_tensor, _ = models["decoder"](
        optical_tensor, fused_base, fused_detail
    )
    return fused_tensor, diagnostics


def evaluate(fused, optical_y, sar, metric_profile="paper"):
    if metric_profile == "paper":
        return PaperEvaluator.metric_vector(fused, optical_y, sar)
    return np.array(
        [
            Evaluator.EN(fused),
            Evaluator.SD(fused),
            Evaluator.SF(fused),
            Evaluator.MI(fused, optical_y, sar),
            Evaluator.SCD(fused, optical_y, sar),
            Evaluator.VIFF(fused, optical_y, sar),
            Evaluator.Qabf(fused, optical_y, sar),
            Evaluator.SSIM(fused, optical_y, sar),
        ],
        dtype=np.float64,
    )


def main():
    args = parse_args()
    if args.timing_warmup < 0:
        raise ValueError("timing-warmup must be non-negative")
    if args.force_demand_value is not None and not 0.0 <= args.force_demand_value <= 1.0:
        raise ValueError("force-demand-value must be in [0,1]")
    if (
        args.force_residual_scale is not None
        and not 0.0 <= args.force_residual_scale <= 1.0
    ):
        raise ValueError("force-residual-scale must be in [0,1]")
    if args.metrics_only and args.save_gate_maps:
        raise ValueError("metrics-only cannot be combined with save-gate-maps")
    metric_names = (
        PAPER_METRIC_NAMES if args.metric_profile == "paper" else LEGACY_METRIC_NAMES
    )
    if args.split_manifest:
        splits, _ = manifest_pairs(args.data_root, args.split_manifest)
        pairs = splits[args.split]
        default_limit = len(pairs)
    else:
        pairs = paired_paths(args.data_root, args.optical_dir, args.sar_dir)
        default_limit = 150
    limit = args.limit if args.limit is not None else default_limit
    selected = pairs[args.offset : args.offset + limit]
    if not selected:
        raise RuntimeError(f"No paired Optical-SAR images selected from {args.split}")
    selected_ids = {optical.stem for optical, _ in selected}
    missing_typical = sorted(set(args.typical_scene_ids) - selected_ids)
    if missing_typical:
        raise ValueError(
            "typical-scene-ids are outside the selected test set: "
            + ", ".join(missing_typical)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_folders = [] if args.metrics_only else ["gray", "rgb", "preview"]
    if args.save_gate_maps:
        output_folders.append("gate_maps")
        if args.force_demand_value is not None:
            output_folders.append("learned_gate_maps")
    for folder in output_folders:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, variant = load_variant_models(args.checkpoint, device, args.variant)
    if args.force_demand_value is not None and variant != "demand_only":
        raise ValueError("force-demand-value is only valid for demand_only")
    if args.force_residual_scale is not None and variant != "residual_only":
        raise ValueError("force-residual-scale is only valid for residual_only")
    if args.save_gate_maps and variant not in (
        "demand_only",
        "srr_detail",
        "r1_spatial_modulation",
        PACMambaCDDFuse.VARIANT,
    ):
        raise ValueError(
            "save-gate-maps is only valid for demand_only, srr_detail, "
            "r1_spatial_modulation, or pac_mamba"
        )
    total_parameters = sum(
        parameter.numel()
        for model in models.values()
        for parameter in model.parameters()
    )
    rows = []
    diagnostic_rows = []
    inference_seconds = []

    with torch.no_grad():
        for index, (optical_path, sar_path) in enumerate(selected, start=1):
            if not sar_path.exists():
                raise FileNotFoundError(f"Missing paired SAR image: {sar_path}")

            optical_bgr = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
            sar_u8 = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
            if optical_bgr is None or sar_u8 is None:
                raise RuntimeError(f"Could not read pair {optical_path.stem}")
            if optical_bgr.shape[:2] != sar_u8.shape:
                raise RuntimeError(
                    f"Shape mismatch for {optical_path.stem}: "
                    f"GF2={optical_bgr.shape[:2]}, GF3={sar_u8.shape}"
                )

            optical_ycrcb = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
            optical_y_u8 = optical_ycrcb[:, :, 0]
            optical_y = optical_y_u8.astype(np.float32) / 255.0
            sar = sar_u8.astype(np.float32) / 255.0

            optical_tensor = torch.from_numpy(optical_y)[None, None].to(device)
            sar_tensor = torch.from_numpy(sar)[None, None].to(device)

            if index == 1:
                for _ in range(args.timing_warmup):
                    forward_variant(
                        models,
                        variant,
                        optical_tensor,
                        sar_tensor,
                        force_demand_value=args.force_demand_value,
                        force_residual_scale=args.force_residual_scale,
                    )
                synchronize_if_cuda(device)

            synchronize_if_cuda(device)
            inference_start = time.perf_counter()
            fused_tensor, diagnostics = forward_variant(
                models,
                variant,
                optical_tensor,
                sar_tensor,
                force_demand_value=args.force_demand_value,
                force_residual_scale=args.force_residual_scale,
                return_maps=args.save_gate_maps,
            )
            synchronize_if_cuda(device)
            inference_seconds.append(time.perf_counter() - inference_start)
            fused_tensor = normalize_fused_tensor(
                fused_tensor, args.output_range_policy
            )
            diagnostics = dict(diagnostics)
            diagnostics["output_optical_mae"] = torch.mean(
                torch.abs(fused_tensor - optical_tensor)
            )
            sar_valid = (sar_tensor > 0.0).to(fused_tensor.dtype)
            diagnostics["output_sar_mae"] = (
                torch.sum(torch.abs(fused_tensor - sar_tensor) * sar_valid)
                / sar_valid.sum().clamp_min(1.0)
            )
            fused_u8 = np.clip(
                np.rint(fused_tensor.squeeze().cpu().numpy() * 255.0), 0, 255
            ).astype(np.uint8)

            fused_ycrcb = optical_ycrcb.copy()
            fused_ycrcb[:, :, 0] = fused_u8
            fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)
            sar_bgr = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)
            preview = np.concatenate((optical_bgr, sar_bgr, fused_bgr), axis=1)

            if not args.metrics_only:
                cv2.imwrite(
                    str(args.output_dir / "gray" / f"{optical_path.stem}.png"),
                    fused_u8,
                )
                cv2.imwrite(
                    str(args.output_dir / "rgb" / f"{optical_path.stem}.png"),
                    fused_bgr,
                )
                cv2.imwrite(
                    str(args.output_dir / "preview" / f"{optical_path.stem}.png"),
                    preview,
                )

            if diagnostics:
                diagnostic_row = {"image": optical_path.stem}
                for name, value in diagnostics.items():
                    if name.endswith("_map"):
                        continue
                    diagnostic_row[name] = float(value.detach().cpu())
                diagnostic_rows.append(diagnostic_row)
                if args.save_gate_maps:
                    if variant == "demand_only":
                        demand_map = diagnostics["demand_map"].squeeze().cpu().numpy()
                        demand_u8 = np.clip(
                            np.rint(demand_map * 255.0), 0, 255
                        ).astype(np.uint8)
                        demand_color = cv2.applyColorMap(
                            demand_u8, cv2.COLORMAP_TURBO
                        )
                        cv2.imwrite(
                            str(
                                args.output_dir
                                / "gate_maps"
                                / f"{optical_path.stem}.png"
                            ),
                            demand_color,
                        )
                    elif variant == "r1_spatial_modulation":
                        modulation_map = (
                            diagnostics["r1_modulation_map"].squeeze().cpu().numpy()
                        )
                        alpha = max(
                            float(models["detail_modulator"].alpha.detach().cpu()),
                            1e-8,
                        )
                        modulation_u8 = np.clip(
                            np.rint(
                                (np.clip(modulation_map / alpha, -1.0, 1.0) + 1.0)
                                * 127.5
                            ),
                            0,
                            255,
                        ).astype(np.uint8)
                        modulation_color = cv2.applyColorMap(
                            modulation_u8, cv2.COLORMAP_TURBO
                        )
                        cv2.imwrite(
                            str(
                                args.output_dir
                                / "gate_maps"
                                / f"{optical_path.stem}_r1_modulation_map.png"
                            ),
                            modulation_color,
                        )
                    elif variant == "srr_detail":
                        for map_name in (
                            "srr_reliability_map",
                            "srr_cross_attribute_map",
                            "srr_complementary_attribute_map",
                            "srr_cross_evidence_map",
                            "srr_match_certainty_map",
                            "srr_structure_prior_map",
                        ):
                            route_map = (
                                diagnostics[map_name].squeeze().cpu().numpy()
                            )
                            route_u8 = np.clip(
                                np.rint(route_map * 255.0), 0, 255
                            ).astype(np.uint8)
                            route_color = cv2.applyColorMap(
                                route_u8, cv2.COLORMAP_TURBO
                            )
                            cv2.imwrite(
                                str(
                                    args.output_dir
                                    / "gate_maps"
                                    / f"{optical_path.stem}_{map_name}.png"
                                ),
                                route_color,
                            )
                    elif variant == PACMambaCDDFuse.VARIANT:
                        for map_name in (
                            "interaction_map",
                            "phase_match_map",
                            "reliability_map",
                            "update_map",
                        ):
                            route_map = diagnostics[map_name].squeeze().cpu().numpy()
                            maximum = float(route_map.max())
                            if map_name == "update_map" and maximum > 0.0:
                                route_map = route_map / maximum
                            route_u8 = np.clip(
                                np.rint(route_map * 255.0), 0, 255
                            ).astype(np.uint8)
                            route_color = cv2.applyColorMap(
                                route_u8, cv2.COLORMAP_TURBO
                            )
                            cv2.imwrite(
                                str(
                                    args.output_dir
                                    / "gate_maps"
                                    / f"{optical_path.stem}_{map_name}.png"
                                ),
                                route_color,
                            )
                    if variant == "demand_only" and args.force_demand_value is not None:
                        learned_map = (
                            diagnostics["learned_demand_map"]
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                        learned_u8 = np.clip(
                            np.rint(learned_map * 255.0), 0, 255
                        ).astype(np.uint8)
                        learned_color = cv2.applyColorMap(
                            learned_u8, cv2.COLORMAP_TURBO
                        )
                        cv2.imwrite(
                            str(
                                args.output_dir
                                / "learned_gate_maps"
                                / f"{optical_path.stem}.png"
                            ),
                            learned_color,
                        )

            metrics = evaluate(
                fused_u8.astype(np.float32),
                optical_y_u8.astype(np.float32),
                sar_u8.astype(np.float32),
                args.metric_profile,
            )
            rows.append((optical_path.stem, metrics))
            if index == 1 or index % 25 == 0 or index == len(selected):
                print(f"processed {index}/{len(selected)}: {optical_path.name}")

    metric_matrix = np.stack([metrics for _, metrics in rows])
    means = metric_matrix.mean(axis=0)
    stds = metric_matrix.std(axis=0)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image",) + metric_names)
        for name, metrics in rows:
            writer.writerow((name,) + tuple(f"{value:.6f}" for value in metrics))
        writer.writerow(("mean",) + tuple(f"{value:.6f}" for value in means))
        writer.writerow(("std",) + tuple(f"{value:.6f}" for value in stds))

    gate_summary = None
    if diagnostic_rows:
        diagnostic_names = sorted(
            name for name in diagnostic_rows[0] if name != "image"
        )
        diagnostic_matrix = np.asarray(
            [
                [row[name] for name in diagnostic_names]
                for row in diagnostic_rows
            ],
            dtype=np.float64,
        )
        diagnostic_means = diagnostic_matrix.mean(axis=0)
        diagnostic_stds = diagnostic_matrix.std(axis=0)
        with (args.output_dir / "gate_stats.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("image",) + tuple(diagnostic_names))
            for row in diagnostic_rows:
                writer.writerow(
                    (row["image"],)
                    + tuple(f"{row[name]:.8f}" for name in diagnostic_names)
                )
            writer.writerow(
                ("mean",)
                + tuple(f"{value:.8f}" for value in diagnostic_means)
            )
            writer.writerow(
                ("std",) + tuple(f"{value:.8f}" for value in diagnostic_stds)
            )
        gate_summary = {
            name: {
                "mean": float(mean),
                "std": float(std),
            }
            for name, mean, std in zip(
                diagnostic_names, diagnostic_means, diagnostic_stds
            )
        }

    metadata = {
        "command": sys.argv,
        "device": device,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_root": str(args.data_root.resolve()),
        "split_manifest": (
            str(args.split_manifest.resolve()) if args.split_manifest else None
        ),
        "split_manifest_sha256": (
            sha256_file(args.split_manifest) if args.split_manifest else None
        ),
        "metric_profile": args.metric_profile,
        "split": args.split,
        "variant": variant,
        "force_demand_value": args.force_demand_value,
        "force_residual_scale": args.force_residual_scale,
        "metrics_only": args.metrics_only,
        "gate_summary": gate_summary,
        "metric_names": metric_names,
        "output_range_policy": args.output_range_policy,
        "offset": args.offset,
        "evaluated_pairs": len(rows),
        "selected_scene_ids": [name for name, _ in rows],
        "typical_scene_ids": list(args.typical_scene_ids),
        "total_parameters": total_parameters,
        "timing_warmup": args.timing_warmup,
        "inference_time_ms": {
            "mean": 1000.0 * float(np.mean(inference_seconds)),
            "std": 1000.0 * float(np.std(inference_seconds)),
            "median": 1000.0 * float(np.median(inference_seconds)),
        },
    }
    with (args.output_dir / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"variant: {variant}")
    print(f"split: {args.split}")
    print(f"evaluated pairs: {len(rows)}")
    print(f"metric profile: {args.metric_profile}")
    print(f"output range policy: {args.output_range_policy}")
    print(f"parameters: {total_parameters}")
    print(f"mean inference time: {metadata['inference_time_ms']['mean']:.3f} ms")
    print("\t".join(metric_names))
    print("\t".join(f"{value:.4f}" for value in means))
    if gate_summary:
        print(
            "diagnostic summary: "
            + ", ".join(
                f"{name}={values['mean']:.4f}"
                for name, values in gate_summary.items()
            )
        )
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
