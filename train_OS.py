"""Fine-tune CDDFuse on paired GF-2 optical and GF-3 SAR images."""

import argparse
import csv
import datetime
import hashlib
import json
import random
import time
from pathlib import Path

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from reliability_modules import (
    DemandDetailInjection,
    OpticalDetailRefinement,
    ResidualDetailInjection,
    ResidualSpatialModulator,
    SRRDetailFusion,
)
from fixed_line_evidence import FixedLineEvidence
from utils.dataset_os import OpticalSARDataset, manifest_pairs, split_pairs
from utils.loss import Sobelxy, cc


HISTORY_COMPONENTS = (
    "fusion",
    "intensity",
    "gradient",
    "optical",
    "optical_l1",
    "optical_ssim",
    "optical_high",
    "optical_constraint",
    "optical_gradient",
    "sar_gradient",
    "structure",
    "sar_mid",
    "line_structure",
    "line_weight_mean",
    "line_reliability_mean",
    "line_reliability_std",
    "line_reliability_detect_fraction",
    "sar_edge_fraction",
    "sar_gradient_retention",
    "sar_gradient_alignment",
    "output_optical_mae",
    "output_sar_mae",
    "gradient_app_norm",
    "gradient_str_norm",
    "gradient_cosine",
    "gradient_conflict",
    "relative_optical",
    "relative_structure",
    "selection_score",
    "demand_mean",
    "demand_std",
    "demand_low_fraction",
    "demand_high_fraction",
    "delta_norm",
    "injection_norm",
    "srr_reliability_mean",
    "srr_reliability_std",
    "srr_cross_attribute_mean",
    "srr_complementary_attribute_mean",
    "srr_cross_evidence_mean",
    "srr_match_certainty_mean",
    "srr_structure_prior_mean",
    "srr_routed_sar_norm",
    "srr_anchor_loss",
    "srr_anchor_bce_loss",
    "srr_anchor_rank_loss",
    "srr_positive_anchor_fraction",
    "srr_negative_anchor_fraction",
    "srr_positive_reliability_mean",
    "srr_negative_reliability_mean",
    "srr_anchor_reliability_gap",
    "r1_modulation_mean",
    "r1_modulation_std",
    "r1_modulation_abs_mean",
    "r1_modulation_min",
    "r1_modulation_max",
    "r1_deviation_loss",
    "decomp",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=(
            "old_baseline",
            "loss_matched_baseline",
            "loss_matched_baseline_v2",
            "loss_matched_baseline_v3",
            "loss_matched_baseline_v4",
            "loss_matched_baseline_v5",
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
        ),
        default="old_baseline",
        help=(
            "old_baseline keeps the original max-intensity/max-gradient loss; "
            "loss_matched_baseline preserves the failed B1-v1 for reproducibility; "
            "loss_matched_baseline_v2 preserves the first anti-collapse pilot; "
            "loss_matched_baseline_v3 uses constrained Optical fidelity and "
            "region-balanced Optical/SAR gradient losses; "
            "loss_matched_baseline_v4 is the plan-aligned two-term common loss; "
            "loss_matched_baseline_v5 keeps V4 losses and coordinates them with normalized PCGrad; "
            "demand_only preserves the rejected A1 content-gate ablation; "
            "residual_only is A1-prime with gamma=0.25 and no DemandGate; "
            "frequency_matched_baseline is C0 with the original DetailFuse; "
            "frequency_minimax_baseline trains C0 with reference-normalized smooth minimax; "
            "base_optical_detail is C1 with Optical detail only; "
            "base_optical_detail_refine is C2 with a zero-initialized Optical refiner."
            "fixed_line_roa is C3a with globally calibrated RoA line evidence; "
            "fixed_line_phase_bridge is C3b with local FA bridge evidence."
            "srr_detail replaces DetailFuse with speckle-aware structure-reliability routing."
            "r1_spatial_modulation keeps C0 frozen and trains a bounded SAR-detail modulator."
        ),
    )
    parser.add_argument(
        "--training-policy",
        choices=("all", "fusion_only", "router_only"),
        help=(
            "Defaults to fusion_only for B1-v2 through B1-v5 and all for older variants. "
            "fusion_only freezes Encoder and Decoder."
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("../datasets/Dongying")
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--optical-dir", default="GF2")
    parser.add_argument("--sar-dir", default="GF3")
    parser.add_argument("--output-dir", type=Path, default=Path("models/optical_sar"))
    parser.add_argument(
        "--run-name",
        help="Optional deterministic subdirectory name, for example B1_seed42.",
    )
    parser.add_argument(
        "--selection-reference-config",
        type=Path,
        help="B1-v5 config whose validation reference is reused by A1/A2/A3.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="models/CDDFuse_IVF.pth",
        help="Pretrained checkpoint, or 'none' to train from scratch.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        help="Defaults to 10 for old_baseline and 0 for loss_matched_baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--srr-router-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the newly initialized SRR detail router.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--decomp-weight",
        type=float,
        help="Defaults to 2.0 for old_baseline and 0.0 for Stage-0 B1.",
    )
    parser.add_argument("--gradient-weight", type=float, default=5.0)
    parser.add_argument(
        "--stage2-gradient-weight",
        type=float,
        default=10.0,
        help="Gradient coefficient used only by the original Stage-II fusion loss.",
    )
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=1.0,
        help="lambda_str in the Stage-0 common loss.",
    )
    parser.add_argument("--structure-sigma", type=float, default=1.0)
    parser.add_argument("--structure-percentile", type=float, default=99.0)
    parser.add_argument(
        "--structure-charbonnier-epsilon", type=float, default=1e-3
    )
    parser.add_argument(
        "--optical-weight",
        type=float,
        default=0.2,
        help="Weight of the low-frequency Optical preservation loss in B1-v2.",
    )
    parser.add_argument("--optical-lowpass-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-highpass-sigma", type=float, default=1.0)
    parser.add_argument("--frequency-midpass-sigma", type=float, default=2.0)
    parser.add_argument("--optical-high-weight", type=float, default=1.0)
    parser.add_argument("--minimax-temperature", type=float, default=0.1)
    parser.add_argument("--line-structure-weight", type=float, default=1.0)
    parser.add_argument("--line-reliability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--srr-anchor-weight",
        type=float,
        default=0.05,
        help=(
            "Weight of sparse reliability-anchor supervision for srr_detail. "
            "Set to zero for the unsupervised SRR ablation."
        ),
    )
    parser.add_argument(
        "--srr-ranking-margin",
        type=float,
        default=0.25,
        help=(
            "Minimum desired reliability gap between positive and negative "
            "SRR anchors."
        ),
    )
    parser.add_argument("--structure-calibration", type=Path)
    parser.add_argument("--phase-calibration", type=Path)
    parser.add_argument(
        "--reference-batches",
        type=int,
        default=0,
        help="Fixed training calibration batches for minimax references; 0 uses all.",
    )
    parser.add_argument(
        "--frequency-reference-config",
        type=Path,
        help=(
            "Reuse fixed frequency_training_reference values from a matching C0 "
            "config. Required by srr_detail for a loss-matched comparison."
        ),
    )
    parser.add_argument("--optical-l1-tolerance", type=float, default=0.02)
    parser.add_argument("--optical-dssim-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--optical-gradient-weight",
        type=float,
        default=0.25,
        help="Weight of the separately normalized non-SAR Optical gradient loss in B1-v3.",
    )
    parser.add_argument(
        "--sar-edge-percentile",
        type=float,
        default=85.0,
        help="Only stronger SAR gradients above this valid-region percentile can replace Optical gradients.",
    )
    parser.add_argument(
        "--sar-edge-margin",
        type=float,
        default=0.02,
        help="Minimum normalized magnitude advantage over Optical for selecting a SAR edge.",
    )
    parser.add_argument(
        "--collapse-mae-threshold",
        type=float,
        default=0.01,
        help=(
            "B1-v2 through B1-v5 checkpoints with validation output/Optical MAE below this "
            "value are rejected as identity collapse."
        ),
    )
    parser.add_argument("--clip-grad", type=float, default=0.01)
    parser.add_argument(
        "--srr-router-clip-grad",
        type=float,
        default=0.1,
        help="Gradient-norm clip used only by the SRR detail router.",
    )
    parser.add_argument(
        "--r1-modulation-alpha",
        type=float,
        default=0.25,
        help="Maximum absolute multiplicative change applied to SAR detail in R1.",
    )
    parser.add_argument(
        "--r1-dev-weight",
        type=float,
        default=0.01,
        help="Weight of the R1 mean absolute modulation penalty.",
    )
    parser.add_argument("--scheduler-step", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help=(
            "Defaults to 50 for old_baseline and unlimited for "
            "loss_matched_baseline; pass 0 for unlimited."
        ),
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, checkpoint, configuration, and model loading without training.",
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Measure B1-v4 Optical/SAR parameter-gradient norms and exit without training.",
    )
    parser.add_argument("--calibration-batches", type=int, default=4)
    args = parser.parse_args()
    is_loss_matched = args.variant.startswith("loss_matched_baseline")
    is_stage2_variant = is_loss_matched or args.variant in (
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
    )
    if args.phase1_epochs is None:
        args.phase1_epochs = 0 if is_stage2_variant else 10
    if args.decomp_weight is None:
        args.decomp_weight = 0.0 if is_stage2_variant else 2.0
    if args.max_val_batches is None:
        args.max_val_batches = 0 if is_stage2_variant else 50
    if args.training_policy is None:
        args.training_policy = (
            "fusion_only"
            if args.variant in (
                "loss_matched_baseline_v2",
                "loss_matched_baseline_v3",
                "loss_matched_baseline_v4",
                "loss_matched_baseline_v5",
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
            )
            else "all"
        )
    if args.variant == "r1_spatial_modulation":
        args.training_policy = "router_only"
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def canonical_state(state):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def checkpoint_model_state(checkpoint, current_key, legacy_key=None):
    state = checkpoint.get(current_key)
    if state is None and legacy_key:
        state = checkpoint.get(legacy_key)
    if state is None:
        raise KeyError(f"Missing {current_key}/{legacy_key} in checkpoint")
    return canonical_state(state)


def build_models(device, variant="old_baseline"):
    models = {
        "encoder": Restormer_Encoder().to(device),
        "decoder": Restormer_Decoder().to(device),
        "base_fuse": BaseFeatureExtraction(dim=64, num_heads=8).to(device),
    }
    if variant == "demand_only":
        models["detail_injector"] = DemandDetailInjection().to(device)
    elif variant == "residual_only":
        models["detail_injector"] = ResidualDetailInjection().to(device)
    elif variant == "base_optical_detail_refine":
        models["detail_refiner"] = OpticalDetailRefinement().to(device)
    elif variant == "base_optical_detail":
        pass
    elif variant == "srr_detail":
        models["detail_router"] = SRRDetailFusion().to(device)
    elif variant == "r1_spatial_modulation":
        models["detail_fuse"] = DetailFeatureExtraction(num_layers=1).to(device)
        models["detail_modulator"] = ResidualSpatialModulator().to(device)
    else:
        models["detail_fuse"] = DetailFeatureExtraction(num_layers=1).to(device)
    return models


def load_model_weights(models, checkpoint):
    models["encoder"].load_state_dict(
        checkpoint_model_state(checkpoint, "CDDF_Encoder", "DIDF_Encoder")
    )
    models["decoder"].load_state_dict(
        checkpoint_model_state(checkpoint, "CDDF_Decoder", "DIDF_Decoder")
    )
    models["base_fuse"].load_state_dict(
        checkpoint_model_state(checkpoint, "BaseFuseLayer")
    )
    if "detail_fuse" in models:
        models["detail_fuse"].load_state_dict(
            checkpoint_model_state(checkpoint, "DetailFuseLayer")
        )
    if "detail_router" in models:
        if checkpoint.get("SRRDetailFuseLayer") is not None:
            models["detail_router"].load_state_dict(
                canonical_state(checkpoint["SRRDetailFuseLayer"])
            )
        else:
            models["detail_router"].load_legacy_detail_fuse(
                checkpoint_model_state(checkpoint, "DetailFuseLayer")
            )
    if "detail_modulator" in models and checkpoint.get("R1SpatialModulator"):
        models["detail_modulator"].load_state_dict(
            canonical_state(checkpoint["R1SpatialModulator"])
        )
    if (
        "detail_injector" in models
        and isinstance(models["detail_injector"], DemandDetailInjection)
        and checkpoint.get("DemandDetailInjector")
    ):
        models["detail_injector"].load_state_dict(
            canonical_state(checkpoint["DemandDetailInjector"])
        )
    if (
        "detail_injector" in models
        and isinstance(models["detail_injector"], ResidualDetailInjection)
        and checkpoint.get("ResidualDetailInjector")
    ):
        models["detail_injector"].load_state_dict(
            canonical_state(checkpoint["ResidualDetailInjector"])
        )
    if "detail_refiner" in models and checkpoint.get("OpticalDetailRefiner"):
        models["detail_refiner"].load_state_dict(
            canonical_state(checkpoint["OpticalDetailRefiner"])
        )


def masked_mean(values, mask):
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(2)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum().clamp_min(1.0)
    return (values * expanded).sum() / denominator


def srr_reliability_anchor_loss(
    reliability, prior, cross, valid, ranking_margin=0.25
):
    """Balanced sparse supervision for SRR reliability only."""
    valid_bool = valid.detach() > 0.5
    prior_detached = prior.detach()
    cross_detached = cross.detach()
    positive = (prior_detached >= 0.5) & valid_bool
    negative_masks = []
    for sample_cross, sample_prior, sample_valid in zip(
        cross_detached, prior_detached, valid_bool
    ):
        valid_values = sample_cross[sample_valid]
        if valid_values.numel():
            low_cross = torch.quantile(valid_values.float(), 0.25).to(
                sample_cross.dtype
            )
            sample_negative = (
                (sample_prior <= 0.05)
                & (sample_cross <= low_cross)
                & sample_valid
            )
        else:
            sample_negative = torch.zeros_like(sample_valid)
        negative_masks.append(sample_negative)
    negative = torch.stack(negative_masks)
    negative = negative & ~positive

    losses = []
    positive_mean = reliability.sum() * 0.0
    negative_mean = reliability.sum() * 0.0
    if torch.any(positive):
        positive_mean = reliability[positive].mean()
        losses.append(
            F.binary_cross_entropy(
                reliability[positive], torch.ones_like(reliability[positive])
            )
        )
    if torch.any(negative):
        negative_mean = reliability[negative].mean()
        losses.append(
            F.binary_cross_entropy(
                reliability[negative], torch.zeros_like(reliability[negative])
            )
        )
    bce_loss = (
        torch.stack(losses).mean()
        if losses
        else reliability.sum() * 0.0
    )
    if torch.any(positive) and torch.any(negative):
        reliability_gap = positive_mean - negative_mean
        rank_loss = F.relu(
            reliability.new_tensor(float(ranking_margin)) - reliability_gap
        )
    else:
        reliability_gap = reliability.sum() * 0.0
        rank_loss = reliability.sum() * 0.0
    anchor_loss = bce_loss + rank_loss
    valid_count = valid_bool.sum().clamp_min(1)
    return (
        anchor_loss,
        bce_loss,
        rank_loss,
        positive.sum().to(reliability.dtype) / valid_count,
        negative.sum().to(reliability.dtype) / valid_count,
        positive_mean,
        negative_mean,
        reliability_gap,
    )


class OpticalSARLosses(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = kornia.losses.SSIM(11, reduction="none")
        self.spatial_gradient = kornia.filters.SpatialGradient()
        self.sobel = Sobelxy()

    def reconstruction(self, source, reconstructed, mask=None):
        ssim_map = self.ssim(source, reconstructed)
        mse_map = (source - reconstructed) ** 2
        if mask is None:
            return 5.0 * ssim_map.mean() + mse_map.mean()
        return 5.0 * masked_mean(ssim_map, mask) + masked_mean(mse_map, mask)

    def optical_gradient(self, optical, reconstructed):
        return F.l1_loss(
            self.spatial_gradient(optical), self.spatial_gradient(reconstructed)
        )

    def normalized_sobel(self, image):
        # Sobelxy sums the absolute x/y responses. Division by eight keeps a
        # [0,1] step response on the same practical scale as the optical loss.
        return self.sobel(image) / 8.0

    @staticmethod
    def gaussian_kernel_size(sigma):
        if sigma <= 0.0:
            raise ValueError("Gaussian sigma must be positive")
        kernel_size = max(3, int(round(6.0 * sigma + 1.0)))
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def gaussian_blur(self, image, sigma):
        size = self.gaussian_kernel_size(sigma)
        return kornia.filters.gaussian_blur2d(
            image, (size, size), (sigma, sigma)
        )

    def gradient_components(self, image):
        return self.spatial_gradient(image)

    @staticmethod
    def clip_gradient_components(gradients, mask, percentile):
        if not 0.0 < percentile <= 100.0:
            raise ValueError("structure-percentile must be in (0, 100]")
        magnitude = torch.sqrt(
            torch.sum(gradients.float() ** 2, dim=2) + 1e-12
        ).to(gradients.dtype)
        clipped = []
        quantile = percentile / 100.0
        for sample_gradients, sample_magnitude, sample_mask in zip(
            gradients, magnitude, mask
        ):
            valid = sample_mask.expand_as(sample_magnitude) > 0.5
            if torch.any(valid):
                threshold = torch.quantile(
                    sample_magnitude[valid].float(), quantile
                ).to(dtype=sample_magnitude.dtype)
                scale = (threshold / sample_magnitude.clamp_min(1e-6)).clamp_max(
                    1.0
                )
                sample_gradients = sample_gradients * scale.unsqueeze(1)
            clipped.append(sample_gradients)
        return torch.stack(clipped), magnitude

    @staticmethod
    def clip_valid_percentile(values, mask, percentile):
        if not 0.0 < percentile <= 100.0:
            raise ValueError("structure-percentile must be in (0, 100]")
        clipped = []
        quantile = percentile / 100.0
        for sample, sample_mask in zip(values, mask):
            valid = sample_mask.expand_as(sample) > 0.5
            if torch.any(valid):
                threshold = torch.quantile(sample[valid].float(), quantile).to(
                    dtype=sample.dtype
                )
                sample = torch.minimum(sample, threshold)
            clipped.append(sample)
        return torch.stack(clipped) * mask

    def sar_structure(self, sar, sar_valid, sigma, percentile):
        valid_mass = self.gaussian_blur(sar_valid, sigma).clamp_min(1e-6)
        smoothed = self.gaussian_blur(sar * sar_valid, sigma) / valid_mass
        structure = self.normalized_sobel(smoothed)
        return self.clip_valid_percentile(structure, sar_valid, percentile)

    def fusion_old(
        self, optical, sar, generated, sar_valid, stage2_gradient_weight
    ):
        intensity_target = torch.where(
            sar_valid > 0.5, torch.maximum(optical, sar), optical
        )
        optical_gradient = self.sobel(optical)
        sar_gradient = self.sobel(sar)
        gradient_target = torch.where(
            sar_valid > 0.5,
            torch.maximum(optical_gradient, sar_gradient),
            optical_gradient,
        )
        intensity_loss = F.l1_loss(generated, intensity_target)
        gradient_loss = F.l1_loss(self.sobel(generated), gradient_target)
        return (
            intensity_loss + stage2_gradient_weight * gradient_loss,
            intensity_loss,
            gradient_loss,
        )

    def fusion_common(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        structure_weight,
        structure_sigma,
        structure_percentile,
    ):
        optical_l1 = F.l1_loss(generated, optical)
        optical_ssim = self.ssim(optical, generated).mean()
        optical_loss = optical_l1 + optical_ssim

        # All three structure tensors must use the same normalized Sobel scale.
        # Mixing raw optical Sobel responses with normalized SAR/output responses
        # creates an unreachable target and drives the network to the optical skip.
        optical_gradient = self.normalized_sobel(optical)
        sar_structure = self.sar_structure(
            sar, sar_valid, structure_sigma, structure_percentile
        )
        structure_target = torch.maximum(optical_gradient, sar_structure)
        structure_loss = masked_mean(
            torch.abs(self.normalized_sobel(generated) - structure_target),
            sar_valid,
        )
        total = optical_loss + structure_weight * structure_loss
        return total, optical_loss, optical_l1, optical_ssim, structure_loss

    def sparse_sar_gradient_targets(
        self,
        optical,
        sar,
        sar_valid,
        structure_sigma,
        structure_percentile,
        sar_edge_percentile,
        sar_edge_margin,
    ):
        valid_mass = self.gaussian_blur(
            sar_valid, structure_sigma
        ).clamp_min(1e-6)
        sar_smoothed = self.gaussian_blur(
            sar * sar_valid, structure_sigma
        ) / valid_mass
        optical_gradients = self.gradient_components(optical)
        sar_gradients = self.gradient_components(sar_smoothed)
        sar_gradients, _ = self.clip_gradient_components(
            sar_gradients,
            sar_valid,
            structure_percentile,
        )
        optical_magnitude = torch.sqrt(
            torch.sum(optical_gradients.float() ** 2, dim=2) + 1e-12
        ).to(optical.dtype)
        sar_magnitude = torch.sqrt(
            torch.sum(sar_gradients.float() ** 2, dim=2) + 1e-12
        ).to(sar.dtype)

        edge_masks = []
        edge_quantile = sar_edge_percentile / 100.0
        for sample_sar, sample_optical, sample_valid in zip(
            sar_magnitude, optical_magnitude, sar_valid
        ):
            valid = sample_valid.expand_as(sample_sar) > 0.5
            if torch.any(valid):
                threshold = torch.quantile(
                    sample_sar[valid].float(), edge_quantile
                ).to(sample_sar.dtype)
                selected = (
                    valid
                    & (sample_sar >= threshold)
                    & (sample_sar > sample_optical + sar_edge_margin)
                )
            else:
                selected = torch.zeros_like(valid)
            edge_masks.append(selected.to(sample_sar.dtype))
        sar_edge_mask = torch.stack(edge_masks)
        return (
            optical_gradients,
            sar_gradients,
            optical_magnitude,
            sar_magnitude,
            sar_edge_mask,
        )

    def fusion_common_v2(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        optical_weight,
        structure_weight,
        structure_sigma,
        structure_percentile,
        optical_lowpass_sigma,
        sar_edge_percentile,
        sar_edge_margin,
    ):
        if not 0.0 < sar_edge_percentile <= 100.0:
            raise ValueError("sar-edge-percentile must be in (0, 100]")

        optical_low = self.gaussian_blur(optical, optical_lowpass_sigma)
        generated_low = self.gaussian_blur(generated, optical_lowpass_sigma)
        optical_l1 = F.l1_loss(generated_low, optical_low)
        optical_ssim = self.ssim(optical_low, generated_low).mean()
        optical_loss = optical_l1 + optical_ssim

        valid_mass = self.gaussian_blur(
            sar_valid, structure_sigma
        ).clamp_min(1e-6)
        sar_smoothed = self.gaussian_blur(
            sar * sar_valid, structure_sigma
        ) / valid_mass
        optical_gradients = self.gradient_components(optical)
        sar_gradients = self.gradient_components(sar_smoothed)
        sar_gradients, sar_magnitude = self.clip_gradient_components(
            sar_gradients,
            sar_valid,
            structure_percentile,
        )
        optical_magnitude = torch.sqrt(
            torch.sum(optical_gradients.float() ** 2, dim=2) + 1e-12
        ).to(optical.dtype)
        sar_magnitude = torch.sqrt(
            torch.sum(sar_gradients.float() ** 2, dim=2) + 1e-12
        ).to(sar.dtype)

        edge_masks = []
        edge_quantile = sar_edge_percentile / 100.0
        for sample_sar, sample_optical, sample_valid in zip(
            sar_magnitude, optical_magnitude, sar_valid
        ):
            valid = sample_valid.expand_as(sample_sar) > 0.5
            if torch.any(valid):
                threshold = torch.quantile(
                    sample_sar[valid].float(), edge_quantile
                ).to(sample_sar.dtype)
                selected = (
                    valid
                    & (sample_sar >= threshold)
                    & (sample_sar > sample_optical + sar_edge_margin)
                )
            else:
                selected = torch.zeros_like(valid)
            edge_masks.append(selected.to(sample_sar.dtype))
        sar_edge_mask = torch.stack(edge_masks)

        target_gradients = torch.where(
            sar_edge_mask.unsqueeze(2) > 0.5,
            sar_gradients,
            optical_gradients,
        )
        generated_gradients = self.gradient_components(generated)
        structure_loss = masked_mean(
            torch.abs(generated_gradients - target_gradients), sar_valid
        )
        sar_edge_fraction = (
            (sar_edge_mask * sar_valid).sum()
            / sar_valid.sum().clamp_min(1.0)
        )
        output_optical_mae = F.l1_loss(generated, optical)
        total = optical_weight * optical_loss + structure_weight * structure_loss
        return (
            total,
            optical_loss,
            optical_l1,
            optical_ssim,
            structure_loss,
            sar_edge_fraction,
            output_optical_mae,
        )

    def fusion_common_v3(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        optical_weight,
        optical_gradient_weight,
        structure_weight,
        structure_sigma,
        structure_percentile,
        optical_lowpass_sigma,
        optical_l1_tolerance,
        optical_dssim_tolerance,
        sar_edge_percentile,
        sar_edge_margin,
    ):
        if not 0.0 < sar_edge_percentile <= 100.0:
            raise ValueError("sar-edge-percentile must be in (0, 100]")

        optical_low = self.gaussian_blur(optical, optical_lowpass_sigma)
        generated_low = self.gaussian_blur(generated, optical_lowpass_sigma)
        optical_l1_per_sample = torch.abs(generated_low - optical_low).flatten(1).mean(1)
        optical_ssim_per_sample = self.ssim(
            optical_low, generated_low
        ).flatten(1).mean(1)
        optical_l1 = optical_l1_per_sample.mean()
        optical_ssim = optical_ssim_per_sample.mean()
        optical_constraint = (
            F.relu(optical_l1_per_sample - optical_l1_tolerance)
            + F.relu(optical_ssim_per_sample - optical_dssim_tolerance)
        ).mean()

        (
            optical_gradients,
            sar_gradients,
            _,
            sar_magnitude,
            sar_edge_mask,
        ) = self.sparse_sar_gradient_targets(
            optical,
            sar,
            sar_valid,
            structure_sigma,
            structure_percentile,
            sar_edge_percentile,
            sar_edge_margin,
        )
        generated_gradients = self.gradient_components(generated)
        optical_region = sar_valid * (1.0 - sar_edge_mask)
        sar_region = sar_valid * sar_edge_mask
        optical_gradient_loss = masked_mean(
            torch.abs(generated_gradients - optical_gradients), optical_region
        )
        sar_gradient_loss = masked_mean(
            torch.abs(generated_gradients - sar_gradients), sar_region
        )
        structure_loss = (
            optical_gradient_weight * optical_gradient_loss
            + structure_weight * sar_gradient_loss
        )

        generated_magnitude = torch.sqrt(
            torch.sum(generated_gradients.float() ** 2, dim=2) + 1e-12
        ).to(generated.dtype)
        sar_gradient_retention = masked_mean(
            (generated_magnitude / sar_magnitude.clamp_min(1e-6)).clamp(0.0, 1.0),
            sar_region,
        )
        gradient_dot = torch.sum(
            generated_gradients.float() * sar_gradients.float(), dim=2
        )
        gradient_denominator = (
            generated_magnitude.float() * sar_magnitude.float()
        ).clamp_min(1e-6)
        sar_gradient_alignment = masked_mean(
            (gradient_dot / gradient_denominator).clamp(-1.0, 1.0),
            sar_region,
        )
        sar_edge_fraction = sar_region.sum() / sar_valid.sum().clamp_min(1.0)
        output_optical_mae = F.l1_loss(generated, optical)
        total = optical_weight * optical_constraint + structure_loss
        return (
            total,
            optical_constraint,
            optical_l1,
            optical_ssim,
            optical_constraint,
            optical_gradient_loss,
            sar_gradient_loss,
            structure_loss,
            sar_edge_fraction,
            sar_gradient_retention,
            sar_gradient_alignment,
            output_optical_mae,
        )

    def fusion_common_v4(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        structure_weight,
        structure_sigma,
        structure_percentile,
        optical_lowpass_sigma,
        charbonnier_epsilon,
        reliability=None,
    ):
        optical_low = self.gaussian_blur(optical, optical_lowpass_sigma)
        generated_low = self.gaussian_blur(generated, optical_lowpass_sigma)
        optical_l1 = F.l1_loss(generated_low, optical_low)
        optical_ssim = self.ssim(optical_low, generated_low).mean()
        optical_loss = optical_l1 + optical_ssim

        sar_structure = self.sar_structure(
            sar,
            sar_valid,
            structure_sigma,
            structure_percentile,
        )
        if reliability is None:
            reliability = torch.ones_like(sar_structure)
        reliability = reliability.detach().clamp(0.0, 1.0)
        optical_gradient = self.normalized_sobel(optical)
        generated_gradient = self.normalized_sobel(generated)
        structure_target = torch.maximum(
            optical_gradient, reliability * sar_structure
        )
        difference = generated_gradient - structure_target
        charbonnier = torch.sqrt(
            difference.float() ** 2 + charbonnier_epsilon ** 2
        ).to(difference.dtype) - charbonnier_epsilon
        structure_loss = masked_mean(charbonnier, sar_valid)
        total = optical_loss + structure_weight * structure_loss
        output_optical_mae = F.l1_loss(generated, optical)
        return (
            total,
            optical_loss,
            optical_l1,
            optical_ssim,
            structure_loss,
            output_optical_mae,
        )

    @staticmethod
    def charbonnier(difference, epsilon):
        return (
            torch.sqrt(difference.float() ** 2 + epsilon ** 2)
            .to(difference.dtype)
            - epsilon
        )

    def fusion_frequency_decoupled(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        optical_lowpass_sigma,
        highpass_sigma,
        midpass_sigma,
        optical_high_weight,
        structure_percentile,
        charbonnier_epsilon,
        reliability=None,
    ):
        """C0-C2 loss: Optical high frequencies and SAR mid-scale structure."""
        optical_low = self.gaussian_blur(optical, optical_lowpass_sigma)
        generated_low = self.gaussian_blur(generated, optical_lowpass_sigma)
        optical_l1 = F.l1_loss(generated_low, optical_low)
        optical_ssim = self.ssim(optical_low, generated_low).mean()

        optical_high = optical - self.gaussian_blur(optical, highpass_sigma)
        generated_high = generated - self.gaussian_blur(
            generated, highpass_sigma
        )
        optical_high_loss = self.charbonnier(
            generated_high - optical_high, charbonnier_epsilon
        ).mean()
        optical_loss = (
            optical_l1 + optical_ssim + optical_high_weight * optical_high_loss
        )

        optical_mid = self.gaussian_blur(optical, midpass_sigma)
        generated_mid = self.gaussian_blur(generated, midpass_sigma)
        valid_mass = self.gaussian_blur(
            sar_valid, midpass_sigma
        ).clamp_min(1e-6)
        sar_mid = self.gaussian_blur(
            sar * sar_valid, midpass_sigma
        ) / valid_mass
        optical_mid_gradient = self.gradient_components(optical_mid)
        generated_mid_gradient = self.gradient_components(generated_mid)
        sar_mid_gradient = self.gradient_components(sar_mid)
        sar_mid_gradient, _ = self.clip_gradient_components(
            sar_mid_gradient,
            sar_valid,
            structure_percentile,
        )
        sar_mid_magnitude = torch.sqrt(
            torch.sum(sar_mid_gradient.float() ** 2, dim=2) + 1e-12
        ).to(sar.dtype)
        optical_mid_magnitude = torch.sqrt(
            torch.sum(optical_mid_gradient.float() ** 2, dim=2) + 1e-12
        ).to(optical.dtype)
        if reliability is None:
            reliability = torch.ones_like(sar_mid_magnitude)
        reliability = reliability.detach().clamp(0.0, 1.0)
        reliable_sar_magnitude = reliability * sar_mid_magnitude
        use_sar = (
            (reliable_sar_magnitude > optical_mid_magnitude)
            & (sar_valid > 0.5)
        )
        mid_target = torch.where(
            use_sar.unsqueeze(2),
            reliability.unsqueeze(2) * sar_mid_gradient,
            optical_mid_gradient,
        )
        sar_mid_loss = masked_mean(
            self.charbonnier(
                generated_mid_gradient - mid_target,
                charbonnier_epsilon,
            ),
            sar_valid,
        )

        output_optical_mae = F.l1_loss(generated, optical)
        output_sar_mae = masked_mean(torch.abs(generated - sar), sar_valid)
        total = optical_loss + sar_mid_loss
        return (
            total,
            optical_loss,
            optical_l1,
            optical_ssim,
            optical_high_loss,
            sar_mid_loss,
            output_optical_mae,
            output_sar_mae,
        )

    def fusion_frequency_selective_line(
        self,
        optical,
        sar,
        generated,
        sar_valid,
        reliability,
        optical_lowpass_sigma,
        highpass_sigma,
        midpass_sigma,
        optical_high_weight,
        line_structure_weight,
        line_reliability_threshold,
        structure_percentile,
        charbonnier_epsilon,
    ):
        """C3: Optical high frequencies except at calibrated SAR line evidence."""
        optical_low = self.gaussian_blur(optical, optical_lowpass_sigma)
        generated_low = self.gaussian_blur(generated, optical_lowpass_sigma)
        optical_l1 = F.l1_loss(generated_low, optical_low)
        optical_ssim = self.ssim(optical_low, generated_low).mean()

        optical_high = optical - self.gaussian_blur(optical, highpass_sigma)
        generated_high = generated - self.gaussian_blur(generated, highpass_sigma)
        valid_mass_high = self.gaussian_blur(
            sar_valid, highpass_sigma
        ).clamp_min(1e-6)
        sar_low_high = self.gaussian_blur(
            sar * sar_valid, highpass_sigma
        ) / valid_mass_high
        sar_high = (sar - sar_low_high) * sar_valid
        optical_high_gradient = self.gradient_components(optical_high)
        generated_high_gradient = self.gradient_components(generated_high)
        sar_high_gradient = self.gradient_components(sar_high)

        reliability = reliability.detach().clamp(0.0, 1.0) * sar_valid
        denominator = max(1.0 - line_reliability_threshold, 1e-6)
        line_weight = (
            (reliability - line_reliability_threshold) / denominator
        ).clamp(0.0, 1.0) * sar_valid
        optical_weight = 1.0 - line_weight
        optical_high_loss = masked_mean(
            self.charbonnier(
                generated_high_gradient - optical_high_gradient,
                charbonnier_epsilon,
            ),
            optical_weight,
        )
        line_structure_loss = masked_mean(
            self.charbonnier(
                generated_high_gradient - sar_high_gradient,
                charbonnier_epsilon,
            ),
            line_weight,
        )
        optical_loss = (
            optical_l1 + optical_ssim + optical_high_weight * optical_high_loss
        )

        optical_mid = self.gaussian_blur(optical, midpass_sigma)
        generated_mid = self.gaussian_blur(generated, midpass_sigma)
        valid_mass_mid = self.gaussian_blur(
            sar_valid, midpass_sigma
        ).clamp_min(1e-6)
        sar_mid = self.gaussian_blur(
            sar * sar_valid, midpass_sigma
        ) / valid_mass_mid
        optical_mid_gradient = self.gradient_components(optical_mid)
        generated_mid_gradient = self.gradient_components(generated_mid)
        sar_mid_gradient, _ = self.clip_gradient_components(
            self.gradient_components(sar_mid),
            sar_valid,
            structure_percentile,
        )
        optical_mid_magnitude = torch.sqrt(
            torch.sum(optical_mid_gradient.float() ** 2, dim=2) + 1e-12
        ).to(optical.dtype)
        sar_mid_magnitude = torch.sqrt(
            torch.sum(sar_mid_gradient.float() ** 2, dim=2) + 1e-12
        ).to(sar.dtype)
        use_sar = (
            (sar_mid_magnitude > optical_mid_magnitude)
            & (sar_valid > 0.5)
        )
        mid_target = torch.where(
            use_sar.unsqueeze(2), sar_mid_gradient, optical_mid_gradient
        )
        sar_mid_loss = masked_mean(
            self.charbonnier(
                generated_mid_gradient - mid_target,
                charbonnier_epsilon,
            ),
            sar_valid,
        )
        structure_loss = sar_mid_loss + line_structure_weight * line_structure_loss
        output_optical_mae = F.l1_loss(generated, optical)
        output_sar_mae = masked_mean(torch.abs(generated - sar), sar_valid)
        valid_mass = sar_valid.sum().clamp_min(1.0)
        reliability_mean = (reliability * sar_valid).sum() / valid_mass
        reliability_variance = (
            ((reliability - reliability_mean) ** 2) * sar_valid
        ).sum() / valid_mass
        reliability_detect_fraction = (
            ((reliability > line_reliability_threshold).to(sar.dtype) * sar_valid).sum()
            / valid_mass
        )
        line_weight_mean = (line_weight * sar_valid).sum() / valid_mass
        total = optical_loss + structure_loss
        return (
            total,
            optical_loss,
            optical_l1,
            optical_ssim,
            optical_high_loss,
            sar_mid_loss,
            line_structure_loss,
            line_weight_mean,
            reliability_mean,
            torch.sqrt(reliability_variance.clamp_min(0.0)),
            reliability_detect_fraction,
            structure_loss,
            output_optical_mae,
            output_sar_mae,
        )


def forward_loss(models, losses, optical, sar, mask, phase, args):
    encoder = models["encoder"]
    decoder = models["decoder"]
    base_fuse = models["base_fuse"]
    detail_fuse = models.get("detail_fuse")
    detail_injector = models.get("detail_injector")
    detail_refiner = models.get("detail_refiner")
    detail_router = models.get("detail_router")
    detail_modulator = models.get("detail_modulator")

    optical_base, optical_detail, _ = encoder(optical)
    sar_base, sar_detail, _ = encoder(sar)
    base_correlation = cc(optical_base, sar_base)
    detail_correlation = cc(optical_detail, sar_detail)
    decomposition = detail_correlation.square() / (1.01 + base_correlation)

    if phase == 1:
        optical_hat, _ = decoder(optical, optical_base, optical_detail)
        sar_hat, _ = decoder(sar, sar_base, sar_detail)
        optical_reconstruction = losses.reconstruction(optical, optical_hat)
        sar_reconstruction = losses.reconstruction(sar, sar_hat, mask)
        gradient = losses.optical_gradient(optical, optical_hat)
        total = (
            optical_reconstruction
            + sar_reconstruction
            + args.decomp_weight * decomposition
            + args.gradient_weight * gradient
        )
        components = {
            "optical_recon": optical_reconstruction,
            "sar_recon": sar_reconstruction,
            "decomp": decomposition,
            "gradient": gradient,
        }
    else:
        fused_base = base_fuse(optical_base + sar_base)
        injection_diagnostics = {}
        if args.variant in ("demand_only", "residual_only"):
            fused_detail, injection_diagnostics = detail_injector(
                optical_detail, sar_detail
            )
        elif args.variant == "base_optical_detail":
            fused_detail = optical_detail
        elif args.variant == "base_optical_detail_refine":
            fused_detail = detail_refiner(optical_detail)
        elif args.variant == "srr_detail":
            with torch.no_grad():
                structure_prior = models["line_evidence"](sar, mask)[
                    "reliability"
                ]
            fused_detail, injection_diagnostics = detail_router(
                optical_detail,
                sar_detail,
                structure_prior,
                valid=mask,
                return_maps=args.srr_anchor_weight > 0.0,
            )
            if args.srr_anchor_weight > 0.0:
                route_maps = {
                    name: value
                    for name, value in injection_diagnostics.items()
                    if name.endswith("_map")
                }
                injection_diagnostics = {
                    name: value
                    for name, value in injection_diagnostics.items()
                    if not name.endswith("_map")
                }
                (
                    anchor_loss,
                    anchor_bce_loss,
                    anchor_rank_loss,
                    positive_fraction,
                    negative_fraction,
                    positive_reliability,
                    negative_reliability,
                    reliability_gap,
                ) = srr_reliability_anchor_loss(
                    route_maps["srr_reliability_map"],
                    route_maps["srr_structure_prior_map"],
                    route_maps["srr_cross_evidence_map"],
                    mask,
                    ranking_margin=args.srr_ranking_margin,
                )
                injection_diagnostics.update(
                    {
                        "srr_anchor_loss": anchor_loss,
                        "srr_anchor_bce_loss": anchor_bce_loss,
                        "srr_anchor_rank_loss": anchor_rank_loss,
                        "srr_positive_anchor_fraction": positive_fraction,
                        "srr_negative_anchor_fraction": negative_fraction,
                        "srr_positive_reliability_mean": positive_reliability,
                        "srr_negative_reliability_mean": negative_reliability,
                        "srr_anchor_reliability_gap": reliability_gap,
                    }
                )
        elif args.variant == "r1_spatial_modulation":
            modulated_sar, injection_diagnostics = detail_modulator(
                optical_detail, sar_detail
            )
            fused_detail = detail_fuse(optical_detail + modulated_sar)
            injection_diagnostics["r1_deviation_loss"] = (
                injection_diagnostics["r1_modulation_abs_mean"]
            )
        else:
            fused_detail = detail_fuse(optical_detail + sar_detail)
        generated, _ = decoder(optical, fused_base, fused_detail)
        if args.variant in ("fixed_line_roa", "fixed_line_phase_bridge"):
            with torch.no_grad():
                line_evidence = models["line_evidence"](sar, mask)["reliability"]
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                optical_high_loss,
                sar_mid_loss,
                line_structure_loss,
                line_weight_mean,
                line_reliability_mean,
                line_reliability_std,
                line_reliability_detect_fraction,
                structure_loss,
                output_optical_mae,
                output_sar_mae,
            ) = losses.fusion_frequency_selective_line(
                optical,
                sar,
                generated,
                mask,
                line_evidence,
                args.optical_lowpass_sigma,
                args.frequency_highpass_sigma,
                args.frequency_midpass_sigma,
                args.optical_high_weight,
                args.line_structure_weight,
                args.line_reliability_threshold,
                args.structure_percentile,
                args.structure_charbonnier_epsilon,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "optical_high": optical_high_loss,
                "structure": structure_loss,
                "sar_mid": sar_mid_loss,
                "line_structure": line_structure_loss,
                "line_weight_mean": line_weight_mean,
                "line_reliability_mean": line_reliability_mean,
                "line_reliability_std": line_reliability_std,
                "line_reliability_detect_fraction": line_reliability_detect_fraction,
                "output_optical_mae": output_optical_mae,
                "output_sar_mae": output_sar_mae,
                "decomp": decomposition,
            }
        elif args.variant in (
            "frequency_matched_baseline",
            "frequency_minimax_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
            "srr_detail",
            "r1_spatial_modulation",
        ):
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                optical_high_loss,
                sar_mid_loss,
                output_optical_mae,
                output_sar_mae,
            ) = losses.fusion_frequency_decoupled(
                optical,
                sar,
                generated,
                mask,
                args.optical_lowpass_sigma,
                args.frequency_highpass_sigma,
                args.frequency_midpass_sigma,
                args.optical_high_weight,
                args.structure_percentile,
                args.structure_charbonnier_epsilon,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "optical_high": optical_high_loss,
                "structure": sar_mid_loss,
                "sar_mid": sar_mid_loss,
                "output_optical_mae": output_optical_mae,
                "output_sar_mae": output_sar_mae,
                "decomp": decomposition,
            }
        elif args.variant in (
            "loss_matched_baseline_v4",
            "loss_matched_baseline_v5",
            "demand_only",
            "residual_only",
        ):
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                structure_loss,
                output_optical_mae,
            ) = losses.fusion_common_v4(
                optical,
                sar,
                generated,
                mask,
                (
                    1.0
                    if args.variant in (
                        "loss_matched_baseline_v5",
                        "demand_only",
                        "residual_only",
                    )
                    else args.structure_weight
                ),
                args.structure_sigma,
                args.structure_percentile,
                args.optical_lowpass_sigma,
                args.structure_charbonnier_epsilon,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "structure": structure_loss,
                "output_optical_mae": output_optical_mae,
                "decomp": decomposition,
            }
        elif args.variant == "loss_matched_baseline_v3":
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                optical_constraint,
                optical_gradient_loss,
                sar_gradient_loss,
                structure_loss,
                sar_edge_fraction,
                sar_gradient_retention,
                sar_gradient_alignment,
                output_optical_mae,
            ) = losses.fusion_common_v3(
                optical,
                sar,
                generated,
                mask,
                args.optical_weight,
                args.optical_gradient_weight,
                args.structure_weight,
                args.structure_sigma,
                args.structure_percentile,
                args.optical_lowpass_sigma,
                args.optical_l1_tolerance,
                args.optical_dssim_tolerance,
                args.sar_edge_percentile,
                args.sar_edge_margin,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "optical_constraint": optical_constraint,
                "optical_gradient": optical_gradient_loss,
                "sar_gradient": sar_gradient_loss,
                "structure": structure_loss,
                "sar_edge_fraction": sar_edge_fraction,
                "sar_gradient_retention": sar_gradient_retention,
                "sar_gradient_alignment": sar_gradient_alignment,
                "output_optical_mae": output_optical_mae,
                "decomp": decomposition,
            }
        elif args.variant == "loss_matched_baseline_v2":
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                structure_loss,
                sar_edge_fraction,
                output_optical_mae,
            ) = losses.fusion_common_v2(
                optical,
                sar,
                generated,
                mask,
                args.optical_weight,
                args.structure_weight,
                args.structure_sigma,
                args.structure_percentile,
                args.optical_lowpass_sigma,
                args.sar_edge_percentile,
                args.sar_edge_margin,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "structure": structure_loss,
                "sar_edge_fraction": sar_edge_fraction,
                "output_optical_mae": output_optical_mae,
                "decomp": decomposition,
            }
        elif args.variant == "loss_matched_baseline":
            (
                fusion,
                optical_loss,
                optical_l1,
                optical_ssim,
                structure_loss,
            ) = losses.fusion_common(
                optical,
                sar,
                generated,
                mask,
                args.structure_weight,
                args.structure_sigma,
                args.structure_percentile,
            )
            components = {
                "fusion": fusion,
                "optical": optical_loss,
                "optical_l1": optical_l1,
                "optical_ssim": optical_ssim,
                "structure": structure_loss,
                "decomp": decomposition,
            }
        else:
            fusion, intensity, gradient = losses.fusion_old(
                optical,
                sar,
                generated,
                mask,
                args.stage2_gradient_weight,
            )
            components = {
                "fusion": fusion,
                "intensity": intensity,
                "gradient": gradient,
                "decomp": decomposition,
            }
        total = fusion + args.decomp_weight * decomposition
        components.update(injection_diagnostics)
    return total, components


def configure_trainability(models, training_policy):
    if training_policy == "router_only":
        trainable = {"detail_modulator"}
    elif training_policy == "fusion_only":
        trainable = {"base_fuse"}
        if "detail_injector" in models:
            trainable.add("detail_injector")
        if "detail_refiner" in models:
            trainable.add("detail_refiner")
        if "detail_fuse" in models:
            trainable.add("detail_fuse")
        if "detail_router" in models:
            trainable.add("detail_router")
    else:
        trainable = set(models)
    for name, model in models.items():
        requires_grad = name in trainable
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad)


def active_names(models, phase):
    phase_names = ("encoder", "decoder") if phase == 1 else tuple(models)
    return tuple(
        name
        for name in phase_names
        if any(parameter.requires_grad for parameter in models[name].parameters())
    )


def normalized_pcgrad_backward(loss_app, loss_structure, parameters):
    gradients_app = torch.autograd.grad(
        loss_app,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gradients_structure = torch.autograd.grad(
        loss_structure,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    gradients_app = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients_app)
    )
    gradients_structure = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients_structure)
    )
    norm_app = torch.sqrt(
        sum(torch.sum(gradient.float() ** 2) for gradient in gradients_app)
    ).clamp_min(1e-12)
    norm_structure = torch.sqrt(
        sum(torch.sum(gradient.float() ** 2) for gradient in gradients_structure)
    ).clamp_min(1e-12)
    unit_app = tuple(gradient / norm_app for gradient in gradients_app)
    unit_structure = tuple(
        gradient / norm_structure for gradient in gradients_structure
    )
    dot = sum(
        torch.sum(left.float() * right.float())
        for left, right in zip(unit_app, unit_structure)
    )
    conflict = dot < 0.0
    if conflict:
        projected_app = tuple(
            left - dot.to(left.dtype) * right
            for left, right in zip(unit_app, unit_structure)
        )
        projected_structure = tuple(
            right - dot.to(right.dtype) * left
            for left, right in zip(unit_app, unit_structure)
        )
    else:
        projected_app = unit_app
        projected_structure = unit_structure
    for parameter, left, right in zip(
        parameters, projected_app, projected_structure
    ):
        parameter.grad = (0.5 * (left + right)).to(parameter.dtype)
    return {
        "gradient_app_norm": norm_app.detach(),
        "gradient_str_norm": norm_structure.detach(),
        "gradient_cosine": dot.detach(),
        "gradient_conflict": conflict.to(dtype=torch.float32).detach(),
    }


def reference_normalized_smooth_minimax(
    loss_optical,
    loss_structure,
    reference_optical,
    reference_structure,
    temperature,
):
    if reference_optical <= 0.0 or reference_structure <= 0.0:
        raise ValueError("minimax reference losses must be positive")
    if temperature <= 0.0:
        raise ValueError("minimax temperature must be positive")
    relative_optical = loss_optical.float() / float(reference_optical)
    relative_structure = loss_structure.float() / float(reference_structure)
    ratios = torch.stack((relative_optical, relative_structure))
    balanced = temperature * torch.logsumexp(ratios / temperature, dim=0)
    return balanced, relative_optical, relative_structure


def set_train_mode(models, training, active):
    active = set(active)
    for name, model in models.items():
        model.train(training and name in active)


def make_optimizers(models, args):
    optimizers = {}
    for name, model in models.items():
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            continue
        if name == "detail_router":
            fast, slow = srr_optimizer_parameters(model)
            param_groups = []
            if slow:
                param_groups.append(
                    {
                        "params": slow,
                        "lr": args.lr,
                        "group_name": "fusion_slow",
                    }
                )
            if fast:
                param_groups.append(
                    {
                        "params": fast,
                        "lr": args.srr_router_lr,
                        "group_name": "evidence_router_fast",
                    }
                )
            optimizers[name] = torch.optim.Adam(
                param_groups,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            optimizers[name] = torch.optim.Adam(
                parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
    return optimizers


def srr_optimizer_parameters(model):
    fast_prefixes = ("cross_evidence.", "router.")
    fast = []
    slow = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = fast if name.startswith(fast_prefixes) else slow
        target.append(parameter)
    return fast, slow


def clip_model_gradients(name, model, args):
    if name != "detail_router":
        nn.utils.clip_grad_norm_(
            model.parameters(), args.clip_grad, norm_type=2
        )
        return
    fast, slow = srr_optimizer_parameters(model)
    if slow:
        nn.utils.clip_grad_norm_(slow, args.clip_grad, norm_type=2)
    if fast:
        nn.utils.clip_grad_norm_(
            fast, args.srr_router_clip_grad, norm_type=2
        )


def make_schedulers(optimizers, args):
    return {
        name: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step,
            gamma=args.scheduler_gamma,
        )
        for name, optimizer in optimizers.items()
    }


def run_epoch(
    models,
    losses,
    loader,
    device,
    phase,
    args,
    optimizers=None,
    scaler=None,
):
    training = optimizers is not None
    use_pcgrad = (
        training
        and phase == 2
        and args.variant in (
            "loss_matched_baseline_v5",
            "demand_only",
            "residual_only",
            "frequency_matched_baseline",
            "base_optical_detail",
            "base_optical_detail_refine",
        )
    )
    use_minimax = phase == 2 and args.variant in (
        "frequency_minimax_baseline",
        "fixed_line_roa",
        "fixed_line_phase_bridge",
        "srr_detail",
        "r1_spatial_modulation",
    )
    active = active_names(models, phase)
    if training and not active:
        raise RuntimeError(f"No trainable modules are active in phase {phase}")
    set_train_mode(models, training, active)
    totals = []
    component_totals = {}
    optimizer_steps = 0
    start = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        optical, sar, mask, _ = batch
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if training:
            for name in active:
                optimizers[name].zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(
                enabled=(
                    args.amp
                    and device.type == "cuda"
                    and not use_pcgrad
                    and not use_minimax
                )
            ):
                total, components = forward_loss(
                    models, losses, optical, sar, mask, phase, args
                )
                if use_minimax:
                    reference = args.frequency_training_reference
                    (
                        total,
                        relative_optical,
                        relative_structure,
                    ) = reference_normalized_smooth_minimax(
                        components["optical"],
                        components["structure"],
                        reference["optical"],
                        reference["structure"],
                        args.minimax_temperature,
                    )
                    components["relative_optical"] = relative_optical
                    components["relative_structure"] = relative_structure
                    components["selection_score"] = torch.maximum(
                        relative_optical, relative_structure
                    )
                if args.variant == "srr_detail" and args.srr_anchor_weight > 0.0:
                    total = total + (
                        args.srr_anchor_weight * components["srr_anchor_loss"]
                    )
                if args.variant == "r1_spatial_modulation":
                    total = total + (
                        args.r1_dev_weight * components["r1_deviation_loss"]
                    )
            if training:
                if use_pcgrad:
                    parameters = [
                        parameter
                        for name in active
                        for parameter in models[name].parameters()
                        if parameter.requires_grad
                    ]
                    gradient_stats = normalized_pcgrad_backward(
                        components["optical"],
                        components["structure"],
                        parameters,
                    )
                    components.update(gradient_stats)
                    for name in active:
                        clip_model_gradients(name, models[name], args)
                        optimizers[name].step()
                    optimizer_steps += 1
                else:
                    scale_before = scaler.get_scale()
                    scaler.scale(total).backward()
                    for name in active:
                        scaler.unscale_(optimizers[name])
                        clip_model_gradients(name, models[name], args)
                    for name in active:
                        scaler.step(optimizers[name])
                    scaler.update()
                    if scaler.get_scale() >= scale_before:
                        optimizer_steps += 1

        totals.append(float(total.detach().cpu()))
        for name, value in components.items():
            component_totals.setdefault(name, []).append(float(value.detach().cpu()))

        if training and (
            batch_index == 1 or batch_index % args.print_every == 0
        ):
            elapsed = time.time() - start
            print(
                f"  batch {batch_index}/{len(loader)} "
                f"loss={np.mean(totals[-args.print_every:]):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if training and args.max_train_batches and batch_index >= args.max_train_batches:
            break
        if not training and args.max_val_batches and batch_index >= args.max_val_batches:
            break

    result = {"loss": float(np.mean(totals))}
    result.update(
        {name: float(np.mean(values)) for name, values in component_totals.items()}
    )
    result["optimizer_steps"] = optimizer_steps
    return result


def loss_parameter_gradient_norm(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + torch.sum(gradient.float() ** 2)
    return float(torch.sqrt(squared_norm).detach().cpu())


def calibrate_v4_structure_weight(models, losses, loader, device, args):
    if args.variant != "loss_matched_baseline_v4":
        raise ValueError("--calibrate-only requires loss_matched_baseline_v4")
    active = active_names(models, phase=2)
    set_train_mode(models, training=False, active=active)
    parameters = [
        parameter
        for name in active
        for parameter in models[name].parameters()
        if parameter.requires_grad
    ]
    rows = []
    for batch_index, batch in enumerate(loader, start=1):
        optical, sar, mask, _ = batch
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.enable_grad():
            _, components = forward_loss(
                models, losses, optical, sar, mask, phase=2, args=args
            )
            optical_norm = loss_parameter_gradient_norm(
                components["optical"], parameters, retain_graph=True
            )
            structure_norm = loss_parameter_gradient_norm(
                components["structure"], parameters, retain_graph=False
            )
        ratio = optical_norm / max(structure_norm, 1e-12)
        rows.append(
            {
                "batch": batch_index,
                "optical_gradient_norm": optical_norm,
                "structure_gradient_norm": structure_norm,
                "recommended_structure_weight": ratio,
            }
        )
        if batch_index >= args.calibration_batches:
            break
    ratios = np.asarray(
        [row["recommended_structure_weight"] for row in rows], dtype=np.float64
    )
    return {
        "batches": rows,
        "recommended_structure_weight": float(np.median(ratios)),
        "mean_structure_weight": float(np.mean(ratios)),
        "calibration_rule": "median(||grad L_app|| / ||grad L_str||)",
    }


def compute_frequency_training_reference(models, losses, loader, device, args):
    """Measure frozen B0 C0 objectives on deterministic training center crops."""
    active = active_names(models, phase=2)
    set_train_mode(models, training=False, active=active)
    optical_values = []
    structure_values = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            optical, sar, mask, _ = batch
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            _, components = forward_loss(
                models, losses, optical, sar, mask, phase=2, args=args
            )
            optical_values.append(float(components["optical"].cpu()))
            structure_values.append(float(components["structure"].cpu()))
            if args.reference_batches and batch_index >= args.reference_batches:
                break
    if not optical_values:
        raise RuntimeError("frequency minimax calibration loader is empty")
    return {
        "optical": float(np.mean(optical_values)),
        "structure": float(np.mean(structure_values)),
        "batches": len(optical_values),
        "samples": min(len(loader.dataset), len(optical_values) * args.batch_size),
        "source": "deterministic_training_center_crops",
    }


def save_checkpoint(
    path, epoch, best_val_by_phase, models, optimizers, schedulers, scaler, args
):
    checkpoint = {
        "epoch": epoch,
        "best_val_by_phase": best_val_by_phase,
        "CDDF_Encoder": models["encoder"].state_dict(),
        "CDDF_Decoder": models["decoder"].state_dict(),
        "BaseFuseLayer": models["base_fuse"].state_dict(),
        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
        "schedulers": {name: value.state_dict() for name, value in schedulers.items()},
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    if getattr(args, "frequency_training_reference", None):
        checkpoint["frequency_training_reference"] = (
            args.frequency_training_reference
        )
    if "detail_fuse" in models:
        checkpoint["DetailFuseLayer"] = models["detail_fuse"].state_dict()
    if "detail_router" in models:
        checkpoint["SRRDetailFuseLayer"] = models["detail_router"].state_dict()
    if "detail_modulator" in models:
        checkpoint["R1SpatialModulator"] = models["detail_modulator"].state_dict()
    if "detail_injector" in models:
        injector_key = (
            "ResidualDetailInjector"
            if args.variant == "residual_only"
            else "DemandDetailInjector"
        )
        checkpoint[injector_key] = models["detail_injector"].state_dict()
    if "detail_refiner" in models:
        checkpoint["OpticalDetailRefiner"] = models["detail_refiner"].state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    if args.phase1_epochs < 0 or args.phase1_epochs > args.epochs:
        raise ValueError("phase1-epochs must be between 0 and epochs")
    if args.stage2_gradient_weight < 0.0 or args.structure_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.optical_weight < 0.0 or args.optical_gradient_weight < 0.0:
        raise ValueError("Optical loss weights must be non-negative")
    if args.optical_high_weight < 0.0:
        raise ValueError("optical-high-weight must be non-negative")
    if args.line_structure_weight < 0.0:
        raise ValueError("line-structure-weight must be non-negative")
    if args.srr_anchor_weight < 0.0:
        raise ValueError("srr-anchor-weight must be non-negative")
    if not 0.0 <= args.srr_ranking_margin <= 1.0:
        raise ValueError("srr-ranking-margin must be in [0,1]")
    if args.srr_router_lr <= 0.0:
        raise ValueError("srr-router-lr must be positive")
    if args.srr_router_clip_grad <= 0.0:
        raise ValueError("srr-router-clip-grad must be positive")
    if not 0.0 <= args.line_reliability_threshold < 1.0:
        raise ValueError("line-reliability-threshold must be in [0,1)")
    if args.frequency_highpass_sigma <= 0.0 or args.frequency_midpass_sigma <= 0.0:
        raise ValueError("frequency sigmas must be positive")
    if args.minimax_temperature <= 0.0:
        raise ValueError("minimax-temperature must be positive")
    if args.reference_batches < 0:
        raise ValueError("reference-batches must be non-negative")
    if args.optical_l1_tolerance < 0.0 or args.optical_dssim_tolerance < 0.0:
        raise ValueError("Optical loss tolerances must be non-negative")
    if args.collapse_mae_threshold < 0.0:
        raise ValueError("collapse-mae-threshold must be non-negative")
    if not 0.0 < args.structure_percentile <= 100.0:
        raise ValueError("structure-percentile must be in (0, 100]")
    if args.structure_charbonnier_epsilon <= 0.0:
        raise ValueError("structure-charbonnier-epsilon must be positive")
    if args.calibration_batches <= 0:
        raise ValueError("calibration-batches must be positive")
    if args.dry_run and args.calibrate_only:
        raise ValueError("Choose only one of --dry-run and --calibrate-only")
    if not 0.0 < args.sar_edge_percentile <= 100.0:
        raise ValueError("sar-edge-percentile must be in (0, 100]")
    if args.training_policy in ("fusion_only", "router_only") and args.phase1_epochs:
        raise ValueError("fusion_only/router_only requires --phase1-epochs 0")
    if args.variant in (
        "loss_matched_baseline_v5",
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
    ) and args.decomp_weight != 0.0:
        raise ValueError("multi-objective training requires --decomp-weight 0")
    if args.variant in ("demand_only", "residual_only") and not args.selection_reference_config:
        raise ValueError("A1 variants require --selection-reference-config from matching B1-v5")
    if args.variant in ("fixed_line_roa", "fixed_line_phase_bridge", "srr_detail"):
        if not args.structure_calibration:
            raise ValueError("Fixed-evidence variants require --structure-calibration")
        if args.variant == "fixed_line_phase_bridge" and not args.phase_calibration:
            raise ValueError("C3b requires --phase-calibration")
    if args.variant == "srr_detail" and not args.frequency_reference_config:
        raise ValueError("srr_detail requires --frequency-reference-config from C0")
    if args.variant == "r1_spatial_modulation" and not args.frequency_reference_config:
        raise ValueError("r1_spatial_modulation requires --frequency-reference-config from C0")
    if not 0.0 < args.r1_modulation_alpha <= 1.0:
        raise ValueError("r1-modulation-alpha must be in (0,1]")
    if args.r1_dev_weight < 0.0:
        raise ValueError("r1-dev-weight must be non-negative")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run_dir = args.resume.resolve().parent
    else:
        run_name = args.run_name or datetime.datetime.now().strftime(
            "CDDFuse_OS_%Y%m%d_%H%M%S"
        )
        run_dir = args.output_dir / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {run_dir}. Use --resume or another --run-name."
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.split_manifest:
        splits, split_manifest = manifest_pairs(args.data_root, args.split_manifest)
    else:
        splits = split_pairs(
            args.data_root,
            args.test_count,
            args.val_count,
            args.optical_dir,
            args.sar_dir,
        )
        split_manifest = None
    train_dataset = OpticalSARDataset(
        splits["train"],
        patch_size=args.patch_size,
        training=True,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
        limit=args.train_limit,
    )
    val_dataset = OpticalSARDataset(
        splits["val"],
        patch_size=args.patch_size,
        training=False,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
        augment=False,
        limit=args.val_limit,
    )
    reference_dataset = None
    if args.variant in (
        "frequency_minimax_baseline",
        "fixed_line_roa",
        "fixed_line_phase_bridge",
        "srr_detail",
        "r1_spatial_modulation",
    ):
        reference_dataset = OpticalSARDataset(
            splits["train"],
            patch_size=args.patch_size,
            training=False,
            invalid_threshold=args.invalid_threshold,
            min_valid_fraction=args.min_valid_fraction,
            augment=False,
            limit=args.train_limit,
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        persistent_workers=args.num_workers > 0,
    )
    reference_loader = None
    if reference_dataset is not None:
        reference_loader = DataLoader(
            reference_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=worker_seed,
            persistent_workers=args.num_workers > 0,
        )

    models = build_models(device, args.variant)
    if args.variant == "r1_spatial_modulation":
        models["detail_modulator"].alpha.fill_(args.r1_modulation_alpha)
    if args.variant in ("fixed_line_roa", "fixed_line_phase_bridge", "srr_detail"):
        models["line_evidence"] = FixedLineEvidence.from_paths(
            args.structure_calibration,
            args.phase_calibration
            if args.variant == "fixed_line_phase_bridge"
            else None,
        ).to(device)
    configure_trainability(models, args.training_policy)
    losses = OpticalSARLosses().to(device)
    optimizers = make_optimizers(models, args)
    schedulers = make_schedulers(optimizers, args)
    scaler = GradScaler(
        enabled=args.amp and device.type == "cuda", init_scale=1024.0
    )
    start_epoch = 0
    best_val_by_phase = {1: float("inf"), 2: float("inf")}
    loaded_checkpoint = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        loaded_checkpoint = checkpoint
        load_model_weights(models, checkpoint)
        for name, state in checkpoint["optimizers"].items():
            optimizers[name].load_state_dict(state)
        for name, state in checkpoint["schedulers"].items():
            schedulers[name].load_state_dict(state)
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_best = checkpoint.get("best_val_by_phase")
        if saved_best:
            best_val_by_phase = {
                int(key): float(value) for key, value in saved_best.items()
            }
        else:
            # Older checkpoints used a single incomparable best value across phases.
            completed_phase = 1 if start_epoch <= args.phase1_epochs else 2
            best_val_by_phase[completed_phase] = float(
                checkpoint.get("best_val", float("inf"))
            )
        print(f"resumed: {args.resume} at epoch {start_epoch}")
    elif args.init_checkpoint.lower() != "none":
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        load_model_weights(models, checkpoint)
        print(f"initialized from: {args.init_checkpoint}")
    else:
        print("training from scratch")

    frequency_training_reference = None
    if args.variant in (
        "frequency_minimax_baseline",
        "fixed_line_roa",
        "fixed_line_phase_bridge",
        "srr_detail",
        "r1_spatial_modulation",
    ):
        if loaded_checkpoint is not None:
            frequency_training_reference = loaded_checkpoint.get(
                "frequency_training_reference"
            )
            if not frequency_training_reference:
                raise ValueError(
                    "Resumed frequency minimax checkpoint has no training reference"
                )
        elif args.frequency_reference_config:
            with args.frequency_reference_config.open(
                "r", encoding="utf-8"
            ) as handle:
                reference_config = json.load(handle)
            frequency_training_reference = reference_config.get(
                "frequency_training_reference"
            )
            if not frequency_training_reference:
                raise ValueError(
                    "Frequency reference config has no frequency_training_reference"
                )
            if reference_config.get("split_manifest_sha256") != sha256_file(
                args.split_manifest
            ):
                raise ValueError(
                    "Frequency reference config uses a different split manifest"
                )
        else:
            frequency_training_reference = compute_frequency_training_reference(
                models, losses, reference_loader, device, args
            )
        args.frequency_training_reference = frequency_training_reference
        print(
            "frequency minimax training reference: "
            f"optical={frequency_training_reference['optical']:.8f}, "
            f"structure={frequency_training_reference['structure']:.8f}, "
            f"batches={frequency_training_reference['batches']}",
            flush=True,
        )

    v5_validation_reference = None
    selection_reference_source = None
    if args.variant == "loss_matched_baseline_v5":
        reference_stats = run_epoch(
            models, losses, val_loader, device, phase=2, args=args
        )
        v5_validation_reference = {
            "optical": reference_stats["optical"],
            "structure": reference_stats["structure"],
            "output_optical_mae": reference_stats["output_optical_mae"],
        }
        selection_reference_source = "computed_from_current_B0_initialization"
        print(
            "V5 validation reference: "
            f"optical={v5_validation_reference['optical']:.6f}, "
            f"structure={v5_validation_reference['structure']:.6f}, "
            f"output_optical_mae={v5_validation_reference['output_optical_mae']:.6f}",
            flush=True,
        )
    elif args.variant in ("demand_only", "residual_only"):
        with args.selection_reference_config.open("r", encoding="utf-8") as handle:
            reference_config = json.load(handle)
        v5_validation_reference = reference_config.get("v5_validation_reference")
        if not v5_validation_reference:
            raise ValueError("Selection reference config has no v5_validation_reference")
        if reference_config.get("split_manifest_sha256") != sha256_file(
            args.split_manifest
        ):
            raise ValueError("Selection reference uses a different split manifest")
        if int(reference_config.get("val_pairs", -1)) != len(val_dataset):
            raise ValueError("Selection reference validation size does not match A1")
        selection_reference_source = str(args.selection_reference_config.resolve())
        print(
            "A1 reused B1-v5 validation reference: "
            f"optical={v5_validation_reference['optical']:.6f}, "
            f"structure={v5_validation_reference['structure']:.6f}, "
            f"source={selection_reference_source}",
            flush=True,
        )

    config = vars(args).copy()
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    trainable_parameter_counts = {
        name: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for name, model in models.items()
    }
    init_checkpoint_path = Path(args.init_checkpoint)
    compact_manifest_metadata = (
        {key: value for key, value in split_manifest.items() if key != "splits"}
        if split_manifest
        else None
    )
    config.update(
        {
            "device": str(device),
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
            "held_out_test_pairs": len(splits["test"]),
            "split_manifest_metadata": compact_manifest_metadata,
            "split_counts": {name: len(values) for name, values in splits.items()},
            "split_manifest_sha256": (
                sha256_file(args.split_manifest) if args.split_manifest else None
            ),
            "init_checkpoint_sha256": (
                sha256_file(init_checkpoint_path)
                if args.init_checkpoint.lower() != "none"
                else None
            ),
            "parameter_counts": parameter_counts,
            "total_parameters": sum(parameter_counts.values()),
            "trainable_parameter_counts": trainable_parameter_counts,
            "trainable_parameters": sum(trainable_parameter_counts.values()),
            "v5_validation_reference": v5_validation_reference,
            "selection_reference_source": selection_reference_source,
            "frequency_training_reference": frequency_training_reference,
            "stage2_loss": (
                "reference-normalized smooth minimax over [Optical low+high,SAR mid]"
                if args.variant == "frequency_minimax_baseline"
                else (
                    "frequency-decoupled objectives=[Optical low+high,SAR mid], optimizer=normalized_PCGrad"
                    if args.variant in (
                    "frequency_matched_baseline",
                    "base_optical_detail",
                    "base_optical_detail_refine",
                    )
                    else (
                    "A1-prime residual-only detail injection (gamma=0.25) with V5 objectives and normalized_PCGrad"
                    if args.variant == "residual_only"
                    else (
                        "A1 demand-gated detail injection with V5 objectives and normalized_PCGrad"
                        if args.variant == "demand_only"
                        else (
                            "V5 objectives=[L_optical_low,L_structure], optimizer=normalized_PCGrad"
                            if args.variant == "loss_matched_baseline_v5"
                            else (
                                "L_common_v4=L_optical_low+lambda_str*Charbonnier(L_gradient_target)"
                                if args.variant == "loss_matched_baseline_v4"
                                else (
                                    "L_common_v3=lambda_opt*L_opt_constraint+lambda_opt_grad*L_opt_grad+lambda_sar_grad*L_sar_grad"
                                    if args.variant == "loss_matched_baseline_v3"
                                    else (
                                        "L_common_v2=lambda_opt*L_opt_low+lambda_str*L_signed_sparse_str"
                                        if args.variant == "loss_matched_baseline_v2"
                                        else (
                                            "L_common=L_opt+lambda_str*L_str"
                                            if args.variant == "loss_matched_baseline"
                                            else "L1(max_intensity)+stage2_gradient_weight*L1(max_gradient)"
                                        )
                                    )
                                )
                            )
                        )
                    )
                    )
                )
            ),
            "common_loss_version": (
                "v7_frequency_reference_normalized_smooth_minimax"
                if args.variant == "frequency_minimax_baseline"
                else (
                    "v6_frequency_decoupled_normalized_pcgrad"
                    if args.variant in (
                    "frequency_matched_baseline",
                    "base_optical_detail",
                    "base_optical_detail_refine",
                    )
                    else (
                    "v5_two_objective_normalized_pcgrad+A1prime_residual_only"
                    if args.variant == "residual_only"
                    else (
                        "v5_two_objective_normalized_pcgrad+A1_demand_injection"
                        if args.variant == "demand_only"
                        else (
                            "v5_two_objective_normalized_pcgrad"
                            if args.variant == "loss_matched_baseline_v5"
                            else (
                                "v4_plan_aligned_two_term"
                                if args.variant == "loss_matched_baseline_v4"
                                else (
                                    "v3_region_balanced_constrained_optical"
                                    if args.variant == "loss_matched_baseline_v3"
                                    else (
                                        "v2_sparse_signed_gradient"
                                        if args.variant == "loss_matched_baseline_v2"
                                        else (
                                            "v1_normalized_sobel"
                                            if args.variant == "loss_matched_baseline"
                                            else None
                                        )
                                    )
                                )
                            )
                        )
                    )
                    )
                )
            ),
            "run_dir": str(run_dir.resolve()),
        }
    )
    if args.variant in ("fixed_line_roa", "fixed_line_phase_bridge"):
        config["stage2_loss"] = (
            "reference-normalized smooth minimax over "
            "[Optical low+masked high,SAR mid+calibrated line]"
        )
        config["common_loss_version"] = (
            "v8_frequency_selective_fixed_line_minimax"
        )
        config["structure_calibration_sha256"] = sha256_file(
            args.structure_calibration
        )
        config["phase_calibration_sha256"] = (
            sha256_file(args.phase_calibration)
            if args.phase_calibration
            else None
        )
    if args.variant == "srr_detail":
        config["stage2_loss"] = (
            "C0 reference-normalized smooth maximum over "
            "[Optical low+high,SAR mid]"
        )
        config["common_loss_version"] = (
            "v11_srr_sparse_anchors_with_reliability_ranking"
            if args.srr_anchor_weight > 0.0
            else "v9_srr_detail_with_c0_frequency_objectives"
        )
        config["detail_router_version"] = (
            "v2_separate_match_certainty_independent_branch_fusion"
        )
        config["reliability_supervision"] = (
            "balanced sparse anchors: prior>=0.5 positive; "
            "prior<=0.05 and bottom-quartile cross evidence negative; "
            f"positive-negative reliability margin={args.srr_ranking_margin}"
            if args.srr_anchor_weight > 0.0
            else None
        )
        config["optimizer_policy"] = {
            "base_fuse_lr": args.lr,
            "detail_fusion_slow_lr": args.lr,
            "evidence_router_fast_lr": args.srr_router_lr,
            "base_fuse_clip_grad": args.clip_grad,
            "detail_fusion_slow_clip_grad": args.clip_grad,
            "evidence_router_fast_clip_grad": args.srr_router_clip_grad,
            "fast_parameter_prefixes": ["cross_evidence", "router"],
        }
        config["structure_calibration_sha256"] = sha256_file(
            args.structure_calibration
        )
        config["frequency_reference_config_sha256"] = sha256_file(
            args.frequency_reference_config
        )
    if args.variant == "r1_spatial_modulation":
        config["stage2_loss"] = (
            "C0 reference-normalized smooth maximum plus bounded-modulation deviation"
        )
        config["common_loss_version"] = "r1_zero_initialized_spatial_modulation"
        config["r1_policy"] = {
            "frozen_modules": ["encoder", "decoder", "base_fuse", "detail_fuse"],
            "trainable_module": "detail_modulator",
            "alpha": args.r1_modulation_alpha,
            "deviation_weight": args.r1_dev_weight,
            "inputs": ["optical_detail", "sar_detail"],
        }
        config["frequency_reference_config_sha256"] = sha256_file(
            args.frequency_reference_config
        )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    if args.calibrate_only:
        calibration = calibrate_v4_structure_weight(
            models, losses, val_loader, device, args
        )
        with (run_dir / "loss_calibration.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(calibration, handle, ensure_ascii=False, indent=2)
        print(
            "loss calibration complete: "
            f"recommended --structure-weight "
            f"{calibration['recommended_structure_weight']:.6f}, "
            f"details={run_dir / 'loss_calibration.json'}",
            flush=True,
        )
        return

    if args.dry_run:
        print(
            f"dry run complete: variant={args.variant}, train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(splits['test'])}, run={run_dir}",
            flush=True,
        )
        return

    history_path = run_dir / "history.csv"
    extended_history = True
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("epoch", "phase", "lr", "train_loss", "val_loss", "seconds")
                + tuple(f"train_{name}" for name in HISTORY_COMPONENTS)
                + tuple(f"val_{name}" for name in HISTORY_COMPONENTS)
            )
    else:
        with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing_header = next(csv.reader(handle), ())
        extended_history = "train_fusion" in existing_header

    print(
        f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"held-out test={len(splits['test'])}, run={run_dir}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        phase = 1 if epoch < args.phase1_epochs else 2
        epoch_start = time.time()
        print(f"epoch {epoch + 1}/{args.epochs}, phase {phase}", flush=True)
        train_stats = run_epoch(
            models,
            losses,
            train_loader,
            device,
            phase,
            args,
            optimizers=optimizers,
            scaler=scaler,
        )
        val_stats = run_epoch(
            models, losses, val_loader, device, phase, args
        )
        if args.variant in (
            "loss_matched_baseline_v5",
            "demand_only",
            "residual_only",
        ) and phase == 2:
            val_stats["relative_optical"] = (
                val_stats["optical"]
                / max(v5_validation_reference["optical"], 1e-12)
            )
            val_stats["relative_structure"] = (
                val_stats["structure"]
                / max(v5_validation_reference["structure"], 1e-12)
            )
            val_stats["selection_score"] = max(
                val_stats["relative_optical"],
                val_stats["relative_structure"],
            )

        if train_stats["optimizer_steps"] > 0:
            for name in active_names(models, phase):
                schedulers[name].step()
                for group in optimizers[name].param_groups:
                    group["lr"] = max(group["lr"], args.min_lr)
        else:
            print(
                "warning: AMP skipped every optimizer step in this epoch; "
                "learning-rate schedulers were not advanced",
                flush=True,
            )

        seconds = time.time() - epoch_start
        lr = next(iter(optimizers.values())).param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1}: train={train_stats['loss']:.6f}, "
            f"val={val_stats['loss']:.6f}, lr={lr:.2e}, {seconds:.1f}s",
            flush=True,
        )
        if args.variant == "srr_detail":
            router_groups = optimizers["detail_router"].param_groups
            slow_lr = next(
                group["lr"]
                for group in router_groups
                if group.get("group_name") == "fusion_slow"
            )
            fast_lr = next(
                group["lr"]
                for group in router_groups
                if group.get("group_name") == "evidence_router_fast"
            )
            print(
                f"  SRR lr slow={slow_lr:.2e}, fast={fast_lr:.2e}; "
                f"clip slow={args.clip_grad:.3g}, "
                f"fast={args.srr_router_clip_grad:.3g}",
                flush=True,
            )
        if args.variant in (
            "loss_matched_baseline_v5",
            "demand_only",
            "residual_only",
        ) and phase == 2:
            print(
                "  V5 relative validation: "
                f"optical={val_stats['relative_optical']:.6f}, "
                f"structure={val_stats['relative_structure']:.6f}, "
                f"minimax={val_stats['selection_score']:.6f}",
                flush=True,
            )
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            row = (
                epoch + 1,
                phase,
                f"{lr:.10g}",
                f"{train_stats['loss']:.8f}",
                f"{val_stats['loss']:.8f}",
                f"{seconds:.3f}",
            )
            if extended_history:
                row += tuple(
                    f"{train_stats[name]:.8f}" if name in train_stats else ""
                    for name in HISTORY_COMPONENTS
                )
                row += tuple(
                    f"{val_stats[name]:.8f}" if name in val_stats else ""
                    for name in HISTORY_COMPONENTS
                )
            csv.writer(handle).writerow(row)

        identity_collapsed = (
            args.variant in (
                "loss_matched_baseline_v2",
                "loss_matched_baseline_v3",
                "loss_matched_baseline_v4",
                "loss_matched_baseline_v5",
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
            )
            and val_stats.get("output_optical_mae", float("inf"))
            < args.collapse_mae_threshold
        )
        if identity_collapsed:
            print(
                "warning: validation output/Optical MAE "
                f"{val_stats['output_optical_mae']:.6f} is below collapse threshold "
                f"{args.collapse_mae_threshold:.6f}; checkpoint is not eligible as best",
                flush=True,
            )
        selection_value = (
            val_stats["selection_score"]
            if args.variant in (
                "loss_matched_baseline_v5",
                "demand_only",
                "residual_only",
            ) and phase == 2
            else val_stats["loss"]
        )
        is_best = (
            not identity_collapsed
            and selection_value < best_val_by_phase[phase]
        )
        if is_best:
            best_val_by_phase[phase] = selection_value
        save_checkpoint(
            run_dir / "last.pth",
            epoch,
            best_val_by_phase,
            models,
            optimizers,
            schedulers,
            scaler,
            args,
        )
        if is_best:
            save_checkpoint(
                run_dir / f"best_phase{phase}.pth",
                epoch,
                best_val_by_phase,
                models,
                optimizers,
                schedulers,
                scaler,
                args,
            )
            if phase == 2:
                save_checkpoint(
                    run_dir / "best.pth",
                    epoch,
                    best_val_by_phase,
                    models,
                    optimizers,
                    schedulers,
                    scaler,
                    args,
                )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch + 1:03d}.pth",
                epoch,
                best_val_by_phase,
                models,
                optimizers,
                schedulers,
                scaler,
                args,
            )

    print(f"training complete: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
