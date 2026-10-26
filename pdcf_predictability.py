"""P1 utilities for balanced cross-modal feature prediction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdcf_modules import BranchNormalizer, CrossModalPredictor


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(1)
    expanded = expanded.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    beta: float = 0.1,
) -> torch.Tensor:
    values = F.smooth_l1_loss(
        prediction, target, reduction="none", beta=beta
    )
    return masked_mean(values, mask)


def masked_nrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    mse = masked_mean((prediction - target).square(), mask)
    target_power = masked_mean(target.square(), mask)
    return torch.sqrt(mse / target_power.clamp_min(epsilon))


class FeatureStatisticsAccumulator:
    """Pooled Optical/SAR per-channel statistics for Base and Detail."""

    def __init__(self, channels: int = 64) -> None:
        self.channels = channels
        self.base_sum = torch.zeros(channels, dtype=torch.float64)
        self.base_square_sum = torch.zeros(channels, dtype=torch.float64)
        self.detail_sum = torch.zeros(channels, dtype=torch.float64)
        self.detail_square_sum = torch.zeros(channels, dtype=torch.float64)
        self.count = 0.0

    @torch.no_grad()
    def update(
        self,
        optical_base: torch.Tensor,
        optical_detail: torch.Tensor,
        sar_base: torch.Tensor,
        sar_detail: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        tensors = (optical_base, optical_detail, sar_base, sar_detail)
        if any(tensor.ndim != 4 for tensor in tensors):
            raise ValueError("Feature statistics require NCHW tensors")
        if any(tensor.shape[1] != self.channels for tensor in tensors):
            raise ValueError(f"Expected {self.channels} feature channels")
        if mask is None:
            weights = torch.ones_like(optical_base[:, :1])
        else:
            weights = mask.to(optical_base.dtype)
            if weights.shape != optical_base[:, :1].shape:
                raise ValueError(
                    f"Mask shape must be {optical_base[:, :1].shape}, got {weights.shape}"
                )
        pooled_count = float(weights.sum().item()) * 2.0
        if pooled_count <= 0:
            return
        weights64 = weights.double()
        for name, first, second in (
            ("base", optical_base, sar_base),
            ("detail", optical_detail, sar_detail),
        ):
            values = torch.cat((first, second), dim=0).double().cpu()
            duplicated_weights = torch.cat((weights64, weights64), dim=0).cpu()
            channel_sum = (values * duplicated_weights).sum(dim=(0, 2, 3))
            square_sum = (
                values.square() * duplicated_weights
            ).sum(dim=(0, 2, 3))
            getattr(self, f"{name}_sum").add_(channel_sum)
            getattr(self, f"{name}_square_sum").add_(square_sum)
        self.count += pooled_count

    def finalize(self, epsilon: float = 1e-6) -> Dict[str, torch.Tensor]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty feature statistics")
        result = {}
        for name in ("base", "detail"):
            mean = getattr(self, f"{name}_sum") / self.count
            second = getattr(self, f"{name}_square_sum") / self.count
            variance = (second - mean.square()).clamp_min(epsilon ** 2)
            result[f"{name}_mean"] = mean.float().view(1, self.channels, 1, 1)
            result[f"{name}_scale"] = variance.sqrt().float().view(
                1, self.channels, 1, 1
            )
        result["count"] = torch.tensor(self.count, dtype=torch.float64)
        return result


class BidirectionalPredictability(nn.Module):
    """Normalized bidirectional predictors used in P1."""

    def __init__(
        self,
        channels: int = 64,
        hidden_channels: int = 32,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        feature_channels = 2 * channels
        self.normalizer = BranchNormalizer(channels)
        self.optical_to_sar = CrossModalPredictor(
            feature_channels, hidden_channels, blocks
        )
        self.sar_to_optical = CrossModalPredictor(
            feature_channels, hidden_channels, blocks
        )

    def set_statistics(self, statistics: Dict[str, torch.Tensor]) -> None:
        self.normalizer.set_statistics(
            statistics["base_mean"],
            statistics["base_scale"],
            statistics["detail_mean"],
            statistics["detail_scale"],
        )

    def normalized_features(
        self,
        optical_base: torch.Tensor,
        optical_detail: torch.Tensor,
        sar_base: torch.Tensor,
        sar_detail: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        optical_parts = self.normalizer(optical_base, optical_detail)
        sar_parts = self.normalizer(sar_base, sar_detail)
        return torch.cat(optical_parts, dim=1), torch.cat(sar_parts, dim=1)

    def forward(
        self,
        optical_base: torch.Tensor,
        optical_detail: torch.Tensor,
        sar_base: torch.Tensor,
        sar_detail: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        optical, sar = self.normalized_features(
            optical_base, optical_detail, sar_base, sar_detail
        )
        predictable_sar = self.optical_to_sar(optical)
        predictable_optical = self.sar_to_optical(sar)
        return {
            "optical": optical,
            "sar": sar,
            "predictable_optical": predictable_optical,
            "predictable_sar": predictable_sar,
            "residual_optical": optical - predictable_optical,
            "residual_sar": sar - predictable_sar,
        }


class BidirectionalLinearBaseline(nn.Module):
    """Matched 1x1 cross-modal baseline required by the P1 protocol."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.optical_to_sar = nn.Conv2d(channels, channels, kernel_size=1)
        self.sar_to_optical = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self, optical: torch.Tensor, sar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "predictable_sar": self.optical_to_sar(optical),
            "predictable_optical": self.sar_to_optical(sar),
        }


def balanced_prediction_loss(
    outputs: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor],
    channels: int = 64,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    optical = outputs["optical"].detach()
    sar = outputs["sar"].detach()
    predictable_optical = outputs["predictable_optical"]
    predictable_sar = outputs["predictable_sar"]
    terms = {
        "o2s_base": masked_smooth_l1(
            predictable_sar[:, :channels], sar[:, :channels], mask, beta
        ),
        "o2s_detail": masked_smooth_l1(
            predictable_sar[:, channels:], sar[:, channels:], mask, beta
        ),
        "s2o_base": masked_smooth_l1(
            predictable_optical[:, :channels], optical[:, :channels], mask, beta
        ),
        "s2o_detail": masked_smooth_l1(
            predictable_optical[:, channels:], optical[:, channels:], mask, beta
        ),
    }
    terms["total"] = torch.stack(tuple(terms.values())).mean()
    return terms


def prediction_metrics(
    outputs: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor],
    channels: int = 64,
) -> Dict[str, torch.Tensor]:
    optical = outputs["optical"].detach()
    sar = outputs["sar"].detach()
    predicted_optical = outputs["predictable_optical"].detach()
    predicted_sar = outputs["predictable_sar"].detach()
    metrics = {}
    for direction, prediction, target in (
        ("o2s", predicted_sar, sar),
        ("s2o", predicted_optical, optical),
    ):
        metrics[f"{direction}_base_nrmse"] = masked_nrmse(
            prediction[:, :channels], target[:, :channels], mask
        )
        metrics[f"{direction}_detail_nrmse"] = masked_nrmse(
            prediction[:, channels:], target[:, channels:], mask
        )
        # With pooled standardized statistics, zero is the fixed mean baseline.
        metrics[f"{direction}_mean_base_nrmse"] = masked_nrmse(
            torch.zeros_like(target[:, :channels]),
            target[:, :channels],
            mask,
        )
        metrics[f"{direction}_mean_detail_nrmse"] = masked_nrmse(
            torch.zeros_like(target[:, channels:]),
            target[:, channels:],
            mask,
        )
    metrics["residual_optical_rms"] = torch.sqrt(
        masked_mean((optical - predicted_optical).square(), mask)
    )
    metrics["residual_sar_rms"] = torch.sqrt(
        masked_mean((sar - predicted_sar).square(), mask)
    )
    return metrics


def residual_intervention(
    residual: torch.Tensor,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Controlled replacements used by the P1 residual audit."""

    if mode == "original":
        return residual
    if mode == "zero":
        return torch.zeros_like(residual)
    if mode == "spatial_shuffle":
        flat = residual.flatten(2)
        permutation = torch.randperm(
            flat.shape[-1], device=flat.device, generator=generator
        )
        return flat[:, :, permutation].view_as(residual)
    if mode == "batch_swap":
        if residual.shape[0] < 2:
            raise ValueError("batch_swap requires at least two samples")
        return residual.roll(shifts=1, dims=0)
    if mode == "rms_noise":
        noise = torch.randn(
            residual.shape,
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        )
        residual_rms = residual.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        noise_rms = noise.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return noise * residual_rms / noise_rms.clamp_min(1e-8)
    raise ValueError(f"Unknown residual intervention: {mode}")

