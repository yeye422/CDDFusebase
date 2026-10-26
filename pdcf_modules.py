"""Core modules for Predictability-Driven Decomposition and Complementary Fusion.

The modules in this file are deliberately independent from the legacy training
entry points.  P0 requires an exact C0 identity at initialization; therefore
the only path that can change the C0 Base/Detail tensors is the zero-initialized
``JointProjection`` output.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    """Small spatial residual block used by predictors and fusion paths."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.body(inputs)


def _zero_module(module: nn.Module) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FeatureRefactor(nn.Module):
    """Invertible additive Base/Detail coupling with exact identity init."""

    def __init__(self, channels: int = 64, hidden_channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.detail_to_base = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        self.base_to_detail = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        _zero_module(self.detail_to_base[-1])
        _zero_module(self.base_to_detail[-1])

    def forward(
        self, base: torch.Tensor, detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate(base, detail)
        new_base = base + self.detail_to_base(detail)
        new_detail = detail + self.base_to_detail(new_base)
        return new_base, new_detail

    def inverse(
        self, new_base: torch.Tensor, new_detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate(new_base, new_detail)
        detail = new_detail - self.base_to_detail(new_base)
        base = new_base - self.detail_to_base(detail)
        return base, detail

    def _validate(self, base: torch.Tensor, detail: torch.Tensor) -> None:
        if base.shape != detail.shape:
            raise ValueError(
                f"Base/Detail shapes must match, got {base.shape} and {detail.shape}"
            )
        if base.ndim != 4 or base.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW tensors with {self.channels} channels, got {base.shape}"
            )


class BranchNormalizer(nn.Module):
    """Frozen per-channel normalization with explicit, checkpointed statistics."""

    def __init__(self, channels: int = 64, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon
        self.register_buffer("base_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("base_scale", torch.ones(1, channels, 1, 1))
        self.register_buffer("detail_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("detail_scale", torch.ones(1, channels, 1, 1))
        self.register_buffer("statistics_ready", torch.tensor(False))

    @torch.no_grad()
    def set_statistics(
        self,
        base_mean: torch.Tensor,
        base_scale: torch.Tensor,
        detail_mean: torch.Tensor,
        detail_scale: torch.Tensor,
    ) -> None:
        values = (base_mean, base_scale, detail_mean, detail_scale)
        names = ("base_mean", "base_scale", "detail_mean", "detail_scale")
        for name, value in zip(names, values):
            expected = (1, self.channels, 1, 1)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        self.base_mean.copy_(base_mean)
        self.base_scale.copy_(base_scale.clamp_min(self.epsilon))
        self.detail_mean.copy_(detail_mean)
        self.detail_scale.copy_(detail_scale.clamp_min(self.epsilon))
        self.statistics_ready.fill_(True)

    def forward(
        self, base: torch.Tensor, detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_base = (base - self.base_mean) / self.base_scale
        normalized_detail = (detail - self.detail_mean) / self.detail_scale
        return normalized_base, normalized_detail

    def inverse(
        self, normalized_base: torch.Tensor, normalized_detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = normalized_base * self.base_scale + self.base_mean
        detail = normalized_detail * self.detail_scale + self.detail_mean
        return base, detail


class CrossModalPredictor(nn.Module):
    """Capacity-limited predictor mapping one modality feature space to another."""

    def __init__(
        self, channels: int = 128, hidden_channels: int = 64, blocks: int = 2
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        ]
        layers.extend(ConvResidualBlock(hidden_channels) for _ in range(blocks))
        layers.append(nn.Conv2d(hidden_channels, channels, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RoleFusionPath(nn.Module):
    """Fusion path shared in shape, but not weights, by C and R representations."""

    def __init__(
        self,
        feature_channels: int = 128,
        work_channels: int = 64,
        condition_channels: int = 0,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.condition_channels = condition_channels
        # Two role tensors plus both complete modality contexts.
        input_channels = 4 * feature_channels + condition_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, work_channels, kernel_size=1),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            ConvResidualBlock(work_channels),
            ConvResidualBlock(work_channels),
        )

    def forward(
        self,
        first_role: torch.Tensor,
        second_role: torch.Tensor,
        optical_context: torch.Tensor,
        sar_context: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tensors = (first_role, second_role, optical_context, sar_context)
        reference_shape = optical_context.shape
        if any(tensor.shape != reference_shape for tensor in tensors):
            raise ValueError("Role and context tensors must have identical shapes")
        if self.condition_channels:
            if condition is None:
                raise ValueError("This fusion path requires a condition tensor")
            if condition.shape[1] != self.condition_channels:
                raise ValueError(
                    f"Expected {self.condition_channels} condition channels, "
                    f"got {condition.shape[1]}"
                )
            if condition.shape[-2:] != reference_shape[-2:]:
                condition = F.interpolate(
                    condition,
                    size=reference_shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            tensors = tensors + (condition,)
        elif condition is not None:
            raise ValueError("Condition supplied to a path configured without one")
        projected = self.input_projection(torch.cat(tensors, dim=1))
        return self.spatial(projected)


class JointProjection(nn.Module):
    """Map both role paths to zero-initialized Base and Detail corrections."""

    def __init__(self, work_channels: int = 64, output_channels: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2 * work_channels, 2 * work_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * work_channels, 2 * output_channels, kernel_size=1),
        )
        _zero_module(self.body[-1])
        self.output_channels = output_channels

    def forward(
        self, correspondence: torch.Tensor, complementary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        corrections = self.body(torch.cat((correspondence, complementary), dim=1))
        return corrections.split(self.output_channels, dim=1)


class PDCFCore(nn.Module):
    """PDCF feature transformation, independent of the legacy C0 backbone."""

    def __init__(
        self,
        channels: int = 64,
        refactor_hidden_channels: int = 32,
        predictor_hidden_channels: int = 32,
        work_channels: int = 32,
        spd_condition_channels: int = 0,
        share_refactor: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.feature_channels = 2 * channels
        self.optical_refactor = FeatureRefactor(
            channels, refactor_hidden_channels
        )
        self.sar_refactor = (
            self.optical_refactor
            if share_refactor
            else FeatureRefactor(channels, refactor_hidden_channels)
        )
        self.normalizer = BranchNormalizer(channels)
        self.optical_to_sar = CrossModalPredictor(
            self.feature_channels, predictor_hidden_channels
        )
        self.sar_to_optical = CrossModalPredictor(
            self.feature_channels, predictor_hidden_channels
        )
        self.correspondence_adapter_optical = nn.Conv2d(
            self.feature_channels, self.feature_channels, kernel_size=1
        )
        self.correspondence_adapter_sar = nn.Conv2d(
            self.feature_channels, self.feature_channels, kernel_size=1
        )
        self.residual_adapter_optical = nn.Conv2d(
            self.feature_channels, self.feature_channels, kernel_size=1
        )
        self.residual_adapter_sar = nn.Conv2d(
            self.feature_channels, self.feature_channels, kernel_size=1
        )
        self.correspondence_fusion = RoleFusionPath(
            self.feature_channels, work_channels
        )
        self.complementary_fusion = RoleFusionPath(
            self.feature_channels, work_channels, spd_condition_channels
        )
        self.joint_projection = JointProjection(work_channels, channels)

    def forward(
        self,
        optical_base: torch.Tensor,
        optical_detail: torch.Tensor,
        sar_base: torch.Tensor,
        sar_detail: torch.Tensor,
        spd_condition: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        optical_base_r, optical_detail_r = self.optical_refactor(
            optical_base, optical_detail
        )
        sar_base_r, sar_detail_r = self.sar_refactor(sar_base, sar_detail)
        optical_base_n, optical_detail_n = self.normalizer(
            optical_base_r, optical_detail_r
        )
        sar_base_n, sar_detail_n = self.normalizer(sar_base_r, sar_detail_r)
        optical = torch.cat((optical_base_n, optical_detail_n), dim=1)
        sar = torch.cat((sar_base_n, sar_detail_n), dim=1)

        predictable_sar = self.optical_to_sar(optical)
        predictable_optical = self.sar_to_optical(sar)
        residual_sar = sar - predictable_sar
        residual_optical = optical - predictable_optical

        correspondence = self.correspondence_fusion(
            self.correspondence_adapter_optical(predictable_optical),
            self.correspondence_adapter_sar(predictable_sar),
            optical,
            sar,
        )
        complementary = self.complementary_fusion(
            self.residual_adapter_optical(residual_optical),
            self.residual_adapter_sar(residual_sar),
            optical,
            sar,
            spd_condition,
        )
        delta_base, delta_detail = self.joint_projection(
            correspondence, complementary
        )
        if not return_diagnostics:
            return delta_base, delta_detail
        diagnostics: Dict[str, torch.Tensor] = {
            "optical_reorganized": optical,
            "sar_reorganized": sar,
            "predictable_optical": predictable_optical,
            "predictable_sar": predictable_sar,
            "residual_optical": residual_optical,
            "residual_sar": residual_sar,
            "correspondence_feature": correspondence,
            "complementary_feature": complementary,
            "delta_base": delta_base,
            "delta_detail": delta_detail,
        }
        return delta_base, delta_detail, diagnostics
