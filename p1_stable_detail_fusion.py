"""P1 provenance-preserving Detail fusion with stable SAR injection."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels, requested=8):
    groups = min(int(channels), int(requested))
    while channels % groups:
        groups -= 1
    return max(groups, 1)


class RouterResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, features):
        return features + self.block(features)


def normalize_channels(features, epsilon=1e-6):
    mean = features.mean(dim=(2, 3), keepdim=True)
    variance = (features - mean).square().mean(dim=(2, 3), keepdim=True)
    return (features - mean) / torch.sqrt(variance + epsilon)


def normalized_feature_sensitivity(first, second, epsilon=1e-5):
    numerator = (first - second).abs().mean(dim=1, keepdim=True)
    denominator = 0.5 * (first.abs() + second.abs()).mean(dim=1, keepdim=True)
    return numerator / denominator.clamp_min(epsilon)


def stability_teacher(
    first,
    second,
    valid=None,
    low_quantile=0.20,
    high_quantile=0.80,
    smooth_kernel=5,
):
    """Convert paired perturbation sensitivity into a detached stable score."""

    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("stability quantiles must satisfy 0 <= low < high <= 1")
    sensitivity = normalized_feature_sensitivity(first, second)
    if valid is None:
        valid = torch.ones_like(sensitivity)
    else:
        valid = F.interpolate(valid.float(), size=sensitivity.shape[-2:], mode="nearest")
    scores = []
    for sample, sample_valid in zip(sensitivity, valid):
        selected = sample[sample_valid > 0.5]
        if not selected.numel():
            raise RuntimeError("stability teacher received an empty valid mask")
        low = torch.quantile(selected.float(), low_quantile).to(sample.dtype)
        high = torch.quantile(selected.float(), high_quantile).to(sample.dtype)
        score = 1.0 - ((sample - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
        scores.append(score)
    teacher = torch.stack(scores)
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be odd")
        teacher = F.avg_pool2d(
            teacher,
            smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2,
            count_include_pad=False,
        )
    return teacher.detach() * valid


class StableSARDetailRouter(nn.Module):
    """Inject the frozen joint expert into an optical expert using a learned gate."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        initial_gate=0.995,
        channel_reduction=4,
    ):
        super().__init__()
        if not 0.5 < initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0.5, 1.0)")
        reduced = max(channels // int(channel_reduction), 1)
        self.input = nn.Conv2d(channels * 4, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[RouterResidualBlock(hidden_channels) for _ in range(int(num_blocks))]
        )
        self.spatial_gate = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, reduced, 1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, 1),
        )
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        initial_suppression = 2.0 * (1.0 - float(initial_gate))
        self.suppression_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_suppression / (1.0 - initial_suppression))
            )
        )

    def predict_gate(self, optical_detail, sar_detail):
        optical = normalize_channels(optical_detail)
        sar = normalize_channels(sar_detail)
        evidence = torch.cat(
            (optical, sar, torch.abs(optical - sar), optical * sar), dim=1
        )
        hidden = self.blocks(F.gelu(self.input(evidence)))
        spatial_logit = self.spatial_gate(hidden)
        channel_logit = self.channel_gate(hidden)
        spatial = torch.sigmoid(spatial_logit)
        channel = torch.sigmoid(channel_logit)
        reliability = torch.sigmoid(spatial_logit + channel_logit)
        suppression_cap = torch.sigmoid(self.suppression_logit)
        gate = 1.0 - suppression_cap * (1.0 - reliability)
        return gate, reliability, spatial, channel, suppression_cap

    def forward(
        self,
        optical_detail,
        sar_detail,
        optical_expert,
        joint_expert,
        return_maps=False,
    ):
        gate, reliability, spatial, channel, suppression_cap = self.predict_gate(
            optical_detail, sar_detail
        )
        sar_contribution = joint_expert - optical_expert
        fused = optical_expert + gate * sar_contribution
        diagnostics = {
            "p1_gate_mean": gate.mean(),
            "p1_gate_std": gate.std(unbiased=False),
            "p1_spatial_gate_mean": spatial.mean(),
            "p1_channel_gate_mean": channel.mean(),
            "p1_reliability_mean": reliability.mean(),
            "p1_reliability_std": reliability.std(unbiased=False),
            "p1_suppression_cap": suppression_cap,
            "p1_sar_contribution_rms": sar_contribution.square().mean().sqrt(),
            "p1_injected_sar_rms": (gate * sar_contribution).square().mean().sqrt(),
            "p1_baseline_deviation_rms": (fused - joint_expert).square().mean().sqrt(),
        }
        if return_maps:
            diagnostics.update(
                {
                    "p1_gate_map": gate,
                    "p1_reliability_map": reliability,
                    "p1_spatial_gate_map": spatial,
                    "p1_channel_gate_map": channel,
                }
            )
        return fused, diagnostics
