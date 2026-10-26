"""P2-A zero-initialized source-preserving Detail residual fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels, requested=8):
    groups = min(int(channels), int(requested))
    while channels % groups:
        groups -= 1
    return max(groups, 1)


def normalize_channels(features, epsilon=1e-6):
    mean = features.mean(dim=(2, 3), keepdim=True)
    variance = (features - mean).square().mean(dim=(2, 3), keepdim=True)
    return (features - mean) / torch.sqrt(variance + epsilon)


class SourceResidualBlock(nn.Module):
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


class SourcePreservingDetailAdapter(nn.Module):
    """Learn a small correction in the frozen B0 fused-Detail coordinates."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_relative_correction=0.03,
        epsilon=1e-6,
    ):
        super().__init__()
        self.max_relative_correction = float(max_relative_correction)
        self.epsilon = float(epsilon)
        self.input = nn.Conv2d(channels * 4, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[SourceResidualBlock(hidden_channels) for _ in range(int(num_blocks))]
        )
        self.delta = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, optical_detail, sar_detail, b0_fused_detail, return_map=False):
        optical = normalize_channels(optical_detail)
        sar = normalize_channels(sar_detail)
        evidence = torch.cat(
            (optical, sar, torch.abs(optical - sar), optical * sar), dim=1
        )
        hidden = self.blocks(F.gelu(self.input(evidence)))
        proposal = torch.tanh(self.delta(hidden))
        channel_rms = b0_fused_detail.detach().square().mean(
            dim=(2, 3), keepdim=True
        ).sqrt().clamp_min(self.epsilon)
        correction = self.max_relative_correction * channel_rms * proposal
        fused = b0_fused_detail + correction
        diagnostics = {
            "p2a_correction_abs_mean": correction.abs().mean(),
            "p2a_correction_rms": correction.square().mean().sqrt(),
            "p2a_relative_correction": (
                correction.square().mean().sqrt()
                / b0_fused_detail.detach().square().mean().sqrt().clamp_min(self.epsilon)
            ),
            "p2a_output_deviation_rms": (fused - b0_fused_detail).square().mean().sqrt(),
        }
        if return_map:
            diagnostics["p2a_correction_map"] = correction
        return fused, diagnostics


class PseudoSensingHeads(nn.Module):
    """Shallow calibrated probes for Optical Detail and stable SAR Detail."""

    def __init__(self, channels=64):
        super().__init__()
        self.optical = nn.Conv2d(channels, channels, 1)
        self.sar = nn.Conv2d(channels, channels, 1)
        self.reset_identity()

    def reset_identity(self):
        with torch.no_grad():
            self.optical.weight.zero_()
            self.sar.weight.zero_()
            diagonal = torch.arange(self.optical.out_channels)
            self.optical.weight[diagonal, diagonal, 0, 0] = 1.0
            self.sar.weight[diagonal, diagonal, 0, 0] = 1.0
            self.optical.bias.zero_()
            self.sar.bias.zero_()

    def forward(self, fused_detail):
        return self.optical(fused_detail), self.sar(fused_detail)


def weighted_normalized_l1(prediction, target, weight, epsilon=1e-6):
    target_rms = target.detach().square().mean(
        dim=(1, 2, 3), keepdim=True
    ).sqrt().clamp_min(epsilon)
    values = torch.abs(prediction - target.detach()) / target_rms
    expanded = weight
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(1)
    expanded = expanded.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)
