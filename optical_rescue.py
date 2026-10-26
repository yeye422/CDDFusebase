"""Optical Detail rescue anchored to a frozen Detail-only SPD fusion path."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedDepthwiseBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation,
                dilation=dilation, groups=channels, bias=False
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, features):
        return features + self.block(features)


class OpticalRescueAdapter(nn.Module):
    def __init__(
        self, channels=64, hidden_channels=32,
        max_relative_correction=0.005, epsilon=1e-6
    ):
        super().__init__()
        self.max_relative_correction = float(max_relative_correction)
        self.epsilon = float(epsilon)
        self.input = nn.Conv2d(channels * 4, hidden_channels, 1)
        self.body = nn.Sequential(
            DilatedDepthwiseBlock(hidden_channels, 1),
            DilatedDepthwiseBlock(hidden_channels, 2),
        )
        self.delta = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self, optical_detail, clean_sar_detail, fused_detail,
        optical_reconstruction_error, return_map=False
    ):
        evidence = torch.cat(
            (
                optical_detail,
                clean_sar_detail,
                fused_detail,
                optical_reconstruction_error.abs(),
            ),
            dim=1,
        )
        proposal = torch.tanh(self.delta(self.body(F.gelu(self.input(evidence)))))
        channel_rms = fused_detail.detach().square().mean(
            dim=(2, 3), keepdim=True
        ).sqrt().clamp_min(self.epsilon)
        correction = self.max_relative_correction * channel_rms * proposal
        rescued = fused_detail + correction
        diagnostics = {
            "rescue_abs_mean": correction.abs().mean(),
            "rescue_rms": correction.square().mean().sqrt(),
            "rescue_relative_rms": correction.square().mean().sqrt()
            / fused_detail.detach().square().mean().sqrt().clamp_min(self.epsilon),
            "rescue_output_deviation_rms": (rescued - fused_detail).square().mean().sqrt(),
        }
        if return_map:
            diagnostics["rescue_map"] = correction
        return rescued, diagnostics


class RescueProbeHeads(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.optical = nn.Conv2d(channels, channels, 1)
        self.sar = nn.Conv2d(channels, channels, 1)
        self.reset_identity()

    def reset_identity(self):
        with torch.no_grad():
            for layer in (self.optical, self.sar):
                layer.weight.zero_(); layer.bias.zero_()
                index = torch.arange(layer.out_channels)
                layer.weight[index, index, 0, 0] = 1.0

    def forward(self, fused_detail):
        return self.optical(fused_detail), self.sar(fused_detail)


def normalized_l1(prediction, target, epsilon=1e-6):
    target_rms = target.detach().square().mean(
        dim=(1, 2, 3), keepdim=True
    ).sqrt().clamp_min(epsilon)
    return torch.mean(torch.abs(prediction - target.detach()) / target_rms)
