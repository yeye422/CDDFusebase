"""Exact two-source provenance decomposition for frozen CDDFuse fusion blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_normalize(features, epsilon=1e-6):
    scale = features.detach().square().mean(
        dim=(2, 3), keepdim=True
    ).sqrt().clamp_min(epsilon)
    return features / scale


def contribution_energy(features, epsilon=1e-12):
    return torch.sqrt(features.float().square().mean() + epsilon).to(features.dtype)


@torch.no_grad()
def shapley_decompose(fusion, optical, sar):
    """Exact symmetric Shapley decomposition of a nonlinear two-input sum fusion.

    For f(x) = fusion(x), decompose f(optical + sar) into a zero-input
    baseline plus optical and SAR contributions. Half of the nonlinear
    interaction is assigned to each modality.
    """
    zero = torch.zeros_like(optical)
    fused_zero = fusion(zero)
    fused_optical = fusion(optical)
    fused_sar = fusion(sar)
    fused_joint = fusion(optical + sar)
    interaction = fused_joint - fused_optical - fused_sar + fused_zero
    optical_contribution = fused_optical - fused_zero + 0.5 * interaction
    sar_contribution = fused_sar - fused_zero + 0.5 * interaction
    reconstruction = fused_zero + optical_contribution + sar_contribution
    return {
        "zero": fused_zero,
        "optical": optical_contribution,
        "sar": sar_contribution,
        "interaction": interaction,
        "joint": fused_joint,
        "reconstruction": reconstruction,
    }


class ProvenanceGate(nn.Module):
    """Bounded source-specific gain corrections, initialized as exact identity."""

    def __init__(self, channels=64, hidden_channels=32, max_gain_delta=0.10):
        super().__init__()
        self.max_gain_delta = float(max_gain_delta)
        self.input = nn.Conv2d(channels * 4, hidden_channels, 1)
        self.body = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, 3, padding=1,
                groups=hidden_channels, bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, optical, sar, interaction):
        evidence = torch.cat((
            channel_normalize(optical),
            channel_normalize(sar),
            channel_normalize(interaction),
            channel_normalize(optical - sar),
        ), dim=1)
        deltas = self.max_gain_delta * torch.tanh(
            self.output(self.body(F.gelu(self.input(evidence))))
        )
        optical_delta, sar_delta = deltas.split(1, dim=1)
        return optical_delta, sar_delta


class ProvenanceAwareFusion(nn.Module):
    """Wrap a frozen CDDFuse block with exact source attribution and routing."""

    def __init__(
        self, fusion, channels=64, hidden_channels=32,
        max_gain_delta=0.10,
    ):
        super().__init__()
        self.fusion = fusion
        self.gate = ProvenanceGate(
            channels=channels,
            hidden_channels=hidden_channels,
            max_gain_delta=max_gain_delta,
        )
        self.fusion.eval()
        for parameter in self.fusion.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.fusion.eval()
        return self

    def forward(self, optical, sar, return_maps=False):
        parts = shapley_decompose(self.fusion, optical, sar)
        optical_delta, sar_delta = self.gate(
            parts["optical"], parts["sar"], parts["interaction"]
        )
        # Express routing as a correction around the original joint tensor.
        # With zero-initialized gates this returns the exact same tensor object
        # values as the legacy fusion path, avoiding reassociation roundoff.
        fused = (
            parts["joint"]
            + optical_delta * parts["optical"]
            + sar_delta * parts["sar"]
        )
        joint_scale = contribution_energy(parts["joint"]).clamp_min(1e-6)
        diagnostics = {
            "provenance_optical_rms": contribution_energy(parts["optical"]),
            "provenance_sar_rms": contribution_energy(parts["sar"]),
            "provenance_interaction_rms": contribution_energy(parts["interaction"]),
            "provenance_zero_rms": contribution_energy(parts["zero"]),
            "provenance_optical_fraction": contribution_energy(parts["optical"]) / joint_scale,
            "provenance_sar_fraction": contribution_energy(parts["sar"]) / joint_scale,
            "provenance_interaction_fraction": contribution_energy(parts["interaction"]) / joint_scale,
            "provenance_optical_gain_mean": (1.0 + optical_delta).mean(),
            "provenance_sar_gain_mean": (1.0 + sar_delta).mean(),
            "provenance_feature_deviation_rms": contribution_energy(
                fused - parts["joint"]
            ),
            "provenance_reconstruction_error": (
                parts["reconstruction"] - parts["joint"]
            ).abs().max(),
        }
        if return_maps:
            diagnostics.update({
                "provenance_optical_map": parts["optical"].abs().mean(dim=1, keepdim=True),
                "provenance_sar_map": parts["sar"].abs().mean(dim=1, keepdim=True),
                "provenance_interaction_map": parts["interaction"].abs().mean(dim=1, keepdim=True),
                "provenance_optical_gain_map": 1.0 + optical_delta,
                "provenance_sar_gain_map": 1.0 + sar_delta,
            })
        return fused, diagnostics
