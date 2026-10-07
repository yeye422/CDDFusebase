"""Detail-fusion modules for the staged Optical-SAR reliability study."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net import DetailFeatureExtraction


def zero_module(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels=64, groups=8):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs):
        residual = self.depthwise(inputs)
        residual = self.activation(self.norm(residual))
        return inputs + self.pointwise(residual)


class ZeroInitializedResidualAligner(nn.Module):
    def __init__(self, channels=64, groups=8):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.output = zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, inputs):
        residual = self.depthwise(inputs)
        residual = self.activation(self.norm(residual))
        return inputs + self.output(residual)


class ResidualCandidateNet(nn.Module):
    def __init__(self, channels=64, groups=8, blocks=2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[DepthwiseResidualBlock(channels, groups) for _ in range(blocks)]
        )
        self.output = zero_module(
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, optical_detail, sar_detail):
        features = self.input(torch.cat((optical_detail, sar_detail), dim=1))
        return torch.tanh(self.output(self.blocks(features)))


class DemandGate(nn.Module):
    def __init__(self, channels=64, hidden_channels=32, groups=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.output = zero_module(nn.Conv2d(hidden_channels, 1, kernel_size=1))

    def forward(self, optical_detail, sar_detail):
        features = self.features(torch.cat((optical_detail, sar_detail), dim=1))
        return torch.sigmoid(self.output(features))


class DemandDetailInjection(nn.Module):
    """A1: content-demand-gated SAR detail residual injection."""

    def __init__(self, channels=64, gamma=0.5):
        super().__init__()
        self.optical_aligner = ZeroInitializedResidualAligner(channels)
        self.sar_aligner = ZeroInitializedResidualAligner(channels)
        self.residual_candidate = ResidualCandidateNet(channels)
        self.demand_gate = DemandGate(channels)
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(
        self,
        optical_detail,
        sar_detail,
        force_demand=None,
        return_maps=False,
    ):
        aligned_optical = self.optical_aligner(optical_detail)
        aligned_sar = self.sar_aligner(sar_detail)
        residual = self.residual_candidate(aligned_optical, aligned_sar)
        learned_demand = self.demand_gate(aligned_optical, aligned_sar)
        demand = (
            torch.full_like(learned_demand, float(force_demand))
            if force_demand is not None
            else learned_demand
        )
        injection = self.gamma * demand * residual
        fused_detail = aligned_optical + injection
        diagnostics = {
            "demand_mean": demand.mean(),
            "demand_std": demand.std(unbiased=False),
            "demand_low_fraction": (demand < 0.05).float().mean(),
            "demand_high_fraction": (demand > 0.95).float().mean(),
            "delta_norm": torch.sqrt(torch.mean(residual.float() ** 2)),
            "injection_norm": torch.sqrt(torch.mean(injection.float() ** 2)),
        }
        if force_demand is not None:
            diagnostics.update(
                {
                    "learned_demand_mean": learned_demand.mean(),
                    "learned_demand_std": learned_demand.std(unbiased=False),
                }
            )
        if return_maps:
            diagnostics["demand_map"] = demand
            diagnostics["learned_demand_map"] = learned_demand
        return fused_detail, diagnostics


class ResidualDetailInjection(nn.Module):
    """A1-prime: bounded residual injection without a separate demand gate."""

    def __init__(self, channels=64, gamma=0.25):
        super().__init__()
        self.optical_aligner = ZeroInitializedResidualAligner(channels)
        self.sar_aligner = ZeroInitializedResidualAligner(channels)
        self.residual_candidate = ResidualCandidateNet(channels)
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(self, optical_detail, sar_detail, force_scale=None):
        aligned_optical = self.optical_aligner(optical_detail)
        aligned_sar = self.sar_aligner(sar_detail)
        residual = self.residual_candidate(aligned_optical, aligned_sar)
        scale = (
            residual.new_tensor(float(force_scale))
            if force_scale is not None
            else self.gamma
        )
        injection = scale * residual
        fused_detail = aligned_optical + injection
        diagnostics = {
            "delta_norm": torch.sqrt(torch.mean(residual.float() ** 2)),
            "injection_norm": torch.sqrt(torch.mean(injection.float() ** 2)),
        }
        return fused_detail, diagnostics


class OpticalDetailRefinement(nn.Module):
    """C2: refine Optical detail without injecting SAR detail features."""

    def __init__(self, channels=64):
        super().__init__()
        self.aligner = ZeroInitializedResidualAligner(channels)

    def forward(self, optical_detail):
        return self.aligner(optical_detail)


class ResidualSpatialModulator(nn.Module):
    """R1: bounded, zero-initialized spatial modulation of SAR detail."""

    def __init__(self, channels=64, hidden_channels=32, alpha=0.25):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0,1]")
        self.features = nn.Sequential(
            nn.Conv2d(2 * channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
        )
        self.output = zero_module(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, optical_detail, sar_detail, return_maps=False):
        if optical_detail.shape != sar_detail.shape:
            raise ValueError("Optical and SAR detail tensors must have the same shape")
        logits = self.output(
            self.features(torch.cat((optical_detail, sar_detail), dim=1))
        )
        modulation = self.alpha * torch.tanh(logits)
        modulated_sar = (1.0 + modulation) * sar_detail
        diagnostics = {
            "r1_modulation_mean": modulation.mean(),
            "r1_modulation_std": modulation.std(unbiased=False),
            "r1_modulation_abs_mean": modulation.abs().mean(),
            "r1_modulation_min": modulation.amin(),
            "r1_modulation_max": modulation.amax(),
        }
        if return_maps:
            diagnostics["r1_modulation_map"] = modulation
        return modulated_sar, diagnostics


class LocalCrossModalEvidence(nn.Module):
    """Estimate local-tolerant Optical/SAR feature correspondence."""

    def __init__(self, channels=64, embedding_channels=16, window_size=3):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.optical_projection = nn.Conv2d(
            channels, embedding_channels, kernel_size=1, bias=False
        )
        self.sar_projection = nn.Conv2d(
            channels, embedding_channels, kernel_size=1, bias=False
        )

    def forward(self, optical_detail, sar_detail):
        optical = F.normalize(self.optical_projection(optical_detail), dim=1)
        sar = self.sar_projection(sar_detail)
        batch, channels, height, width = optical.shape
        neighbors = F.unfold(
            sar,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).view(
            batch,
            channels,
            self.window_size * self.window_size,
            height,
            width,
        )
        neighbors = F.normalize(neighbors, dim=1)
        similarities = (optical.unsqueeze(2) * neighbors).sum(dim=1)
        weights = torch.softmax(similarities, dim=1)
        matched = (weights * similarities).sum(dim=1, keepdim=True)
        matched = ((matched + 1.0) * 0.5).clamp(0.0, 1.0)
        if self.window_size == 1:
            certainty = torch.ones_like(matched)
        else:
            entropy = -(
                weights * torch.log(weights.clamp_min(1e-8))
            ).sum(dim=1, keepdim=True)
            certainty = 1.0 - entropy / math.log(
                self.window_size * self.window_size
            )
        optical_energy = optical_detail.square().mean(dim=1, keepdim=True)
        sar_energy = sar_detail.square().mean(dim=1, keepdim=True)
        energy = 1.0 - torch.exp(-(optical_energy + sar_energy).float())
        energy = energy.to(optical_detail.dtype)
        support = (matched * energy).clamp(0.0, 1.0)
        return support, certainty.clamp(0.0, 1.0)


class CrossSupportedCandidate(nn.Module):
    """Align SAR detail using Optical/SAR context while preserving initialization."""

    def __init__(self, channels=64, groups=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.correction = zero_module(
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, optical_detail, sar_detail):
        context = self.features(torch.cat((optical_detail, sar_detail), dim=1))
        return sar_detail + self.correction(context)


class SRRDetailFusion(nn.Module):
    """Speckle-aware structure-reliability routing replacement for DetailFuse."""

    def __init__(self, channels=64, hidden_channels=16, window_size=3):
        super().__init__()
        self.cross_evidence = LocalCrossModalEvidence(
            channels=channels,
            embedding_channels=hidden_channels,
            window_size=window_size,
        )
        self.router = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        self.cross_candidate = CrossSupportedCandidate(channels=channels)
        self.complementary_candidate = ZeroInitializedResidualAligner(channels)
        self.route_fusion = nn.Conv2d(
            3 * channels, channels, kernel_size=1, bias=False
        )
        self.detail_transform = DetailFeatureExtraction(num_layers=1)
        self._initialize_route_fusion(channels)
        self._initialize_router()

    def _initialize_router(self):
        output = self.router[-1]
        nn.init.normal_(output.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            output.bias.copy_(output.bias.new_tensor((2.2, 2.2, -2.2)))

    def _initialize_route_fusion(self, channels):
        nn.init.zeros_(self.route_fusion.weight)
        identity = torch.eye(channels).view(channels, channels, 1, 1)
        with torch.no_grad():
            self.route_fusion.weight[:, :channels].copy_(identity)
            self.route_fusion.weight[:, channels : 2 * channels].copy_(identity)
            self.route_fusion.weight[:, 2 * channels :].copy_(identity)

    def load_legacy_detail_fuse(self, state):
        """Initialize the terminal detail transform from a C0/B0 checkpoint."""
        self.detail_transform.load_state_dict(state)

    def forward(
        self,
        optical_detail,
        sar_detail,
        structure_prior,
        valid=None,
        return_maps=False,
        force_reliability=None,
    ):
        if optical_detail.shape != sar_detail.shape:
            raise ValueError("Optical and SAR detail tensors must have the same shape")
        if structure_prior.ndim != 4 or structure_prior.shape[1] != 1:
            raise ValueError("structure_prior must have shape [B,1,H,W]")
        if structure_prior.shape[-2:] != optical_detail.shape[-2:]:
            structure_prior = F.interpolate(
                structure_prior,
                size=optical_detail.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        prior = structure_prior.detach().clamp(0.0, 1.0)
        if valid is not None:
            if valid.shape[-2:] != prior.shape[-2:]:
                valid = F.interpolate(valid, size=prior.shape[-2:], mode="nearest")
            prior = prior * valid

        cross, match_certainty = self.cross_evidence(
            optical_detail, sar_detail
        )
        optical_energy = 1.0 - torch.exp(
            -optical_detail.square().mean(dim=1, keepdim=True).float()
        )
        sar_energy = 1.0 - torch.exp(
            -sar_detail.square().mean(dim=1, keepdim=True).float()
        )
        evidence = torch.cat(
            (
                cross,
                match_certainty,
                prior,
                optical_energy.to(optical_detail.dtype),
                sar_energy.to(sar_detail.dtype),
            ),
            dim=1,
        )
        reliability, cross_attribute, complementary_attribute = torch.sigmoid(
            self.router(evidence)
        ).chunk(3, dim=1)
        if force_reliability is not None:
            if torch.is_tensor(force_reliability):
                reliability = force_reliability.to(
                    device=reliability.device, dtype=reliability.dtype
                )
                if reliability.shape != prior.shape:
                    reliability = F.interpolate(
                        reliability,
                        size=prior.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                reliability = reliability.expand_as(prior).clamp(0.0, 1.0)
            else:
                reliability = torch.full_like(
                    reliability, float(force_reliability)
                )
        if valid is not None:
            reliability = reliability * valid

        cross_detail = self.cross_candidate(optical_detail, sar_detail)
        complementary_detail = self.complementary_candidate(sar_detail)
        routed_cross = reliability * cross_attribute * cross_detail
        routed_complementary = (
            reliability * complementary_attribute * complementary_detail
        )
        fusion_input = self.route_fusion(
            torch.cat(
                (optical_detail, routed_cross, routed_complementary), dim=1
            )
        )
        fused_detail = self.detail_transform(fusion_input)
        routed_sar = routed_cross + routed_complementary

        diagnostics = {
            "srr_reliability_mean": reliability.mean(),
            "srr_reliability_std": reliability.std(unbiased=False),
            "srr_cross_attribute_mean": cross_attribute.mean(),
            "srr_complementary_attribute_mean": complementary_attribute.mean(),
            "srr_cross_evidence_mean": cross.mean(),
            "srr_match_certainty_mean": match_certainty.mean(),
            "srr_structure_prior_mean": prior.mean(),
            "srr_routed_sar_norm": torch.sqrt(
                torch.mean(routed_sar.float().square())
            ),
        }
        if return_maps:
            diagnostics.update(
                {
                    "srr_reliability_map": reliability,
                    "srr_cross_attribute_map": cross_attribute,
                    "srr_complementary_attribute_map": complementary_attribute,
                    "srr_cross_evidence_map": cross,
                    "srr_match_certainty_map": match_certainty,
                    "srr_structure_prior_map": prior,
                }
            )
        return fused_detail, diagnostics
