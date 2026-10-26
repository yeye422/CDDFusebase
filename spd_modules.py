"""Structure-preserving denoising modules for the V2.1 SPD stage.

The frozen CDDFuse encoder produces the original base/detail features.  SPD
adds bounded, zero-initialized residual corrections, so construction preserves
the C0 representation exactly.  The auxiliary denoising head is intentionally
separate from the fusion decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, requested=8):
    groups = min(int(requested), int(channels))
    while channels % groups:
        groups -= 1
    return max(groups, 1)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        norm_groups = _group_count(channels, groups)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, channels),
        )

    def forward(self, features):
        return features + self.block(features)


class BoundedFeatureCorrection(nn.Module):
    """Zero-initialized, RMS-bounded correction for one feature branch."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_relative_correction=0.10,
        epsilon=1e-6,
    ):
        super().__init__()
        self.max_relative_correction = float(max_relative_correction)
        self.epsilon = float(epsilon)
        self.input = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[
                DepthwiseResidualBlock(hidden_channels)
                for _ in range(int(num_blocks))
            ]
        )
        self.delta = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, features, return_maps=False):
        hidden = self.blocks(F.gelu(self.input(features)))
        gate = torch.sigmoid(self.gate(hidden))
        feature_rms = features.detach().square().mean(
            dim=(1, 2, 3), keepdim=True
        ).sqrt().clamp_min(self.epsilon)
        correction = (
            self.max_relative_correction
            * feature_rms
            * gate
            * torch.tanh(self.delta(hidden))
        )
        cleaned = features + correction
        diagnostics = {
            "gate_mean": gate.mean(),
            "gate_std": gate.std(unbiased=False),
            "correction_abs_mean": correction.abs().mean(),
            "correction_rms": correction.square().mean().sqrt(),
            "feature_rms": feature_rms.mean(),
        }
        if return_maps:
            diagnostics["gate_map"] = gate
            diagnostics["correction_map"] = correction
        return cleaned, diagnostics


class StructurePreservingDenoiser(nn.Module):
    """Independent bounded corrections for CDDFuse base and detail features."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_relative_correction=0.10,
    ):
        super().__init__()
        branch_args = dict(
            channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            max_relative_correction=max_relative_correction,
        )
        self.base = BoundedFeatureCorrection(**branch_args)
        self.detail = BoundedFeatureCorrection(**branch_args)

    def forward(self, base, detail, return_maps=False):
        clean_base, base_diagnostics = self.base(base, return_maps)
        clean_detail, detail_diagnostics = self.detail(detail, return_maps)
        diagnostics = {
            f"base_{name}": value for name, value in base_diagnostics.items()
        }
        diagnostics.update(
            {
                f"detail_{name}": value
                for name, value in detail_diagnostics.items()
            }
        )
        return clean_base, clean_detail, diagnostics


class DetailOnlyStructurePreservingDenoiser(nn.Module):
    """Leave Base untouched and learn only a bounded Detail correction."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_relative_correction=0.10,
    ):
        super().__init__()
        self.detail = BoundedFeatureCorrection(
            channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            max_relative_correction=max_relative_correction,
        )

    def forward(self, base, detail, return_maps=False):
        clean_detail, detail_diagnostics = self.detail(
            detail, return_maps=return_maps
        )
        zero = base.sum() * 0.0
        diagnostics = {
            "base_gate_mean": zero,
            "base_gate_std": zero,
            "base_correction_abs_mean": zero,
            "base_correction_rms": zero,
            "base_feature_rms": base.detach().square().mean().sqrt(),
        }
        diagnostics.update(
            {
                f"detail_{name}": value
                for name, value in detail_diagnostics.items()
            }
        )
        return base, clean_detail, diagnostics


class SelectiveDetailCorrection(nn.Module):
    """Frequency-conditioned, channel-selective bounded Detail correction.

    The original SPD uses one spatial gate for all feature channels and scales
    its correction with one image-level RMS value.  This module instead builds
    evidence from several fixed high-pass residuals, predicts independent
    spatial and channel confidence, and bounds every feature channel by its own
    RMS.  The predicted correction is projected back to the high-pass domain so
    that the module cannot introduce a DC/Base-like shift into Detail features.
    """

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_relative_correction=0.10,
        scale_kernels=(3, 5, 9),
        correction_highpass_kernel=5,
        channel_reduction=4,
        epsilon=1e-6,
    ):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in scale_kernels)
        if not kernels or any(kernel < 3 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("scale_kernels must contain odd integers >= 3")
        correction_kernel = int(correction_highpass_kernel)
        if correction_kernel < 3 or correction_kernel % 2 == 0:
            raise ValueError("correction_highpass_kernel must be odd and >= 3")
        reduction = max(int(channel_reduction), 1)
        reduced_channels = max(int(channels) // reduction, 1)

        self.max_relative_correction = float(max_relative_correction)
        self.scale_kernels = kernels
        self.correction_highpass_kernel = correction_kernel
        self.epsilon = float(epsilon)
        evidence_channels = int(channels) * (1 + len(kernels))
        self.input = nn.Conv2d(evidence_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[
                DepthwiseResidualBlock(hidden_channels)
                for _ in range(int(num_blocks))
            ]
        )
        self.delta = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.spatial_confidence = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.channel_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, reduced_channels, 1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, 1),
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.spatial_confidence.weight)
        nn.init.zeros_(self.spatial_confidence.bias)
        nn.init.zeros_(self.channel_confidence[-1].weight)
        nn.init.zeros_(self.channel_confidence[-1].bias)

    @staticmethod
    def _lowpass(features, kernel):
        return F.avg_pool2d(
            features,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
            count_include_pad=False,
        )

    def forward(self, features, return_maps=False):
        evidence = [features]
        evidence.extend(
            features - self._lowpass(features, kernel)
            for kernel in self.scale_kernels
        )
        hidden = self.blocks(F.gelu(self.input(torch.cat(evidence, dim=1))))
        spatial = torch.sigmoid(self.spatial_confidence(hidden))
        channel = torch.sigmoid(self.channel_confidence(hidden))
        confidence = spatial * channel

        proposal = self.delta(hidden)
        proposal = proposal - self._lowpass(
            proposal, self.correction_highpass_kernel
        )
        proposal = torch.tanh(proposal)
        channel_rms = features.detach().square().mean(
            dim=(2, 3), keepdim=True
        ).sqrt().clamp_min(self.epsilon)
        correction = (
            self.max_relative_correction
            * channel_rms
            * confidence
            * proposal
        )
        cleaned = features + correction
        correction_lowpass = self._lowpass(
            correction, self.correction_highpass_kernel
        )
        diagnostics = {
            "gate_mean": confidence.mean(),
            "gate_std": confidence.std(unbiased=False),
            "spatial_confidence_mean": spatial.mean(),
            "channel_confidence_mean": channel.mean(),
            "correction_abs_mean": correction.abs().mean(),
            "correction_rms": correction.square().mean().sqrt(),
            "correction_lowpass_abs_mean": correction_lowpass.abs().mean(),
            "feature_rms": channel_rms.mean(),
        }
        if return_maps:
            diagnostics.update(
                {
                    "gate_map": confidence,
                    "spatial_confidence_map": spatial,
                    "channel_confidence_map": channel,
                    "correction_map": correction,
                }
            )
        return cleaned, diagnostics


class SelectiveDetailStructurePreservingDenoiser(nn.Module):
    """Strict Base bypass with multiscale selective Detail purification."""

    def __init__(self, **correction_kwargs):
        super().__init__()
        self.detail = SelectiveDetailCorrection(**correction_kwargs)

    def forward(self, base, detail, return_maps=False):
        clean_detail, detail_diagnostics = self.detail(
            detail, return_maps=return_maps
        )
        zero = base.sum() * 0.0
        diagnostics = {
            "base_gate_mean": zero,
            "base_gate_std": zero,
            "base_correction_abs_mean": zero,
            "base_correction_rms": zero,
            "base_feature_rms": base.detach().square().mean().sqrt(),
        }
        diagnostics.update(
            {
                f"detail_{name}": value
                for name, value in detail_diagnostics.items()
            }
        )
        return base, clean_detail, diagnostics


def build_spd_variant(variant="base_detail", **kwargs):
    """Construct an SPD variant while keeping legacy checkpoints compatible."""

    if variant == "base_detail":
        return StructurePreservingDenoiser(
            max_relative_correction=kwargs.get("max_relative_correction", 0.10)
        )
    if variant == "detail_only":
        return DetailOnlyStructurePreservingDenoiser(
            max_relative_correction=kwargs.get("max_relative_correction", 0.10)
        )
    if variant == "selective_detail":
        return SelectiveDetailStructurePreservingDenoiser(
            max_relative_correction=kwargs.get("max_relative_correction", 0.10),
            scale_kernels=kwargs.get("scale_kernels", (3, 5, 9)),
            correction_highpass_kernel=kwargs.get(
                "correction_highpass_kernel", 5
            ),
        )
    raise ValueError(f"Unknown SPD variant: {variant}")


def build_spd_from_checkpoint_args(args):
    """Recreate the exact SPD architecture recorded in checkpoint metadata."""

    args = args or {}
    return build_spd_variant(
        args.get("spd_variant", "base_detail"),
        max_relative_correction=float(
            args.get("max_relative_correction", 0.10)
        ),
        scale_kernels=tuple(args.get("selective_scale_kernels", (3, 5, 9))),
        correction_highpass_kernel=int(
            args.get("selective_correction_highpass_kernel", 5)
        ),
    )


class SPDDenoisingHead(nn.Module):
    """Decode corrected features into a bounded residual in SAR logit space."""

    def __init__(
        self,
        channels=64,
        hidden_channels=64,
        num_blocks=2,
        max_logit_residual=2.0,
        input_epsilon=1.0 / 255.0,
    ):
        super().__init__()
        self.max_logit_residual = float(max_logit_residual)
        self.input_epsilon = float(input_epsilon)
        self.reduce = nn.Conv2d(channels * 2, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[
                DepthwiseResidualBlock(hidden_channels)
                for _ in range(int(num_blocks))
            ]
        )
        self.output = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy, clean_base, clean_detail, return_residual=False):
        hidden = self.reduce(torch.cat((clean_base, clean_detail), dim=1))
        hidden = self.blocks(F.gelu(hidden))
        logit_residual = self.max_logit_residual * torch.tanh(
            self.output(hidden)
        )
        clipped = noisy.clamp(
            self.input_epsilon, 1.0 - self.input_epsilon
        )
        denoised = torch.sigmoid(torch.logit(clipped) + logit_residual)
        if return_residual:
            return denoised, logit_residual
        return denoised


class SPDModel(nn.Module):
    """Frozen-encoder wrapper used by the standalone Stage-I trainer."""

    def __init__(self, encoder, spd=None, denoising_head=None):
        super().__init__()
        self.encoder = encoder
        self.spd = spd or StructurePreservingDenoiser()
        self.denoising_head = denoising_head or SPDDenoisingHead()
        self.freeze_encoder()

    def freeze_encoder(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, sar, return_maps=False):
        with torch.no_grad():
            base, detail, _ = self.encoder(sar)
        clean_base, clean_detail, diagnostics = self.spd(
            base, detail, return_maps=return_maps
        )
        denoised, logit_residual = self.denoising_head(
            sar, clean_base, clean_detail, return_residual=True
        )
        diagnostics.update(
            {
                "image_logit_residual_abs_mean": logit_residual.abs().mean(),
                "image_change_abs_mean": (denoised - sar).abs().mean(),
            }
        )
        return {
            "base": clean_base,
            "detail": clean_detail,
            "denoised": denoised,
            "diagnostics": diagnostics,
        }
