"""Optical-guided dynamic Detail reconstruction for scratch optical-Base LWPD-E2.

All variants retain the original E2 Detail fusion, Base anchor, Decoder, and loss.
Only the fused 64-channel Detail tensor is locally reconstructed; no new SAR
high-frequency values are injected.  The guide is computed from *optical Detail*
feature activity rather than from the raw optical image, so the existing
fuse_features/Stage-II training interface remains unchanged.
"""

import torch
from torch import nn
import torch.nn.functional as F

from e2_base_anchor import OpticalBaseAnchoredE2


MODES = ("conv", "dynamic", "guided", "shifted")
VARIANT_PREFIX = "lwpd_e2_ogdr_v1_"
OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
GROUPS = 4
CHANNELS = 64
MAX_RESIDUAL_SCALE = 0.10


class OpticalGuidedDetailReconstruction(nn.Module):
    """A spatially shared / spatially adaptive 3x3 grouped local difference.

    - conv: learned spatially constant kernel (ordinary fixed-kernel control).
    - dynamic: position-wise kernel conditioned on fused Detail and fused Base.
    - guided: dynamic plus aligned optical Detail activity and projected features.
    - shifted: same network and optical guide, but deterministically misaligned.

    The 1x1 output projection is zero-initialized so all four modes exactly
    reproduce the unmodified E2 head before training.
    """

    def __init__(self, keep_fusion, mode="guided"):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown OGDR mode: " + str(mode))
        self.mode = mode
        self.keep_fusion = keep_fusion
        self.optical_projection = nn.Conv2d(CHANNELS, 8, 1)
        self.detail_projection = nn.Conv2d(CHANNELS, 8, 1)
        self.base_projection = nn.Conv2d(CHANNELS, 8, 1)
        # 8 optical + 8 fused Detail + 8 Base + 3 fixed optical evidence.
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(27, 24, 1), nn.GELU(), nn.Conv2d(24, GROUPS * 9, 1),
        )
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(27, 16, 1), nn.GELU(), nn.Conv2d(16, GROUPS, 1),
        )
        self.stationary_input = nn.Parameter(torch.zeros(1, 27, 1, 1))
        self.output_projection = nn.Conv2d(CHANNELS, CHANNELS, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)

        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sy = sx.transpose(0, 1).contiguous()
        self.register_buffer("sobel_x", sx.reshape(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sy.reshape(1, 1, 3, 3), persistent=False)

    def _optical_structure(self, optical_detail):
        # Take magnitude BEFORE any local averaging; signed Haar Detail can
        # cancel under simple 2x2 pooling, as observed in earlier PAC pilots.
        with torch.autocast(device_type=optical_detail.device.type, enabled=False):
            activity = optical_detail.float().abs().mean(dim=1, keepdim=True)
            smooth = F.avg_pool2d(
                F.pad(activity, (1, 1, 1, 1), mode="replicate"), 3, stride=1,
            )
            gx = F.conv2d(F.pad(smooth, (1, 1, 1, 1), mode="replicate"),
                          self.sobel_x.float())
            gy = F.conv2d(F.pad(smooth, (1, 1, 1, 1), mode="replicate"),
                          self.sobel_y.float())
            magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
            normalizer = magnitude.detach().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
            cue = torch.cat((gx, gy, magnitude), dim=1) / normalizer
        return cue.clamp(-4, 4).to(dtype=optical_detail.dtype)

    @staticmethod
    def _shift(tensor):
        height, width = tensor.shape[-2:]
        return tensor.roll((height // 2 if height > 1 else 0,
                            width // 2 if width > 1 else 0), dims=(-2, -1))

    def _condition(self, optical_detail, fused_detail, fused_base, mode):
        batch = fused_detail.shape[0]
        if mode == "conv":
            return self.stationary_input.expand(batch, -1, -1, -1), fused_detail.new_zeros(())

        detail = F.gelu(self.detail_projection(fused_detail))
        base = F.gelu(self.base_projection(fused_base))
        if mode in ("guided", "shifted"):
            optical = F.gelu(self.optical_projection(optical_detail))
            cue = self._optical_structure(optical_detail)
            if mode == "shifted":
                optical, cue = self._shift(optical), self._shift(cue)
            guide_mean = cue[:, 2:3].mean().detach()
        else:
            optical = torch.zeros_like(detail)
            cue = fused_detail.new_zeros((batch, 3, *fused_detail.shape[-2:]))
            guide_mean = fused_detail.new_zeros(())
        return torch.cat((optical, detail, base, cue), dim=1), guide_mean

    def _local_difference(self, fused_detail, kernels, gate):
        batch, channels, height, width = fused_detail.shape
        if channels != CHANNELS:
            raise ValueError("OGDR requires a 64-channel fused Detail")
        center = fused_detail.reshape(batch, GROUPS, channels // GROUPS, height, width)
        padded = F.pad(fused_detail, (1, 1, 1, 1), mode="replicate")
        accumulated = torch.zeros_like(center)
        for index, (dy, dx) in enumerate(OFFSETS):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[..., 1 + dy:1 + dy + height,
                             1 + dx:1 + dx + width]
            shifted = shifted.reshape_as(center)
            accumulated = accumulated + kernels[:, :, index:index + 1] * (shifted - center)
        return (accumulated * gate).reshape(batch, channels, height, width)

    def forward(self, optical_detail, sar_detail, fused_base, *,
                condition=None, return_diagnostics=False):
        fused_detail = self.keep_fusion(optical_detail, sar_detail, fused_base)
        if condition == "off":
            if not return_diagnostics:
                return fused_detail
            zero = fused_detail.new_zeros(())
            return fused_detail, dict(update_rms=zero, relative_update_rms=zero,
                                      guide_mean=zero, kernel_entropy=zero)
        if condition not in (None, "aligned", "shifted"):
            raise ValueError("condition must be None, 'aligned', 'shifted', or 'off'")
        mode = self.mode if condition is None else ("guided" if condition == "aligned" else "shifted")
        context, guide_mean = self._condition(optical_detail, fused_detail, fused_base, mode)
        batch = fused_detail.shape[0]
        logits = self.kernel_predictor(context)
        spatial = logits.shape[-2:]
        kernels = logits.reshape(batch, GROUPS, 9, *spatial).softmax(dim=2)
        gates = torch.sigmoid(self.gate_predictor(context)).reshape(batch, GROUPS, 1, *spatial)
        reconstructed = self._local_difference(fused_detail, kernels, gates)
        update = MAX_RESIDUAL_SCALE * self.output_projection(reconstructed)
        result = fused_detail + update
        if not return_diagnostics:
            return result
        with torch.no_grad():
            update_rms = update.float().square().mean().sqrt()
            detail_rms = fused_detail.float().square().mean().sqrt().clamp_min(1e-8)
            entropy = -(kernels.float() * kernels.float().clamp_min(1e-8).log()).sum(dim=2).mean()
        return result, dict(update_rms=update_rms, relative_update_rms=update_rms / detail_rms,
                            guide_mean=guide_mean, kernel_entropy=entropy)


class OGDRE2(OpticalBaseAnchoredE2):
    """Full LWPD-E2 with optical Base anchor and the OGDR Detail head."""

    def __init__(self, mode="guided"):
        if mode not in MODES:
            raise ValueError("Unknown OGDR mode: " + str(mode))
        super().__init__(OpticalBaseAnchoredE2.VARIANT)
        self.variant = VARIANT_PREFIX + mode
        self.detail_fusion = OpticalGuidedDetailReconstruction(self.detail_fusion, mode)

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail,
                      *, condition=None, return_diagnostics=False):
        fused_base = optical_base + self.base_fusion(optical_base, sar_base)
        fused_detail = self.detail_fusion(optical_detail, sar_detail, fused_base,
                                         condition=condition, return_diagnostics=return_diagnostics)
        return fused_base, fused_detail
