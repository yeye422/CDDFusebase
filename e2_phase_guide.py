"""Matched phase guidance controls on signed E2 Detail subbands."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor_pac import OpticalBaseAnchoredPACE2
from lwpd import DirectionalPhaseSymmetry, HaarDWT2D
from pac_mamba import PACMambaDetailFusion


MODES = ("feature", "pa", "pc", "pa_shifted")
VARIANT_PREFIX = "lwpd_e2_signed_pac_v1_"


class DirectionalPhaseCongruency(DirectionalPhaseSymmetry):
    """PC from the same even/Riesz-odd filter responses used by PAC's PA."""

    @torch.no_grad()
    def forward(self, image):
        height, width = image.shape[-2:]
        pad = min(self.reflect_padding, height - 1, width - 1)
        work = F.pad(image.float(), (pad,) * 4, mode="reflect") if pad else image.float()
        height, width = work.shape[-2:]
        radial, rx, ry = self._filters(height, width, work.device, work.dtype)
        filtered = torch.fft.fft2(work).unsqueeze(2) * radial
        even = torch.fft.ifft2(filtered).real
        odd_x, odd_y = (torch.fft.ifft2(filtered * r).real for r in (rx, ry))
        responses = []
        for index in range(self.orientations):
            angle = math.pi * index / self.orientations
            odd = odd_x * math.cos(angle) + odd_y * math.sin(angle)
            total_even, total_odd = even.sum(dim=2, keepdim=True), odd.sum(dim=2, keepdim=True)
            angle_norm = torch.hypot(total_even, total_odd).clamp_min(self.epsilon)
            cos_phase, sin_phase = total_even / angle_norm, total_odd / angle_norm
            deviation_energy = even * cos_phase + odd * sin_phase - (even * sin_phase - odd * cos_phase).abs()
            amplitude = torch.hypot(even, odd)
            amplitude_sum = amplitude.sum(dim=2)
            spread = (amplitude_sum / amplitude.amax(dim=2).clamp_min(self.epsilon) - 1) / max(1, len(self.wavelengths) - 1)
            weight = torch.sigmoid(10 * (spread - 0.5))
            congruency = weight * (deviation_energy.sum(dim=2) - self.noise_threshold).relu()
            responses.append(congruency / amplitude_sum.clamp_min(self.epsilon))
        phase = torch.cat(responses, dim=1).clamp(0, 1)
        if pad:
            phase = phase[..., pad:-pad, pad:-pad]
        return phase.to(image.dtype)


class SignedPhaseDetailFusion(nn.Module):
    def __init__(self, legacy, mode):
        super().__init__()
        self.mode = mode
        self.haar = HaarDWT2D()
        self.keep_fusion = legacy.keep_fusion
        channels = legacy.channels
        self.joint_projection = nn.Sequential(
            nn.Conv2d(9 * channels, channels, 1), *list(legacy.joint_projection.children())[1:]
        )
        self.feature_gate = nn.Sequential(nn.Conv2d(channels, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        self.ssm = legacy.ssm
        self.update_projection = legacy.update_projection
        self.update_logit = legacy.update_logit

    def signed_bands(self, detail):
        # Pooling the synthesized Detail cancels every zero-mean 2x2 Haar block.
        with torch.autocast(device_type=detail.device.type, enabled=False):
            subbands, _ = self.haar(detail.float())
        return torch.cat(subbands[1:], dim=1).to(detail.dtype)

    def phase_factor(self, optical_phase, sar_phase, size):
        match = PACMambaDetailFusion.phase_match(optical_phase, sar_phase)
        low = F.interpolate(match.float(), size=size, mode="area")
        if self.mode == "pa_shifted":
            h, w = size
            low = low.roll((h // 2, w // 3), dims=(-2, -1))
        centered = low - low.mean(dim=(-2, -1), keepdim=True)
        scale = centered.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(0.05)
        return low, 1 + 0.5 * centered / scale

    def forward(self, optical_detail, sar_detail, fused_base, optical_pa, sar_pa, return_diagnostics=False):
        keep = self.keep_fusion(optical_detail, sar_detail, fused_base)
        optical, sar = map(self.signed_bands, (optical_detail, sar_detail))
        joint = self.joint_projection(torch.cat((optical, sar, (optical - sar).abs()), dim=1))
        update = self.update_projection(self.ssm(joint))
        reliability = torch.sigmoid(self.feature_gate(joint)) * (2 / 3)
        if self.mode == "feature":
            phase_match = torch.zeros_like(reliability)
            interaction = reliability
        else:
            phase_match, factor = self.phase_factor(optical_pa, sar_pa, update.shape[-2:])
            interaction = reliability * factor
        gated = F.interpolate(interaction.to(update.dtype) * update, size=keep.shape[-2:],
                              mode="bilinear", align_corners=False)
        scale = torch.sigmoid(self.update_logit)
        fused = keep + scale * gated
        if not return_diagnostics:
            return fused
        return fused, dict(phase_match=phase_match, reliability=reliability, interaction=interaction,
                           update=gated, update_scale=scale, keep=keep)


class SignedPhaseGuideE2(OpticalBaseAnchoredPACE2):
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unsupported phase guide: {mode}")
        super().__init__()
        self.mode = mode
        self.variant = VARIANT_PREFIX + mode
        self.detail_fusion = SignedPhaseDetailFusion(self.detail_fusion, mode)
        if mode == "pc":
            self.phase_evidence = DirectionalPhaseCongruency(orientations=4)

    def compute_phase_evidence(self, optical, sar):
        if self.mode == "feature":
            return dict(optical_pa=None, sar_pa=None, sar_ps=None)
        if self.mode == "pc":
            return dict(optical_pa=self.phase_evidence(optical), sar_pa=self.phase_evidence(sar), sar_ps=None)
        return super().compute_phase_evidence(optical, sar)
