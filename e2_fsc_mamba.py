"""Transfer same-line Haar LL context to the signed-Detail SSM."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_phase_guide import SignedPhaseGuideE2


MODES = ("ll_feature", "state", "shuffled_state")
VARIANT_PREFIX = "lwpd_e2_fsc_mamba_v1_"
CONTEXT_SCALE = 0.1


def scan_with_state(scan, sequence, initial_state):
    """Use the signed-PAC scan's parameters with an explicit initial state."""
    value, gate = scan.input_projection(scan.norm(sequence)).chunk(2, dim=-1)
    delta, b, c = scan.parameter_projection(value).split(
        (scan.inner, scan.state_channels, scan.state_channels), dim=-1,
    )
    delta = F.softplus(delta + scan.delta_bias).clamp(max=1.0)
    transition = -scan.a_log.exp()
    state = initial_state.to(transition.dtype)
    outputs = []
    for t in range(sequence.shape[1]):
        dt = delta[:, t, :, None]
        decay = torch.exp(dt * transition[None])
        drive = dt * b[:, t, None, :] * value[:, t, :, None]
        state = decay * state + drive
        outputs.append((state * c[:, t, None, :]).sum(-1) + scan.skip * value[:, t])
    return scan.output_projection(torch.stack(outputs, dim=1) * F.silu(gate))


class LLStateTransfer(nn.Module):
    """Use one low-frequency summary per row/column and scan direction."""

    def __init__(self, coarse_scan, mode):
        super().__init__()
        channels = coarse_scan.channels
        if any(core.inner != channels for core in (coarse_scan.row_scan, coarse_scan.column_scan)):
            raise ValueError("This matched control expects unexpanded Detail states.")
        self.mode = mode
        self.core = coarse_scan
        self.ll_projection = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.decay_logit = nn.Parameter(torch.full((channels,), math.log(0.95 / 0.05)))

    def terminal(self, sequence):
        # Closed form of q_t = rho*q_(t-1) + (1-rho)*x_t, with q_0=0.
        rho = self.decay_logit.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        distance = torch.arange(sequence.shape[1] - 1, -1, -1, device=sequence.device)
        weights = (1 - rho) * rho.pow(distance[:, None])
        return (sequence.float() * weights).sum(1)

    def initial_state(self, summary, scan):
        return summary[:, :, None].expand(-1, -1, scan.state_channels) / math.sqrt(scan.state_channels)

    def scan_lines(self, detail, low, scan):
        outputs = []
        for reverse in (False, True):
            x = detail.flip(2) if reverse else detail
            coarse = low.flip(2) if reverse else low
            batch, lines, length, channels = x.shape
            x = x.reshape(batch * lines, length, channels).float()
            summary = self.terminal(coarse.reshape_as(x)).reshape(batch, lines, channels)
            if self.mode == "shuffled_state":
                summary = summary.roll(lines // 2, dims=1)
            summary = CONTEXT_SCALE * summary.reshape(batch * lines, channels)
            initial = self.initial_state(summary, scan)
            if self.mode == "ll_feature":
                x = x + summary[:, None, :]
                initial = torch.zeros_like(initial)
            result = scan_with_state(scan, x, initial).reshape(batch, lines, length, channels)
            outputs.append(result.flip(2) if reverse else result)
        return 0.5 * (outputs[0] + outputs[1])

    def forward(self, detail, optical_ll, sar_ll):
        with torch.autocast(device_type=detail.device.type, enabled=False):
            low = self.ll_projection(torch.cat((optical_ll.float(), sar_ll.float()), dim=1)).tanh()
        d_row = detail.permute(0, 2, 3, 1)
        l_row = low.permute(0, 2, 3, 1)
        rows = self.scan_lines(d_row, l_row, self.core.row_scan).permute(0, 3, 1, 2)
        columns = self.scan_lines(d_row.transpose(1, 2), l_row.transpose(1, 2),
                                  self.core.column_scan).permute(0, 3, 2, 1)
        return detail + self.core.merge(torch.cat((self.core.local(detail), rows, columns), dim=1))


class FSCDetailFusion(nn.Module):
    def __init__(self, signed_detail, mode):
        super().__init__()
        self.haar = signed_detail.haar
        self.keep_fusion = signed_detail.keep_fusion
        self.joint_projection = signed_detail.joint_projection
        self.feature_gate = signed_detail.feature_gate
        self.update_projection = signed_detail.update_projection
        self.update_logit = signed_detail.update_logit
        self.ssm = LLStateTransfer(signed_detail.ssm, mode)

    def bands(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            bands, _ = self.haar(x.float())
        return tuple(value.to(x.dtype) for value in bands)

    def signed_joint(self, optical_detail, sar_detail):
        od, sd = (torch.cat(self.bands(x)[1:], dim=1) for x in (optical_detail, sar_detail))
        return self.joint_projection(torch.cat((od, sd, (od - sd).abs()), dim=1))

    def update(self, optical_detail, sar_detail, optical_base, sar_base):
        joint = self.signed_joint(optical_detail, sar_detail)
        ll = (self.bands(x)[0] for x in (optical_base, sar_base))
        contextual = self.ssm(joint, *ll)
        reliability = torch.sigmoid(self.feature_gate(joint)) * (2 / 3)
        residual = reliability * self.update_projection(contextual)
        return torch.sigmoid(self.update_logit) * F.interpolate(
            residual, size=optical_detail.shape[-2:], mode="bilinear", align_corners=False,
        )

    def forward(self, optical_detail, sar_detail, fused_base, optical_base, sar_base):
        keep = self.keep_fusion(optical_detail, sar_detail, fused_base)
        return keep + self.update(optical_detail, sar_detail, optical_base, sar_base)


class FSCMambaE2(OpticalBaseAnchoredE2):
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unsupported FSC mode: {mode}")
        nn.Module.__init__(self)
        backbone = SignedPhaseGuideE2("feature")
        self.variant = VARIANT_PREFIX + mode
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.lwpd = backbone.lwpd
        self.base_fusion = backbone.base_fusion.correction
        self.detail_fusion = FSCDetailFusion(backbone.detail_fusion, mode)

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail):
        base = optical_base + self.base_fusion(optical_base, sar_base)
        detail = self.detail_fusion(optical_detail, sar_detail, base, optical_base, sar_base)
        return base, detail
