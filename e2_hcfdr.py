"""CFDR with patchwise adaptation of the SAR response along the optical direction."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from e2_cfdr import CFDRE2, CenterPreservingReconstruction, OFFSETS


VARIANT = "lwpd_e2_hcfdr_v1"
EPSILON = 1e-4


def adapt_sar(optical, sar, beta):
    """Adapt patches with the nine samples along dimension 1."""
    optical, sar, beta = optical.float(), sar.float(), beta.float()
    dot = (optical * sar).sum(dim=1, keepdim=True)
    negative = (dot / (optical.square().sum(dim=1, keepdim=True) + EPSILON)).clamp_max(0)
    return sar - (2 * beta * negative) * optical, negative < 0


class HeterogeneousReconstruction(CenterPreservingReconstruction):
    def __init__(self, cf2_head):
        super().__init__(dynamic=True)
        self.load_state_dict(cf2_head.state_dict())
        self.adaptation = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.adaptation.weight)
        nn.init.constant_(self.adaptation.bias, math.log(0.05 / 0.95))

    def forward(self, prediction, hidden, fused_detail, optical, sar, *, return_stats=False):
        context = self.condition(torch.cat((hidden, fused_detail), dim=1))
        gain = self.gain(context).softmax(dim=1, dtype=torch.float32)
        kernel = self.neighbors(context).unflatten(1, (2, 8)).softmax(dim=2, dtype=torch.float32)
        rho = 0.5 * self.strength(context).float().sigmoid()
        beta = self.adaptation(context).float().sigmoid()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            inputs = torch.cat((prediction.float(), optical.float(), sar.float()), dim=1)
            base, means = self.smooth(inputs).split((1, 2), dim=1)
            centered = inputs[:, 1:] - means
            height, width = centered.shape[-2:]
            padded = F.pad(centered, (1, 1, 1, 1), mode="replicate")
            neighbors = [padded[..., y:y + height, x:x + width] for y, x in OFFSETS]
            optical_patch, sar_patch = torch.stack([centered, *neighbors], dim=2).unbind(dim=1)
            adapted, active = adapt_sar(optical_patch, sar_patch, beta)
            patches = torch.stack((optical_patch, adapted), dim=1)
            detail = torch.lerp(patches[:, :, 0], (kernel * patches[:, :, 1:]).sum(dim=2), rho)
            residual = (gain[:, :2] * detail).sum(dim=1, keepdim=True)
            fused = base + residual
        if not return_stats:
            return fused
        with torch.no_grad():
            return fused, dict(hcfdr_beta=beta.mean(), hcfdr_negative_fraction=active.float().mean(),
                               hcfdr_beta_on_negative=(beta * active).sum() / active.sum().clamp_min(1),
                               hcfdr_patch_change_abs=(adapted - sar_patch).abs().mean(),
                               cfdr_gain_s=gain[:, 1].mean(), cfdr_rho_o=rho[:, 0].mean(),
                               cfdr_rho_s=rho[:, 1].mean(), cfdr_min_center=(1 - rho).min(),
                               cfdr_out_of_range=((fused < 0) | (fused > 1)).float().mean())


class HCFDRE2(CFDRE2):
    def __init__(self):
        super().__init__("CF2")
        self.cfdr = HeterogeneousReconstruction(self.cfdr)
        self.variant = VARIANT
