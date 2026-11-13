"""Fusion-only SAR Detail shrinkage guided by directional support in the SAR image."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from e2_base_anchor import OpticalBaseAnchoredE2


WINDOW = 5
MAX_ATTENUATION = 0.25
INITIAL_ATTENUATION = 0.05
MODES = ("spatial", "mean", "off")
STATS = ("sar_support_mean", "sar_attenuation_mean")
SPEC = dict(window=WINDOW, max_attenuation=MAX_ATTENUATION,
            initial_attenuation=INITIAL_ATTENUATION, mode="spatial")


class SARDirectionalShrinkage(nn.Module):
    def __init__(self):
        super().__init__()
        initial_fraction = INITIAL_ATTENUATION / MAX_ATTENUATION
        self.strength_logits = nn.Parameter(torch.full(
            (3,), math.log(initial_fraction / (1.0 - initial_fraction))
        ))
        horizontal = torch.zeros(WINDOW, WINDOW)
        horizontal[WINDOW // 2] = 1.0
        diagonal = torch.eye(WINDOW)
        windows = torch.stack((horizontal, horizontal.T, diagonal, diagonal.flip(1)))
        self.register_buffer("windows", windows[:, None].repeat(4, 1, 1, 1), persistent=False)

    @torch.no_grad()
    def support(self, sar, haar):
        """Compute signed continuity and spatial coverage on the Detail sub-band grid."""
        with autocast(enabled=False):
            valid = (sar > 0).float()
            image = torch.log1p(sar.float().clamp_min(0.0))
            # Overlapping Haar windows cover both decimation phases on each axis.
            bands = F.conv2d(F.pad(image, (0, 1, 0, 1), mode="replicate"), haar.filters[1:].float())
            bands = bands[:, (0, 1, 2, 2)]
            valid = F.avg_pool2d(F.pad(valid, (0, 1, 0, 1), mode="replicate"), 2, stride=1)
            samples = torch.cat((bands, bands.abs(), bands.square(), valid.expand_as(bands)), dim=1)
            sums = F.conv2d(
                F.pad(samples, (WINDOW // 2,) * 4, mode="replicate"),
                self.windows.float(), groups=samples.shape[1],
            )
            signed, magnitude, energy, coverage = sums.chunk(4, dim=1)
            # The energy term prevents a single large coefficient from scoring as a full line.
            score = (signed.abs() * magnitude / (WINDOW * energy).clamp_min(1e-12)).clamp(0, 1)
            score = torch.cat((score[:, :2], score[:, 2:3].maximum(score[:, 3:4])), dim=1)
            complete = coverage == WINDOW
            complete = torch.cat((complete[:, :2], complete[:, 2:3] & complete[:, 3:4]), dim=1)
            height, width = sar.shape[-2:]
            # A sliding 2x2 window can start one pixel before its target Haar cell.
            pad = (1, width % 2, 1, height % 2)
            score = F.max_pool2d(F.pad(score, pad, mode="replicate"), 3, stride=2)
            incomplete = F.max_pool2d(F.pad((~complete).float(), pad, mode="replicate"), 3, stride=2)
            return score, incomplete == 0

    def forward(self, sar, detail, haar, mode="spatial"):
        if mode == "off":
            return detail, {}
        support, eligible = self.support(sar, haar)
        attenuation = MAX_ATTENUATION * self.strength_logits.float().sigmoid()[None, :, None, None]
        attenuation = attenuation * (1.0 - support) * eligible
        if mode == "mean":
            count = eligible.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
            attenuation = eligible * (attenuation.sum(dim=(-2, -1), keepdim=True) / count)
        with autocast(enabled=False):
            (low, *high), output_size = haar(detail.float())
            delta_bands = tuple(-weight * band for weight, band in zip(attenuation.split(1, dim=1), high))
            correction = haar.inverse((torch.zeros_like(low), *delta_bands), output_size)
        return detail + correction.to(detail.dtype), {
            "sar_support_mean": (support * eligible).sum() / eligible.sum().clamp_min(1),
            "sar_attenuation_mean": attenuation.mean(),
        }


class SARDirectionalE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_sar_directional_v1"

    def __init__(self):
        super().__init__(variant=OpticalBaseAnchoredE2.VARIANT)
        self.variant = self.VARIANT
        self.sar_directional = SARDirectionalShrinkage()
        self.sar_detail_mode = "spatial"

    def prepare_fusion_inputs(self, encoded, sar):
        optical_base, optical_detail, sar_base, sar_detail = encoded
        sar_detail, diagnostics = self.sar_directional(
            sar, sar_detail, self.lwpd.haar, self.sar_detail_mode
        )
        return (optical_base, optical_detail, sar_base, sar_detail), diagnostics
