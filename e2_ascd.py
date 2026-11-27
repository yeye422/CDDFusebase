"""ASCD-V1 on the frozen optical-Base-anchored LWPD-E2 frontend."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2


def refinement(width, dilation=1):
    return nn.Sequential(
        nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width),
        nn.GELU(),
        nn.Conv2d(width, width, 1),
    )


class MultiScaleStructuralCollaboration(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = refinement(width)
        self.regional = refinement(width, dilation=2)
        self.sar_gate = nn.Conv2d(2 * width, width, 1)
        self.sar_value = nn.Conv2d(width, width, 1)
        self.context_projection = nn.Conv2d(width, width, 1)
        self.full_resolution = refinement(width)

    def forward(self, optical, sar):
        # Degenerate image axes still need one context cell.
        pool_size = tuple(min(2, size) for size in optical.shape[-2:])
        low_optical = F.avg_pool2d(optical, pool_size)
        low_sar = F.avg_pool2d(sar, pool_size)
        low_optical = low_optical + self.local(low_optical)
        low_optical = low_optical + self.regional(low_optical)
        gate = self.sar_gate(torch.cat((low_optical, low_sar), dim=1)).sigmoid()
        regional = low_optical + gate * self.sar_value(low_sar)
        context = self.context_projection(F.interpolate(
            regional, size=optical.shape[-2:], mode="bilinear", align_corners=False,
        ))
        structure = optical + context
        return structure + self.full_resolution(structure), context


class StructureConditionedComplementaryRefinement(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.optical_gate = nn.Conv2d(2 * width, width, 1)
        self.refine = refinement(width)

    def forward(self, context, sar):
        gate = self.optical_gate(torch.cat((context, sar), dim=1)).sigmoid()
        return sar + gate * self.refine(sar)


class SourceConstrainedGeneration(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.structure_head = nn.Conv2d(width, 1, 3, padding=1)
        self.optical_head = nn.Conv2d(width, 1, 1)
        self.sar_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.structure_head.weight)
        nn.init.zeros_(self.structure_head.bias)
        for head in (self.optical_head, self.sar_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(9.5))
        taps = torch.tensor([1., 4., 6., 4., 1.]) / 16
        self.register_buffer("blur_taps", taps.view(1, 1, 1, 5), persistent=False)

    def smooth(self, images):
        channels = images.shape[1]
        horizontal = self.blur_taps.to(dtype=images.dtype).expand(channels, 1, 1, 5)
        images = F.conv2d(F.pad(images, (2, 2, 0, 0), mode="replicate"), horizontal, groups=channels)
        return F.conv2d(F.pad(images, (0, 0, 2, 2), mode="replicate"),
                        horizontal.transpose(-1, -2), groups=channels)

    def forward(self, optical, sar, structure, complementary, *, return_stats=False):
        optical_logit = self.optical_head(structure)
        sar_logit = self.sar_head(complementary)
        logits = torch.cat((optical_logit, sar_logit, torch.zeros_like(optical_logit)), dim=1)
        gain = logits.softmax(dim=1, dtype=torch.float32)
        correction = self.structure_head(structure)
        with torch.autocast(device_type=optical.device.type, enabled=False):
            optical, sar = optical.float(), sar.float()
            base, optical_mean, sar_mean = self.smooth(
                torch.cat((optical + correction.float(), optical, sar), dim=1)
            ).split(1, dim=1)
            fused = base + gain[:, :1] * (optical - optical_mean) + gain[:, 1:2] * (sar - sar_mean)
        if not return_stats:
            return fused
        with torch.no_grad():
            stats = dict(gain_optical=gain[:, 0].mean(), gain_sar=gain[:, 1].mean(),
                         gain_null=gain[:, 2].mean(), structure_correction_abs=correction.float().abs().mean(),
                         out_of_range=((fused < 0) | (fused > 1)).float().mean())
        return fused, stats


class ASCDDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.optical_projection = nn.Conv2d(128, width, 1)
        self.sar_projection = nn.Conv2d(128, width, 1)
        self.mssc = MultiScaleStructuralCollaboration(width)
        self.scc = StructureConditionedComplementaryRefinement(width)
        self.scg = SourceConstrainedGeneration(width)

    def forward(self, optical, sar, fused_base, optical_detail, sar_base, sar_detail, *, return_stats=False):
        optical_features = self.optical_projection(torch.cat((fused_base, optical_detail), dim=1))
        sar_features = self.sar_projection(torch.cat((sar_base, sar_detail), dim=1))
        structure, context = self.mssc(optical_features, sar_features)
        complementary = self.scc(context, sar_features)
        return self.scg(optical, sar, structure, complementary, return_stats=return_stats)


class ASCDE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_ascd_v1"
    FROZEN_PREFIXES = ("encoder.", "lwpd.", "base_fusion.")

    def __init__(self):
        super().__init__(variant=self.VARIANT)
        del self.detail_fusion, self.decoder
        self.requires_grad_(False)
        self.ascd = ASCDDecoder()
        self.train(False)

    @classmethod
    def frontend_state(cls, state):
        return {key: value for key, value in state.items() if key.startswith(cls.FROZEN_PREFIXES)}

    def load_frontend(self, backbone):
        missing, unexpected = self.load_state_dict(self.frontend_state(backbone), strict=False)
        if unexpected or set(missing) != {key for key in self.state_dict() if key.startswith("ascd.")}:
            raise ValueError(f"Unexpected ASCD frontend state: missing={missing}, unexpected={unexpected}")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.lwpd.eval()
        self.base_fusion.eval()
        return self

    @torch.no_grad()
    def frontend(self, optical, sar):
        optical_base, optical_detail, sar_base, sar_detail = self.encode_pair(optical, sar)
        fused_base = optical_base + self.base_fusion(optical_base, sar_base)
        return fused_base, optical_detail, sar_base, sar_detail

    def forward(self, optical, sar, *, return_stats=False):
        return self.ascd(optical, sar, *self.frontend(optical, sar), return_stats=return_stats)
