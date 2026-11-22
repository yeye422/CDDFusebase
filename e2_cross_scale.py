"""Small source-wise cross-scale conditioning probe for optical-Base E2.

The signed, full-resolution content path stays in the original fusion head.
Only descriptors of Haar magnitudes are pooled. Relation and ordinary-context
controls share parameters, resolutions, and convolutional work; their three
nonlinear summary channels differ. These are descriptors, not quality labels.
"""

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2
from lwpd import HaarDWT2D


MODES = ("relation", "wide", "shuffled")
ALIGNMENTS = ("aligned", "shifted", "disabled")
VARIANT_PREFIX = "lwpd_e2_cross_scale_v1_"
SPEC = dict(descriptor_channels=8, hidden_channels=16, scales=[1, 2, 4],
            max_modulation=0.1, descriptor="normalized_absolute_signed_haar_bands",
            context="source_wise_descriptor_pyramid", value="original_projected_detail",
            shuffle="half_image_roll_of_full_resolution_modulation",
            initialization="zero_final_condition_projection")
DIAGNOSTICS = ("optical_modulation_rms", "sar_modulation_rms",
               "optical_modulation_mean", "sar_modulation_mean")


def resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def cross_scale_summaries(contexts, kind):
    """Return informative, equally sized summaries for both controls."""
    if kind == "relation":
        normalized = [F.normalize(x, dim=1, eps=1e-6) for x in contexts]
        return torch.cat([(normalized[i] * normalized[j]).sum(1, keepdim=True)
                          for i, j in ((0, 1), (0, 2), (1, 2))], dim=1)
    if kind == "wide":
        return torch.cat([x.mean(1, keepdim=True) for x in contexts], dim=1)
    raise ValueError(f"Unknown context kind: {kind}")


class CrossScaleConditioner(nn.Module):
    def __init__(self, channels=64, projected_channels=32, mode="relation"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown cross-scale mode: {mode}")
        self.mode = mode
        self.haar = HaarDWT2D()
        width = SPEC["descriptor_channels"]
        self.descriptor = nn.Conv2d(3 * channels, width, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(3 * width + 3, SPEC["hidden_channels"], 1), nn.GELU(),
            nn.Conv2d(SPEC["hidden_channels"], projected_channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def describe(self, detail):
        bands, _ = self.haar(detail)
        magnitudes = torch.cat([band.abs() for band in bands[1:]], dim=1)
        # Normalize channel energy at each coordinate, without interpreting it
        # as SAR reliability. Never average-pool reconstructed signed Detail.
        normalized = F.normalize(magnitudes, dim=1, eps=1e-6)
        return F.gelu(self.descriptor(normalized))

    @staticmethod
    def contexts(descriptor):
        contexts = [descriptor]
        coarse = descriptor
        for _ in range(2):
            coarse = F.avg_pool2d(coarse, 2, stride=2, ceil_mode=True,
                                  count_include_pad=False)
            contexts.append(resize(coarse, descriptor.shape[-2:]))
        return contexts

    def forward(self, detail, output_size, alignment=None):
        alignment = alignment or ("shifted" if self.mode == "shuffled" else "aligned")
        if alignment not in ALIGNMENTS:
            raise ValueError(f"Unknown condition alignment: {alignment}")
        if alignment == "disabled":
            return detail.new_zeros(detail.shape[0], self.gate[-1].out_channels, *output_size)
        with torch.autocast(device_type=detail.device.type, enabled=False):
            descriptor = self.describe(detail.float())
            contexts = self.contexts(descriptor)
            kind = "wide" if self.mode == "wide" else "relation"
            summaries = cross_scale_summaries(contexts, kind)
            logits = self.gate(torch.cat((*contexts, summaries), dim=1))
            delta = SPEC["max_modulation"] * resize(logits.tanh(), output_size)
        if alignment == "shifted":
            delta = delta.roll(tuple(size // 2 for size in output_size), dims=(-2, -1))
        return delta


class CrossScaleDetailFusion(nn.Module):
    def __init__(self, keep_fusion, mode="relation"):
        super().__init__()
        self.keep_fusion = keep_fusion
        projected = keep_fusion.optical_projection[0].out_channels
        self.conditioner = CrossScaleConditioner(keep_fusion.channels, projected, mode)

    def forward(self, optical_detail, sar_detail, fused_base, *, condition=None,
                return_diagnostics=False):
        keep = self.keep_fusion
        optical, sar, base = keep._project_inputs(optical_detail, sar_detail, fused_base)
        optical_delta = self.conditioner(optical_detail, optical.shape[-2:], condition).to(optical.dtype)
        sar_delta = self.conditioner(sar_detail, sar.shape[-2:], condition).to(sar.dtype)
        optical = optical + optical * optical_delta
        sar = sar + sar * sar_delta
        joint = torch.cat((optical, sar), dim=1)
        scales = [branch(joint) for branch in keep.scale_branches]
        fused = keep.output(torch.cat((optical, sar, *scales, base), dim=1))
        if not return_diagnostics:
            return fused
        return fused, dict(
            optical_modulation_rms=optical_delta.float().square().mean().sqrt(),
            sar_modulation_rms=sar_delta.float().square().mean().sqrt(),
            optical_modulation_mean=optical_delta.float().mean(),
            sar_modulation_mean=sar_delta.float().mean(),
        )


class CrossScaleE2(OpticalBaseAnchoredE2):
    def __init__(self, mode="relation"):
        if mode not in MODES:
            raise ValueError(f"Unknown cross-scale mode: {mode}")
        # Construct every original module before the new branch, so the same
        # seed gives the same original-module initialization as archived E2.
        super().__init__()
        self.variant = VARIANT_PREFIX + mode
        self.detail_fusion = CrossScaleDetailFusion(self.detail_fusion, mode)

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail, *,
                      condition=None, return_diagnostics=False):
        base = optical_base + self.base_fusion(optical_base, sar_base)
        return base, self.detail_fusion(optical_detail, sar_detail, base,
                                      condition=condition, return_diagnostics=return_diagnostics)
