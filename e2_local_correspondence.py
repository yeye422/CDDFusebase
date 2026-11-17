"""E2 Detail fusion with bidirectional correspondence and a no-match option."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct_fusion import MultiScaleDetailFusion
from e2_base_anchor import OpticalBaseAnchoredE2
from lwpd import HaarDWT2D


OFFSETS = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1))
CENTER = OFFSETS.index((0, 0))
TEMPERATURE = 0.12
NO_MATCH_SCORE = 0.45
EPSILON = 1e-6


def neighbors(tensor):
    height, width = tensor.shape[-2:]
    padded = F.pad(tensor, (1, 1, 1, 1))
    return tuple(padded[..., 1 + y:1 + y + height, 1 + x:1 + x + width] for y, x in OFFSETS)


def reverse_edges(edges, fill):
    """Convert edge p -> p + offset to the inverse edge on the neighbor grid."""
    height, width = edges.shape[-2:]
    padded = F.pad(edges.flip((1,)), (1, 1, 1, 1), value=fill)
    return torch.cat(tuple(
        padded[:, i:i + 1, 1 + y:1 + y + height, 1 + x:1 + x + width]
        for i, (y, x) in enumerate(OFFSETS)
    ), dim=1)


def reciprocal_weights(optical, sar, sar_valid):
    """Cosine matching on unit descriptors with abstention and soft reciprocity."""
    scores = torch.cat(tuple((optical * neighbor).sum(dim=1, keepdim=True)
                             for neighbor in neighbors(sar)), dim=1)
    valid = torch.cat(neighbors(sar_valid), dim=1)
    scores = scores.masked_fill(~valid | (scores <= NO_MATCH_SCORE), -torch.inf)
    reverse = reverse_edges(scores, -torch.inf)

    def probabilities(logits):
        null = torch.full_like(logits[:, :1], NO_MATCH_SCORE)
        return (torch.cat((logits, null), dim=1) / TEMPERATURE).softmax(dim=1)[:, :-1]

    optical_to_sar = probabilities(scores)
    sar_to_optical = probabilities(reverse)
    return (optical_to_sar * reverse_edges(sar_to_optical, 0.0),
            sar_to_optical * reverse_edges(optical_to_sar, 0.0))


class LocalCorrespondenceDetailFusion(MultiScaleDetailFusion):
    def __init__(self):
        super().__init__()
        # Replace the largest-dilation branch with a 32-channel exchange branch.
        del self.scale_branches[-1]
        self.haar = HaarDWT2D()
        self.optical_value = nn.Conv2d(self.channels, 16, 1, bias=False)
        self.sar_value = nn.Conv2d(self.channels, 16, 1, bias=False)

    def _describe(self, optical_detail, sar_detail, sar_valid):
        with torch.autocast(device_type=optical_detail.device.type, enabled=False):
            optical, size = self.haar(optical_detail.float())
            sar, _ = self.haar(sar_detail.float())
            optical_bands, sar_bands = (torch.stack(bands[1:], dim=1) for bands in (optical, sar))
            if sar_valid is None:
                valid = torch.ones_like(sar[0][:, :1], dtype=torch.bool)
            else:
                h, w = size
                valid = F.avg_pool2d(F.pad(sar_valid.float(), (0, w % 2, 0, h % 2), mode="replicate"), 2) == 1
            # Average magnitudes on the subband grid. Pooling the spatial Detail
            # itself would cancel the three zero-mean Haar high bands.
            descriptors = tuple(F.normalize(
                F.avg_pool2d(bands.flatten(1, 2).abs() * mask, 3, stride=1, padding=1,
                             count_include_pad=False),
                dim=1, eps=EPSILON,
            ) for bands, mask in ((optical_bands, 1), (sar_bands, valid)))
            weights = reciprocal_weights(*descriptors, valid)
        return optical_bands, sar_bands, size, weights

    def _exchange(self, bands, projection, weights, size):
        batch, directions = bands.shape[:2]
        values = projection(bands.flatten(0, 1)).reshape(batch, directions * projection.out_channels, *bands.shape[-2:])
        exchanged = sum(weights[:, i:i + 1].to(values.dtype) * value for i, value in enumerate(neighbors(values)))
        high = exchanged.unflatten(1, (directions, projection.out_channels)).unbind(1)
        return self.haar.inverse((torch.zeros_like(high[0]), *high), size)

    def forward(self, optical_detail, sar_detail, fused_base, *, sar_valid=None, return_diagnostics=False):
        optical, sar, base = self._project_inputs(optical_detail, sar_detail, fused_base)
        if sar_valid is not None and sar_valid.shape != optical_detail[:, :1].shape:
            raise ValueError("Expected a single-channel SAR validity map on the Detail grid")
        joint = torch.cat((optical, sar), dim=1)
        local = [branch(joint) for branch in self.scale_branches]
        o_bands, s_bands, size, (os, so) = self._describe(optical_detail, sar_detail, sar_valid)
        sar_at_optical = self._exchange(s_bands, self.sar_value, os, size)
        optical_at_sar = self._exchange(o_bands, self.optical_value, so, size)
        fused = self.output(torch.cat((optical, sar, *local, sar_at_optical, optical_at_sar, base), dim=1))
        if not return_diagnostics:
            return fused
        optical_mass, sar_mass = (weights.sum(dim=1) for weights in (os, so))
        return fused, {
            "match_mass": optical_mass.mean(),
            "offcenter_mass": (optical_mass - os[:, CENTER]).mean(),
            "optical_zero_exchange_fraction": (optical_mass == 0).float().mean(),
            "sar_zero_exchange_fraction": (sar_mass == 0).float().mean(),
        }


class LocalCorrespondenceE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_local_correspondence_v1"

    def __init__(self):
        super().__init__(self.VARIANT)
        self.detail_fusion = LocalCorrespondenceDetailFusion()

    def prepare_fusion_inputs(self, encoded, sar):
        return (*encoded, sar > 0), {}

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail,
                      sar_valid=None, *, return_diagnostics=False):
        base = optical_base + self.base_fusion(optical_base, sar_base)
        detail = self.detail_fusion(optical_detail, sar_detail, base, sar_valid=sar_valid,
                                    return_diagnostics=return_diagnostics)
        return base, detail
