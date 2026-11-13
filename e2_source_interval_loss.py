"""Conservative source-specific gradient floors and a tolerant L2 upper bound."""

import torch
from torch.nn import functional as F

from e2_structure_joint import SourceStructureEvidence
from train_direct_fusion import DirectFusionLosses


INTERVAL = {
    "common_threshold": 0.12,
    "private_threshold": 0.30,
    "private_margin": 0.10,
    "floor_fraction": 0.8,
    "regional_blend": 0.5,
    "upper_ratio": 1.10,
    "upper_slack": 0.02,
    "upper_weight": 2.5,
}
INTERVAL_STATS = (
    "legacy_gradient", "optical_region_fraction", "sar_region_fraction",
    "common_region_fraction", "source_floor_loss", "overshoot_loss", "overshoot_fraction",
)


class SourceIntervalFusionLosses(DirectFusionLosses):
    def __init__(self):
        super().__init__()
        self.evidence = SourceStructureEvidence()
        self.interval_stats = {}

    def signed_sobel(self, image):
        gx = F.conv2d(image, self.sobel.weightx, padding=1)
        gy = F.conv2d(image, self.sobel.weighty, padding=1)
        return torch.cat((gx, gy), dim=1)

    def source_regions(self, optical, sar, valid):
        _, eligible, _, _ = self.evidence(optical, sar, valid)
        # Each location uses at most one source normal. A common response
        # takes precedence over a private response in the same neighborhood.
        common, opt, radar = eligible.amax(dim=1).unbind(dim=1)
        common = common >= INTERVAL["common_threshold"]
        opt_private = (~common & (opt >= INTERVAL["private_threshold"])
                       & ((opt - radar) >= INTERVAL["private_margin"]))
        sar_private = (~common & (radar >= INTERVAL["private_threshold"])
                       & ((radar - opt) >= INTERVAL["private_margin"]))
        masks = torch.stack((common, opt_private, sar_private), dim=1)
        masks = F.interpolate(masks.float(), scale_factor=2, mode="nearest").bool()
        h, w = valid.shape[-2:]
        # Zero padding rejects the frame border as well as incomplete SAR neighborhoods.
        support = F.avg_pool2d(valid.float(), 3, stride=1, padding=1) == 1
        return (masks[..., :h, :w] & support).split(1, dim=1), support

    def fusion(self, optical, sar, fused, valid, gradient_weight):
        with torch.autocast(device_type=fused.device.type, enabled=False):
            optical, sar, fused = optical.float(), sar.float(), fused.float()
            valid = valid > 0.5
            with torch.no_grad():
                go, gs = self.signed_sobel(optical), self.signed_sobel(sar)
                (common, opt_private, sar_private), support = self.source_regions(optical, sar, valid)
                source = torch.where(sar_private, gs, go)
                # Common matches use the optical grid and its normal once.
                normal = F.normalize(source, dim=1, eps=1e-6)
                magnitude = torch.linalg.vector_norm(source, dim=1, keepdim=True)
                # A half-grid match does not imply an edge at every pixel in the cell.
                has_edge = magnitude > INTERVAL["upper_slack"]
                common, opt_private, sar_private = (mask & has_edge for mask in (common, opt_private, sar_private))
                regions = common | opt_private | sar_private
                floor = INTERVAL["floor_fraction"] * magnitude
                max_source = torch.maximum(
                    torch.linalg.vector_norm(go, dim=1, keepdim=True),
                    torch.linalg.vector_norm(gs, dim=1, keepdim=True),
                )
                ceiling = INTERVAL["upper_ratio"] * max_source + INTERVAL["upper_slack"]
                intensity_target = torch.where(valid, torch.maximum(optical, sar), optical)
                go_l1, gs_l1 = (g.abs().sum(dim=1, keepdim=True) for g in (go, gs))
                gradient_target = torch.where(valid, torch.maximum(go_l1, gs_l1), go_l1)

            gf = self.signed_sobel(fused)
            legacy = (gf.abs().sum(dim=1, keepdim=True) - gradient_target).abs()
            along_normal = (gf * normal).sum(dim=1, keepdim=True).abs()
            shortfall = (floor - along_normal).relu()
            blend = INTERVAL["regional_blend"] * regions
            gradient = torch.lerp(legacy, shortfall, blend).mean()
            magnitude_f = torch.linalg.vector_norm(gf, dim=1, keepdim=True)
            excess = (magnitude_f - ceiling).relu().masked_fill(~support, 0)
            upper = excess.sum() / support.sum().clamp_min(1)
            intensity = F.l1_loss(fused, intensity_target)
            total = intensity + float(gradient_weight) * gradient + INTERVAL["upper_weight"] * upper
            self.interval_stats = {
                "legacy_gradient": legacy.mean().detach(),
                "optical_region_fraction": opt_private.float().mean(),
                "sar_region_fraction": sar_private.float().mean(),
                "common_region_fraction": common.float().mean(),
                "source_floor_loss": (shortfall.detach() * regions).sum() / regions.sum().clamp_min(1),
                "overshoot_loss": upper.detach(),
                "overshoot_fraction": (excess.detach() > 0).float().mean(),
            }
            return total, intensity, gradient
