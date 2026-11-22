"""V2 with source-only conflict masking of both max-Sobel supervision terms."""

import torch
from torch.nn import functional as F

import e2_local_correction_v2 as v2


RULE = dict(min_contrast=6 / 255, flank_offset=2, tangent_length=7, min_tangent_votes=5,
            search_radius=4, min_peak_separation=3, conflict_weight=0.0)
SPEC = dict(v2.SPEC, conflict_rule=RULE)
FIELDS = (*v2.FIELDS, "sobel_conflict_fraction")
reconstruction_forward = v2.reconstruction_forward


class LocalCorrectionV3(v2.LocalCorrectionV2):
    VARIANT = "lwpd_e2_local_correction_v3"


def horizontal_conflict(optical, sar, valid):
    """Search the two signed peaks along x, with tangential support along y."""
    # A 3x3 flank mean at +/-2, then seven vertical samples and +/-4 normal search.
    offset, radius = RULE["flank_offset"], RULE["search_radius"]
    length = RULE["tangent_length"]
    images = torch.cat((optical, sar), 1)
    smooth = F.avg_pool2d(images, 3, 1, 1)
    extended = F.pad(smooth, (offset, offset, 0, 0), mode="replicate")
    contrast = extended[..., 2 * offset:] - extended[..., :-2 * offset]
    def tangent_mean(x):
        return F.avg_pool2d(x, (length, 1), 1, (length // 2, 0))
    signed = tangent_mean(contrast)
    # Only signed peaks supported by enough individual tangent samples can compete.
    threshold = RULE["min_contrast"]
    pos = tangent_mean((contrast >= threshold).float()) * length
    neg = tangent_mean((contrast <= -threshold).float()) * length
    stable = ((pos >= RULE["min_tangent_votes"] - 1e-5) & (signed >= threshold)) | (
        (neg >= RULE["min_tangent_votes"] - 1e-5) & (signed <= -threshold))
    strength, indices = F.max_pool2d(signed.abs() * stable, (1, 2 * radius + 1), 1,
                                     (0, radius), return_indices=True)
    peak = signed.flatten(2).gather(2, indices.flatten(2)).reshape_as(signed)
    opposite = peak[:, :1] * peak[:, 1:] < 0
    separated = (indices[:, :1] - indices[:, 1:]).abs() >= RULE["min_peak_separation"]
    # Ignore borders and SAR nodata throughout both flank/search footprints.
    ry, rx = 1 + length // 2, 1 + offset + radius
    valid_footprint = F.avg_pool2d(valid.float(), (2 * ry + 1, 2 * rx + 1), 1, (ry, rx)) == 1
    return opposite & separated & (strength.amin(1, keepdim=True) >= threshold) & valid_footprint


@torch.no_grad()
def conflict_mask(optical, sar, valid):
    o, s = optical.detach().float(), sar.detach().float()
    x = horizontal_conflict(o, s, valid)
    y = horizontal_conflict(o.transpose(-1, -2), s.transpose(-1, -2), valid.transpose(-1, -2))
    return x | y.transpose(-1, -2)


class LocalFusionObjective:
    """Reuse the original reconstruction criterion; prepare source targets once per batch."""
    def __init__(self, criterion, optical, sar, valid):
        self.reconstruction = criterion.reconstruction
        self.sobel = criterion.sobel
        with torch.autocast(device_type=optical.device.type, enabled=False), torch.no_grad():
            o, s, active = optical.float(), sar.float(), valid > .5
            self.mask = conflict_mask(o, s, active)
            self.weight = 1 - self.mask.float() * (1 - RULE["conflict_weight"])
            go, gs = self.sobel(o), self.sobel(s)
            self.intensity_target = torch.where(active, torch.maximum(o, s), o)
            self.gradient_target = torch.where(active, torch.maximum(go, gs), go)

    def fusion(self, optical, sar, image, valid, gradient_weight):
        with torch.autocast(device_type=image.device.type, enabled=False):
            intensity = F.l1_loss(image.float(), self.intensity_target)
            gradient = ((self.sobel(image.float()) - self.gradient_target).abs() * self.weight).mean()
        return intensity + gradient_weight * gradient, intensity, gradient


def fusion_forward(model, losses, optical, sar, valid, args):
    objective = LocalFusionObjective(losses, optical, sar, valid)
    total, stats = v2.fusion_forward(model, objective, optical, sar, valid, args)
    stats["sobel_conflict_fraction"] = objective.mask.float().mean()
    return total, stats
