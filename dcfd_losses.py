"""FP32 source-structure and frozen-CF1 reconstruction losses for DCFD."""

import torch
from torch import nn
from torch.nn import functional as F


def full_support(valid, width):
    radius = width // 2
    invalid = F.pad((~valid).float(), (radius, radius, radius, radius), mode="replicate")
    return F.max_pool2d(invalid, width, stride=1) == 0


class DCFDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        coords = torch.arange(-5, 6, dtype=torch.float32)
        weights = (-0.5 * (coords / 1.5).square()).softmax(dim=0)
        self.register_buffer("gaussian", weights.view(1, 1, 1, 11), persistent=False)

    def ssim_map(self, fused, source):
        moments = torch.cat((fused, source, fused.square(), source.square(), fused * source), dim=0)
        moments = F.conv2d(F.pad(moments, (5, 5, 0, 0), mode="replicate"), self.gaussian)
        moments = F.conv2d(F.pad(moments, (0, 0, 5, 5), mode="replicate"), self.gaussian.transpose(-1, -2))
        mean_f, mean_s, square_f, square_s, product = moments.chunk(5, dim=0)
        var_f = (square_f - mean_f.square()).clamp_min(0)
        var_s = (square_s - mean_s.square()).clamp_min(0)
        covariance = product - mean_f * mean_s
        return ((2 * mean_f * mean_s + 0.01**2) * (2 * covariance + 0.03**2)
                / ((mean_f.square() + mean_s.square() + 0.01**2) * (var_f + var_s + 0.03**2)))

    def structure(self, fused, optical, sar, valid):
        optical_error = (1 - self.ssim_map(fused, optical)).mean(dim=(1, 2, 3))
        sar_error = 1 - self.ssim_map(fused, sar)
        windows = full_support(valid, 11)
        support = windows.sum(dim=(1, 2, 3))
        sar_error = (sar_error * windows).sum(dim=(1, 2, 3)) / support.clamp_min(1)
        return torch.where(support > 0, 0.5 * (optical_error + sar_error), optical_error).mean()

    def forward(self, terms, reference, optical, sar, valid):
        with torch.autocast(device_type=optical.device.type, enabled=False):
            fused, target = terms.fused.float(), reference.fused.float()
            valid = valid.bool()
            full = self.structure(fused, optical.float(), sar.float(), valid)
            smooth_valid = full_support(valid, 5)
            base_stack = torch.cat((terms.base.float(), terms.source_base.float()), dim=1)
            base = fused.new_zeros(())
            for factor in (2, 4):
                low, low_optical, low_sar = F.avg_pool2d(base_stack, factor).split(1, dim=1)
                low_valid = F.avg_pool2d(smooth_valid.float(), factor) == 1
                base = base + 0.5 * self.structure(low, low_optical, low_sar, low_valid)
            detail_delta = F.l1_loss(terms.detail.float(), reference.detail.float())
            gradient = sum(F.l1_loss(fused.diff(dim=dim), target.diff(dim=dim)) for dim in (-1, -2))
            detail = detail_delta + gradient
            tether = F.l1_loss(terms.base.float(), reference.base.float())
            total = base + 0.5 * full + 2 * detail + 0.15 * tether
        return total, dict(loss_base=base, loss_full=full, loss_detail=detail, loss_tether=tether,
                           base_delta_mae=tether, detail_delta_mae=detail_delta)
