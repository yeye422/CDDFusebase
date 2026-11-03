"""Dual-source valid-window SSIM loss; independent of evaluation metrics."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class FusionStructureLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        t = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-t.square() / (2 * sigma**2))
        g = g / g.sum()
        self.register_buffer('window', (g[:, None] * g[None, :])[None, None])

    def dissimilarity(self, source, fused):
        kernel = self.window.float()
        conv = lambda x: F.conv2d(x, kernel)
        ms, mf = conv(source), conv(fused)
        vs = conv(source.square()) - ms.square()
        vf = conv(fused.square()) - mf.square()
        cov = conv(source * fused) - ms * mf
        similarity = ((2 * ms * mf + .01**2) * (2 * cov + .03**2)) / (
            (ms.square() + mf.square() + .01**2) * (vs + vf + .03**2)
        )
        # 1-SSIM, not DSSIM. Only trim floating-point excursions below zero.
        return (1 - similarity).clamp_min(0)

    def forward(self, optical, sar, fused, valid):
        if optical.shape != sar.shape or optical.shape != fused.shape or valid.shape != sar.shape:
            raise ValueError('Structure loss expects matching Bx1xHxW tensors')
        if optical.ndim != 4 or optical.shape[1] != 1 or min(optical.shape[-2:]) < self.window_size:
            raise ValueError('Structure loss requires grayscale images at least 11x11')
        with autocast(enabled=False):
            optical, sar, fused = optical.float(), sar.float(), fused.float()
            valid = (valid.detach() > .5).float()
            # Invalid SAR values cannot enter a selected SSIM window.
            sar = torch.where(valid.bool(), sar, torch.zeros_like(sar))
            window_valid = 1 - F.max_pool2d(1 - valid, self.window_size, stride=1)
            optical_loss = self.dissimilarity(optical, fused).flatten(1).mean(1)
            sar_map = self.dissimilarity(sar, fused)
            count = window_valid.flatten(1).sum(1)
            sar_loss = (sar_map * window_valid).flatten(1).sum(1) / count.clamp_min(1)
            available = (count > 0).float()
            per_image = (optical_loss + available * sar_loss) / (1 + available)
            return per_image.mean(), {
                'structure_optical': optical_loss.mean(),
                'structure_sar': sar_loss.mean(),
                'structure_sar_available': available.mean(),
                'structure_valid_fraction': window_valid.mean(),
            }
