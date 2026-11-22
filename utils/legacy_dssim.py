"""Kornia 0.2 SSIM-loss semantics for environments without losses.SSIM.

In particular the historical loss clamps (1-SSIM) to [0,1] *before*
dividing by two. Newer SSIMLoss classes have a different upper clamp.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LegacyDSSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("Legacy DSSIM requires a positive odd window")
        x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-x.square() / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        self.register_buffer("window", (gaussian[:, None] * gaussian[None, :])[None, None], persistent=False)
        self.padding = window_size // 2

    def forward(self, source, image):
        if source.ndim != 4 or source.shape != image.shape or source.dtype != image.dtype or source.device != image.device:
            raise ValueError("Legacy DSSIM requires matching NCHW tensors, dtypes and devices")
        kernel = self.window.to(source).expand(source.shape[1], 1, -1, -1)

        def smooth(value):
            return F.conv2d(F.pad(value, (self.padding,) * 4, mode="reflect"), kernel, groups=source.shape[1])

        ms, mi = smooth(source), smooth(image)
        vs, vi = smooth(source.square()) - ms.square(), smooth(image.square()) - mi.square()
        covariance = smooth(source * image) - ms * mi
        similarity = ((2 * ms * mi + 0.01**2) * (2 * covariance + 0.03**2)
                      / ((ms.square() + mi.square() + 0.01**2) * (vs + vi + 0.03**2)))
        return (1 - similarity).clamp(0, 1) / 2
