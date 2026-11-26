"""Center-preserving image-space dynamic reconstruction for optical-Base E2."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2


ARMS = ("CF1", "CF2")
VARIANTS = {arm: f"lwpd_e2_cfdr_v21_{arm.lower()}" for arm in ARMS}
OFFSETS = tuple((y, x) for y in range(3) for x in range(3) if (y, x) != (1, 1))


class CenterPreservingReconstruction(nn.Module):
    def __init__(self, dynamic):
        super().__init__()
        self.condition = nn.Sequential(nn.Conv2d(128, 32, 1), nn.GELU(),
                                       nn.Conv2d(32, 32, 3, padding=1, groups=32, padding_mode="replicate"), nn.GELU())
        self.gain = nn.Conv2d(32, 3, 1)
        nn.init.zeros_(self.gain.weight)
        with torch.no_grad():
            self.gain.bias.copy_(torch.tensor([0.475, 0.475, 0.05]).log())
        self.dynamic = dynamic
        if dynamic:
            self.neighbors = nn.Conv2d(32, 16, 1)
            self.strength = nn.Conv2d(32, 2, 1)
            nn.init.zeros_(self.neighbors.weight)
            nn.init.zeros_(self.neighbors.bias)
            nn.init.zeros_(self.strength.weight)
            nn.init.constant_(self.strength.bias, math.log(0.2 / 0.8))
        taps = torch.tensor([1., 4., 6., 4., 1.]) / 16
        self.register_buffer("blur_kernel", torch.outer(taps, taps)[None, None], persistent=False)

    def smooth(self, images):
        channels = images.shape[1]
        return F.conv2d(F.pad(images, (2, 2, 2, 2), mode="replicate"),
                        self.blur_kernel.expand(channels, 1, 5, 5), groups=channels)

    def forward(self, prediction, hidden, fused_detail, optical, sar, *, return_stats=False):
        context = self.condition(torch.cat((hidden, fused_detail), dim=1))
        gain = self.gain(context).softmax(dim=1, dtype=torch.float32)
        if self.dynamic:
            logits = self.neighbors(context).unflatten(1, (2, 8))
            kernel = logits.softmax(dim=2, dtype=torch.float32)
            rho = 0.5 * self.strength(context).float().sigmoid()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            inputs = torch.cat((prediction.float(), optical.float(), sar.float()), dim=1)
            base, means = self.smooth(inputs).split((1, 2), dim=1)
            centered = inputs[:, 1:] - means
            if self.dynamic:
                height, width = centered.shape[-2:]
                padded = F.pad(centered, (1, 1, 1, 1), mode="replicate")
                neighborhood = torch.stack([padded[..., y:y + height, x:x + width] for y, x in OFFSETS], dim=2)
                detail = torch.lerp(centered, (kernel * neighborhood).sum(dim=2), rho)
            else:
                detail = centered
            residual = (gain[:, :2] * detail).sum(dim=1, keepdim=True)
            fused = base + residual
        if not return_stats:
            return fused
        with torch.no_grad():
            stats = dict(cfdr_gain_o=gain[:, 0].mean(), cfdr_gain_s=gain[:, 1].mean(),
                         cfdr_gain_null=gain[:, 2].mean(), cfdr_residual_abs=residual.abs().mean(),
                         cfdr_out_of_range=((fused < 0) | (fused > 1)).float().mean())
            if self.dynamic:
                stats.update(cfdr_rho_o=rho[:, 0].mean(), cfdr_rho_s=rho[:, 1].mean(),
                             cfdr_min_center=(1 - rho).min())
        return fused, stats


class CFDRE2(OpticalBaseAnchoredE2):
    def __init__(self, arm):
        super().__init__()
        self.variant = VARIANTS[arm]
        self.cfdr = CenterPreservingReconstruction(dynamic=arm == "CF2")

    def fuse_and_reconstruct(self, optical, sar, encoded, *, return_stats=False):
        base, detail = self.fuse_features(*encoded)
        # Read the same decoder body and sigmoid prediction as the original E2.
        hidden = self.decoder.encoder_level2(self.decoder.reduce_channel(torch.cat((base, detail), dim=1)))
        logits = self.decoder.output(hidden)
        if self.uses_image_skip:
            logits = logits + optical
        prediction = self.decoder.sigmoid(logits)
        return self.cfdr(prediction, hidden, detail, optical, sar, return_stats=return_stats)

    def forward(self, optical, sar):
        return self.fuse_and_reconstruct(optical, sar, self.encode_pair(optical, sar))
