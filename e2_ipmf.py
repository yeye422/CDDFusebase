"""Information-preserving parallel multi-scale refinement of frozen CF1."""

import torch
from torch import nn
from torch.nn import functional as F

from e2_cfdr import CFDRE2, VARIANTS


class ResidualRefinement(nn.Sequential):
    def __init__(self, width, dilation=1):
        super().__init__(
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, features):
        return features + super().forward(features)


def unshuffle(features):
    height, width = features.shape[-2:]
    if height % 2 or width % 2:
        features = F.pad(features, (0, width % 2, 0, height % 2), mode="replicate")
    return F.pixel_unshuffle(features, 2)


class ParallelMultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.optical_projection = nn.Conv2d(512, width, 1)
        self.sar_projection = nn.Conv2d(512, width, 1)
        self.low_projection = nn.Conv2d(2 * width, width, 1)
        self.low_refine = ResidualRefinement(width)
        self.full_projection = nn.Conv2d(128, width, 1)
        self.full_refine = ResidualRefinement(width)
        self.full_to_low = nn.Conv2d(4 * width, width, 1)
        self.regional_refine = ResidualRefinement(width, dilation=2)
        self.low_to_full = nn.Conv2d(width, width, 1)
        self.output_refine = ResidualRefinement(width)
        self.base_head = nn.Conv2d(width, 64, 1)
        self.detail_head = nn.Conv2d(width, 64, 1)
        for head in (self.base_head, self.detail_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, encoded, base, detail):
        bo, do, bs, ds = encoded
        optical = self.optical_projection(unshuffle(torch.cat((bo, do), dim=1)))
        sar = self.sar_projection(unshuffle(torch.cat((bs, ds), dim=1)))
        low = self.low_refine(self.low_projection(torch.cat((optical, sar), dim=1)))
        full = self.full_refine(self.full_projection(torch.cat((base, detail), dim=1)))
        low = self.regional_refine(low + self.full_to_low(unshuffle(full)))
        size = tuple(length + length % 2 for length in base.shape[-2:])
        context = F.interpolate(low, size=size, mode="bilinear", align_corners=False)
        height, width = base.shape[-2:]
        full = self.output_refine(full + self.low_to_full(context[..., :height, :width]))
        delta_base = F.interpolate(self.base_head(low), size=size, mode="bilinear", align_corners=False)
        return delta_base[..., :height, :width], self.detail_head(full)


class IPMFE2(CFDRE2):
    VARIANT = "lwpd_e2_ipmf_v1"

    def __init__(self):
        super().__init__("CF1")
        self.variant = self.VARIANT
        self.requires_grad_(False)
        self.ipmf = ParallelMultiScaleFusion()
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.training = mode
        self.ipmf.train(mode)
        return self

    def load_cf1(self, checkpoint):
        if checkpoint["arm"] != "CF1" or checkpoint["variant"] != VARIANTS["CF1"] or checkpoint["epoch"] != 4:
            raise ValueError("IPMF initialization requires the complete CF1 epoch-4 checkpoint")
        missing, unexpected = self.load_state_dict(checkpoint["model"], strict=False)
        if unexpected or set(missing) != {key for key in self.state_dict() if key.startswith("ipmf.")}:
            raise ValueError(f"CF1 checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    @torch.no_grad()
    def encode_pair(self, optical, sar, return_diagnostics=False):
        return super().encode_pair(optical, sar, return_diagnostics=return_diagnostics)

    def fuse_and_reconstruct(self, optical, sar, encoded, *, return_stats=False):
        with torch.no_grad():
            base, detail = super().fuse_features(*encoded)
        delta_base, delta_detail = self.ipmf(encoded, base, detail)
        fused_detail = detail + delta_detail
        features = torch.cat((base + delta_base, fused_detail), dim=1)
        hidden = self.decoder.encoder_level2(self.decoder.reduce_channel(features))
        logits = self.decoder.output(hidden)
        if self.uses_image_skip:
            logits = logits + optical
        prediction = self.decoder.sigmoid(logits)
        result = self.cfdr(prediction, hidden, fused_detail, optical, sar, return_stats=return_stats)
        if not return_stats:
            return result
        fused, stats = result
        with torch.no_grad():
            for name, delta, initial in (("base", delta_base, base), ("detail", delta_detail, detail)):
                stats[f"{name}_correction_rms_ratio"] = (
                    delta.float().square().mean().sqrt() / initial.float().square().mean().sqrt().clamp_min(1e-8)
                )
        return fused, stats


def require_frozen_cf1(state, cf1):
    frozen = {key for key in state if not key.startswith("ipmf.")}
    if frozen != cf1.keys() or any(not torch.equal(state[key].detach().cpu(), value) for key, value in cf1.items()):
        raise ValueError("IPMF frozen parameters or buffers differ from CF1")
