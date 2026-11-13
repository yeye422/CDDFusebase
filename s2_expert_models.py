"""S2 pilot: matched convolution and heterogeneous expert Detail fusion operators.

No changes to LWPD or Base fusion. Top-2 is a ROUTING ablation, not an
inference-speed claim: this reference implementation evaluates all experts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 64
ROUTING_TILE = 16
MODES = ("e2_ft", "conv", "dense", "top2")


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def _inputs(o, s, b):
    if o.ndim != 4 or o.shape != s.shape or o.shape != b.shape or o.shape[1] != CHANNELS:
        raise ValueError("Expected Optical/SAR/Base tensors with same [B,64,H,W] shape")


class OpticalDirectionalExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical = nn.Conv2d(64, 48, 1)
        self.context = nn.Conv2d(128, 48, 1)
        self.horizontal = nn.Conv2d(48, 48, (1, 5), padding=(0, 2), groups=48)
        self.vertical = nn.Conv2d(48, 48, (5, 1), padding=(2, 0), groups=48)
        self.out = nn.Conv2d(144, 64, 1)

    def forward(self, o, s, b):
        x = F.gelu(self.optical(o))
        context = torch.sigmoid(self.context(torch.cat((s, b), 1)))
        return self.out(torch.cat((self.horizontal(x), self.vertical(x), x * context), 1))


class SARContextExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.sar = nn.Conv2d(64 * 3, 48, 1)
        self.spatial = nn.Conv2d(48, 48, 3, padding=1, groups=48)
        self.optical_context = nn.Conv2d(128, 48, 1)
        self.out = nn.Conv2d(96, 64, 1)

    def forward(self, o, s, b):
        coarse = F.avg_pool2d(s, 5, stride=1, padding=2)
        x = F.gelu(self.sar(torch.cat((s, coarse, b), 1)))
        context = torch.sigmoid(self.optical_context(torch.cat((o, b), 1)))
        return self.out(torch.cat((F.gelu(self.spatial(x)), x * context), 1))


class CooperativeLocalExpert(nn.Module):
    """Cross-source 3x3 local matching at half resolution; no hard match labels."""
    def __init__(self):
        super().__init__()
        self.q = nn.Conv2d(64, 24, 1)
        self.k = nn.Conv2d(64, 24, 1)
        self.v = nn.Conv2d(64, 24, 1)
        self.out = nn.Conv2d(64 * 3 + 24, 64, 1)

    def forward(self, o, s, b):
        h, w = o.shape[-2:]
        o2 = F.avg_pool2d(o, 2)
        s2 = F.avg_pool2d(s, 2)
        q = F.normalize(self.q(o2).float(), dim=1, eps=1e-6)
        k = F.normalize(self.k(s2).float(), dim=1, eps=1e-6)
        v = self.v(s2)
        batch, channels, hh, ww = q.shape
        keys = F.unfold(k, 3, padding=1).view(batch, channels, 9, hh, ww)
        values = F.unfold(v, 3, padding=1).view(batch, channels, 9, hh, ww)
        attention = torch.softmax((q.unsqueeze(2) * keys).sum(1) / 0.25, dim=1)
        matched = (attention.unsqueeze(1).to(values.dtype) * values).sum(2)
        matched = F.interpolate(matched, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(torch.cat((o, s, b, matched), 1))


class HeterogeneousExpertFusion(nn.Module):
    """Dense/Top2 differ only in normalized router; weights match by seed."""
    def __init__(self, mode="dense", tile=ROUTING_TILE):
        super().__init__()
        if mode not in ("dense", "top2"):
            raise ValueError(mode)
        self.mode = mode
        self.tile = int(tile)
        self.experts = nn.ModuleList((
            OpticalDirectionalExpert(), SARContextExpert(), CooperativeLocalExpert()
        ))
        self.router = nn.Sequential(nn.Conv2d(64 * 3 + 2, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1))
        self.synthesis = nn.Conv2d(64, 64, 3, padding=1)
        self.last_route = None
        self.last_expert_cosine = None

    def routing(self, o, s, b, *, mode=None):
        _inputs(o, s, b)
        if o.shape[-2] % self.tile or o.shape[-1] % self.tile:
            raise ValueError("Feature H/W must be divisible by routing tile")
        pool = lambda x: F.avg_pool2d(x, self.tile, stride=self.tile)
        eo = pool(o.float().square().mean(1, keepdim=True)).sqrt()
        es = pool(s.float().square().mean(1, keepdim=True)).sqrt()
        relative = (es + 1e-6).log() - (eo + 1e-6).log()
        coarse_b = pool(b.float())
        expanded_b = coarse_b.repeat_interleave(self.tile, -2).repeat_interleave(self.tile, -1)
        local_b = pool((b.float() - expanded_b).square().mean(1, keepdim=True)).sqrt()
        x = torch.cat((pool(o), pool(s), coarse_b, relative.to(o.dtype), local_b.to(o.dtype)), 1)
        logits = self.router(x)
        use = self.mode if mode is None else mode
        if use == "top2":
            indices = logits.topk(2, dim=1).indices
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, indices, True)
            logits = logits.masked_fill(~keep, -torch.inf)
        elif use != "dense":
            raise ValueError(use)
        return logits.float().softmax(1).to(o.dtype)

    def forward(self, o, s, b):
        _inputs(o, s, b)
        weights = self.routing(o, s, b)
        up = F.interpolate(weights, size=o.shape[-2:], mode="nearest")
        outputs = [e(o, s, b) for e in self.experts]
        mixture = sum(up[:, i:i + 1] * output for i, output in enumerate(outputs))
        if not self.training:
            self.last_route = weights.detach()
            flat = [F.normalize((v.detach().float() - v.detach().float().mean((2, 3), keepdim=True)).flatten(1), dim=1) for v in outputs]
            self.last_expert_cosine = [float((flat[i] * flat[j]).sum(1).mean()) for i, j in ((0, 1), (0, 2), (1, 2))]
        return self.synthesis(mixture)


class ConvCapacityControl(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Conv2d(192, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 64, 3, padding=1),
        )

    def forward(self, o, s, b):
        _inputs(o, s, b)
        return self.net(torch.cat((o, s, b), 1))


def matched_conv_hidden(target):
    candidates = range(32, 385, 8)
    return min(candidates, key=lambda h: abs(count_params(ConvCapacityControl(h)) - target))


def make_detail_fusion(mode):
    if mode in ("dense", "top2"):
        return HeterogeneousExpertFusion(mode)
    if mode == "conv":
        target = count_params(HeterogeneousExpertFusion("dense"))
        control = ConvCapacityControl(matched_conv_hidden(target))
        if abs(count_params(control) / target - 1) > 0.10:
            raise RuntimeError("Capacity-matched control differs by more than 10%")
        return control
    raise ValueError(mode)


class S2ModelFactory:
    @staticmethod
    def build(mode):
        # Imported here so operator tests only require PyTorch.
        from e2_base_anchor import OpticalBaseAnchoredE2
        if mode == "e2_ft":
            return OpticalBaseAnchoredE2()
        if mode not in MODES:
            raise ValueError(mode)
        model = OpticalBaseAnchoredE2()
        model.detail_fusion = make_detail_fusion(mode)
        model.variant = f"s2_{mode}_v1"
        return model
