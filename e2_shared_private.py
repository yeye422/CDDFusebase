"""Patch-aligned shared Detail and native, signed private Detail for optical-Base E2."""
import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2


SPEC = dict(shared_channels=16, private_channels=16, patch=16, temperature=0.2, align_weight=0.05)


def patch_alignment(optical, sar, valid):
    """Symmetric within-scene patch retrieval on the shared channels only."""
    with torch.autocast(device_type=optical.device.type, enabled=False):
        mask = valid.float()
        kernel = tuple(min(SPEC["patch"], n) for n in mask.shape[-2:])
        def pool(x):
            return F.avg_pool2d(x, kernel, kernel, ceil_mode=True, count_include_pad=False).flatten(2)
        masses = pool(mask)
        def describe(x):
            # Magnitudes survive patch pooling even if signed high bands cancel.
            tokens = pool(x.float().abs() * mask) / masses.clamp_min(1e-8)
            mean = (tokens * masses).sum(2, keepdim=True) / masses.sum(2, keepdim=True).clamp_min(1e-8)
            return F.normalize((tokens - mean).transpose(1, 2), dim=-1, eps=1e-6)
        o, s = describe(optical), describe(sar)
        valid_patch = masses[:, 0] > 0
        logits = o @ s.transpose(1, 2) / SPEC["temperature"]
        labels = torch.arange(o.size(1), device=o.device).expand(o.size(0), -1)
        def loss(scores):
            # Finite padding also handles a scene with no valid keys.
            return F.cross_entropy(scores.masked_fill(~valid_patch[:, None], -1e4).transpose(1, 2),
                                   labels, reduction="none")
        weights = masses[:, 0] * (valid_patch.sum(1, keepdim=True) >= 2)
        return ((loss(logits) + loss(logits.transpose(1, 2))) * weights).sum() / (2 * weights.sum().clamp_min(1e-8))


class SharedPrivateDetail(nn.Module):
    def __init__(self, e2_head):
        super().__init__()
        width = SPEC["shared_channels"] + SPEC["private_channels"]
        # Keep the E2 per-source 1x1 projections; use their signed outputs.
        self.optical_projection = e2_head.optical_projection[0]
        self.sar_projection = e2_head.sar_projection[0]
        self.scale_branches = e2_head.scale_branches
        self.base_projection = e2_head.base_projection
        self.output = e2_head.output
        self.common = nn.Sequential(nn.Conv2d(2 * SPEC["shared_channels"], width, 3, padding=1), nn.GELU())
        self.private_gain = nn.Conv2d(width, 2 * SPEC["private_channels"], 1)
        nn.init.zeros_(self.private_gain.weight)
        nn.init.zeros_(self.private_gain.bias)
        self.optical_readout = nn.Conv2d(width, e2_head.channels, 1)
        self.sar_readout = nn.Conv2d(width, e2_head.channels, 1)

    def project(self, do, ds):
        return self.optical_projection(do), self.sar_projection(ds)

    def aggregate(self, zo, zs, base):
        co, uo = zo.split((SPEC["shared_channels"], SPEC["private_channels"]), dim=1)
        cs, us = zs.split((SPEC["shared_channels"], SPEC["private_channels"]), dim=1)
        common = self.common(torch.cat((co, cs), dim=1))
        go, gs = self.private_gain(common).tanh().chunk(2, dim=1)
        native = torch.cat((common, uo * (1 + go), us * (1 + gs)), dim=1)
        scales = [branch(native) for branch in self.scale_branches]
        return self.output(torch.cat((native, *scales, self.base_projection(base)), dim=1))

    def forward(self, do, ds, base):
        return self.aggregate(*self.project(do, ds), base)


class SharedPrivateE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_shared_private_patch_v1"

    def __init__(self):
        super().__init__(self.VARIANT)
        with torch.random.fork_rng(devices=[]):
            self.detail_fusion = SharedPrivateDetail(self.detail_fusion)


def stage_loss(model, losses, optical, sar, valid, args, *, fusion):
    (bo, do, bs, ds), diagnostics = model.encode_pair(optical, sar, return_diagnostics=True)
    head = model.detail_fusion
    zo, zs = head.project(do, ds)
    optical_hat = model.reconstruct(optical, bo, head.optical_readout(zo))
    sar_hat = model.reconstruct(sar, bs, head.sar_readout(zs))
    lo, ls = losses.reconstruction(optical, optical_hat), losses.reconstruction(sar, sar_hat, valid)
    rec = (lo + ls) * 0.5
    k = SPEC["shared_channels"]
    align = patch_alignment(zo[:, :k], zs[:, :k], valid)
    stats = dict(reconstruction=rec, optical_reconstruction=lo, sar_reconstruction=ls, patch_alignment=align)
    if not fusion:
        return rec + SPEC["align_weight"] * align, stats
    base = bo + model.base_fusion(bo, bs)
    detail = head.aggregate(zo, zs, base)
    out = model.decoder(None, base, detail)[0]
    value, intensity, gradient = losses.fusion(optical, sar, out, valid, args.gradient_weight)
    stats.update(fusion=value, intensity=intensity, gradient=gradient,
                 gate_mean=(diagnostics["optical_gates"].mean() + diagnostics["sar_gates"].mean()) / 2)
    return value + args.reconstruction_weight * rec + SPEC["align_weight"] * align, stats
