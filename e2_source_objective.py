"""Experimental source-contrast supervision and optional late source readout.

The fixed evidence is a proxy, not a SAR noise model or a truth/confidence map.
Legacy E2 models and their loss remain unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2
from train_direct_fusion import DirectFusionLosses


SPEC = {
    "version": 1,
    "low_weight": 1.0,
    "preserve_weight": 5.0,
    "uncertain_weight": 1.0,
    "contrast_beta": 0.02,
    "evidence_scale_floor": 0.01,
    "low_operator": "replicate_pad_2x2_mean_after_valid_pixelwise_max",
    "evidence": "fixed_signed_strip_contrast_coherence_and_scale_agreement",
    "directions": ((0, 1), (1, 0), (1, 1), (1, -1)),
}
READOUT_SPEC = {"branch_width": 32, "max_gate": 1.0, "logit_bound": 2.0}
LOSS_STATS = (
    "low_intensity", "preserve", "preserve_optical", "preserve_sar", "preserve_common",
    "uncertain", "optical_support", "sar_support", "common_support", "ambiguous_support",
    "uncertain_support", "sar_valid_support", "legacy_intensity", "legacy_gradient",
)
READOUT_STATS = ("optical_readout_gate", "sar_readout_gate", "optical_logit_rms", "sar_logit_rms")


def lowpass(image):
    """Image-domain Haar LL/2, including the original odd-size padding policy."""
    h, w = image.shape[-2:]
    return F.avg_pool2d(F.pad(image, (0, w % 2, 0, h % 2), mode="replicate"), 2)


def eroded_support(valid, radius=3):
    width = 2 * radius + 1
    return F.avg_pool2d(valid.float(), width, stride=1, padding=radius) == 1


class SourceContrastLosses(DirectFusionLosses):
    def __init__(self):
        super().__init__()
        kernels = torch.zeros(4, 1, 5, 5)
        for index, (ny, nx) in enumerate(SPEC["directions"]):
            ty, tx = -nx, ny
            for t in (-1, 0, 1):
                kernels[index, 0, 2 + ny + t * ty, 2 + nx + t * tx] += 1 / 3
                kernels[index, 0, 2 - ny + t * ty, 2 - nx + t * tx] -= 1 / 3
        self.register_buffer("contrast_kernels", kernels)
        self.stats = {}

    def contrast(self, image):
        return F.conv2d(image, self.contrast_kernels.float(), padding=2)

    @torch.no_grad()
    def evidence(self, optical, sar, valid):
        """No model output, learnable masks, or manual labels enter this path."""
        active = valid > 0.5
        radar = torch.where(active, sar, torch.zeros_like(sar))
        o_support = eroded_support(torch.ones_like(valid))
        s_support = eroded_support(active)

        def source(image, support):
            signed = self.contrast(image)
            blurred = F.avg_pool2d(image, 3, stride=1, padding=1)
            coarse = self.contrast(blurred)
            amplitude = signed.abs()
            count = support.flatten(1).sum(1) * signed.shape[1]
            scale = (amplitude * support).flatten(1).sum(1) / count.clamp_min(1)
            prominence = amplitude / (amplitude + scale[:, None, None, None] + SPEC["evidence_scale_floor"])
            coherence = (F.avg_pool2d(signed, 3, stride=1, padding=1).abs()
                         / F.avg_pool2d(amplitude, 3, stride=1, padding=1).clamp_min(1e-6)).clamp(0, 1)
            agreement = (signed * coarse > 0).float() * (
                torch.minimum(amplitude, coarse.abs()) / torch.maximum(amplitude, coarse.abs()).clamp_min(1e-6))
            return signed, prominence * coherence * agreement * support

        co, wo = source(optical, o_support)
        cs, ws = source(radar, s_support)
        shared = torch.minimum(wo, ws)
        common = shared * (co * cs > 0)
        ambiguous = shared * (co * cs < 0)
        optical_private = (wo - shared).clamp_min(0)
        sar_private = (ws - shared).clamp_min(0)
        uncertain = (1 - torch.maximum(wo, ws) + ambiguous).clamp(0, 1) * o_support
        common_target = (wo * co + ws * cs) / (wo + ws).clamp_min(1e-6)
        return {"optical_target": co, "sar_target": cs, "common_target": common_target,
                "optical": optical_private, "sar": sar_private, "common": common,
                "ambiguous": ambiguous, "uncertain": uncertain, "sar_valid": s_support}

    def fusion(self, optical, sar, fused, valid, gradient_weight):
        if optical.shape != sar.shape or optical.shape != fused.shape or optical.shape != valid.shape:
            raise ValueError("Source contrast loss requires equal Bx1xHxW inputs")
        if optical.ndim != 4 or optical.shape[1] != 1 or min(optical.shape[-2:]) < 7:
            raise ValueError("Source contrast loss requires grayscale images at least 7x7")
        with torch.autocast(device_type=fused.device.type, enabled=False):
            optical, sar, fused, valid = (x.float() for x in (optical, sar, fused, valid))
            active = valid > 0.5
            if not (torch.isfinite(optical).all() and torch.isfinite(fused).all()
                    and torch.isfinite(sar[active]).all() and torch.isfinite(valid).all()):
                raise ValueError("Non-finite source/output on valid support")
            radar = torch.where(active, sar, torch.zeros_like(sar))
            with torch.no_grad():
                targets = torch.where(active, torch.maximum(optical, radar), optical)
                evidence = self.evidence(optical, radar, valid)
                gradient_target = torch.where(eroded_support(active, radius=1),
                                              torch.maximum(self.sobel(optical), self.sobel(radar)),
                                              self.sobel(optical))
            low = (lowpass(fused) - lowpass(targets)).abs().mean()
            response = self.contrast(fused)
            group_values, available = [], []
            for name in ("optical", "sar", "common"):
                error = F.smooth_l1_loss(response, evidence[name + "_target"],
                                         beta=SPEC["contrast_beta"], reduction="none")
                mass = evidence[name].flatten(1).sum(1)
                group_values.append((error * evidence[name]).flatten(1).sum(1) / mass.clamp_min(1e-6))
                available.append(mass > 1e-6)
            values, present = torch.stack(group_values, 1), torch.stack(available, 1)
            preserve = ((values * present).sum(1) / present.sum(1).clamp_min(1)).mean()
            legacy_map = (self.sobel(fused) - gradient_target).abs()
            uncertainty = evidence["uncertain"].mean(1, keepdim=True)
            uncertain = ((legacy_map * uncertainty).flatten(1).sum(1)
                         / uncertainty.flatten(1).sum(1).clamp_min(1e-6)).mean()
            total = SPEC["low_weight"] * low + SPEC["preserve_weight"] * preserve + SPEC["uncertain_weight"] * uncertain
            self.stats = dict(low_intensity=low.detach(), preserve=preserve.detach(),
                              uncertain=uncertain.detach(), legacy_intensity=(fused - targets).abs().mean().detach(),
                              legacy_gradient=legacy_map.mean().detach())
            for name, value in zip(("optical", "sar", "common"), group_values):
                self.stats["preserve_" + name] = value.mean().detach()
                self.stats[name + "_support"] = evidence[name].mean().detach()
            self.stats.update(ambiguous_support=evidence["ambiguous"].mean().detach(),
                              uncertain_support=uncertainty.mean().detach(),
                              sar_valid_support=evidence["sar_valid"].float().mean().detach())
        # The third component is the new contrast term, not the old max-Sobel.
        return total, low, preserve


class SourceObjectiveE2(OpticalBaseAnchoredE2):
    """Original E2 architecture with an explicit checkpoint variant for this loss."""
    VARIANT = "lwpd_e2_source_objective_v1"

    def __init__(self):
        super().__init__()
        self.variant = self.VARIANT


class SourceReadoutDecoder(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        channels = core.reduce_channel.out_channels
        width = READOUT_SPEC["branch_width"]

        def branch():
            layers = nn.Sequential(nn.Conv2d(channels * 2, width, 1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GELU(),
                                   nn.Conv2d(width, 1, 1))
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            return layers

        self.optical_head, self.sar_head = branch(), branch()
        self.optical_gate = nn.Conv2d(channels * 2, 1, 1)
        self.sar_gate = nn.Conv2d(channels * 2, 1, 1)
        for gate in (self.optical_gate, self.sar_gate):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.stats = {}

    def forward(self, inp_img, base, details):
        if inp_img is not None or not isinstance(details, tuple) or len(details) != 2:
            raise ValueError("Source readout expects no image skip and an explicit (optical, SAR) pair")
        context = self.core.reduce_channel(torch.cat((base, torch.zeros_like(base)), dim=1))
        hidden = self.core.encoder_level2(context)
        logits = self.core.output(hidden)
        stats = {}
        for name, detail, head, gate in (("optical", details[0], self.optical_head, self.optical_gate),
                                         ("sar", details[1], self.sar_head, self.sar_gate)):
            if detail is None:
                continue
            inputs = torch.cat((detail, hidden), dim=1)
            coefficient = READOUT_SPEC["max_gate"] * torch.sigmoid(gate(inputs))
            residual = READOUT_SPEC["logit_bound"] * torch.tanh(head(inputs))
            logits = logits + coefficient * residual
            stats[name + "_readout_gate"] = coefficient.detach().mean()
            stats[name + "_logit_rms"] = residual.detach().float().square().mean().sqrt()
        self.stats = stats
        return torch.sigmoid(logits), context


class SourceReadoutE2(SourceObjectiveE2):
    VARIANT = "lwpd_e2_source_readout_v1"

    def __init__(self):
        super().__init__()
        # New heads must not change subsequent sampler/augmentation RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.decoder = SourceReadoutDecoder(self.decoder)
        # The old joint Detail compression is absent, not a frozen unused head.
        self.detail_fusion = nn.Identity()
        self.variant = self.VARIANT

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail):
        base = optical_base + self.base_fusion(optical_base, sar_base)
        return base, (optical_detail, sar_detail)

    def reconstruct_source(self, base, detail, modality):
        if modality not in ("optical", "sar"):
            raise ValueError("Unknown reconstruction source")
        pair = (detail, None) if modality == "optical" else (None, detail)
        return self.decoder(None, base, pair)[0]


def reconstructions(model, optical, sar, encoded):
    bo, do, bs, ds = encoded
    if isinstance(model, SourceReadoutE2):
        return model.reconstruct_source(bo, do, "optical"), model.reconstruct_source(bs, ds, "sar")
    return model.reconstruct(optical, bo, do), model.reconstruct(sar, bs, ds)


def reconstruction_forward(model, losses, optical, sar, valid, args):
    encoded = model.encode_pair(optical, sar)
    oh, sh = reconstructions(model, optical, sar, encoded)
    lo, ls = losses.reconstruction(optical, oh), losses.reconstruction(sar, sh, valid)
    value = 0.5 * (lo + ls)
    return value, dict(optical_reconstruction=lo, sar_reconstruction=ls, reconstruction=value)


def forward_losses(model, losses, optical, sar, valid, args):
    encoded, diagnostics = model.encode_pair(optical, sar, return_diagnostics=True)
    oh, sh = reconstructions(model, optical, sar, encoded)
    lo, ls = losses.reconstruction(optical, oh), losses.reconstruction(sar, sh, valid)
    reconstruction = 0.5 * (lo + ls)
    base, detail = model.fuse_features(*encoded)
    fused = model.decoder(None, base, detail)[0]
    fusion, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    total = fusion + args.reconstruction_weight * reconstruction
    components = dict(optical_reconstruction=lo, sar_reconstruction=ls, reconstruction=reconstruction,
                      fusion=fusion, intensity=intensity, gradient=gradient,
                      gate_mean=diagnostics["sar_gates"].mean())
    components.update(getattr(losses, "stats", {}))
    components.update(getattr(model.decoder, "stats", {}))
    return total, components
