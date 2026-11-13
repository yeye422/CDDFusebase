"""Structure-guided Base and directional Detail fusion for scratch LWPD E2."""

import torch
from torch import nn
from torch.nn import functional as F

from direct_fusion import DirectFusionCDDFuse
from e2_base_anchor import OpticalBaseAnchoredE2


OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
GEOMETRY = dict(gaussian_sigmas=(1.0, 2.0), rms_floor=0.01,
                match_temperature=0.125, match_margin=0.1,
                routing="evidence_prior_softmax")
ROUTING_STATS = ("base_gate_mean", "common_weight_mean", "optical_weight_mean",
                 "sar_weight_mean", "suppression_weight_mean")


def neighborhood(tensor):
    """Read nine shifts in dy/dx order with zero outside the image."""
    h, w = tensor.shape[-2:]
    padded = F.pad(tensor, (1, 1, 1, 1))
    return tuple(padded[..., 1 + y:1 + y + h, 1 + x:1 + x + w] for y, x in OFFSETS)


class SourceStructureEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        for i, sigma in enumerate(GEOMETRY["gaussian_sigmas"]):
            coords = torch.arange(-int(3 * sigma), int(3 * sigma) + 1).float()
            kernel = torch.exp(-coords.square() / (2 * sigma ** 2))
            kernel /= kernel.sum()
            self.register_buffer(f"gaussian_{i}", kernel, persistent=False)
        self.register_buffer("sobel", torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]
        ).unsqueeze(1) / 8, persistent=False)

    @staticmethod
    def blur(image, kernel):
        radius = kernel.numel() // 2
        image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"),
                         kernel.view(1, 1, 1, -1))
        return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"),
                        kernel.view(1, 1, -1, 1))

    @staticmethod
    def half_grid(tensor):
        h, w = tensor.shape[-2:]
        tensor = F.pad(tensor, (0, w % 2, 0, h % 2), mode="replicate")
        return F.avg_pool2d(tensor, 2)

    def responses(self, image, valid):
        directions, strengths = [], []
        for kernel in (self.gaussian_0, self.gaussian_1):
            smooth = self.blur(image * valid, kernel) / self.blur(valid, kernel).clamp_min(1e-6)
            gx, gy = F.conv2d(F.pad(smooth, (1, 1, 1, 1), mode="replicate"), self.sobel).split(1, dim=1)
            energy = gx.square() + gy.square()
            rms = F.avg_pool2d(energy, 5, stride=1, padding=2, count_include_pad=False).sqrt()
            scale = rms + GEOMETRY["rms_floor"]
            # Haar LH responds vertically, HL horizontally, HH to diagonal variation.
            strengths.append((torch.cat((gy.abs(), gx.abs(), torch.minimum(gx.abs(), gy.abs())), dim=1)
                              / scale).clamp(0, 1))
            directions.append(F.normalize(torch.cat((gx, gy), dim=1), dim=1, eps=1e-6))
        coherence = (directions[0] * directions[1]).sum(dim=1, keepdim=True).abs().clamp(0, 1)
        reliable = self.half_grid(torch.minimum(*strengths) * coherence)
        orientation = F.normalize(self.half_grid(directions[1]), dim=1, eps=1e-6)
        return reliable, orientation, self.half_grid(strengths[1])

    @staticmethod
    def match_scores(ref_strength, ref_direction, src_strength, src_direction):
        scores = []
        for candidate, direction in zip(neighborhood(src_strength), neighborhood(src_direction)):
            agreement = (ref_direction * direction).sum(dim=1, keepdim=True).abs().clamp(0, 1)
            scores.append(agreement * (ref_strength * candidate).sqrt())
        return torch.stack(scores, dim=2)

    @staticmethod
    def match_confidence(scores, reverse_best):
        top, indices = scores.topk(2, dim=2)
        best = indices[:, :, :1]
        reverse = torch.stack(neighborhood(reverse_best.squeeze(2)), dim=2).gather(2, best)
        reciprocal = (reverse == (len(OFFSETS) - 1 - best)).squeeze(2)
        margin = ((top[:, :, 0] - top[:, :, 1]) / GEOMETRY["match_margin"]).clamp(0, 1)
        peak = top[:, :, 0]
        match = peak * (0.5 + 0.5 * margin) * (0.5 + 0.5 * reciprocal.float())
        return match, (peak - match).clamp_min(0)

    @torch.no_grad()
    def forward(self, optical, sar, valid):
        with torch.autocast(device_type=optical.device.type, enabled=False):
            valid = (valid > 0.5).float()
            ro, do, coarse_o = self.responses(optical.float(), torch.ones_like(valid))
            rs, ds, coarse_s = self.responses(sar.float(), valid)
            # All pixels in the immediate gradient support must be valid.
            support = F.avg_pool2d(F.pad(valid, (1, 1, 1, 1), mode="replicate"), 3, stride=1) == 1
            valid_half = self.half_grid(support.float()) == 1
            rs = rs * valid_half
            coarse_s = coarse_s * valid_half
            scores_o = self.match_scores(ro, do, rs, ds)
            scores_s = self.match_scores(rs, ds, ro, do)
            mo, uo = self.match_confidence(scores_o, scores_s.argmax(dim=2, keepdim=True))
            ms, us = self.match_confidence(scores_s, scores_o.argmax(dim=2, keepdim=True))
            alignment = (scores_o / GEOMETRY["match_temperature"]).softmax(dim=2)
            alignment = alignment * (scores_o > 0)
            alignment = alignment / alignment.sum(dim=2, keepdim=True).clamp_min(1e-6)
            aligned_rs = (torch.stack(neighborhood(rs), dim=2) * alignment).sum(dim=2)
            eligible = torch.stack((
                mo * torch.minimum(ro, aligned_rs),
                ro * (1 - mo) * (1 - uo),
                rs * (1 - ms) * (1 - us),
            ), dim=2)
            # Common coarse geometry shares the direction estimates used by Detail.
            coarse_match = self.match_scores(coarse_o, do, coarse_s, ds).amax(dim=2)
            coarse = torch.cat((coarse_match, coarse_o, coarse_s), dim=1)
            coarse = F.avg_pool2d(coarse, 5, stride=1, padding=2, count_include_pad=False)
            return coarse, eligible, alignment, valid_half


class DirectionalDetailFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.source_projection = nn.Conv2d(channels, 8, 1)
        self.router = nn.Sequential(nn.Conv2d(19, 16, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 4, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.common = nn.Conv2d(2 * channels, channels, 1, bias=False)
        with torch.no_grad():
            self.common.weight.zero_()
            eye = 0.5 * torch.eye(channels)
            self.common.weight[:, :channels, 0, 0].copy_(eye)
            self.common.weight[:, channels:, 0, 0].copy_(eye)
        self.base_projection = nn.Conv2d(channels, 16, 1)
        self.synthesis = nn.Sequential(nn.Conv2d(3 * channels + 16, channels, 1), nn.GELU(),
                                       nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, optical, sar, base, geometry, haar, shape):
        _, eligible, alignment, valid_half = geometry
        streams = [[], [], []]
        weights = []
        fallback = torch.cat((torch.zeros_like(valid_half), torch.ones_like(valid_half),
                              torch.zeros_like(valid_half), torch.zeros_like(valid_half)), dim=1)
        for d, (o, s) in enumerate(zip(optical, sar)):
            evidence = eligible[:, d]
            logits = self.router(torch.cat((self.source_projection(o), self.source_projection(s),
                                           evidence.to(o.dtype)), dim=1))
            suppression = (1 - evidence.amax(dim=1, keepdim=True)).square()
            prior = torch.cat((evidence, suppression), dim=1)
            # Evidence competes for one budget, rather than attenuating an
            # already normalized weight a second time.
            log_prior = prior.clamp_min(1e-8).log().masked_fill(prior == 0, -torch.inf)
            w = (logits.float() + log_prior).softmax(dim=1)
            w = torch.where(valid_half, w, fallback.to(w.dtype))
            weights.append(w)
            sar_aligned = sum(candidate * alignment[:, d, i:i + 1].to(s.dtype)
                              for i, candidate in enumerate(neighborhood(s)))
            common = self.common(torch.cat((o, sar_aligned), dim=1))
            for i, value in enumerate((common, o, s)):
                streams[i].append(value * w[:, i:i + 1].to(value.dtype))
        spatial = [haar.inverse((torch.zeros_like(bands[0]), *bands), shape) for bands in streams]
        detail = self.synthesis(torch.cat((self.base_projection(base), *spatial), dim=1))
        return detail, torch.stack(weights, dim=1)


class StructureGuidedE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_struct_joint_v1"

    def __init__(self):
        super().__init__(variant=self.VARIANT)
        self.detail_fusion = DirectionalDetailFusion()
        self.structure_evidence = SourceStructureEvidence()
        # Group the spatial gate with the existing Base fusion optimizer group.
        self.base_fusion.structure_gate = nn.Sequential(nn.Conv2d(10, 16, 3, padding=1),
                                                        nn.GELU(), nn.Conv2d(16, 1, 1))

    def encode_bands(self, optical, sar):
        haar = self.lwpd.haar
        opt, shape = haar(self.encoder.encoder_level1(self.encoder.patch_embed(optical)))
        sar, _ = haar(self.encoder.encoder_level1(self.encoder.patch_embed(sar)))
        gates_o, gates_s = self.lwpd._gates(self.lwpd._context(opt), self.lwpd._context(sar))
        ll_o, ll_s, _ = self.lwpd._modulate_low_bands(opt[0], sar[0])
        high_o = tuple(band * gate for band, gate in zip(opt[1:], gates_o))
        high_s = tuple(band * gate for band, gate in zip(sar[1:], gates_s))
        gate_mean = torch.stack([g.mean() for g in (*gates_o, *gates_s)]).mean()
        return (ll_o, high_o), (ll_s, high_s), shape, gate_mean

    def forward(self, optical, sar, return_features=False, valid=None):
        if optical.shape != sar.shape or optical.ndim != 4 or optical.shape[1] != 1:
            raise ValueError("Expected paired N×1×H×W images")
        valid = (sar > 0) if valid is None else (valid > 0.5)
        if valid.shape != sar.shape:
            raise ValueError("SAR valid mask must match the input shape")
        sar = sar.masked_fill(~valid, 0)
        (ll_o, high_o), (ll_s, high_s), shape, gate_mean = self.encode_bands(optical, sar)
        haar = self.lwpd.haar
        zero = torch.zeros_like(ll_o)
        base_o = haar.inverse((ll_o, zero, zero, zero), shape)
        base_s = haar.inverse((ll_s, zero, zero, zero), shape)
        geometry = self.structure_evidence(optical, sar, valid)
        coarse, _, _, valid_half = geometry
        ll_agreement = F.cosine_similarity(ll_o.float(), ll_s.float(), dim=1, eps=1e-6).unsqueeze(1)
        gate_input = torch.cat((coarse, ll_agreement), dim=1).to(base_o.dtype)
        coarse_match = coarse[:, :3].amax(dim=1, keepdim=True)
        base_gate = self.base_fusion.structure_gate(gate_input).sigmoid() * coarse_match * valid_half
        base_gate = F.interpolate(base_gate, scale_factor=2, mode="nearest")[..., :shape[0], :shape[1]]
        base_gate = base_gate * valid
        fused_base = base_o + base_gate * self.base_fusion(base_o, base_s)
        fused_detail, weights = self.detail_fusion(high_o, high_s, fused_base, geometry, haar, shape)
        output = self.decoder(None, fused_base, fused_detail)[0]
        if not return_features:
            return output
        return output, dict(
            optical_base=base_o, sar_base=base_s, fused_base=fused_base, fused_detail=fused_detail,
            optical_detail=haar.inverse((zero, *high_o), shape),
            sar_detail=haar.inverse((zero, *high_s), shape),
            gate_mean=gate_mean, base_gate=base_gate, detail_weights=weights,
        )


def routing_stats(features):
    values = features["detail_weights"].mean(dim=(0, 1, 3, 4)).unbind()
    return dict(zip(ROUTING_STATS, (features["base_gate"].mean(), *values)))


def forward_losses(model, losses, optical, sar, valid, args):
    fused, features = model(optical, sar, return_features=True, valid=valid)
    optical_hat = model.reconstruct(optical, features["optical_base"], features["optical_detail"])
    sar_hat = model.reconstruct(sar, features["sar_base"], features["sar_detail"])
    reconstruction = 0.5 * (losses.reconstruction(optical, optical_hat)
                            + losses.reconstruction(sar, sar_hat, valid))
    fusion, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    total = fusion + args.reconstruction_weight * reconstruction
    return total, dict(fusion=fusion, intensity=intensity, gradient=gradient,
                       reconstruction=reconstruction, gate_mean=features["gate_mean"],
                       **routing_stats(features))


def main():
    from unittest.mock import patch
    import test_OS

    class EvaluationModel:
        VARIANTS = (*DirectFusionCDDFuse.VARIANTS, StructureGuidedE2.VARIANT)

        def __new__(cls, variant):
            if variant == StructureGuidedE2.VARIANT:
                return StructureGuidedE2()
            return DirectFusionCDDFuse(variant)

    original_forward = test_OS.forward_variant

    def evaluate_forward(models, variant, optical, sar, **kwargs):
        if variant == StructureGuidedE2.VARIANT:
            output, features = models["direct"](optical, sar, return_features=True)
            return output, routing_stats(features)
        return original_forward(models, variant, optical, sar, **kwargs)

    with patch.object(test_OS, "DirectFusionCDDFuse", EvaluationModel), patch.object(
        test_OS, "forward_variant", evaluate_forward
    ):
        test_OS.main()


if __name__ == "__main__":
    main()
