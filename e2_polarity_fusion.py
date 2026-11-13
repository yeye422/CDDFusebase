"""Optical/SAR boundary fusion with explicit contrast polarity and gradient reconstruction."""

import math

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

from e2_aspr import gradient, gradient_adjoint
from e2_base_anchor import OpticalBaseAnchoredE2


SPEC = dict(sigmas=[0.8, 1.6], angle_degrees=30, confidence=0.55,
            weak_confidence=0.25, amplitude=4 / 255, minimum_contrast=2 / 255,
            epsilon=1 / 255, pool=4, poisson_weight=4.0, poisson_steps=16,
            coarse_scale=0.1, coarse_weight=0.5, range_weight=0.05)
MODES = ("aligned", "signed_mix", "no_sar_private")
REGIONS = ("common_same", "common_reversed", "optical_private", "sar_private")
STATS = tuple(f"{region}_fraction" for region in REGIONS) + (
    "common_sar_weight", "sar_private_gain", "reversed_polarity_accuracy",
    "reversed_polarity_pixels", "reversed_polarity_correct_pixels",
    "solver_relative_residual", "field_reconstruction_error", "out_of_range_fraction",
    "boundary_loss", "coarse_loss", "range_loss",
)


def dot_axis(a, b):
    return (a * b).sum(dim=1, keepdim=True)


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def coarse_pool(image):
    return F.avg_pool2d(image, SPEC["pool"])


def coarse_adjoint(coarse):
    size = SPEC["pool"]
    return coarse.repeat_interleave(size, -2).repeat_interleave(size, -1) / size**2


def poisson_operator(image, weight):
    return gradient_adjoint(gradient(image)) + weight * coarse_adjoint(coarse_pool(image))


def poisson_solve(rhs, initial, weight, steps):
    """Batched Jacobi-PCG with fixed iteration budget and safe zero residuals."""
    diagonal = torch.full_like(rhs, 4.0 + weight / SPEC["pool"]**4)
    diagonal[..., 0, :] -= 1
    diagonal[..., -1, :] -= 1
    diagonal[..., :, 0] -= 1
    diagonal[..., :, -1] -= 1

    def product(left, right):
        return (left * right).sum(dim=(-2, -1), keepdim=True)

    result = initial
    residual = rhs - poisson_operator(result, weight)
    direction = residual / diagonal
    rz = product(residual, direction)
    tolerance = product(rhs, rhs).clamp_min(1e-20) * 1e-14
    for _ in range(steps):
        response = poisson_operator(direction, weight)
        denominator = product(direction, response)
        active = (product(residual, residual) > tolerance) & (denominator > 0) & (rz > 0)
        step = torch.where(active, rz, 0.0) / torch.where(active, denominator, 1.0)
        result = result + step * direction
        residual = residual - step * response
        preconditioned = residual / diagonal
        next_rz = product(residual, preconditioned)
        ratio = torch.where(active, next_rz, 0.0) / torch.where(active, rz, 1.0)
        direction = preconditioned + ratio * direction
        rz = next_rz
    return result


class GradientReconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, coarse, weight, steps):
        ctx.weight, ctx.steps = weight, steps
        with torch.autocast(field.device.type, enabled=False):
            rhs = gradient_adjoint(field) + weight * coarse_adjoint(coarse)
            initial = F.interpolate(coarse, size=rhs.shape[-2:], mode="bilinear", align_corners=False)
            image = poisson_solve(rhs, initial, weight, steps)
            error = (poisson_operator(image, weight) - rhs).flatten(1).norm(dim=1)
            relative = (error / rhs.flatten(1).norm(dim=1).clamp_min(1e-10)).mean()
        ctx.mark_non_differentiable(relative)
        return image, relative

    @staticmethod
    def backward(ctx, upstream, _relative):
        with torch.autocast(upstream.device.type, enabled=False):
            adjoint = poisson_solve(upstream, torch.zeros_like(upstream), ctx.weight, ctx.steps)
            return gradient(adjoint), ctx.weight * coarse_pool(adjoint), None, None


def blur(image, sigma):
    radius = math.ceil(3 * sigma)
    size = 2 * radius + 1
    return kornia.filters.gaussian_blur2d(image, (size, size), (sigma, sigma), border_type="replicate")


def source_evidence(source, validity):
    fine, coarse = (
        gradient(blur(source * validity, sigma) / blur(validity, sigma).clamp_min(1e-6))
        for sigma in SPEC["sigmas"]
    )
    radius = math.ceil(3 * max(SPEC["sigmas"]))
    complete = F.avg_pool2d(validity, 2 * radius + 1, stride=1, padding=radius) == 1
    complete = complete & F.pad(complete[..., 1:], (0, 1), value=False) & F.pad(
        complete[..., 1:, :], (0, 0, 0, 1), value=False
    )
    axis = F.normalize(fine, dim=1, eps=1e-8)
    magnitude = fine.square().sum(dim=1, keepdim=True).sqrt()
    rms = F.avg_pool2d(magnitude.square(), 7, stride=1, padding=3, count_include_pad=False).sqrt()
    limit = math.cos(math.radians(SPEC["angle_degrees"]))
    stability = ((dot_axis(axis, F.normalize(coarse, dim=1, eps=1e-8)) - limit) / (1 - limit)).clamp(0, 1)
    strength = (magnitude / SPEC["amplitude"]).clamp(0, 1) * (magnitude / (rms + SPEC["epsilon"])).clamp(0, 1)
    return axis, strength * stability * complete


def nearby_match(axis, other_axis, other_confidence):
    height, width = other_confidence.shape[-2:]
    padded = F.pad(torch.cat((other_axis, other_confidence), dim=1), (1, 1, 1, 1))
    limit = math.cos(math.radians(SPEC["angle_degrees"]))
    found = torch.zeros_like(other_confidence, dtype=torch.bool)
    for dy in range(3):
        for dx in range(3):
            normal, strength = padded[..., dy:dy + height, dx:dx + width].split((2, 1), dim=1)
            found |= (strength >= SPEC["weak_confidence"]) & (dot_axis(axis, normal).abs() >= limit)
    return found


@torch.no_grad()
def boundary_evidence(optical, sar, valid):
    """Fixed evidence in source-image coordinates, with independent private supports."""
    with torch.autocast(optical.device.type, enabled=False):
        optical, sar, valid = optical.float(), sar.float(), valid.float()
        no, ro = source_evidence(optical, torch.ones_like(valid))
        ns, rs = source_evidence(sar, valid)
        go, gs = gradient(optical), gradient(sar)
        axis_dot = dot_axis(no, ns)
        nc = F.normalize(ro * no + rs * torch.where(axis_dot < 0, -ns, ns), dim=1, eps=1e-8)
        po, ps = dot_axis(nc, go), dot_axis(nc, gs)
        confidence, weak = SPEC["confidence"], SPEC["weak_confidence"]
        threshold = SPEC["minimum_contrast"]
        limit = math.cos(math.radians(SPEC["angle_degrees"]))
        common = ((ro >= confidence) & (rs >= confidence) & (axis_dot.abs() >= limit)
                  & (po > threshold) & (ps.abs() > threshold) & (ps * dot_axis(nc, ns) > 0))
        optical_amplitude = dot_axis(no, go).clamp_min(0)
        sar_amplitude = dot_axis(ns, gs).clamp_min(0)
        optical_private = ((ro >= confidence) & (rs < weak) & (optical_amplitude > threshold)
                           & ~nearby_match(no, ns, rs))
        sar_private = ((rs >= confidence) & (ro < weak) & (sar_amplitude > threshold)
                        & ~nearby_match(ns, no, ro))
        uncertain = ~(common | optical_private | sar_private)
        optical_field, sar_field = optical_amplitude * no, sar_amplitude * ns
        neutral_field = (ro * optical_field + rs * sar_field) / (ro + rs + SPEC["epsilon"])
        return {
            "common_same": common & (ps > 0), "common_reversed": common & (ps < 0),
            "optical_private": optical_private, "sar_private": sar_private, "uncertain": uncertain,
            "common_axis": nc, "optical_axis": no, "sar_axis": ns,
            "common_optical_amplitude": po.abs(), "common_sar_amplitude": ps.abs(),
            "sar_polarity": ps.sign(), "optical_amplitude": optical_amplitude, "sar_amplitude": sar_amplitude,
            "optical_field": optical_field, "sar_field": sar_field, "neutral_field": neutral_field,
        }


def compose_field(evidence, logits, mode="aligned"):
    if mode not in MODES:
        raise ValueError(f"Unknown polarity fusion mode: {mode}")
    common, optical, sar = logits.float().sigmoid().split(1, dim=1)
    common_weight = 0.1 + 0.8 * common
    optical_gain, sar_gain = 0.6 + 0.8 * optical, 0.6 + 0.8 * sar
    common_mask = evidence["common_same"] | evidence["common_reversed"]
    sar_amplitude = evidence["common_sar_amplitude"]
    if mode == "signed_mix":
        sar_amplitude = sar_amplitude * evidence["sar_polarity"]
    common_amplitude = ((1 - common_weight) * evidence["common_optical_amplitude"]
                        + common_weight * sar_amplitude)
    field = common_mask * common_amplitude * evidence["common_axis"]
    field = field + evidence["optical_private"] * optical_gain * evidence["optical_field"]
    if mode != "no_sar_private":
        field = field + evidence["sar_private"] * sar_gain * evidence["sar_field"]
    field = field + evidence["uncertain"] * evidence["neutral_field"]
    dx, dy = field.split(1, dim=1)
    field = torch.cat((F.pad(dx[..., :-1], (0, 1)), F.pad(dy[..., :-1, :], (0, 0, 0, 1))), dim=1)
    return field, common_weight, sar_gain


class PolarityBoundaryFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.base_projection = nn.Conv2d(channels, 16, 1)
        self.amplitude = nn.Sequential(
            nn.Conv2d(2 * channels + 16, 32, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
            nn.Conv2d(32, 3, 1),
        )
        self.coarse = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 1, 3, padding=1))
        for last in (self.amplitude[-1], self.coarse[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, optical, sar, valid, base, optical_detail, sar_detail, mode):
        logits = self.amplitude(torch.cat((optical_detail, sar_detail, self.base_projection(base)), dim=1))
        coarse_residual = self.coarse(coarse_pool(base)).float()
        coarse = (coarse_pool(optical.float()) + SPEC["coarse_scale"] * coarse_residual.tanh()).clamp(0, 1)
        evidence = boundary_evidence(optical, sar, valid)
        field, common_weight, sar_gain = compose_field(evidence, logits, mode)
        fused, relative = GradientReconstruction.apply(field, coarse, SPEC["poisson_weight"], SPEC["poisson_steps"])
        with torch.no_grad():
            fused_gradient = gradient(fused)
            stats = {f"{name}_fraction": evidence[name].float().mean() for name in REGIONS}
            reversed_mask = evidence["common_reversed"]
            correct = reversed_mask & (dot_axis(fused_gradient, evidence["common_axis"]) > 0)
            stats.update(
                common_sar_weight=masked_mean(common_weight, evidence["common_same"] | reversed_mask),
                sar_private_gain=masked_mean(sar_gain, evidence["sar_private"]),
                reversed_polarity_accuracy=correct.sum() / reversed_mask.sum().clamp_min(1),
                reversed_polarity_pixels=reversed_mask.sum() / fused.shape[0],
                reversed_polarity_correct_pixels=correct.sum() / fused.shape[0],
                solver_relative_residual=relative,
                field_reconstruction_error=(fused_gradient - field).abs().mean(),
                out_of_range_fraction=((fused < 0) | (fused > 1)).float().mean(),
            )
        return fused, coarse, evidence, stats


class PolarityFusionE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_polarity_v1"

    def __init__(self):
        super().__init__(variant=self.VARIANT)
        self.detail_fusion = PolarityBoundaryFusion()
        self.polarity_mode = "aligned"

    def fuse_features(self, optical_base, optical_detail, sar_base, sar_detail, *, optical, sar, valid):
        base = optical_base + self.base_fusion(optical_base, sar_base)
        return self.detail_fusion(optical, sar, valid, base, optical_detail, sar_detail, self.polarity_mode)

    def forward(self, optical, sar, return_features=False, valid=None):
        if optical.shape != sar.shape or optical.ndim != 4 or optical.shape[1] != 1:
            raise ValueError("Expected paired N×1×H×W inputs")
        valid = (sar > 0) if valid is None else (valid > 0.5)
        if valid.shape != sar.shape:
            raise ValueError("SAR validity must match the input shape")
        valid = valid & torch.isfinite(sar)
        sar = sar.masked_fill(~valid, 0)
        height, width = optical.shape[-2:]
        padding = (0, -width % SPEC["pool"], 0, -height % SPEC["pool"])
        if any(padding):
            optical = F.pad(optical, padding, mode="replicate")
            sar = F.pad(sar, padding, mode="replicate")
            valid = F.pad(valid, padding, value=False)
        encoded = self.encode_pair(optical, sar)
        fused, coarse, evidence, stats = self.fuse_features(*encoded, optical=optical, sar=sar, valid=valid)
        cropped = fused[..., :height, :width]
        if return_features:
            return cropped, dict(fused=fused, coarse=coarse, evidence=evidence, stats=stats,
                                 optical=optical, sar=sar, valid=valid, encoded=encoded)
        return cropped


def boundary_loss(fused, evidence):
    grad = gradient(fused)
    projection = dot_axis(grad, evidence["common_axis"])
    a, b = evidence["common_optical_amplitude"], evidence["common_sar_amplitude"]
    common = (0.8 * torch.minimum(a, b) - projection).relu() + (projection - 1.1 * torch.maximum(a, b)).relu()
    optical = dot_axis(grad, evidence["optical_axis"]) - evidence["optical_amplitude"]
    sar = dot_axis(grad, evidence["sar_axis"]) - evidence["sar_amplitude"]
    error = torch.cat((common, common, optical, sar), dim=1)
    masks = torch.cat([evidence[region] for region in REGIONS], dim=1)
    epsilon = SPEC["epsilon"]
    penalty = (error.square() + epsilon**2).sqrt() - epsilon
    count = masks.sum(dim=(0, 2, 3))
    values = (penalty * masks).sum(dim=(0, 2, 3)) / count.clamp_min(1)
    return values.sum() / (count > 0).sum().clamp_min(1)


def forward_losses(model, losses, optical, sar, valid, args):
    fused, data = model(optical, sar, valid=valid, return_features=True)
    ob, od, sb, sd = data["encoded"]
    h, w = fused.shape[-2:]
    opt_hat = model.reconstruct(data["optical"], ob, od)[..., :h, :w]
    sar_hat = model.reconstruct(data["sar"], sb, sd)[..., :h, :w]
    opt_rec = losses.reconstruction(optical, opt_hat)
    sar_rec = losses.reconstruction(data["sar"][..., :h, :w], sar_hat, data["valid"][..., :h, :w])
    reconstruction = 0.5 * (opt_rec + sar_rec)
    with torch.autocast(fused.device.type, enabled=False):
        evidence = {key: value[..., :h, :w] for key, value in data["evidence"].items()}
        edge = boundary_loss(fused, evidence)
        coarse = data["coarse"]
        coarse_loss = (coarse_pool(data["fused"]) - coarse).abs().mean()
        coarse_loss = coarse_loss + 0.1 * (coarse - coarse_pool(data["optical"].float())).abs().mean()
        range_loss = ((-fused).relu() + (fused - 1).relu()).mean()
        fusion = edge + SPEC["coarse_weight"] * coarse_loss + SPEC["range_weight"] * range_loss
        total = fusion + args.reconstruction_weight * reconstruction
    return total, {**data["stats"], "fusion": fusion, "reconstruction": reconstruction,
                   "optical_reconstruction": opt_rec, "sar_reconstruction": sar_rec,
                   "boundary_loss": edge, "coarse_loss": coarse_loss, "range_loss": range_loss}
