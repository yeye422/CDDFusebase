"""OCAF-v1: optical-identity fusion with attributable SAR innovations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from competitive_structure_routing import (
    RoutingBlock,
    charbonnier,
    gaussian_blur,
    gradient_magnitude,
    spatial_gradient,
    weighted_mean,
)
from competitive_structure_routing_v2 import (
    spatial_percentile_normalize,
    spd_evidence_maps_v2,
)


def structure_tensor_coherence(image, sigma=1.5, epsilon=1e-6):
    gradients = spatial_gradient(gaussian_blur(image, 1.0))
    gx, gy = gradients[:, :, 0], gradients[:, :, 1]
    jxx = gaussian_blur(gx.square(), sigma)
    jyy = gaussian_blur(gy.square(), sigma)
    jxy = gaussian_blur(gx * gy, sigma)
    numerator = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square() + epsilon)
    return (numerator / (jxx + jyy + epsilon)).clamp(0.0, 1.0)


def band_decompose(image, high_sigma=1.0, low_sigma=3.0):
    high_blur = gaussian_blur(image, high_sigma)
    low = gaussian_blur(image, low_sigma)
    return low, high_blur - low, image - high_blur


def source_priors(
    optical, clean_sar, raw_detail, clean_detail, epsilon=1e-6
):
    size = optical.shape[-2:]
    removed, reliability = spd_evidence_maps_v2(
        raw_detail, clean_detail, size
    )
    # Evidence is deliberately smoothed after feature-to-image projection;
    # otherwise low-resolution feature blocks become false pixel-level masks.
    removed = gaussian_blur(removed, 1.5).clamp(0.0, 1.0)
    reliability = gaussian_blur(reliability, 1.5).clamp(0.0, 1.0)

    optical_grad = spatial_gradient(gaussian_blur(optical, 1.0))
    sar_grad = spatial_gradient(gaussian_blur(clean_sar, 1.0))
    optical_mag = torch.sqrt(
        optical_grad.float().square().sum(dim=2) + epsilon
    ).to(optical.dtype)
    sar_mag = torch.sqrt(
        sar_grad.float().square().sum(dim=2) + epsilon
    ).to(clean_sar.dtype)
    optical_evidence = spatial_percentile_normalize(optical_mag, 0.50, 0.95)
    sar_evidence = spatial_percentile_normalize(sar_mag, 0.50, 0.95)
    optical_local = F.max_pool2d(optical_evidence, 5, stride=1, padding=2)
    sar_local = F.max_pool2d(sar_evidence, 5, stride=1, padding=2)

    dot = (optical_grad * sar_grad).sum(dim=2)
    denominator = optical_mag * sar_mag + epsilon
    direction_agreement = (dot / denominator).abs().clamp(0.0, 1.0)
    direction_agreement = F.avg_pool2d(
        direction_agreement, 3, stride=1, padding=1
    )
    common = (
        torch.minimum(optical_local, sar_local)
        * (0.5 + 0.5 * direction_agreement)
    ).clamp(0.0, 1.0)

    sar_coherence = structure_tensor_coherence(clean_sar)
    sar_scale2 = spatial_percentile_normalize(
        gradient_magnitude(gaussian_blur(clean_sar, 2.0)), 0.50, 0.95
    )
    persistence = torch.minimum(sar_evidence, sar_scale2)
    exclusive = (
        F.relu(sar_evidence - optical_local)
        * reliability
        * (1.0 - removed)
        * sar_coherence
        * persistence
    ).clamp(0.0, 1.0)
    return {
        "removed": removed,
        "reliability": reliability,
        "optical_structure": optical_evidence,
        "sar_structure": sar_evidence,
        "common": common,
        "exclusive": exclusive,
        "coherence": sar_coherence,
        "persistence": persistence,
    }


def positive_excess(reference_band, source_band):
    """Add source energy only where it exceeds the optical canvas."""
    magnitude = F.relu(source_band.abs() - reference_band.abs())
    return source_band.sign() * magnitude


class OpticalCanvasFusion(nn.Module):
    def __init__(
        self, hidden_channels=32, max_common_mid=0.10,
        max_exclusive_mid=0.05, max_exclusive_high=0.02,
        high_sigma=1.0, low_sigma=3.0,
    ):
        super().__init__()
        self.max_common_mid = float(max_common_mid)
        self.max_exclusive_mid = float(max_exclusive_mid)
        self.max_exclusive_high = float(max_exclusive_high)
        self.high_sigma = float(high_sigma)
        self.low_sigma = float(low_sigma)
        self.input = nn.Conv2d(9, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            RoutingBlock(hidden_channels, 1), RoutingBlock(hidden_channels, 2)
        )
        self.logits = nn.Conv2d(hidden_channels, 3, 1)
        with torch.no_grad():
            self.logits.weight.zero_()
            self.logits.bias.copy_(torch.tensor((3.0, 0.0, -1.0)))
        self.common_strength = nn.Parameter(torch.zeros(()))
        self.exclusive_mid_strength = nn.Parameter(torch.zeros(()))
        self.exclusive_high_strength = nn.Parameter(torch.zeros(()))

    def forward(
        self, optical, clean_sar, baseline, raw_sar_detail,
        clean_sar_detail, valid=None, return_maps=False,
    ):
        priors = source_priors(
            optical, clean_sar, raw_sar_detail, clean_sar_detail
        )
        evidence = torch.cat((
            optical, clean_sar, baseline, priors["common"],
            priors["exclusive"], priors["removed"], priors["reliability"],
            priors["optical_structure"], priors["sar_structure"],
        ), dim=1)
        probabilities = torch.softmax(
            self.logits(self.body(F.gelu(self.input(evidence)))), dim=1
        )
        optical_probability, common_probability, sar_probability = probabilities.split(
            1, dim=1
        )
        common_gate = common_probability * priors["common"]
        sar_gate = sar_probability * priors["exclusive"]
        if valid is not None:
            common_gate = common_gate * valid
            sar_gate = sar_gate * valid

        _, optical_mid, optical_high = band_decompose(
            optical, self.high_sigma, self.low_sigma
        )
        _, sar_mid, sar_high = band_decompose(
            clean_sar, self.high_sigma, self.low_sigma
        )
        common_candidate = optical_mid.sign() * F.relu(
            sar_mid.abs() - optical_mid.abs()
        )
        exclusive_mid_candidate = positive_excess(optical_mid, sar_mid)
        exclusive_high_candidate = positive_excess(optical_high, sar_high)
        common_scale = self.max_common_mid * torch.tanh(self.common_strength)
        exclusive_mid_scale = self.max_exclusive_mid * torch.tanh(
            self.exclusive_mid_strength
        )
        exclusive_high_scale = self.max_exclusive_high * torch.tanh(
            self.exclusive_high_strength
        )
        common_delta = common_scale * common_gate * common_candidate
        exclusive_delta = sar_gate * (
            exclusive_mid_scale * exclusive_mid_candidate
            + exclusive_high_scale * exclusive_high_candidate
        )
        output = optical + common_delta + exclusive_delta
        valid_weight = valid if valid is not None else torch.ones_like(optical)
        diagnostics = {
            "ocaf_common_scale": common_scale,
            "ocaf_exclusive_mid_scale": exclusive_mid_scale,
            "ocaf_exclusive_high_scale": exclusive_high_scale,
            "ocaf_common_gate_mean": weighted_mean(common_gate, valid_weight),
            "ocaf_sar_gate_mean": weighted_mean(sar_gate, valid_weight),
            "ocaf_optical_probability": optical_probability.mean(),
            "ocaf_common_probability": common_probability.mean(),
            "ocaf_sar_probability": sar_probability.mean(),
            "ocaf_common_prior_mean": priors["common"].mean(),
            "ocaf_exclusive_prior_mean": priors["exclusive"].mean(),
            "ocaf_removed_mean": priors["removed"].mean(),
            "ocaf_reliability_mean": priors["reliability"].mean(),
            "ocaf_common_delta_rms": common_delta.square().mean().sqrt(),
            "ocaf_exclusive_delta_rms": exclusive_delta.square().mean().sqrt(),
            "ocaf_total_delta_rms": (output - optical).square().mean().sqrt(),
        }
        if return_maps:
            diagnostics.update({
                "ocaf_removed_map": priors["removed"],
                "ocaf_reliability_map": priors["reliability"],
                "ocaf_common_prior_map": priors["common"],
                "ocaf_exclusive_prior_map": priors["exclusive"],
                "ocaf_common_gate_map": common_gate,
                "ocaf_sar_gate_map": sar_gate,
                "ocaf_optical_probability_map": optical_probability,
                "ocaf_common_probability_map": common_probability,
                "ocaf_sar_probability_map": sar_probability,
                "ocaf_common_candidate": common_candidate,
                "ocaf_exclusive_mid_candidate": exclusive_mid_candidate,
                "ocaf_exclusive_high_candidate": exclusive_high_candidate,
            })
        return output, diagnostics


def routing_targets(common, exclusive, valid, epsilon=1e-6):
    common = common * valid
    exclusive = exclusive * valid
    optical = (1.0 - torch.maximum(common, exclusive)).clamp(0.0, 1.0) * valid
    targets = torch.cat((optical, common, exclusive), dim=1)
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(epsilon)


def ocaf_losses(
    output, optical, valid, diagnostics, counterfactual_output=None,
    high_sigma=1.0, low_sigma=3.0, sar_floor_ratio=0.25,
    epsilon=1e-3,
):
    common = diagnostics["ocaf_common_prior_map"].detach()
    exclusive = diagnostics["ocaf_exclusive_prior_map"].detach()
    removed = diagnostics["ocaf_removed_map"].detach()
    common_gate = diagnostics["ocaf_common_gate_map"]
    sar_gate = diagnostics["ocaf_sar_gate_map"]
    optical_probability = diagnostics["ocaf_optical_probability_map"]
    optical_mask = (1.0 - torch.maximum(common, exclusive)).clamp(0.0, 1.0) * valid

    optical_low, optical_mid, optical_high = band_decompose(
        optical, high_sigma, low_sigma
    )
    output_low, output_mid, output_high = band_decompose(
        output, high_sigma, low_sigma
    )
    low_identity = weighted_mean(torch.abs(output_low - optical_low), valid)
    optical_preservation = weighted_mean(
        charbonnier(spatial_gradient(output) - spatial_gradient(optical), epsilon),
        optical_mask,
    )
    target_common = optical_mid + (
        diagnostics["ocaf_common_candidate"].detach()
        * common * 0.10
    )
    common_injection = weighted_mean(
        charbonnier(output_mid - target_common, epsilon), common * valid
    )
    target_exclusive_mid = optical_mid + (
        diagnostics["ocaf_exclusive_mid_candidate"].detach()
        * exclusive * 0.05
    )
    target_exclusive_high = optical_high + (
        diagnostics["ocaf_exclusive_high_candidate"].detach()
        * exclusive * 0.02
    )
    sar_injection = weighted_mean(
        charbonnier(output_mid - target_exclusive_mid, epsilon), exclusive * valid
    ) + weighted_mean(
        charbonnier(output_high - target_exclusive_high, epsilon), exclusive * valid
    )
    supported = torch.maximum(
        gradient_magnitude(optical),
        gradient_magnitude(optical + target_common - optical_mid
                           + target_exclusive_mid - optical_mid
                           + target_exclusive_high - optical_high),
    ).detach()
    unsupported = weighted_mean(
        F.relu(gradient_magnitude(output) - supported), valid
    )
    probabilities = torch.cat(
        (
            optical_probability,
            diagnostics["ocaf_common_probability_map"],
            diagnostics["ocaf_sar_probability_map"],
        ), dim=1
    ).clamp_min(1e-6)
    targets = routing_targets(common, exclusive, valid)
    routing = weighted_mean(
        -(targets * probabilities.log()).sum(dim=1, keepdim=True), valid
    )
    delta = output - optical
    target_delta = (
        common * diagnostics["ocaf_common_candidate"].detach() * 0.10
        + exclusive * diagnostics["ocaf_exclusive_mid_candidate"].detach() * 0.05
        + exclusive * diagnostics["ocaf_exclusive_high_candidate"].detach() * 0.02
    )
    sar_floor = F.relu(
        float(sar_floor_ratio) * weighted_mean(target_delta.abs(), exclusive * valid)
        - weighted_mean(delta.abs(), exclusive * valid)
    )
    removed_suppression = weighted_mean(delta.abs(), removed * valid)
    sparse = weighted_mean(common_gate + sar_gate, valid)
    invariance = output.new_zeros(())
    if counterfactual_output is not None:
        invariance = weighted_mean(
            torch.abs(output - counterfactual_output), valid
        )
    negative_scale = (
        F.relu(-diagnostics["ocaf_common_scale"])
        + F.relu(-diagnostics["ocaf_exclusive_mid_scale"])
        + F.relu(-diagnostics["ocaf_exclusive_high_scale"])
    )
    return {
        "low_identity": low_identity,
        "optical_preservation": optical_preservation,
        "common_injection": common_injection,
        "sar_injection": sar_injection,
        "unsupported": unsupported,
        "routing": routing,
        "sar_floor": sar_floor,
        "removed_suppression": removed_suppression,
        "sparse": sparse,
        "invariance": invariance,
        "negative_scale": negative_scale,
    }
