"""OCAF-v2: independent additive gates with amplitude-normalized SAR innovations."""

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
from ocaf_v1 import band_decompose, structure_tensor_coherence


def evidence_scores(
    optical, clean_sar, counterfactual_clean_sar,
    raw_detail, clean_detail, epsilon=1e-6,
):
    size = optical.shape[-2:]
    removed, reliability = spd_evidence_maps_v2(raw_detail, clean_detail, size)
    removed = gaussian_blur(removed, 1.5).clamp(0.0, 1.0)
    reliability = gaussian_blur(reliability, 1.5).clamp(0.0, 1.0)
    stable_sar = 0.5 * (clean_sar + counterfactual_clean_sar)
    difference = (clean_sar - counterfactual_clean_sar).abs()
    difference_scale = difference.detach().mean(
        dim=(2, 3), keepdim=True
    ).clamp_min(epsilon)
    perturbation_stability = torch.exp(-difference / difference_scale)

    optical_grad = spatial_gradient(gaussian_blur(optical, 1.0))
    sar_grad = spatial_gradient(gaussian_blur(stable_sar, 1.0))
    optical_mag = torch.sqrt(
        optical_grad.float().square().sum(dim=2) + epsilon
    ).to(optical.dtype)
    sar_mag = torch.sqrt(
        sar_grad.float().square().sum(dim=2) + epsilon
    ).to(stable_sar.dtype)
    optical_structure = spatial_percentile_normalize(optical_mag, 0.50, 0.95)
    sar_structure = spatial_percentile_normalize(sar_mag, 0.50, 0.95)
    optical_local = F.max_pool2d(optical_structure, 5, stride=1, padding=2)
    sar_local = F.max_pool2d(sar_structure, 5, stride=1, padding=2)
    dot = (optical_grad * sar_grad).sum(dim=2)
    direction = (dot / (optical_mag * sar_mag + epsilon)).abs().clamp(0.0, 1.0)
    direction = F.avg_pool2d(direction, 3, stride=1, padding=1)
    common = (
        torch.minimum(optical_local, sar_local) * (0.5 + 0.5 * direction)
    ).clamp(0.0, 1.0)

    novelty = F.relu(sar_structure - optical_local)
    coherence = structure_tensor_coherence(stable_sar)
    scale2 = spatial_percentile_normalize(
        gradient_magnitude(gaussian_blur(stable_sar, 2.0)), 0.50, 0.95
    )
    persistence = torch.minimum(sar_structure, scale2)
    reliability_support = (
        reliability + (1.0 - removed) + coherence
        + persistence + perturbation_stability
    ) / 5.0
    # Novel SAR structure is mandatory, but confidence terms modulate it with
    # a nonzero floor instead of collapsing through five-way multiplication.
    exclusive = novelty * (0.40 + 0.60 * reliability_support)
    return {
        "stable_sar": stable_sar,
        "removed": removed,
        "reliability": reliability,
        "perturbation_stability": perturbation_stability,
        "optical_structure": optical_structure,
        "sar_structure": sar_structure,
        "direction": direction,
        "common": common,
        "novelty": novelty,
        "coherence": coherence,
        "persistence": persistence,
        "exclusive": exclusive.clamp(0.0, 1.0),
    }


def positive_excess(reference_band, source_band):
    return source_band.sign() * F.relu(source_band.abs() - reference_band.abs())


def normalized_gated_delta(candidate, gate, epsilon=1e-6):
    """Normalize candidate energy inside a gate without amplifying tiny gates."""
    numerator = (gate * candidate.square()).sum(dim=(1, 2, 3), keepdim=True)
    denominator = gate.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    regional_rms = torch.sqrt(numerator / denominator + epsilon)
    return gate * torch.tanh(candidate / regional_rms)


def stable_rms(values, epsilon=1e-12):
    root = torch.sqrt(values.square().mean() + epsilon)
    return root - values.new_tensor(epsilon).sqrt()


class OpticalCanvasFusionV2(nn.Module):
    def __init__(
        self, hidden_channels=32,
        max_common_amplitude=2.0 / 255.0,
        max_exclusive_mid_amplitude=1.5 / 255.0,
        max_exclusive_high_amplitude=1.0 / 255.0,
        high_sigma=1.0, low_sigma=3.0,
    ):
        super().__init__()
        self.max_common_amplitude = float(max_common_amplitude)
        self.max_exclusive_mid_amplitude = float(max_exclusive_mid_amplitude)
        self.max_exclusive_high_amplitude = float(max_exclusive_high_amplitude)
        self.high_sigma = float(high_sigma)
        self.low_sigma = float(low_sigma)
        self.input = nn.Conv2d(13, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            RoutingBlock(hidden_channels, 1), RoutingBlock(hidden_channels, 2)
        )
        self.gate_logits = nn.Conv2d(hidden_channels, 2, 1)
        with torch.no_grad():
            self.gate_logits.weight.zero_()
            self.gate_logits.bias.copy_(torch.tensor((-1.5, -2.5)))
        self.common_amplitude = nn.Parameter(torch.zeros(()))
        self.exclusive_mid_amplitude = nn.Parameter(torch.zeros(()))
        self.exclusive_high_amplitude = nn.Parameter(torch.zeros(()))

    def forward(
        self, optical, clean_sar, counterfactual_clean_sar, baseline,
        raw_sar_detail, clean_sar_detail, valid=None, return_maps=False,
    ):
        scores = evidence_scores(
            optical, clean_sar, counterfactual_clean_sar,
            raw_sar_detail, clean_sar_detail,
        )
        evidence = torch.cat((
            optical, scores["stable_sar"], baseline, scores["removed"],
            scores["reliability"], scores["perturbation_stability"],
            scores["optical_structure"], scores["sar_structure"],
            scores["direction"], scores["common"], scores["novelty"],
            scores["coherence"], scores["persistence"],
        ), dim=1)
        learned_common, learned_sar = self.gate_logits(
            self.body(F.gelu(self.input(evidence)))
        ).split(1, dim=1)
        common_logit = (
            learned_common + 2.5 * scores["common"]
            + 0.5 * scores["direction"] - 0.5 * scores["removed"]
        )
        sar_logit = (
            learned_sar + 3.0 * scores["novelty"]
            + 1.0 * scores["reliability"]
            - 1.5 * scores["removed"]
            + 0.75 * scores["coherence"]
            + 0.75 * scores["persistence"]
            + 0.75 * scores["perturbation_stability"]
        )
        common_gate = torch.sigmoid(common_logit)
        sar_gate = torch.sigmoid(sar_logit)
        if valid is not None:
            common_gate = common_gate * valid
            sar_gate = sar_gate * valid

        _, optical_mid, optical_high = band_decompose(
            optical, self.high_sigma, self.low_sigma
        )
        _, sar_mid, sar_high = band_decompose(
            scores["stable_sar"], self.high_sigma, self.low_sigma
        )
        common_candidate = optical_mid.sign() * F.relu(
            sar_mid.abs() - optical_mid.abs()
        )
        exclusive_mid_candidate = positive_excess(optical_mid, sar_mid)
        exclusive_high_candidate = positive_excess(optical_high, sar_high)
        common_unit = normalized_gated_delta(common_candidate, common_gate)
        exclusive_mid_unit = normalized_gated_delta(
            exclusive_mid_candidate, sar_gate
        )
        exclusive_high_unit = normalized_gated_delta(
            exclusive_high_candidate, sar_gate
        )
        common_amplitude = self.max_common_amplitude * torch.tanh(
            self.common_amplitude
        )
        exclusive_mid_amplitude = self.max_exclusive_mid_amplitude * torch.tanh(
            self.exclusive_mid_amplitude
        )
        exclusive_high_amplitude = self.max_exclusive_high_amplitude * torch.tanh(
            self.exclusive_high_amplitude
        )
        common_delta = common_amplitude * common_unit
        exclusive_delta = (
            exclusive_mid_amplitude * exclusive_mid_unit
            + exclusive_high_amplitude * exclusive_high_unit
        )
        output = optical + common_delta + exclusive_delta
        valid_weight = valid if valid is not None else torch.ones_like(optical)
        diagnostics = {
            "ocaf_common_amplitude": common_amplitude,
            "ocaf_exclusive_mid_amplitude": exclusive_mid_amplitude,
            "ocaf_exclusive_high_amplitude": exclusive_high_amplitude,
            "ocaf_common_gate_mean": weighted_mean(common_gate, valid_weight),
            "ocaf_sar_gate_mean": weighted_mean(sar_gate, valid_weight),
            "ocaf_common_gate_coverage_050": weighted_mean(
                (common_gate > 0.50).to(common_gate.dtype), valid_weight
            ),
            "ocaf_sar_gate_coverage_050": weighted_mean(
                (sar_gate > 0.50).to(sar_gate.dtype), valid_weight
            ),
            "ocaf_common_score_mean": scores["common"].mean(),
            "ocaf_exclusive_score_mean": scores["exclusive"].mean(),
            "ocaf_novelty_mean": scores["novelty"].mean(),
            "ocaf_removed_mean": scores["removed"].mean(),
            "ocaf_reliability_mean": scores["reliability"].mean(),
            "ocaf_stability_mean": scores["perturbation_stability"].mean(),
            "ocaf_common_delta_rms": stable_rms(common_delta),
            "ocaf_exclusive_delta_rms": stable_rms(exclusive_delta),
            "ocaf_total_delta_rms": stable_rms(output - optical),
        }
        if return_maps:
            diagnostics.update({
                "ocaf_removed_map": scores["removed"],
                "ocaf_reliability_map": scores["reliability"],
                "ocaf_stability_map": scores["perturbation_stability"],
                "ocaf_common_score_map": scores["common"],
                "ocaf_exclusive_score_map": scores["exclusive"],
                "ocaf_common_gate_map": common_gate,
                "ocaf_sar_gate_map": sar_gate,
                "ocaf_common_candidate": common_candidate,
                "ocaf_exclusive_mid_candidate": exclusive_mid_candidate,
                "ocaf_exclusive_high_candidate": exclusive_high_candidate,
                "ocaf_common_unit": common_unit,
                "ocaf_exclusive_mid_unit": exclusive_mid_unit,
                "ocaf_exclusive_high_unit": exclusive_high_unit,
            })
        return output, diagnostics


def soft_gate_bce(gate, target, valid, epsilon=1e-6):
    gate = gate.clamp(epsilon, 1.0 - epsilon)
    values = -(target * gate.log() + (1.0 - target) * (1.0 - gate).log())
    return weighted_mean(values, valid)


def ocaf_v2_losses(
    output, optical, valid, diagnostics,
    target_common_rms=0.5 / 255.0,
    target_exclusive_rms=0.25 / 255.0,
    high_sigma=1.0, low_sigma=3.0, epsilon=1e-3,
):
    common_score = diagnostics["ocaf_common_score_map"].detach()
    exclusive_score = diagnostics["ocaf_exclusive_score_map"].detach()
    removed = diagnostics["ocaf_removed_map"].detach()
    common_gate = diagnostics["ocaf_common_gate_map"]
    sar_gate = diagnostics["ocaf_sar_gate_map"]
    optical_mask = (
        1.0 - torch.maximum(common_score, exclusive_score)
    ).clamp(0.0, 1.0) * valid

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
    common_target = optical_mid + (
        diagnostics["ocaf_common_unit"].detach() * float(target_common_rms)
    )
    common_injection = weighted_mean(
        charbonnier(output_mid - common_target, epsilon), common_score * valid
    )
    exclusive_mid_target = optical_mid + (
        diagnostics["ocaf_exclusive_mid_unit"].detach()
        * float(target_exclusive_rms)
    )
    exclusive_high_target = optical_high + (
        diagnostics["ocaf_exclusive_high_unit"].detach()
        * float(target_exclusive_rms)
    )
    sar_injection = weighted_mean(
        charbonnier(output_mid - exclusive_mid_target, epsilon),
        exclusive_score * valid,
    ) + weighted_mean(
        charbonnier(output_high - exclusive_high_target, epsilon),
        exclusive_score * valid,
    )
    gate_alignment = soft_gate_bce(common_gate, common_score, valid) + soft_gate_bce(
        sar_gate, exclusive_score, valid
    )
    common_budget = torch.abs(
        diagnostics["ocaf_common_delta_rms"] - float(target_common_rms)
        * torch.sqrt(common_score.mean().detach().clamp_min(1e-6))
    )
    exclusive_budget = torch.abs(
        diagnostics["ocaf_exclusive_delta_rms"] - float(target_exclusive_rms)
        * torch.sqrt(exclusive_score.mean().detach().clamp_min(1e-6))
    )
    removed_suppression = weighted_mean(
        torch.abs(output - optical), removed * valid
    )
    supported = gradient_magnitude(
        optical
        + float(target_common_rms) * diagnostics["ocaf_common_unit"].detach()
        + float(target_exclusive_rms) * (
            diagnostics["ocaf_exclusive_mid_unit"].detach()
            + diagnostics["ocaf_exclusive_high_unit"].detach()
        )
    ).detach()
    unsupported = weighted_mean(
        F.relu(gradient_magnitude(output) - torch.maximum(
            gradient_magnitude(optical), supported
        )), valid
    )
    negative_amplitude = (
        F.relu(-diagnostics["ocaf_common_amplitude"])
        + F.relu(-diagnostics["ocaf_exclusive_mid_amplitude"])
        + F.relu(-diagnostics["ocaf_exclusive_high_amplitude"])
    )
    return {
        "low_identity": low_identity,
        "optical_preservation": optical_preservation,
        "common_injection": common_injection,
        "sar_injection": sar_injection,
        "gate_alignment": gate_alignment,
        "common_budget": common_budget,
        "exclusive_budget": exclusive_budget,
        "removed_suppression": removed_suppression,
        "unsupported": unsupported,
        "negative_amplitude": negative_amplitude,
    }
