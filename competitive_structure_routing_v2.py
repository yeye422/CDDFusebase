"""CSR-v2: calibrated SPD evidence and logit-space competitive routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from competitive_structure_routing import (
    RoutingBlock,
    charbonnier,
    gaussian_blur,
    gradient_magnitude,
    optical_missing_map,
    spatial_gradient,
    weighted_mean,
)


def spatial_percentile_normalize(values, low=0.50, high=0.95, epsilon=1e-6):
    """Robust per-image calibration for sparse, frozen evidence maps."""
    calibrated = []
    for sample in values.float():
        flattened = sample.flatten()
        lower = torch.quantile(flattened, float(low))
        upper = torch.quantile(flattened, float(high))
        normalized = (sample - lower) / (upper - lower).clamp_min(epsilon)
        calibrated.append(normalized.clamp(0.0, 1.0))
    return torch.stack(calibrated).to(values.dtype)


def spd_evidence_maps_v2(
    raw_detail, clean_detail, output_size, topk_fraction=0.125, epsilon=1e-6
):
    """Preserve sparse SPD removal evidence instead of averaging it away."""
    raw = raw_detail.float()
    clean = clean_detail.float()
    removed = (raw - clean).abs()
    raw_scale = raw.abs().mean(dim=(2, 3), keepdim=True).clamp_min(epsilon)
    relative_removed = removed / (raw.abs() + 0.25 * raw_scale)
    k = max(1, int(round(raw.shape[1] * float(topk_fraction))))
    removed_topk = relative_removed.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    raw_rms = raw.square().mean(dim=1, keepdim=True).sqrt()
    clean_rms = clean.square().mean(dim=1, keepdim=True).sqrt()
    raw_salience = spatial_percentile_normalize(raw_rms)
    clean_salience = spatial_percentile_normalize(clean_rms)
    removed_evidence = spatial_percentile_normalize(removed_topk) * (
        0.25 + 0.75 * raw_salience
    )
    # Max pooling keeps thin deleted structures alive before image-scale resize.
    removed_evidence = F.max_pool2d(removed_evidence, 3, stride=1, padding=1)

    retained_fraction = (clean_rms / (raw_rms + epsilon)).clamp(0.0, 1.0)
    coherence = (
        (raw * clean).sum(dim=1, keepdim=True)
        / (
            raw.square().sum(dim=1, keepdim=True).sqrt()
            * clean.square().sum(dim=1, keepdim=True).sqrt()
            + epsilon
        )
    ).clamp(0.0, 1.0)
    sar_reliability = (
        0.25 + 0.75 * clean_salience
    ) * retained_fraction * coherence
    sar_reliability = spatial_percentile_normalize(sar_reliability, 0.40, 0.95)

    maps = (removed_evidence, sar_reliability)
    return tuple(
        F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
        .clamp(0.0, 1.0)
        .to(raw_detail.dtype)
        for value in maps
    )


def routing_targets_v2(missing, removed, sar_reliability, valid, epsilon=1e-6):
    optical = missing * (0.5 + 0.5 * removed)
    sar = sar_reliability * (1.0 - missing)
    anchor = (1.0 - torch.maximum(optical, sar)).clamp(0.0, 1.0)
    target = torch.cat((anchor, optical, sar), dim=1) * valid
    return target / target.sum(dim=1, keepdim=True).clamp_min(epsilon)


class CompetitiveStructureRouterV2(nn.Module):
    """Use calibrated SPD priors in logits; never multiply the optical gate away."""

    def __init__(
        self, hidden_channels=32, max_mid_replacement=0.10,
        max_high_replacement=0.03, mid_sigma=3.0, high_sigma=1.0,
        missing_logit_gain=3.0, removed_logit_gain=1.5,
        reliability_logit_gain=2.0,
    ):
        super().__init__()
        self.max_mid_replacement = float(max_mid_replacement)
        self.max_high_replacement = float(max_high_replacement)
        self.mid_sigma = float(mid_sigma)
        self.high_sigma = float(high_sigma)
        self.missing_logit_gain = float(missing_logit_gain)
        self.removed_logit_gain = float(removed_logit_gain)
        self.reliability_logit_gain = float(reliability_logit_gain)
        self.input = nn.Conv2d(6, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            RoutingBlock(hidden_channels, 1), RoutingBlock(hidden_channels, 2)
        )
        self.logits = nn.Conv2d(hidden_channels, 3, 1)
        with torch.no_grad():
            self.logits.weight.zero_()
            self.logits.bias.copy_(torch.tensor((2.0, -1.0, 1.0)))
        self.mid_strength = nn.Parameter(torch.zeros(()))
        self.high_strength = nn.Parameter(torch.zeros(()))

    def forward(
        self, optical, sar, baseline, raw_sar_detail, clean_sar_detail,
        valid=None, return_maps=False,
    ):
        output_size = baseline.shape[-2:]
        removed, sar_reliability = spd_evidence_maps_v2(
            raw_sar_detail, clean_sar_detail, output_size
        )
        missing = optical_missing_map(optical, baseline)
        evidence = torch.cat(
            (optical, sar, baseline, missing, removed, sar_reliability), dim=1
        )
        logits = self.logits(self.body(F.gelu(self.input(evidence))))
        anchor_logit, optical_logit, sar_logit = logits.split(1, dim=1)
        optical_logit = (
            optical_logit
            + self.missing_logit_gain * missing
            + self.removed_logit_gain * removed
            - self.reliability_logit_gain * sar_reliability
        )
        sar_logit = sar_logit + self.reliability_logit_gain * sar_reliability
        probabilities = torch.softmax(
            torch.cat((anchor_logit, optical_logit, sar_logit), dim=1), dim=1
        )
        anchor_gate, optical_gate, sar_gate = probabilities.split(1, dim=1)
        if valid is not None:
            optical_gate = optical_gate * valid

        optical_high = optical - gaussian_blur(optical, self.high_sigma)
        baseline_high = baseline - gaussian_blur(baseline, self.high_sigma)
        optical_mid = gaussian_blur(optical, self.high_sigma) - gaussian_blur(
            optical, self.mid_sigma
        )
        baseline_mid = gaussian_blur(baseline, self.high_sigma) - gaussian_blur(
            baseline, self.mid_sigma
        )
        mid_scale = self.max_mid_replacement * torch.tanh(self.mid_strength)
        high_scale = self.max_high_replacement * torch.tanh(self.high_strength)
        correction = optical_gate * (
            mid_scale * (optical_mid - baseline_mid)
            + high_scale * (optical_high - baseline_high)
        )
        output = baseline + correction
        valid_weight = valid if valid is not None else torch.ones_like(optical_gate)
        missing_recall = weighted_mean(optical_gate, missing * valid_weight)
        sar_false_activation = weighted_mean(
            optical_gate, sar_reliability * valid_weight
        )
        coverage = weighted_mean(optical_gate, valid_weight)

        diagnostics = {
            "csr_mid_scale": mid_scale,
            "csr_high_scale": high_scale,
            "csr_optical_gate_mean": coverage,
            "csr_optical_gate_coverage_010": weighted_mean(
                (optical_gate > 0.10).to(optical_gate.dtype), valid_weight
            ),
            "csr_missing_recall": missing_recall,
            "csr_sar_false_activation": sar_false_activation,
            "csr_anchor_probability": anchor_gate.mean(),
            "csr_sar_probability": sar_gate.mean(),
            "csr_missing_mean": missing.mean(),
            "csr_removed_mean": removed.mean(),
            "csr_removed_spatial_std": removed.flatten(2).std(dim=2).mean(),
            "csr_sar_reliability_mean": sar_reliability.mean(),
            "csr_sar_reliability_spatial_std": sar_reliability.flatten(2).std(dim=2).mean(),
            "csr_correction_rms": correction.square().mean().sqrt(),
        }
        if return_maps:
            diagnostics.update({
                "csr_missing_map": missing,
                "csr_removed_map": removed,
                "csr_sar_reliability_map": sar_reliability,
                "csr_anchor_gate_map": anchor_gate,
                "csr_optical_gate_map": optical_gate,
                "csr_sar_gate_map": sar_gate,
            })
        return output, diagnostics


def competitive_structure_losses_v2(
    output, baseline, optical, valid, diagnostics, target_coverage=0.04,
    low_sigma=3.0, epsilon=1e-3,
):
    missing = diagnostics["csr_missing_map"].detach()
    removed = diagnostics["csr_removed_map"].detach()
    sar_reliability = diagnostics["csr_sar_reliability_map"].detach()
    optical_mask = missing * valid
    sar_mask = sar_reliability * (1.0 - missing) * valid
    output_gradient = spatial_gradient(output)
    optical_gradient = spatial_gradient(optical)
    baseline_gradient = spatial_gradient(baseline)
    optical_visibility = weighted_mean(
        charbonnier(output_gradient - optical_gradient, epsilon), optical_mask
    )
    sar_retention = weighted_mean(
        charbonnier(output_gradient - baseline_gradient, epsilon), sar_mask
    )
    supported = torch.maximum(
        gradient_magnitude(optical), gradient_magnitude(baseline)
    ).detach()
    unsupported = weighted_mean(
        F.relu(gradient_magnitude(output) - supported), valid
    )
    anchor = weighted_mean(
        torch.abs(output - baseline), (1.0 - missing).clamp(0.0, 1.0) * valid
    )
    low_frequency = weighted_mean(
        torch.abs(gaussian_blur(output, low_sigma) - gaussian_blur(baseline, low_sigma)),
        valid,
    )
    probabilities = torch.cat(
        (
            diagnostics["csr_anchor_gate_map"],
            diagnostics["csr_optical_gate_map"],
            diagnostics["csr_sar_gate_map"],
        ), dim=1,
    ).clamp_min(1e-6)
    targets = routing_targets_v2(missing, removed, sar_reliability, valid)
    routing = weighted_mean(
        -(targets * probabilities.log()).sum(dim=1, keepdim=True), valid
    )
    coverage = torch.abs(
        diagnostics["csr_optical_gate_mean"] - float(target_coverage)
    )
    negative_scale = F.relu(-diagnostics["csr_mid_scale"]) + F.relu(
        -diagnostics["csr_high_scale"]
    )
    return {
        "optical_visibility": optical_visibility,
        "sar_retention": sar_retention,
        "unsupported": unsupported,
        "anchor": anchor,
        "low_frequency": low_frequency,
        "routing": routing,
        "coverage": coverage,
        "negative_scale": negative_scale,
    }
