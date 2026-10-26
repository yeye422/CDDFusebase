"""SPD-guided competitive routing for visible optical-structure recovery."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels, requested=8):
    groups = min(int(channels), int(requested))
    while channels % groups:
        groups -= 1
    return max(groups, 1)


def gaussian_kernel(sigma, dtype, device):
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * float(sigma) ** 2))
    kernel = kernel / kernel.sum()
    return kernel[:, None] @ kernel[None, :]


def gaussian_blur(image, sigma):
    kernel = gaussian_kernel(sigma, image.dtype, image.device)
    size = kernel.shape[0]
    kernel = kernel.view(1, 1, size, size).expand(image.shape[1], 1, size, size)
    pad = size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def spatial_gradient(image):
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    kernel_y = kernel_x.t()
    kernels = torch.stack((kernel_x, kernel_y)).unsqueeze(1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    gradients = F.conv2d(padded, kernels)
    return gradients.view(image.shape[0], image.shape[1], 2, *image.shape[-2:])


def gradient_magnitude(image, epsilon=1e-6):
    gradients = spatial_gradient(image).float()
    return torch.sqrt(gradients.square().sum(dim=2) + epsilon).to(image.dtype)


def bounded_evidence(values, epsilon=1e-6):
    """Map nonnegative evidence to [0,1] without dataset-level calibration."""
    scale = values.detach().mean(dim=(2, 3), keepdim=True).clamp_min(epsilon)
    return 1.0 - torch.exp(-values / scale)


def spd_evidence_maps(raw_detail, clean_detail, output_size, epsilon=1e-6):
    removed = (raw_detail - clean_detail).float().square().mean(
        dim=1, keepdim=True
    ).sqrt()
    retained = clean_detail.float().square().mean(dim=1, keepdim=True).sqrt()
    removed_fraction = removed / (removed + retained + epsilon)
    removed_evidence = bounded_evidence(removed) * removed_fraction
    sar_reliability = bounded_evidence(retained) * (1.0 - removed_fraction)
    maps = (removed_evidence, sar_reliability)
    return tuple(
        F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
        .clamp(0.0, 1.0)
        .to(raw_detail.dtype)
        for value in maps
    )


def optical_missing_map(optical, baseline, epsilon=1e-6):
    optical_structure = bounded_evidence(gradient_magnitude(optical))
    baseline_structure = bounded_evidence(gradient_magnitude(baseline))
    relative_deficit = F.relu(optical_structure - baseline_structure) / (
        optical_structure + epsilon
    )
    return (optical_structure * relative_deficit).clamp(0.0, 1.0)


class RoutingBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=dilation, dilation=dilation,
                groups=channels, bias=False,
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, features):
        return features + self.block(features)


class CompetitiveStructureRouter(nn.Module):
    """Route anchor/Optical/SAR evidence, then replace only output high bands."""

    def __init__(
        self, hidden_channels=32, max_mid_replacement=0.10,
        max_high_replacement=0.03, mid_sigma=3.0, high_sigma=1.0,
    ):
        super().__init__()
        self.max_mid_replacement = float(max_mid_replacement)
        self.max_high_replacement = float(max_high_replacement)
        self.mid_sigma = float(mid_sigma)
        self.high_sigma = float(high_sigma)
        self.input = nn.Conv2d(6, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            RoutingBlock(hidden_channels, 1), RoutingBlock(hidden_channels, 2)
        )
        self.logits = nn.Conv2d(hidden_channels, 3, 1)
        with torch.no_grad():
            self.logits.weight.zero_()
            self.logits.bias.copy_(torch.tensor((2.0, 0.0, 1.0)))
        # Exact identity at initialization, while retaining a nonzero derivative.
        self.mid_strength = nn.Parameter(torch.zeros(()))
        self.high_strength = nn.Parameter(torch.zeros(()))

    def forward(
        self, optical, sar, baseline, raw_sar_detail, clean_sar_detail,
        valid=None, return_maps=False,
    ):
        output_size = baseline.shape[-2:]
        removed, sar_reliability = spd_evidence_maps(
            raw_sar_detail, clean_sar_detail, output_size
        )
        missing = optical_missing_map(optical, baseline)
        evidence = torch.cat(
            (optical, sar, baseline, missing, removed, sar_reliability), dim=1
        )
        probabilities = torch.softmax(
            self.logits(self.body(F.gelu(self.input(evidence)))), dim=1
        )
        anchor_gate, optical_gate, sar_gate = probabilities.split(1, dim=1)
        optical_gate = optical_gate * missing * (1.0 - sar_reliability)
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
        mid_correction = mid_scale * optical_gate * (optical_mid - baseline_mid)
        high_correction = high_scale * optical_gate * (optical_high - baseline_high)
        output = baseline + mid_correction + high_correction

        diagnostics = {
            "csr_mid_scale": mid_scale,
            "csr_high_scale": high_scale,
            "csr_optical_gate_mean": optical_gate.mean(),
            "csr_anchor_probability": anchor_gate.mean(),
            "csr_sar_probability": sar_gate.mean(),
            "csr_missing_mean": missing.mean(),
            "csr_removed_mean": removed.mean(),
            "csr_sar_reliability_mean": sar_reliability.mean(),
            "csr_correction_rms": (mid_correction + high_correction).square().mean().sqrt(),
        }
        if return_maps:
            diagnostics.update({
                "csr_missing_map": missing,
                "csr_removed_map": removed,
                "csr_sar_reliability_map": sar_reliability,
                "csr_anchor_gate_map": anchor_gate,
                "csr_optical_probability_map": probabilities[:, 1:2],
                "csr_optical_gate_map": optical_gate,
                "csr_sar_gate_map": sar_gate,
            })
        return output, diagnostics


def weighted_mean(values, weight, epsilon=1e-6):
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(1)
    weight = weight.expand_as(values).to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(epsilon)


def charbonnier(values, epsilon=1e-3):
    return torch.sqrt(values.float().square() + epsilon ** 2).to(values.dtype) - epsilon


def routing_targets(missing, removed, sar_reliability, valid, epsilon=1e-6):
    optical = missing * (1.0 - sar_reliability) * valid
    sar = sar_reliability * (1.0 - missing) * valid
    anchor = (1.0 - torch.maximum(optical, sar)).clamp(0.0, 1.0) * valid
    target = torch.cat((anchor, optical, sar), dim=1)
    return target / target.sum(dim=1, keepdim=True).clamp_min(epsilon)


def competitive_structure_losses(
    output, baseline, optical, valid, diagnostics,
    low_sigma=3.0, epsilon=1e-3,
):
    missing = diagnostics["csr_missing_map"].detach()
    removed = diagnostics["csr_removed_map"].detach()
    sar_reliability = diagnostics["csr_sar_reliability_map"].detach()
    optical_mask = missing * (1.0 - sar_reliability) * valid
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
    output_magnitude = gradient_magnitude(output)
    supported = torch.maximum(
        gradient_magnitude(optical), gradient_magnitude(baseline)
    ).detach()
    unsupported = weighted_mean(F.relu(output_magnitude - supported), valid)
    anchor = weighted_mean(
        torch.abs(output - baseline), (1.0 - optical_mask).clamp(0.0, 1.0) * valid
    )
    low_frequency = weighted_mean(
        torch.abs(gaussian_blur(output, low_sigma) - gaussian_blur(baseline, low_sigma)),
        valid,
    )

    probabilities = torch.cat(
        (
            diagnostics["csr_anchor_gate_map"],
            diagnostics["csr_optical_probability_map"],
            diagnostics["csr_sar_gate_map"],
        ),
        dim=1,
    ).clamp_min(1e-6)
    targets = routing_targets(missing, removed, sar_reliability, valid)
    routing = weighted_mean(
        -(targets * probabilities.log()).sum(dim=1, keepdim=True), valid
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
        "negative_scale": negative_scale,
    }
