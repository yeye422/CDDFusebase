"""Globally calibrated fixed SAR line-structure evidence (FSE-v2a)."""

import hashlib
import json
from pathlib import Path

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd_size(value):
    value = int(value)
    if value < 3 or value % 2 == 0:
        raise ValueError("kernel sizes must be odd integers >= 3")
    return value


def _directional_half_kernels(size):
    radius = size // 2
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    kernels = []
    for coordinate in (x, y, x + y, x - y):
        kernels.append((coordinate < 0).float())
        kernels.append((coordinate > 0).float())
    return torch.stack(kernels).unsqueeze(1)


def _directional_smoothing_kernels(size, sigma):
    radius = size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    kernels = torch.zeros(4, 1, size, size)
    kernels[0, 0, radius, :] = weights
    kernels[1, 0, :, radius] = weights
    indices = torch.arange(size)
    kernels[2, 0, indices, indices] = weights
    kernels[3, 0, indices, size - 1 - indices] = weights
    return kernels


def load_calibration(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    payload = calibration.copy()
    expected = payload.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected and expected != actual:
        raise ValueError("structure calibration SHA-256 mismatch")
    return calibration


class FixedSARLineEvidenceV2(nn.Module):
    """FSE-v2a line evidence with shared thresholds calibrated on training data."""

    def __init__(
        self,
        calibration=None,
        scales=(3, 5, 9),
        smoothing_size=7,
        smoothing_sigma=2.0,
        direction_temperature=0.1,
        tensor_window=5,
        epsilon=1e-6,
    ):
        super().__init__()
        self.scales = tuple(_odd_size(scale) for scale in scales)
        self.smoothing_size = _odd_size(smoothing_size)
        self.smoothing_sigma = float(smoothing_sigma)
        self.direction_temperature = float(direction_temperature)
        self.tensor_window = _odd_size(tensor_window)
        self.epsilon = float(epsilon)
        if self.smoothing_sigma <= 0.0 or self.direction_temperature <= 0.0:
            raise ValueError("smoothing sigma and direction temperature must be positive")
        self.calibration = calibration
        self.spatial_gradient = kornia.filters.SpatialGradient()
        self.register_buffer(
            "smoothing_kernels",
            _directional_smoothing_kernels(
                self.smoothing_size, self.smoothing_sigma
            ),
        )
        for scale in self.scales:
            self.register_buffer(
                f"roa_kernels_{scale}", _directional_half_kernels(scale)
            )

    def _masked_filter(self, values, valid, kernels, padding):
        kernels = kernels.to(device=values.device, dtype=values.dtype)
        numerator = F.conv2d(values * valid, kernels, padding=padding)
        denominator = F.conv2d(valid, kernels.abs(), padding=padding)
        return numerator / denominator.clamp_min(self.epsilon), denominator

    def directional_coarse(self, sar, valid):
        filtered, support = self._masked_filter(
            sar,
            valid,
            self.smoothing_kernels,
            self.smoothing_size // 2,
        )
        local_mean = F.avg_pool2d(sar, 5, stride=1, padding=2)
        local_variance = F.avg_pool2d(
            (sar - local_mean) ** 2, 5, stride=1, padding=2
        ).clamp_min(self.epsilon)
        residual_energy = F.avg_pool2d(
            (sar - filtered) ** 2, 5, stride=1, padding=2
        )
        normalized_energy = residual_energy / local_variance
        normalized_energy = normalized_energy.masked_fill(support <= 0.5, 1e6)
        weights = torch.softmax(
            -normalized_energy / self.direction_temperature, dim=1
        )
        coarse = torch.sum(weights * filtered, dim=1, keepdim=True) * valid
        fine = (sar - coarse) * valid
        return coarse, fine, weights

    def raw_roa(self, coarse, valid, scale):
        kernels = getattr(self, f"roa_kernels_{scale}").to(
            device=coarse.device, dtype=coarse.dtype
        )
        values, masses = self._masked_filter(
            coarse, valid, kernels, scale // 2
        )
        values = values.view(
            coarse.shape[0], 4, 2, coarse.shape[-2], coarse.shape[-1]
        )
        masses = masses.view_as(values)
        response = torch.abs(
            torch.log(
                (values[:, :, 0] + self.epsilon)
                / (values[:, :, 1] + self.epsilon)
            )
        )
        supported = (masses[:, :, 0] > 0.5) & (masses[:, :, 1] > 0.5)
        response = response * supported.to(response.dtype)
        return response.max(dim=1, keepdim=True).values * valid

    def tensor_evidence(self, coarse, valid):
        gradients = self.spatial_gradient(coarse)
        gx = gradients[:, :, 0]
        gy = gradients[:, :, 1]
        size = self.tensor_window
        jxx = F.avg_pool2d(gx * gx, size, stride=1, padding=size // 2)
        jyy = F.avg_pool2d(gy * gy, size, stride=1, padding=size // 2)
        jxy = F.avg_pool2d(gx * gy, size, stride=1, padding=size // 2)
        anisotropy = torch.sqrt(
            ((jxx - jyy) ** 2 + 4.0 * jxy ** 2).clamp_min(0.0)
        )
        coherence = (anisotropy / (jxx + jyy + self.epsilon)).clamp(0.0, 1.0)
        cos2 = (jxx - jyy) / anisotropy.clamp_min(self.epsilon)
        sin2 = (2.0 * jxy) / anisotropy.clamp_min(self.epsilon)
        mean_cos2 = F.avg_pool2d(cos2, size, stride=1, padding=size // 2)
        mean_sin2 = F.avg_pool2d(sin2, size, stride=1, padding=size // 2)
        direction_consistency = torch.sqrt(
            (mean_cos2 ** 2 + mean_sin2 ** 2).clamp_min(0.0)
        ).clamp(0.0, 1.0)
        tensor_energy = torch.sqrt((jxx + jyy).clamp_min(0.0))
        return (
            coherence * valid,
            direction_consistency * valid,
            tensor_energy * valid,
        )

    def raw_features(self, sar, valid=None):
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError("FSE-v2a expects Bx1xHxW SAR input")
        if valid is None:
            valid = (sar > 0.0).to(sar.dtype)
        valid = (valid > 0.5).to(sar.dtype)
        coarse, fine, direction_weights = self.directional_coarse(sar, valid)
        raw_roa = {
            f"roa_raw_s{scale}": self.raw_roa(coarse, valid, scale)
            for scale in self.scales
        }
        coherence, direction_consistency, tensor_energy = self.tensor_evidence(
            coarse, valid
        )
        return {
            "sar_coarse": coarse,
            "sar_fine": fine,
            "direction_weights": direction_weights,
            **raw_roa,
            "coherence": coherence,
            "direction_consistency": direction_consistency,
            "tensor_energy": tensor_energy,
            "valid": valid,
        }

    def _calibrated_probability(self, raw, name):
        if self.calibration is None:
            raise RuntimeError("FSE-v2a probability maps require calibration")
        entry = self.calibration["responses"][name]
        threshold = float(entry["threshold"])
        transition = max(float(entry["transition"]), self.epsilon)
        return torch.sigmoid((raw - threshold) / transition)

    def forward(self, sar, valid=None):
        result = self.raw_features(sar, valid)
        probabilities = []
        for scale in self.scales:
            name = f"roa_raw_s{scale}"
            probability = self._calibrated_probability(result[name], name)
            probability = probability * result["valid"]
            result[f"roa_prob_s{scale}"] = probability
            probabilities.append(probability)
        stacked = torch.stack(probabilities, dim=0)
        count = min(2, len(probabilities))
        scale_persistence = torch.topk(stacked, count, dim=0).values[-1]
        line_score = (
            scale_persistence
            * result["coherence"]
            * result["direction_consistency"]
            * result["valid"]
        ).clamp_min(0.0).pow(1.0 / 3.0) * result["valid"]
        r_line = self._calibrated_probability(
            line_score, "line_score"
        ) * result["valid"]
        result["scale_persistence"] = scale_persistence
        result["line_score"] = line_score
        result["r_line"] = r_line
        # The globally calibrated q99.5 flat-background threshold maps to 0.5.
        # Values above 0.7 remain available as a stricter soft-confidence tier.
        result["line_high_mask"] = (r_line > 0.5).to(r_line.dtype)
        result["line_low_mask"] = (r_line < 0.2).to(r_line.dtype) * result["valid"]
        return result
