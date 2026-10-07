"""Globally calibrated, interpretable SAR structure evidence for C3."""

import json
from pathlib import Path

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


CALIBRATION_VERSION = "v2_global_empirical_thresholds"


def _odd_kernel_size(sigma):
    size = max(3, int(round(6.0 * sigma + 1.0)))
    return size + 1 if size % 2 == 0 else size


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


class FixedSARStructureEvidence(nn.Module):
    """Generate globally comparable line and point evidence from SAR only.

    ``r_fixed`` intentionally equals the conservative line evidence. Point
    evidence remains separate until a dedicated point-target protocol passes.
    """

    def __init__(
        self,
        scales=(3, 5, 9),
        calibration=None,
        coherence_sigma=1.5,
        coarse_sigma=1.0,
        epsilon=1e-6,
    ):
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        if not self.scales or any(scale < 3 or scale % 2 == 0 for scale in self.scales):
            raise ValueError("RoA scales must be odd integers >= 3")
        if coherence_sigma <= 0.0 or coarse_sigma <= 0.0:
            raise ValueError("smoothing sigmas must be positive")
        self.coherence_sigma = float(coherence_sigma)
        self.coarse_sigma = float(coarse_sigma)
        self.epsilon = float(epsilon)
        self.spatial_gradient = kornia.filters.SpatialGradient()
        self.calibration = self._validate_calibration(calibration)
        for scale in self.scales:
            self.register_buffer(
                f"roa_kernels_{scale}", _directional_half_kernels(scale)
            )

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            calibration = json.load(handle)
        return cls(scales=calibration["scales"], calibration=calibration)

    def _validate_calibration(self, calibration):
        if calibration is None:
            return None
        if calibration.get("version") != CALIBRATION_VERSION:
            raise ValueError("Unsupported structure calibration version")
        if tuple(calibration.get("scales", ())) != self.scales:
            raise ValueError("Calibration scales do not match evidence scales")
        thresholds = calibration.get("thresholds", {})
        temperatures = calibration.get("temperatures", {})
        required = {
            *(f"roa_s{scale}" for scale in self.scales),
            *(f"point_s{scale}" for scale in self.scales),
            "tensor_energy",
        }
        missing = sorted(required - set(thresholds))
        if missing:
            raise ValueError("Calibration is missing thresholds: " + ", ".join(missing))
        if any(float(thresholds[name]) <= 0.0 for name in required):
            raise ValueError("All calibration thresholds must be positive")
        if any(float(temperatures.get(name, 0.0)) <= 0.0 for name in required):
            raise ValueError("All calibration temperatures must be positive")
        return calibration

    def _normalized_gaussian(self, values, valid, sigma):
        size = _odd_kernel_size(sigma)
        numerator = kornia.filters.gaussian_blur2d(
            values * valid, (size, size), (sigma, sigma)
        )
        denominator = kornia.filters.gaussian_blur2d(
            valid, (size, size), (sigma, sigma)
        ).clamp_min(self.epsilon)
        return numerator / denominator

    def _coarse_sar(self, sar, valid):
        return self._normalized_gaussian(sar, valid, self.coarse_sigma)

    def _roa_raw(self, sar, valid, scale):
        kernels = getattr(self, f"roa_kernels_{scale}").to(
            device=sar.device, dtype=sar.dtype
        )
        padding = scale // 2
        sums = F.conv2d(sar * valid, kernels, padding=padding)
        masses = F.conv2d(valid, kernels, padding=padding)
        means = sums / masses.clamp_min(self.epsilon)
        means = means.view(sar.shape[0], 4, 2, sar.shape[-2], sar.shape[-1])
        masses = masses.view_as(means)
        response = torch.abs(
            torch.log(
                (means[:, :, 0] + self.epsilon)
                / (means[:, :, 1] + self.epsilon)
            )
        )
        supported = (masses[:, :, 0] > 0.5) & (masses[:, :, 1] > 0.5)
        return response.mul(supported).max(dim=1, keepdim=True).values * valid

    def _tensor_raw(self, sar, valid):
        gradients = self.spatial_gradient(sar)
        gx = gradients[:, :, 0]
        gy = gradients[:, :, 1]
        jxx = self._normalized_gaussian(gx * gx, valid, self.coherence_sigma)
        jyy = self._normalized_gaussian(gy * gy, valid, self.coherence_sigma)
        jxy = self._normalized_gaussian(gx * gy, valid, self.coherence_sigma)
        numerator = torch.sqrt(((jxx - jyy) ** 2 + 4.0 * jxy ** 2).clamp_min(0.0))
        denominator = jxx + jyy + self.epsilon
        coherence = (numerator / denominator).clamp(0.0, 1.0) * valid
        energy = torch.sqrt((jxx + jyy).clamp_min(0.0)) * valid
        return coherence, energy

    def _point_raw(self, sar, valid, scale):
        padding = scale // 2
        local_sum = F.avg_pool2d(sar * valid, scale, stride=1, padding=padding)
        local_mass = F.avg_pool2d(valid, scale, stride=1, padding=padding)
        local_mean = local_sum / local_mass.clamp_min(self.epsilon)
        contrast = torch.relu(
            torch.log((sar + self.epsilon) / (local_mean + self.epsilon))
        )
        return contrast * (local_mass > 0.5).to(contrast.dtype) * valid

    def raw_evidence(self, sar, valid=None):
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError("SAR evidence expects Bx1xHxW input")
        if valid is None:
            valid = (sar > 0.0).to(sar.dtype)
        valid = (valid > 0.5).to(sar.dtype)
        coarse = self._coarse_sar(sar, valid)
        result = {
            f"roa_s{scale}": self._roa_raw(coarse, valid, scale)
            for scale in self.scales
        }
        result.update(
            {
                f"point_s{scale}": self._point_raw(sar, valid, scale)
                for scale in self.scales
            }
        )
        coherence, tensor_energy = self._tensor_raw(coarse, valid)
        result["coherence"] = coherence
        result["tensor_energy"] = tensor_energy
        result["coarse_sar"] = coarse * valid
        return result

    def _probability(self, values, name, valid):
        threshold = float(self.calibration["thresholds"][name])
        temperature = float(self.calibration["temperatures"][name])
        return torch.sigmoid((values - threshold) / temperature) * valid

    @staticmethod
    def _geometric_mean(maps, epsilon=1e-8):
        logs = torch.stack(
            [torch.log(values.clamp_min(epsilon)) for values in maps], dim=0
        )
        return torch.exp(logs.mean(dim=0))

    @staticmethod
    def _top2_mean(maps):
        stacked = torch.stack(maps, dim=0)
        return torch.topk(stacked, min(2, len(maps)), dim=0).values.mean(dim=0)

    def forward(self, sar, valid=None):
        if self.calibration is None:
            raise RuntimeError("Calibrated forward requires a structure calibration")
        if valid is None:
            valid = (sar > 0.0).to(sar.dtype)
        valid = (valid > 0.5).to(sar.dtype)
        raw = self.raw_evidence(sar, valid)

        roa_maps = [
            self._probability(raw[f"roa_s{scale}"], f"roa_s{scale}", valid)
            for scale in self.scales
        ]
        point_maps = [
            self._probability(
                raw[f"point_s{scale}"], f"point_s{scale}", valid
            )
            for scale in self.scales
        ]
        energy_score = self._probability(raw["tensor_energy"], "tensor_energy", valid)
        line_persistence = self._geometric_mean(roa_maps) * valid
        r_line = self._geometric_mean(
            (line_persistence, raw["coherence"], energy_score)
        ) * valid

        point_persistence = self._top2_mean(point_maps) * valid
        neighborhood_sum = (
            F.avg_pool2d(point_persistence, 3, stride=1, padding=1) * 9.0
            - point_persistence
        )
        neighborhood_mean = neighborhood_sum / 8.0
        point_support = torch.sigmoid((neighborhood_mean - 0.5) / 0.1) * valid
        local_peak = F.max_pool2d(point_persistence, 3, stride=1, padding=1)
        peak_support = (
            point_persistence / local_peak.clamp_min(self.epsilon)
        ).clamp(0.0, 1.0)
        r_point = (
            point_persistence * point_support * peak_support * valid
        ).clamp(0.0, 1.0)

        result = {
            f"roa_s{scale}": evidence
            for scale, evidence in zip(self.scales, roa_maps)
        }
        result.update(
            {
                "coarse_sar": raw["coarse_sar"],
                "coherence": raw["coherence"],
                "tensor_energy": energy_score,
                "roa_peak": torch.stack(roa_maps, dim=0).max(dim=0).values,
                "line_persistence": line_persistence,
                "point_persistence": point_persistence,
                "point_support": point_support,
                "r_line": r_line.clamp(0.0, 1.0),
                "r_point": r_point,
                # Conservative v2 policy: point evidence is not fused yet.
                "r_fixed": r_line.clamp(0.0, 1.0),
            }
        )
        return result
