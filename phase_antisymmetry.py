"""Monogenic phase antisymmetry adapted from the inspected MATLAB MSPA code."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_phase_bridge(
    roa_probability,
    phase_probability,
    kernel_size=5,
    seed_threshold=0.5,
):
    """Allow phase evidence to extend, but not independently create, RoA lines."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("bridge kernel_size must be an odd integer >= 3")
    roa_probability = roa_probability.clamp(0.0, 1.0)
    phase_probability = phase_probability.clamp(0.0, 1.0)
    seed = (roa_probability > seed_threshold).to(roa_probability.dtype)
    support = F.max_pool2d(
        seed,
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    bridge = roa_probability + (
        (1.0 - roa_probability) * phase_probability * support
    )
    return bridge.clamp(0.0, 1.0), support


class MonogenicPhaseAntisymmetry(nn.Module):
    """Continuous odd-dominant phase evidence without per-image thresholding."""

    def __init__(
        self,
        wavelengths=None,
        sigma_on_f=0.3,
        reflect_padding=32,
        epsilon=1e-6,
    ):
        super().__init__()
        if wavelengths is None:
            wavelengths = tuple(15.0 * 1.5 ** exponent for exponent in (-2, -1, 0, 1))
        self.wavelengths = tuple(float(value) for value in wavelengths)
        self.sigma_on_f = float(sigma_on_f)
        self.reflect_padding = int(reflect_padding)
        self.epsilon = float(epsilon)
        if not self.wavelengths or any(value <= 0.0 for value in self.wavelengths):
            raise ValueError("wavelengths must be positive")
        if not 0.0 < self.sigma_on_f < 1.0:
            raise ValueError("sigma_on_f must be in (0,1)")
        if self.reflect_padding < 0:
            raise ValueError("reflect_padding must be non-negative")
        self._filter_cache = {}

    def _filters(self, height, width, device, dtype):
        key = (height, width, device.type, device.index, dtype)
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(x * x + y * y)
        safe_radius = radius.clone()
        safe_radius[0, 0] = 1.0
        filters = []
        denominator = 2.0 * torch.log(
            torch.tensor(self.sigma_on_f, device=device, dtype=dtype)
        ) ** 2
        for wavelength in self.wavelengths:
            center_frequency = 1.0 / wavelength
            log_gabor = torch.exp(
                -(torch.log(safe_radius / center_frequency) ** 2) / denominator
            )
            log_gabor[0, 0] = 0.0
            if height % 2 == 0:
                log_gabor[height // 2, :] = 0.0
            if width % 2 == 0:
                log_gabor[:, width // 2] = 0.0
            filters.append(log_gabor)
        filters = torch.stack(filters)
        filters = filters / filters.sum(dim=0).max().clamp_min(self.epsilon)
        riesz = (1j * y - x) / safe_radius
        riesz[0, 0] = 0.0
        cached = (filters, riesz)
        self._filter_cache[key] = cached
        return cached

    def forward(self, image, valid=None):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("phase antisymmetry expects Bx1xHxW input")
        if valid is None:
            valid = torch.ones_like(image)
        valid = (valid > 0.5).to(image.dtype)
        padding = self.reflect_padding
        if padding:
            image_work = F.pad(image, (padding,) * 4, mode="reflect")
            valid_work = F.pad(valid, (padding,) * 4, mode="reflect")
        else:
            image_work = image
            valid_work = valid
        height, width = image_work.shape[-2:]
        filters, riesz = self._filters(height, width, image.device, image.dtype)
        image_fft = torch.fft.fft2(image_work)
        filtered_fft = image_fft.unsqueeze(2) * filters.view(1, 1, -1, height, width)
        even = torch.fft.ifft2(filtered_fft).real
        odd_complex = torch.fft.ifft2(filtered_fft * riesz)
        odd_y = odd_complex.real
        odd_x = odd_complex.imag
        odd = torch.sqrt((odd_x ** 2 + odd_y ** 2).clamp_min(0.0))
        even_abs = even.abs()
        amplitude = torch.sqrt((even_abs ** 2 + odd ** 2).clamp_min(0.0))
        fa_scales = torch.relu(odd - even_abs) / amplitude.clamp_min(self.epsilon)
        fa_mean = fa_scales.mean(dim=2)
        best_values, best_scale = fa_scales.max(dim=2)
        best_odd_y = torch.gather(odd_y, 2, best_scale.unsqueeze(2)).squeeze(2)
        best_odd_x = torch.gather(odd_x, 2, best_scale.unsqueeze(2)).squeeze(2)
        orientation = torch.atan2(best_odd_y, best_odd_x).remainder(torch.pi)
        if padding:
            crop = (..., slice(padding, -padding), slice(padding, -padding))
            fa_scales = fa_scales[crop]
            fa_mean = fa_mean[crop]
            best_values = best_values[crop]
            orientation = orientation[crop]
            valid_work = valid_work[crop]
        fa_scales = fa_scales * valid_work.unsqueeze(2)
        return {
            "fa_scales": fa_scales,
            "fa_mean": fa_mean * valid,
            "fa_peak": best_values * valid,
            "orientation": orientation * valid,
        }
