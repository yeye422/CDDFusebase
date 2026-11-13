"""Learnable wavelet prior decomposition for optical/SAR feature pairs.

The module operates on the shared full-resolution encoder features and returns
Base/Detail tensors with the same shape.  Four modes match the planned
ablation sequence:

* ``fixed``: fixed Haar Base/Detail decomposition (E1);
* ``gated``: context-gated directional sub-bands (E2);
* ``gated_prl``: E2 with a Haar-initialized perfect-reconstruction lifting bank;
* ``recovery``: gated sub-bands plus controlled SAR recovery (E3);
* ``phase``: recovery additionally conditioned on fixed phase symmetry (E4).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Orthonormal one-level Haar DWT/IDWT with right/bottom padding."""

    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) / 2.0
        self.register_buffer("filters", filters, persistent=False)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("Haar DWT expects an NCHW tensor")
        batch, channels, height, width = features.shape
        pad_bottom = height % 2
        pad_right = width % 2
        if pad_bottom or pad_right:
            features = F.pad(
                features,
                (0, pad_right, 0, pad_bottom),
                mode="replicate",
            )
        work = features.reshape(batch * channels, 1, *features.shape[-2:])
        bands = F.conv2d(work, self.filters.to(features.dtype), stride=2)
        bands = bands.reshape(batch, channels, 4, *bands.shape[-2:])
        return tuple(bands[:, :, index] for index in range(4)), (height, width)

    def inverse(self, bands, output_size=None):
        if len(bands) != 4:
            raise ValueError("Haar IDWT expects four sub-bands")
        reference = bands[0]
        if any(band.shape != reference.shape for band in bands):
            raise ValueError("All Haar sub-bands must have identical shapes")
        batch, channels, height, width = reference.shape
        work = torch.stack(tuple(bands), dim=2).reshape(
            batch * channels, 4, height, width
        )
        reconstructed = F.conv_transpose2d(
            work,
            self.filters.to(reference.dtype),
            stride=2,
        ).reshape(batch, channels, height * 2, width * 2)
        if output_size is not None:
            reconstructed = reconstructed[..., : output_size[0], : output_size[1]]
        return reconstructed


class LearnableLiftingDWT2D(nn.Module):
    """Perfect-reconstruction 2D lifting transform initialized as Haar.

    The predict/update filters are shared by every feature channel and by both
    modalities.  Their residuals are zero-sum, so the predict filter keeps unit
    DC gain and the update filter keeps its Haar gain while learning a small
    spatial correction.  Invertibility follows from the lifting construction
    and does not depend on a reconstruction loss.
    """

    def __init__(self, kernel_size=3, deviation_scale=0.1):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("lifting kernel_size must be an odd integer >= 3")
        if not 0.0 < deviation_scale <= 1.0:
            raise ValueError("deviation_scale must be in (0, 1]")
        self.kernel_size = int(kernel_size)
        self.deviation_scale = float(deviation_scale)
        center = kernel_size // 2
        predict = torch.zeros(kernel_size)
        update = torch.zeros(kernel_size)
        predict[center] = 1.0
        update[center] = 0.5
        self.register_buffer("predict_base", predict)
        self.register_buffer("update_base", update)
        self.predict_horizontal_delta = nn.Parameter(torch.zeros(kernel_size))
        self.update_horizontal_delta = nn.Parameter(torch.zeros(kernel_size))
        self.predict_vertical_delta = nn.Parameter(torch.zeros(kernel_size))
        self.update_vertical_delta = nn.Parameter(torch.zeros(kernel_size))

    def _kernel(self, base, delta):
        residual = torch.tanh(delta)
        residual = residual - residual.mean()
        return base + self.deviation_scale * residual

    def kernels(self):
        return {
            "predict_horizontal": self._kernel(
                self.predict_base, self.predict_horizontal_delta
            ),
            "update_horizontal": self._kernel(
                self.update_base, self.update_horizontal_delta
            ),
            "predict_vertical": self._kernel(
                self.predict_base, self.predict_vertical_delta
            ),
            "update_vertical": self._kernel(
                self.update_base, self.update_vertical_delta
            ),
        }

    @staticmethod
    def _padding_mode(features, axis, padding):
        size = features.shape[-1] if axis == "horizontal" else features.shape[-2]
        return "reflect" if padding < size else "replicate"

    def _filter(self, features, kernel, axis):
        channels = features.shape[1]
        padding = self.kernel_size // 2
        mode = self._padding_mode(features, axis, padding)
        if axis == "horizontal":
            features = F.pad(features, (padding, padding, 0, 0), mode=mode)
            weight = kernel.view(1, 1, 1, -1)
        elif axis == "vertical":
            features = F.pad(features, (0, 0, padding, padding), mode=mode)
            weight = kernel.view(1, 1, -1, 1)
        else:
            raise ValueError(f"Unknown lifting axis: {axis}")
        weight = weight.to(device=features.device, dtype=features.dtype)
        weight = weight.repeat(channels, 1, 1, 1)
        return F.conv2d(features, weight, groups=channels)

    @staticmethod
    def _split(features, axis):
        if axis == "horizontal":
            return features[..., 0::2], features[..., 1::2]
        return features[..., 0::2, :], features[..., 1::2, :]

    @staticmethod
    def _interleave(even, odd, axis):
        if axis == "horizontal":
            return torch.stack((even, odd), dim=-1).flatten(-2, -1)
        stacked = torch.stack((even, odd), dim=-2)
        return stacked.flatten(-3, -2)

    def _forward_axis(self, features, predict, update, axis):
        even, odd = self._split(features, axis)
        detail = odd - self._filter(even, predict, axis)
        smooth = even + self._filter(detail, update, axis)
        root_two = math.sqrt(2.0)
        return smooth * root_two, detail / root_two

    def _inverse_axis(self, low, high, predict, update, axis):
        root_two = math.sqrt(2.0)
        smooth = low / root_two
        detail = high * root_two
        even = smooth - self._filter(detail, update, axis)
        odd = detail + self._filter(even, predict, axis)
        return self._interleave(even, odd, axis)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("Learnable lifting DWT expects an NCHW tensor")
        height, width = features.shape[-2:]
        pad_bottom = height % 2
        pad_right = width % 2
        if pad_bottom or pad_right:
            features = F.pad(
                features,
                (0, pad_right, 0, pad_bottom),
                mode="replicate",
            )
        kernels = self.kernels()
        low_x, high_x = self._forward_axis(
            features,
            kernels["predict_horizontal"],
            kernels["update_horizontal"],
            "horizontal",
        )
        ll, lh = self._forward_axis(
            low_x,
            kernels["predict_vertical"],
            kernels["update_vertical"],
            "vertical",
        )
        hl, hh = self._forward_axis(
            high_x,
            kernels["predict_vertical"],
            kernels["update_vertical"],
            "vertical",
        )
        return (ll, lh, hl, hh), (height, width)

    def inverse(self, bands, output_size=None):
        if len(bands) != 4:
            raise ValueError("Learnable lifting IDWT expects four sub-bands")
        if any(band.shape != bands[0].shape for band in bands):
            raise ValueError("All lifting sub-bands must have identical shapes")
        ll, lh, hl, hh = bands
        kernels = self.kernels()
        low_x = self._inverse_axis(
            ll,
            lh,
            kernels["predict_vertical"],
            kernels["update_vertical"],
            "vertical",
        )
        high_x = self._inverse_axis(
            hl,
            hh,
            kernels["predict_vertical"],
            kernels["update_vertical"],
            "vertical",
        )
        reconstructed = self._inverse_axis(
            low_x,
            high_x,
            kernels["predict_horizontal"],
            kernels["update_horizontal"],
            "horizontal",
        )
        if output_size is not None:
            reconstructed = reconstructed[..., : output_size[0], : output_size[1]]
        return reconstructed

    def deviation(self):
        kernels = self.kernels()
        return torch.stack(
            (
                (kernels["predict_horizontal"] - self.predict_base).abs().mean(),
                (kernels["update_horizontal"] - self.update_base).abs().mean(),
                (kernels["predict_vertical"] - self.predict_base).abs().mean(),
                (kernels["update_vertical"] - self.update_base).abs().mean(),
            )
        ).mean()


class DirectionalPhaseSymmetry(nn.Module):
    """Fixed multi-scale log-Gabor/Riesz phase-symmetry evidence."""

    def __init__(
        self,
        wavelengths=(4.0, 8.0, 16.0),
        orientations=4,
        sigma_on_f=0.55,
        reflect_padding=16,
        noise_threshold=0.0,
        epsilon=1e-6,
    ):
        super().__init__()
        if orientations < 1:
            raise ValueError("orientations must be positive")
        if not wavelengths or any(value <= 0 for value in wavelengths):
            raise ValueError("wavelengths must contain positive values")
        if not 0.0 < sigma_on_f < 1.0:
            raise ValueError("sigma_on_f must be in (0, 1)")
        self.wavelengths = tuple(float(value) for value in wavelengths)
        self.orientations = int(orientations)
        self.sigma_on_f = float(sigma_on_f)
        self.reflect_padding = int(reflect_padding)
        self.noise_threshold = float(noise_threshold)
        self.epsilon = float(epsilon)
        self._filter_cache = {}

    def _filters(self, height, width, device, dtype):
        key = (height, width, device.type, device.index, dtype)
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(x.square() + y.square())
        safe_radius = radius.clone()
        safe_radius[0, 0] = 1.0
        denominator = 2.0 * math.log(self.sigma_on_f) ** 2
        radial_filters = []
        for wavelength in self.wavelengths:
            center_frequency = 1.0 / wavelength
            log_gabor = torch.exp(
                -(torch.log(safe_radius / center_frequency).square()) / denominator
            )
            log_gabor[0, 0] = 0.0
            radial_filters.append(log_gabor)
        radial_filters = torch.stack(radial_filters)
        riesz_x = 1j * x / safe_radius
        riesz_y = 1j * y / safe_radius
        riesz_x[0, 0] = 0.0
        riesz_y[0, 0] = 0.0
        cached = (radial_filters, riesz_x, riesz_y)
        self._filter_cache[key] = cached
        return cached

    def forward(self, image, return_both=False):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Phase symmetry expects a Bx1xHxW image")
        input_dtype = image.dtype
        height, width = image.shape[-2:]
        padding = min(self.reflect_padding, height - 1, width - 1)
        with torch.no_grad():
            work = image.float()
            if padding:
                work = F.pad(work, (padding,) * 4, mode="reflect")
            work_height, work_width = work.shape[-2:]
            radial, riesz_x, riesz_y = self._filters(
                work_height, work_width, work.device, work.dtype
            )
            image_fft = torch.fft.fft2(work)
            filtered = image_fft.unsqueeze(2) * radial.view(
                1, 1, -1, work_height, work_width
            )
            even = torch.fft.ifft2(filtered).real
            odd_x = torch.fft.ifft2(filtered * riesz_x).real
            odd_y = torch.fft.ifft2(filtered * riesz_y).real
            symmetry_responses = []
            antisymmetry_responses = []
            for index in range(self.orientations):
                angle = math.pi * index / self.orientations
                odd = odd_x * math.cos(angle) + odd_y * math.sin(angle)
                amplitude = torch.sqrt(even.square() + odd.square())
                symmetry_numerator = torch.relu(
                    even.abs().sum(dim=2)
                    - odd.abs().sum(dim=2)
                    - self.noise_threshold
                )
                antisymmetry_numerator = torch.relu(
                    odd.abs().sum(dim=2)
                    - even.abs().sum(dim=2)
                    - self.noise_threshold
                )
                denominator = amplitude.sum(dim=2).clamp_min(self.epsilon)
                symmetry_responses.append(symmetry_numerator / denominator)
                antisymmetry_responses.append(antisymmetry_numerator / denominator)
            symmetry = torch.cat(symmetry_responses, dim=1).clamp(0.0, 1.0)
            antisymmetry = torch.cat(antisymmetry_responses, dim=1).clamp(
                0.0, 1.0
            )
            if padding:
                crop = (..., slice(padding, -padding), slice(padding, -padding))
                symmetry = symmetry[crop]
                antisymmetry = antisymmetry[crop]
        symmetry = symmetry.to(input_dtype)
        if not return_both:
            return symmetry
        return {
            "symmetry": symmetry,
            "antisymmetry": antisymmetry.to(input_dtype),
        }


class _Projection(nn.Sequential):
    def __init__(self, input_channels, hidden_channels, output_channels, bias=0.0):
        super().__init__(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
        nn.init.zeros_(self[-1].weight)
        nn.init.constant_(self[-1].bias, float(bias))


class LearnableWaveletPriorDecomposition(nn.Module):
    """Direction-aware Base/Detail construction for an optical/SAR pair."""

    MODES = ("fixed", "gated", "gated_prl", "recovery", "phase")

    def __init__(
        self,
        channels=64,
        mode="phase",
        hidden_channels=32,
        gate_floor=0.1,
        base_scale=0.1,
        recovery_scale=0.2,
        phase_orientations=4,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown LWPD mode: {mode}")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        if not 0.0 <= recovery_scale <= 1.0:
            raise ValueError("recovery_scale must be in [0, 1]")
        self.channels = int(channels)
        self.mode = mode
        self.gate_floor = float(gate_floor)
        self.base_scale = float(base_scale)
        self.recovery_scale = float(recovery_scale)
        self.haar = (
            LearnableLiftingDWT2D()
            if mode == "gated_prl"
            else HaarDWT2D()
        )

        if mode != "fixed":
            context_channels = channels * 4
            self.optical_base = _Projection(channels, hidden_channels, channels)
            self.sar_base = _Projection(channels, hidden_channels, channels)
            self.optical_self_gate = _Projection(
                context_channels, hidden_channels, channels * 3, bias=2.0
            )
            self.sar_self_gate = _Projection(
                context_channels, hidden_channels, channels * 3, bias=2.0
            )
            self.optical_cross_gate = _Projection(
                context_channels * 2, hidden_channels, channels * 3
            )
            self.sar_cross_gate = _Projection(
                context_channels * 2, hidden_channels, channels * 3
            )

        if mode in ("recovery", "phase"):
            recovery_inputs = channels * 4
            if mode == "phase":
                self.phase_symmetry = DirectionalPhaseSymmetry(
                    orientations=phase_orientations
                )
                self.phase_projection = nn.Sequential(
                    nn.Conv2d(phase_orientations, channels, 1),
                    nn.GELU(),
                )
                recovery_inputs += channels
            self.sar_recovery = _Projection(
                recovery_inputs,
                hidden_channels,
                channels * 3,
                bias=-2.0,
            )

    @staticmethod
    def _context(bands):
        return torch.cat(tuple(bands), dim=1)

    @staticmethod
    def _split_directions(values):
        return values.chunk(3, dim=1)

    def _gates(self, optical_context, sar_context):
        optical_logits = self.optical_self_gate(optical_context)
        optical_logits = optical_logits + self.optical_cross_gate(
            torch.cat((optical_context, sar_context), dim=1)
        )
        sar_logits = self.sar_self_gate(sar_context)
        sar_logits = sar_logits + self.sar_cross_gate(
            torch.cat((sar_context, optical_context), dim=1)
        )
        optical = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(
            optical_logits
        )
        sar = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(sar_logits)
        return self._split_directions(optical), self._split_directions(sar)

    def _modulate_low_bands(self, optical_ll, sar_ll):
        optical = (1.0 + self.base_scale * torch.tanh(self.optical_base(optical_ll))) * optical_ll
        sar = (1.0 + self.base_scale * torch.tanh(self.sar_base(sar_ll))) * sar_ll
        return optical, sar, {}

    def forward(
        self,
        optical_features,
        sar_features,
        sar_image=None,
        phase_symmetry=None,
        return_diagnostics=False,
    ):
        if optical_features.shape != sar_features.shape:
            raise ValueError("Optical and SAR LWPD features must have identical shapes")
        if optical_features.ndim != 4 or optical_features.shape[1] != self.channels:
            raise ValueError(
                f"LWPD expects NCHW features with {self.channels} channels"
            )
        optical_bands, output_size = self.haar(optical_features)
        sar_bands, sar_output_size = self.haar(sar_features)
        if output_size != sar_output_size:
            raise ValueError("Optical and SAR feature sizes do not match")

        optical_ll, *optical_high = optical_bands
        sar_ll, *sar_high = sar_bands
        phase = None
        low_diagnostics = {}
        recovery = tuple(torch.zeros_like(band) for band in sar_high)

        if self.mode == "fixed":
            optical_gates = tuple(torch.ones_like(band) for band in optical_high)
            sar_gates = tuple(torch.ones_like(band) for band in sar_high)
            optical_ll_hat = optical_ll
            sar_ll_hat = sar_ll
        else:
            optical_context = self._context(optical_bands)
            sar_context = self._context(sar_bands)
            optical_gates, sar_gates = self._gates(optical_context, sar_context)
            optical_ll_hat, sar_ll_hat, low_diagnostics = self._modulate_low_bands(optical_ll, sar_ll)

            if self.mode in ("recovery", "phase"):
                recovery_inputs = [sar_context]
                if self.mode == "phase":
                    if phase_symmetry is None:
                        if sar_image is None:
                            raise ValueError(
                                "phase mode requires a SAR image or phase symmetry"
                            )
                        phase = self.phase_symmetry(sar_image)
                    else:
                        phase = phase_symmetry
                        if phase.ndim != 4:
                            raise ValueError("phase_symmetry must be an NCHW tensor")
                    phase = F.interpolate(
                        phase,
                        size=sar_ll.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    recovery_inputs.append(self.phase_projection(phase))
                recovery = self._split_directions(
                    torch.sigmoid(self.sar_recovery(torch.cat(recovery_inputs, dim=1)))
                )

        optical_high_hat = tuple(
            gate * band for gate, band in zip(optical_gates, optical_high)
        )
        sar_coefficients = tuple(
            gate + (1.0 - gate) * self.recovery_scale * residual
            for gate, residual in zip(sar_gates, recovery)
        )
        sar_high_hat = tuple(
            coefficient * band
            for coefficient, band in zip(sar_coefficients, sar_high)
        )
        zeros_optical = torch.zeros_like(optical_ll_hat)
        zeros_sar = torch.zeros_like(sar_ll_hat)
        optical_base = self.haar.inverse(
            (optical_ll_hat, zeros_optical, zeros_optical, zeros_optical),
            output_size,
        )
        sar_base = self.haar.inverse(
            (sar_ll_hat, zeros_sar, zeros_sar, zeros_sar),
            output_size,
        )
        optical_detail = self.haar.inverse(
            (zeros_optical, *optical_high_hat), output_size
        )
        sar_detail = self.haar.inverse((zeros_sar, *sar_high_hat), output_size)

        outputs = (optical_base, optical_detail, sar_base, sar_detail)
        if not return_diagnostics:
            return outputs
        return outputs, {
            **low_diagnostics,
            "optical_gates": torch.stack(optical_gates, dim=2),
            "sar_gates": torch.stack(sar_gates, dim=2),
            "sar_recovery": torch.stack(recovery, dim=2),
            "sar_coefficients": torch.stack(sar_coefficients, dim=2),
            "wavelet_deviation": (
                self.haar.deviation()
                if isinstance(self.haar, LearnableLiftingDWT2D)
                else optical_features.new_zeros(())
            ),
            "phase_symmetry": phase,
        }
