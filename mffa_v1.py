"""MFFA v1: frequency-factorized provenance-preserving symmetric fusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net import TransformerBlock


def gaussian_kernel(kernel_size=7, sigma=1.0):
    coordinates = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)


class FixedFrequencyAnalysis(nn.Module):
    """Exactly reconstructable two-level analysis in bounded image logit space."""

    def __init__(self, kernel_size=7, sigma=1.0, epsilon=1.0 / 255.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = int(kernel_size)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "kernel", gaussian_kernel(kernel_size, sigma)[None, None]
        )

    def blur(self, values, valid):
        pad = self.kernel_size // 2
        padded_values = F.pad(values * valid, (pad, pad, pad, pad), mode="reflect")
        padded_valid = F.pad(valid, (pad, pad, pad, pad), mode="reflect")
        numerator = F.conv2d(padded_values, self.kernel)
        denominator = F.conv2d(padded_valid, self.kernel).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, image, valid=None):
        if image.shape[-2] % 2 or image.shape[-1] % 2:
            raise ValueError("MFFA input height and width must be even")
        if valid is None:
            valid = torch.ones_like(image)
        valid = valid.to(image.dtype)
        clipped = image.clamp(self.epsilon, 1.0 - self.epsilon)
        image_logit = torch.logit(clipped) * valid
        low_1 = self.blur(image_logit, valid)
        high = (image_logit - low_1) * valid
        low_1_blurred = self.blur(low_1, valid)
        low = F.avg_pool2d(low_1_blurred, 2, 2)
        low_up = F.interpolate(low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        mid = (low_1 - low_up) * valid
        low_valid = F.avg_pool2d(valid, 2, 2)
        return {
            "low": low,
            "mid": mid,
            "high": high,
            "valid": valid,
            "low_valid": low_valid,
            "image_logit": image_logit,
        }

    @staticmethod
    def synthesize(low, mid, high):
        low_up = F.interpolate(low, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        logit = low_up + mid + high
        return torch.sigmoid(logit), logit


class LocalResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, features):
        return features + self.block(features)


def transformer_stack(channels, blocks, heads=8):
    return nn.Sequential(
        *[
            TransformerBlock(
                dim=channels,
                num_heads=heads,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type="WithBias",
            )
            for _ in range(int(blocks))
        ]
    )


class SharedLowEncoder(nn.Module):
    def __init__(self, hidden=48, out_channels=64, blocks=4):
        super().__init__()
        self.input = nn.Conv2d(1, hidden, 3, padding=1)
        self.body = transformer_stack(hidden, blocks, heads=8)
        self.output = nn.Conv2d(hidden, out_channels, 3, padding=1)

    def forward(self, low):
        return self.output(self.body(self.input(low)))


class ModalityDetailEncoder(nn.Module):
    def __init__(self, channels=64, mid_blocks=2, high_blocks=2):
        super().__init__()
        self.mid_input = nn.Conv2d(1, channels, 3, padding=1)
        self.mid_body = transformer_stack(channels, mid_blocks, heads=8)
        self.high_input = nn.Conv2d(1, channels, 3, padding=1)
        self.mid_to_high = nn.Conv2d(channels, channels, 1)
        self.high_body = nn.Sequential(
            *[LocalResidualBlock(channels) for _ in range(int(high_blocks))]
        )

    def forward(self, mid, high):
        mid_feature = self.mid_body(self.mid_input(mid))
        high_feature = self.high_input(high) + self.mid_to_high(mid_feature)
        high_feature = self.high_body(high_feature)
        return mid_feature, high_feature


class ProvenanceFusion(nn.Module):
    def __init__(self, channels=64, blocks=2):
        super().__init__()
        self.input = nn.Conv2d(channels * 4, channels, 1)
        self.body = nn.Sequential(
            *[LocalResidualBlock(channels) for _ in range(int(blocks))]
        )

    def forward(self, optical, sar):
        evidence = torch.cat(
            (optical, sar, torch.abs(optical - sar), optical * sar), dim=1
        )
        return self.body(self.input(evidence))


class OpticalAnchoredHighFusion(nn.Module):
    def __init__(self, channels=64, blocks=2, max_relative_sar=0.05, epsilon=1e-6):
        super().__init__()
        self.max_relative_sar = float(max_relative_sar)
        self.epsilon = float(epsilon)
        self.sar_body = nn.Sequential(
            *[LocalResidualBlock(channels) for _ in range(int(blocks))]
        )
        self.sar_delta = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.sar_delta.weight)
        nn.init.zeros_(self.sar_delta.bias)

    def forward(self, optical, sar, return_diagnostics=False):
        proposal = torch.tanh(self.sar_delta(self.sar_body(sar)))
        optical_rms = optical.detach().square().mean(
            dim=(2, 3), keepdim=True
        ).sqrt().clamp_min(self.epsilon)
        sar_residual = self.max_relative_sar * optical_rms * proposal
        fused = optical + sar_residual
        if not return_diagnostics:
            return fused
        return fused, {
            "high_sar_residual_rms": sar_residual.square().mean().sqrt(),
            "high_optical_rms": optical_rms.mean(),
            "high_relative_sar": sar_residual.square().mean().sqrt()
            / optical.detach().square().mean().sqrt().clamp_min(self.epsilon),
        }


class SymmetricFrequencyDecoder(nn.Module):
    def __init__(self, channels=64, low_hidden=48):
        super().__init__()
        self.low_input = nn.Conv2d(channels, low_hidden, 3, padding=1)
        self.low_body = transformer_stack(low_hidden, 4, heads=8)
        self.low_output = nn.Conv2d(low_hidden, 1, 3, padding=1)
        self.mid_body = transformer_stack(channels, 2, heads=8)
        self.mid_output = nn.Conv2d(channels, 1, 3, padding=1)
        self.high_body = nn.Sequential(LocalResidualBlock(channels), LocalResidualBlock(channels))
        self.high_output = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, low, mid, high):
        low_hat = self.low_output(self.low_body(self.low_input(low)))
        mid_hat = self.mid_output(self.mid_body(mid))
        high_hat = self.high_output(self.high_body(high))
        image, logit = FixedFrequencyAnalysis.synthesize(low_hat, mid_hat, high_hat)
        return image, {
            "low": low_hat,
            "mid": mid_hat,
            "high": high_hat,
            "logit": logit,
        }


class MFFAV1(nn.Module):
    def __init__(self, channels=64, max_relative_sar=0.05):
        super().__init__()
        self.analysis = FixedFrequencyAnalysis()
        self.low_encoder = SharedLowEncoder(out_channels=channels)
        self.optical_detail_encoder = ModalityDetailEncoder(channels=channels)
        self.sar_detail_encoder = ModalityDetailEncoder(channels=channels)
        self.low_fusion = ProvenanceFusion(channels=channels)
        self.mid_fusion = ProvenanceFusion(channels=channels)
        self.high_fusion = OpticalAnchoredHighFusion(
            channels=channels, max_relative_sar=max_relative_sar
        )
        self.decoder = SymmetricFrequencyDecoder(channels=channels)

    def encode(self, image, modality, valid=None):
        bands = self.analysis(image, valid)
        low = self.low_encoder(bands["low"])
        if modality == "optical":
            mid, high = self.optical_detail_encoder(bands["mid"], bands["high"])
        elif modality == "sar":
            mid, high = self.sar_detail_encoder(bands["mid"], bands["high"])
        else:
            raise ValueError("modality must be optical or sar")
        return {"low": low, "mid": mid, "high": high, "bands": bands}

    def reconstruct(self, image, modality, valid=None):
        encoded = self.encode(image, modality, valid)
        reconstructed, decoded = self.decoder(
            encoded["low"], encoded["mid"], encoded["high"]
        )
        return reconstructed, encoded, decoded

    def fuse(self, optical, sar, sar_valid=None, return_diagnostics=False):
        optical_encoded = self.encode(optical, "optical")
        sar_encoded = self.encode(sar, "sar", sar_valid)
        fused_low = self.low_fusion(optical_encoded["low"], sar_encoded["low"])
        fused_mid = self.mid_fusion(optical_encoded["mid"], sar_encoded["mid"])
        high_output = self.high_fusion(
            optical_encoded["high"],
            sar_encoded["high"],
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            fused_high, diagnostics = high_output
        else:
            fused_high, diagnostics = high_output, {}
        fused, decoded = self.decoder(fused_low, fused_mid, fused_high)
        return fused, {
            "optical": optical_encoded,
            "sar": sar_encoded,
            "fused": {"low": fused_low, "mid": fused_mid, "high": fused_high},
            "decoded": decoded,
            "diagnostics": diagnostics,
        }
