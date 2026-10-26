"""P2 minimal single-path fusion and capacity-matched raw-feature control."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pdcf_modules import ConvResidualBlock


def stable_rms(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """RMS with a finite zero-point gradient and an exact zero value."""

    return torch.sqrt(values.square().mean() + epsilon ** 2) - epsilon


class SinglePathFusionAdapter(nn.Module):
    """Map four feature-role tensors to bounded C0 Base/Detail corrections."""

    INPUT_MODES = ("predictable_residual", "raw_evidence")

    def __init__(
        self,
        feature_channels: int = 128,
        output_channels: int = 64,
        hidden_channels: int = 64,
        blocks: int = 2,
        max_relative_correction: float = 0.25,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < max_relative_correction <= 1.0:
            raise ValueError("max_relative_correction must be in (0,1]")
        self.input_projection = nn.Sequential(
            nn.Conv2d(4 * feature_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ConvResidualBlock(hidden_channels) for _ in range(blocks)]
        )
        self.output = nn.Conv2d(
            hidden_channels, 2 * output_channels, kernel_size=1
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.output_channels = output_channels
        self.max_relative_correction = float(max_relative_correction)
        self.epsilon = float(epsilon)

    def compose_inputs(
        self, predictor_outputs: Dict[str, torch.Tensor], input_mode: str
    ) -> torch.Tensor:
        if input_mode == "predictable_residual":
            tensors = (
                predictor_outputs["predictable_optical"],
                predictor_outputs["predictable_sar"],
                predictor_outputs["residual_optical"],
                predictor_outputs["residual_sar"],
            )
        elif input_mode == "raw_evidence":
            optical = predictor_outputs["optical"]
            sar = predictor_outputs["sar"]
            tensors = (optical, sar, torch.abs(optical - sar), optical * sar)
        else:
            raise ValueError(f"Unknown P2 input mode: {input_mode}")
        if any(tensor.shape != tensors[0].shape for tensor in tensors):
            raise ValueError("All P2 input tensors must have identical shapes")
        return torch.cat(tensors, dim=1)

    def forward(
        self,
        predictor_outputs: Dict[str, torch.Tensor],
        c0_base: torch.Tensor,
        c0_detail: torch.Tensor,
        input_mode: str,
    ):
        hidden = self.blocks(
            self.input_projection(
                self.compose_inputs(predictor_outputs, input_mode)
            )
        )
        proposal_base, proposal_detail = torch.tanh(self.output(hidden)).split(
            self.output_channels, dim=1
        )
        base_scale = torch.sqrt(
            c0_base.detach().square().mean(dim=(2, 3), keepdim=True)
            + self.epsilon ** 2
        )
        detail_scale = torch.sqrt(
            c0_detail.detach().square().mean(dim=(2, 3), keepdim=True)
            + self.epsilon ** 2
        )
        delta_base = self.max_relative_correction * base_scale * proposal_base
        delta_detail = self.max_relative_correction * detail_scale * proposal_detail
        fused_base = c0_base + delta_base
        fused_detail = c0_detail + delta_detail
        delta_base_rms = stable_rms(delta_base, self.epsilon)
        delta_detail_rms = stable_rms(delta_detail, self.epsilon)
        c0_base_rms = torch.sqrt(c0_base.detach().square().mean() + self.epsilon ** 2)
        c0_detail_rms = torch.sqrt(
            c0_detail.detach().square().mean() + self.epsilon ** 2
        )
        diagnostics = {
            "delta_base_rms": delta_base_rms,
            "delta_detail_rms": delta_detail_rms,
            "relative_base_correction": (
                delta_base_rms / c0_base_rms
            ),
            "relative_detail_correction": (
                delta_detail_rms / c0_detail_rms
            ),
        }
        return fused_base, fused_detail, diagnostics


class PDCFSinglePathModel(nn.Module):
    """Frozen C0 + frozen P1 predictor + trainable matched P2 adapter."""

    def __init__(
        self,
        c0,
        predictor,
        input_mode: str = "predictable_residual",
        hidden_channels: int = 64,
        blocks: int = 2,
        max_relative_correction: float = 0.25,
    ) -> None:
        super().__init__()
        if input_mode not in SinglePathFusionAdapter.INPUT_MODES:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        self.c0 = c0
        self.predictor = predictor
        channels = int(predictor.channels)
        self.adapter = SinglePathFusionAdapter(
            feature_channels=2 * channels,
            output_channels=channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            max_relative_correction=max_relative_correction,
        )
        self.input_mode = input_mode
        self.freeze_sources()

    def freeze_sources(self) -> None:
        for module in (self.c0, self.predictor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.c0.eval()
        self.predictor.eval()
        self.adapter.train(mode)
        return self

    @torch.no_grad()
    def frozen_features(self, optical, sar):
        features = self.c0.extract_features(optical, sar)
        c0_base, c0_detail = self.c0.c0_features(*features)
        predictor_outputs = self.predictor(*features)
        return c0_base, c0_detail, predictor_outputs

    def forward(self, optical, sar, return_diagnostics=False):
        c0_base, c0_detail, predictor_outputs = self.frozen_features(optical, sar)
        fused_base, fused_detail, diagnostics = self.adapter(
            predictor_outputs, c0_base, c0_detail, self.input_mode
        )
        generated, decoder_feature = self.c0.decoder(
            optical, fused_base, fused_detail
        )
        if not return_diagnostics:
            return generated, decoder_feature
        with torch.no_grad():
            c0_output, _ = self.c0.decoder(optical, c0_base, c0_detail)
        diagnostics.update(
            {
                "output_deviation_rms": stable_rms(
                    generated - c0_output
                ),
                "c0_output": c0_output,
            }
        )
        return generated, decoder_feature, diagnostics
