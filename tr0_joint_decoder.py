"""State-conditioned optical/SAR synthesis interleaved with Restormer blocks."""

import math

import torch
from torch import nn

from net import Restormer_Decoder


class SourceConditionedSynthesis(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.state_projection = nn.Conv2d(channels, channels, 1)
        self.optical_message = self._source_branch(channels)
        self.sar_message = self._source_branch(channels)

    @staticmethod
    def _source_branch(channels):
        return nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, state, optical, sar):
        if optical is None and sar is None:
            raise ValueError("At least one detail source is required")
        condition = self.state_projection(state)
        if optical is None:
            return state + self.sar_message(torch.cat((condition, sar), dim=1))
        optical_message = self.optical_message(torch.cat((condition, optical), dim=1))
        if sar is None:
            return state + optical_message
        sar_message = self.sar_message(torch.cat((condition, sar), dim=1))
        return state + (optical_message + sar_message) / math.sqrt(2)


class ProgressiveJointDecoder(nn.Module):
    def __init__(self, channels=64, *, early=False, restormer=None):
        super().__init__()
        if restormer is None:
            restormer = Restormer_Decoder(dim=channels)
        self.early = early
        self.base_projection = nn.Conv2d(channels, channels, 1)
        self.optical_projection = nn.Conv2d(channels, channels, 1)
        self.sar_projection = nn.Conv2d(channels, channels, 1)
        for projection in (self.base_projection, self.optical_projection, self.sar_projection):
            nn.init.dirac_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.synthesis_stages = nn.ModuleList(SourceConditionedSynthesis(channels) for _ in range(2))
        blocks = restormer.encoder_level2
        self.decoder_stages = nn.ModuleList((blocks[:2], blocks[2:]))
        self.output = restormer.output
        self.sigmoid = restormer.sigmoid

    def forward(self, fused_base, optical_detail, sar_detail, active_sources=(True, True)):
        if len(active_sources) != 2 or not any(active_sources):
            raise ValueError("At least one detail source must be active")
        for enabled, detail in zip(active_sources, (optical_detail, sar_detail)):
            if enabled and (detail is None or detail.shape != fused_base.shape):
                raise ValueError("Active detail sources must match the Base shape")
        state = self.base_projection(fused_base)
        optical = self.optical_projection(optical_detail) if active_sources[0] else None
        sar = self.sar_projection(sar_detail) if active_sources[1] else None
        first, second = self.synthesis_stages
        first_blocks, second_blocks = self.decoder_stages
        state = first(state, optical, sar)
        if self.early:
            state = first_blocks(second(state, optical, sar))
        else:
            state = second(first_blocks(state), optical, sar)
        return self.sigmoid(self.output(second_blocks(state)))
