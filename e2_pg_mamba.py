"""Fixed, paired 2-D position codes at the signed-Detail SSM input."""

import math

import torch
from torch import nn

from e2_phase_guide import SignedPhaseGuideE2


VARIANT_PREFIX = "lwpd_e2_pg_mamba_v1_"
POSITION_SCALE = 0.02
SHUFFLE_SEED = 1729
MODES = ("pg", "sar_shuffled")


def coordinates(height, width, channels, device):
    if channels % 8:
        raise ValueError("The SSM width must be divisible by eight.")
    frequencies = torch.arange(1, channels // 8 + 1, device=device, dtype=torch.float32)[:, None]

    def encode(axis_length):
        axis = (torch.arange(axis_length, device=device, dtype=torch.float32) + 0.5) / axis_length
        phase = 2 * math.pi * frequencies * axis
        return torch.cat((phase.sin(), phase.cos()), dim=0)

    y = encode(height)[:, :, None].expand(-1, -1, width)
    x = encode(width)[:, None, :].expand(-1, height, -1)
    return torch.cat((y, x), dim=0)


class PositionalSSM(nn.Module):
    def __init__(self, core, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown SSM positional mode: {mode}")
        self.core = core
        self.mode = mode

    def position_code(self, joint):
        _, channels, height, width = joint.shape
        optical = coordinates(height, width, channels, joint.device)
        sar = optical
        if self.mode == "sar_shuffled":
            rng = torch.Generator().manual_seed(SHUFFLE_SEED)
            permutation = torch.randperm(height * width, generator=rng).to(joint.device)
            sar = optical.flatten(1).index_select(1, permutation).reshape_as(optical)
        return torch.cat((optical, sar), dim=0).unsqueeze(0).to(joint.dtype)

    def forward(self, joint):
        return self.core(joint + POSITION_SCALE * self.position_code(joint))


class PGMambaE2(SignedPhaseGuideE2):
    def __init__(self, mode):
        super().__init__("feature")
        self.variant = VARIANT_PREFIX + mode
        # Wrap only the scan: keep the original joint inputs, gate and residual.
        self.detail_fusion.ssm = PositionalSSM(self.detail_fusion.ssm, mode)
