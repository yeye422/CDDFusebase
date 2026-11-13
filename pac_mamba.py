"""Phase-antisymmetry-conditioned selective SSM fusion on top of LWPD.

The implementation is self-contained PyTorch.  It uses input-dependent
selective recurrent state updates along both image axes, avoiding the compiled
``mamba_ssm`` dependency while retaining linear sequence complexity and
bidirectional long-range modelling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from direct_fusion import (
    CrossModalBaseFusion,
    MultiScaleDetailFusion,
)
from lwpd import DirectionalPhaseSymmetry, LearnableWaveletPriorDecomposition
from net import Restormer_Decoder, Restormer_Encoder


def _canonical_state(state):
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


class SelectiveScan1D(nn.Module):
    """Pure-PyTorch Mamba selective SSM for BxLxC sequences.

    The discretization step and the input/output state projections are all
    token-dependent.  State transitions use a learned negative diagonal A,
    matching the core selective-SSM mechanism without a fused CUDA kernel.
    """

    def __init__(self, channels, expansion=1, state_channels=4):
        super().__init__()
        inner = int(channels * expansion)
        self.channels = int(channels)
        self.inner = inner
        self.state_channels = int(state_channels)
        self.norm = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, inner * 2)
        self.parameter_projection = nn.Linear(
            inner, inner + state_channels * 2
        )
        base = torch.arange(1, state_channels + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(base.log().repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.delta_bias = nn.Parameter(torch.full((inner,), -3.0))
        self.output_projection = nn.Linear(inner, channels)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.channels:
            raise ValueError(
                f"SelectiveScan1D expects BxLx{self.channels} sequences"
            )
        projected = self.input_projection(self.norm(sequence))
        value, output_gate = projected.chunk(2, dim=-1)
        parameters = self.parameter_projection(value)
        delta_logits = parameters[..., : self.inner]
        state_input = parameters[
            ..., self.inner : self.inner + self.state_channels
        ]
        state_output = parameters[..., self.inner + self.state_channels :]
        delta = F.softplus(delta_logits + self.delta_bias).clamp(max=1.0)
        transition = -torch.exp(self.a_log)
        state = value.new_zeros(
            value.shape[0], self.inner, self.state_channels
        )
        outputs = []
        for index in range(sequence.shape[1]):
            current_delta = delta[:, index]
            discrete_a = torch.exp(
                current_delta.unsqueeze(-1) * transition.unsqueeze(0)
            )
            drive = (
                current_delta.unsqueeze(-1)
                * state_input[:, index].unsqueeze(1)
                * value[:, index].unsqueeze(-1)
            )
            state = discrete_a * state + drive
            output = (
                state * state_output[:, index].unsqueeze(1)
            ).sum(dim=-1)
            output = output + self.skip * value[:, index]
            outputs.append(output)
        output = torch.stack(outputs, dim=1)
        output = output * F.silu(output_gate)
        return self.output_projection(output)


class BiAxialSelectiveStateSpace(nn.Module):
    """Row/column forward-backward selective scans with local 2D context."""

    def __init__(self, channels, expansion=1, state_channels=4):
        super().__init__()
        self.channels = int(channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.row_scan = SelectiveScan1D(channels, expansion, state_channels)
        self.column_scan = SelectiveScan1D(channels, expansion, state_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    @staticmethod
    def _bidirectional(scan, sequence):
        forward = scan(sequence)
        backward = torch.flip(scan(torch.flip(sequence, dims=(1,))), dims=(1,))
        return 0.5 * (forward + backward)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f"BiAxialSelectiveStateSpace expects NCHW with {self.channels} channels"
            )
        batch, channels, height, width = features.shape
        local = self.local(features)
        row_sequence = features.permute(0, 2, 3, 1).reshape(
            batch * height, width, channels
        )
        row = self._bidirectional(self.row_scan, row_sequence)
        row = row.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        column_sequence = features.permute(0, 3, 2, 1).reshape(
            batch * width, height, channels
        )
        column = self._bidirectional(self.column_scan, column_sequence)
        column = column.reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        return features + self.merge(torch.cat((local, row, column), dim=1))


class PACMambaDetailFusion(nn.Module):
    """Keep the trained LWPD fusion and add a PA-gated SSM update."""

    def __init__(
        self,
        channels=64,
        phase_orientations=4,
        hidden_channels=32,
        initial_update_scale=0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.phase_orientations = int(phase_orientations)
        self.keep_fusion = MultiScaleDetailFusion(channels)
        reliability_inputs = channels * 2 + phase_orientations * 2
        self.reliability = nn.Sequential(
            nn.Conv2d(reliability_inputs, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.joint_projection = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
        )
        self.ssm = BiAxialSelectiveStateSpace(channels)
        self.update_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.update_projection[-1].weight)
        nn.init.zeros_(self.update_projection[-1].bias)
        initial_logit = math.log(
            float(initial_update_scale) / (1.0 - float(initial_update_scale))
        )
        self.update_logit = nn.Parameter(torch.tensor(initial_logit))

    @staticmethod
    def phase_match(optical_pa, sar_pa, neighborhood=3):
        if optical_pa.shape != sar_pa.shape:
            raise ValueError("Optical and SAR PA maps must have identical shapes")
        if neighborhood < 1 or neighborhood % 2 == 0:
            raise ValueError("neighborhood must be a positive odd integer")
        padding = neighborhood // 2
        optical_support = F.max_pool2d(
            optical_pa, neighborhood, stride=1, padding=padding
        )
        sar_support = F.max_pool2d(sar_pa, neighborhood, stride=1, padding=padding)
        optical_to_sar = torch.sqrt((optical_pa * sar_support).clamp_min(0.0))
        sar_to_optical = torch.sqrt((sar_pa * optical_support).clamp_min(0.0))
        directional = 0.5 * (optical_to_sar + sar_to_optical)
        return directional.max(dim=1, keepdim=True).values.clamp(0.0, 1.0)

    def forward(
        self,
        optical_detail,
        sar_detail,
        fused_base,
        optical_pa,
        sar_pa,
        return_diagnostics=False,
    ):
        if optical_detail.shape != sar_detail.shape:
            raise ValueError("PAC detail inputs must have identical shapes")
        keep = self.keep_fusion(optical_detail, sar_detail, fused_base)
        phase_match = self.phase_match(optical_pa, sar_pa)
        phase_features = torch.cat((optical_pa, sar_pa), dim=1)
        reliability = torch.sigmoid(
            self.reliability(
                torch.cat((optical_detail, sar_detail, phase_features), dim=1)
            )
        )
        interaction = (phase_match * reliability).clamp(0.0, 1.0)

        optical_low = F.avg_pool2d(optical_detail, 2, stride=2)
        sar_low = F.avg_pool2d(sar_detail, 2, stride=2)
        joint = self.joint_projection(
            torch.cat((optical_low, sar_low, (optical_low - sar_low).abs()), dim=1)
        )
        contextual = self.ssm(joint)
        update = self.update_projection(contextual)
        interaction_low = F.interpolate(
            interaction,
            size=update.shape[-2:],
            mode="area",
        )
        gated_update = interaction_low * update
        gated_update = F.interpolate(
            gated_update,
            size=keep.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        update_scale = torch.sigmoid(self.update_logit)
        fused = keep + update_scale * gated_update
        if not return_diagnostics:
            return fused
        return fused, {
            "phase_match": phase_match,
            "reliability": reliability,
            "interaction": interaction,
            "update": gated_update,
            "update_scale": update_scale,
            "keep": keep,
        }


class PACMambaCDDFuse(nn.Module):
    """LWPD E4 backbone plus PA-conditioned axial selective SSM update."""

    VARIANT = "pac_mamba"
    BACKBONE_VARIANT = "lwpd_e4"
    LWPD_MODE = "phase"

    def __init__(self, channels=64, num_heads=8, phase_orientations=4):
        super().__init__()
        self.encoder = Restormer_Encoder(dim=channels)
        self.decoder = Restormer_Decoder(dim=channels)
        self.lwpd = LearnableWaveletPriorDecomposition(
            channels=channels,
            mode=self.LWPD_MODE,
            phase_orientations=phase_orientations,
        )
        self.base_fusion = CrossModalBaseFusion(channels, num_heads)
        self.detail_fusion = PACMambaDetailFusion(
            channels=channels,
            phase_orientations=phase_orientations,
        )
        self.phase_evidence = DirectionalPhaseSymmetry(
            orientations=phase_orientations
        )

    def compute_phase_evidence(self, optical, sar):
        optical_phase = self.phase_evidence(optical, return_both=True)
        sar_phase = self.phase_evidence(sar, return_both=True)
        return {
            "optical_pa": optical_phase["antisymmetry"],
            "sar_pa": sar_phase["antisymmetry"],
            "sar_ps": sar_phase["symmetry"],
        }

    def encode_pair(self, optical, sar, evidence=None, return_diagnostics=False):
        if evidence is None:
            evidence = self.compute_phase_evidence(optical, sar)
        optical_shared = self.encoder.encoder_level1(self.encoder.patch_embed(optical))
        sar_shared = self.encoder.encoder_level1(self.encoder.patch_embed(sar))
        decomposed = self.lwpd(
            optical_shared,
            sar_shared,
            sar_image=sar,
            phase_symmetry=evidence["sar_ps"],
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            decomposed, diagnostics = decomposed
            return decomposed, evidence, diagnostics
        return decomposed

    def reconstruct(self, source, base, detail):
        return self.decoder(None, base, detail)[0]

    def forward(self, optical, sar, return_features=False):
        evidence = self.compute_phase_evidence(optical, sar)
        if return_features:
            encoded, evidence, lwpd_diagnostics = self.encode_pair(
                optical,
                sar,
                evidence=evidence,
                return_diagnostics=True,
            )
        else:
            encoded = self.encode_pair(optical, sar, evidence=evidence)
        optical_base, optical_detail, sar_base, sar_detail = encoded
        fused_base = self.base_fusion(optical_base, sar_base)
        detail = self.detail_fusion(
            optical_detail,
            sar_detail,
            fused_base,
            evidence["optical_pa"],
            evidence["sar_pa"],
            return_diagnostics=return_features,
        )
        if return_features:
            fused_detail, pac_diagnostics = detail
        else:
            fused_detail = detail
        fused = self.decoder(None, fused_base, fused_detail)[0]
        if not return_features:
            return fused
        return fused, {
            "optical_base": optical_base,
            "optical_detail": optical_detail,
            "sar_base": sar_base,
            "sar_detail": sar_detail,
            "fused_base": fused_base,
            "fused_detail": fused_detail,
            "phase": evidence,
            "lwpd": lwpd_diagnostics,
            "pac": pac_diagnostics,
        }

    def load_lwpd_checkpoint(self, checkpoint):
        variant = checkpoint.get("variant", checkpoint.get("args", {}).get("variant"))
        if variant != self.BACKBONE_VARIANT:
            raise ValueError(f"{self.VARIANT} requires {self.BACKBONE_VARIANT}, got {variant!r}")
        self.encoder.load_state_dict(_canonical_state(checkpoint["CDDF_Encoder"]))
        self.decoder.load_state_dict(_canonical_state(checkpoint["CDDF_Decoder"]))
        self.lwpd.load_state_dict(_canonical_state(checkpoint["LWPD"]))
        self.base_fusion.load_state_dict(
            _canonical_state(checkpoint["DirectBaseFusion"])
        )
        self.detail_fusion.keep_fusion.load_state_dict(
            _canonical_state(checkpoint["DirectDetailFusion"])
        )


class PACMambaE2CDDFuse(PACMambaCDDFuse):
    """E2 gated Haar backbone with the same zero-initialized PAC residual.

    PA conditions the additive interaction only. No E3 SAR recovery or E4 PS
    recovery module is introduced into the underlying E2 decomposition.
    """

    VARIANT = "pac_mamba_e2"
    BACKBONE_VARIANT = "lwpd_e2"
    LWPD_MODE = "gated"


PAC_MODEL_TYPES = {cls.VARIANT: cls for cls in (PACMambaCDDFuse, PACMambaE2CDDFuse)}
