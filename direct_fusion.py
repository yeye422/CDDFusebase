"""Source-preserving direct-fusion variants for optical/SAR CDDFuse.

The historical CDDFuse path adds the two modality features before a learnable
fusion layer.  This module keeps the two sources separate until the fusion
operator and always makes the image-domain skip decision explicit.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    LayerNorm,
    Restormer_Decoder,
    Restormer_Encoder,
)
from lwpd import LearnableWaveletPriorDecomposition
from lwpd_ll_context import LLContextWaveletDecomposition


def _check_pair(left, right, name):
    if left.shape != right.shape:
        raise ValueError(
            f"{name} inputs must have identical shapes, got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.ndim != 4:
        raise ValueError(f"{name} expects NCHW tensors")


def _inverse_softplus(value):
    return math.log(math.expm1(float(value)))


class CrossModalBaseFusion(nn.Module):
    """Bidirectional Restormer-style channel interaction for two Base tensors."""

    def __init__(self, channels=64, num_heads=8, epsilon=1e-6, bias=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.epsilon = float(epsilon)

        self.optical_norm = LayerNorm(channels, "WithBias")
        self.sar_norm = LayerNorm(channels, "WithBias")
        self.optical_qk = self._qk_projection(channels, bias)
        self.sar_qk = self._qk_projection(channels, bias)
        self.optical_value = nn.Conv2d(channels, channels, 1, bias=bias)
        self.sar_value = nn.Conv2d(channels, channels, 1, bias=bias)

        initial_raw = _inverse_softplus(1.0 - self.epsilon)
        self.raw_temperature_os = nn.Parameter(
            torch.full((num_heads, 1, 1), initial_raw)
        )
        self.raw_temperature_so = nn.Parameter(
            torch.full((num_heads, 1, 1), initial_raw)
        )
        self.compress = nn.Conv2d(channels * 4, channels, 1, bias=bias)
        self.transform = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=bias
            ),
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.GELU(),
        )

    @staticmethod
    def _qk_projection(channels, bias):
        return nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=bias),
            nn.Conv2d(
                channels * 2,
                channels * 2,
                3,
                padding=1,
                groups=channels * 2,
                bias=bias,
            ),
        )

    def _heads(self, tensor):
        return rearrange(
            tensor,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )

    def _interaction(self, query, key, value, raw_temperature):
        query = F.normalize(self._heads(query), dim=-1, eps=self.epsilon)
        key = F.normalize(self._heads(key), dim=-1, eps=self.epsilon)
        value = self._heads(value)
        temperature = F.softplus(raw_temperature) + self.epsilon
        attention = (query @ key.transpose(-2, -1)) / temperature
        attention = attention.softmax(dim=-1)
        output = attention @ value
        return output, attention

    def forward(self, optical_base, sar_base, return_diagnostics=False):
        _check_pair(optical_base, sar_base, "Base fusion")
        if optical_base.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} Base channels, got "
                f"{optical_base.shape[1]}"
            )
        height, width = optical_base.shape[-2:]
        optical_q, optical_k = self.optical_qk(
            self.optical_norm(optical_base)
        ).chunk(2, dim=1)
        sar_q, sar_k = self.sar_qk(self.sar_norm(sar_base)).chunk(2, dim=1)
        optical_value = self.optical_value(optical_base)
        sar_value = self.sar_value(sar_base)

        optical_from_sar, attention_os = self._interaction(
            optical_q, sar_k, sar_value, self.raw_temperature_os
        )
        sar_from_optical, attention_so = self._interaction(
            sar_q, optical_k, optical_value, self.raw_temperature_so
        )
        optical_from_sar = rearrange(
            optical_from_sar,
            "b head c (h w) -> b (head c) h w",
            h=height,
            w=width,
        )
        sar_from_optical = rearrange(
            sar_from_optical,
            "b head c (h w) -> b (head c) h w",
            h=height,
            w=width,
        )
        fused = self.transform(
            self.compress(
                torch.cat(
                    (
                        optical_value,
                        sar_value,
                        optical_from_sar,
                        sar_from_optical,
                    ),
                    dim=1,
                )
            )
        )
        if not return_diagnostics:
            return fused
        return fused, {
            "attention_os": attention_os,
            "attention_so": attention_so,
            "temperature_os": F.softplus(self.raw_temperature_os) + self.epsilon,
            "temperature_so": F.softplus(self.raw_temperature_so) + self.epsilon,
        }


class MultiScaleDetailFusion(nn.Module):
    """Joint multi-scale Detail aggregation with fused Base context."""

    def __init__(self, channels=64, projected_channels=32, base_channels=16):
        super().__init__()
        self.channels = int(channels)
        self.optical_projection = nn.Sequential(
            nn.Conv2d(channels, projected_channels, 1), nn.GELU()
        )
        self.sar_projection = nn.Sequential(
            nn.Conv2d(channels, projected_channels, 1), nn.GELU()
        )
        joint_channels = projected_channels * 2
        self.scale_branches = nn.ModuleList(
            self._scale_branch(joint_channels, projected_channels, dilation)
            for dilation in (1, 2, 3)
        )
        self.base_projection = nn.Sequential(
            nn.Conv2d(channels, base_channels, 1), nn.GELU()
        )
        aggregate_channels = projected_channels * 5 + base_channels
        self.output = nn.Sequential(
            nn.Conv2d(aggregate_channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    @staticmethod
    def _scale_branch(input_channels, output_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=input_channels,
            ),
            nn.Conv2d(input_channels, output_channels, 1),
            nn.GELU(),
        )

    def _project_inputs(self, optical_detail, sar_detail, fused_base):
        _check_pair(optical_detail, sar_detail, "Detail fusion")
        if fused_base.shape != optical_detail.shape:
            raise ValueError(
                "Fused Base and Detail tensors must have identical shapes, got "
                f"{tuple(fused_base.shape)} and {tuple(optical_detail.shape)}"
            )
        if optical_detail.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} Detail channels, got "
                f"{optical_detail.shape[1]}"
            )
        optical = self.optical_projection(optical_detail)
        sar = self.sar_projection(sar_detail)
        return optical, sar, self.base_projection(fused_base)

    def forward(self, optical_detail, sar_detail, fused_base):
        optical, sar, base = self._project_inputs(optical_detail, sar_detail, fused_base)
        joint = torch.cat((optical, sar), dim=1)
        scales = [branch(joint) for branch in self.scale_branches]
        return self.output(torch.cat((optical, sar, *scales, base), dim=1))


class SourcePreservingConcatFusion(nn.Module):
    """Simple source-preserving R2 control used for both feature branches."""

    def __init__(self, channels=64, hidden_channels=128):
        super().__init__()
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, optical, sar):
        _check_pair(optical, sar, "Concat fusion")
        return self.network(torch.cat((optical, sar), dim=1))


class DirectFusionCDDFuse(nn.Module):
    """R0-R3 controls with a shared Encoder and explicit decoder skip policy."""

    LWPD_VARIANTS = {
        "lwpd_e1": "fixed",
        "lwpd_e2": "gated",
        "lwpd_e2_llctx_v1": "gated",
        "lwpd_e2_prl": "gated_prl",
        "lwpd_e3": "recovery",
        "lwpd_e4": "phase",
    }
    VARIANTS = ("r0", "r1", "r2", "r3", *LWPD_VARIANTS)

    def __init__(self, variant="r3", channels=64, num_heads=8):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown direct-fusion variant: {variant}")
        self.variant = variant
        self.encoder = Restormer_Encoder(dim=channels)
        self.decoder = Restormer_Decoder(dim=channels)
        self.lwpd = None
        if variant in ("r0", "r1"):
            self.base_fusion = BaseFeatureExtraction(channels, num_heads)
            self.detail_fusion = DetailFeatureExtraction(num_layers=1)
        elif variant == "r2":
            # 56/121 make the combined R2 fusion parameter count match R3 to
            # within 0.1% at 64 channels; they are capacity controls, not
            # architectural recommendations.
            self.base_fusion = SourcePreservingConcatFusion(
                channels, hidden_channels=56
            )
            self.detail_fusion = SourcePreservingConcatFusion(
                channels, hidden_channels=121
            )
        else:
            self.base_fusion = CrossModalBaseFusion(channels, num_heads)
            self.detail_fusion = MultiScaleDetailFusion(channels)
            if variant in self.LWPD_VARIANTS:
                decomposition = LLContextWaveletDecomposition if variant == "lwpd_e2_llctx_v1" else LearnableWaveletPriorDecomposition
                self.lwpd = decomposition(
                    channels=channels,
                    mode=self.LWPD_VARIANTS[variant],
                )

    @property
    def uses_image_skip(self):
        return self.variant == "r0"

    def load_e2_llctx_checkpoint(self, checkpoint):
        if self.variant!='lwpd_e2_llctx_v1' or checkpoint.get('variant')!='lwpd_e2':
            raise ValueError('LL-context initialization requires an original E2 checkpoint')
        state={k[7:] if k.startswith('module.') else k:v for k,v in checkpoint['model'].items()}
        missing,unexpected=self.load_state_dict(state,strict=False)
        expected={k for k in self.state_dict() if k.startswith('lwpd.ll_context.')}
        if set(missing)!=expected or unexpected:
            raise RuntimeError(f'Incomplete E2 initialization: missing={missing}, unexpected={unexpected}')

    def encode_pair(self, optical, sar, return_diagnostics=False):
        diagnostics = {}
        if self.lwpd is not None:
            # LWPD replaces the legacy Base/Detail heads, so compute only the
            # shared encoder representation instead of wasting memory on two
            # feature branches whose outputs would immediately be discarded.
            optical_shared = self.encoder.encoder_level1(
                self.encoder.patch_embed(optical)
            )
            sar_shared = self.encoder.encoder_level1(self.encoder.patch_embed(sar))
            decomposed = self.lwpd(
                optical_shared,
                sar_shared,
                sar_image=sar,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                decomposed, diagnostics = decomposed
            optical_base, optical_detail, sar_base, sar_detail = decomposed
        else:
            optical_base, optical_detail, _ = self.encoder(optical)
            sar_base, sar_detail, _ = self.encoder(sar)
        encoded = (optical_base, optical_detail, sar_base, sar_detail)
        if return_diagnostics:
            return encoded, diagnostics
        return encoded

    def fuse_features(
        self, optical_base, optical_detail, sar_base, sar_detail
    ):
        if self.variant in ("r0", "r1"):
            fused_base = self.base_fusion(optical_base + sar_base)
            fused_detail = self.detail_fusion(optical_detail + sar_detail)
        elif self.variant == "r2":
            fused_base = self.base_fusion(optical_base, sar_base)
            fused_detail = self.detail_fusion(optical_detail, sar_detail)
        else:
            fused_base = self.base_fusion(optical_base, sar_base)
            fused_detail = self.detail_fusion(
                optical_detail, sar_detail, fused_base
            )
        return fused_base, fused_detail

    def reconstruct(self, source, base, detail):
        decoder_input = source if self.uses_image_skip else None
        return self.decoder(decoder_input, base, detail)[0]

    def prepare_fusion_inputs(self, encoded, sar):
        """Allow fusion-only conditioning after the source reconstruction split."""
        return encoded, {}

    def forward(self, optical, sar, return_features=False):
        if return_features:
            encoded, lwpd_diagnostics = self.encode_pair(
                optical, sar, return_diagnostics=True
            )
        else:
            encoded = self.encode_pair(optical, sar)
        fusion_inputs, fusion_diagnostics = self.prepare_fusion_inputs(encoded, sar)
        fused_base, fused_detail = self.fuse_features(*fusion_inputs)
        decoder_input = optical if self.uses_image_skip else None
        fused, _ = self.decoder(decoder_input, fused_base, fused_detail)
        if not return_features:
            return fused
        optical_base, optical_detail, sar_base, sar_detail = encoded
        features = {
            "optical_base": optical_base,
            "optical_detail": optical_detail,
            "sar_base": sar_base,
            "sar_detail": sar_detail,
            "fused_base": fused_base,
            "fused_detail": fused_detail,
        }
        if self.lwpd is not None:
            features["lwpd"] = lwpd_diagnostics
        features.update(fusion_diagnostics)
        return fused, features

    def load_backbone_checkpoint(self, checkpoint):
        encoder = checkpoint.get("CDDF_Encoder", checkpoint.get("DIDF_Encoder"))
        decoder = checkpoint.get("CDDF_Decoder", checkpoint.get("DIDF_Decoder"))
        if encoder is None or decoder is None:
            raise KeyError("Checkpoint must contain CDDF/DIDF Encoder and Decoder")
        def canonical_state(state):
            return {
                key[len("module.") :] if key.startswith("module.") else key: value
                for key, value in state.items()
            }
        self.encoder.load_state_dict(
            canonical_state(encoder)
        )
        self.decoder.load_state_dict(
            canonical_state(decoder)
        )

    def freeze_encoder(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
