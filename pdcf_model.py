"""C0-compatible PDCF model wrapper and checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from pdcf_modules import PDCFCore


PathLike = Union[str, Path]


def _canonical_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _checkpoint_state(checkpoint: dict, key: str, legacy_key: Optional[str] = None):
    state = checkpoint.get(key)
    if state is None and legacy_key is not None:
        state = checkpoint.get(legacy_key)
    if state is None:
        raise KeyError(f"Checkpoint is missing {key}/{legacy_key}")
    return _canonical_state(state)


class PDCFModel(nn.Module):
    """Wrap the frozen C0 path and add zero-initialized PDCF corrections."""

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        base_fuse: Optional[nn.Module] = None,
        detail_fuse: Optional[nn.Module] = None,
        core: Optional[PDCFCore] = None,
        freeze_c0: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else Restormer_Encoder()
        self.decoder = decoder if decoder is not None else Restormer_Decoder()
        self.base_fuse = (
            base_fuse
            if base_fuse is not None
            else BaseFeatureExtraction(dim=64, num_heads=8)
        )
        self.detail_fuse = (
            detail_fuse
            if detail_fuse is not None
            else DetailFeatureExtraction(num_layers=1)
        )
        self.core = core if core is not None else PDCFCore()
        if freeze_c0:
            self.freeze_c0()

    def freeze_c0(self) -> None:
        for module in (
            self.encoder,
            self.decoder,
            self.base_fuse,
            self.detail_fuse,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen C0 modules must not silently switch to training behavior.
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
            self.decoder.eval()
            self.base_fuse.eval()
            self.detail_fuse.eval()
        return self

    def extract_features(self, optical: torch.Tensor, sar: torch.Tensor):
        optical_base, optical_detail, _ = self.encoder(optical)
        sar_base, sar_detail, _ = self.encoder(sar)
        return optical_base, optical_detail, sar_base, sar_detail

    def c0_features(
        self,
        optical_base: torch.Tensor,
        optical_detail: torch.Tensor,
        sar_base: torch.Tensor,
        sar_detail: torch.Tensor,
    ):
        fused_base = self.base_fuse(optical_base + sar_base)
        fused_detail = self.detail_fuse(optical_detail + sar_detail)
        return fused_base, fused_detail

    def forward_c0(self, optical: torch.Tensor, sar: torch.Tensor):
        features = self.extract_features(optical, sar)
        fused_base, fused_detail = self.c0_features(*features)
        output, decoder_feature = self.decoder(optical, fused_base, fused_detail)
        return output, decoder_feature

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        spd_condition: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        features = self.extract_features(optical, sar)
        fused_base_c0, fused_detail_c0 = self.c0_features(*features)
        core_result = self.core(
            *features,
            spd_condition=spd_condition,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            delta_base, delta_detail, diagnostics = core_result
        else:
            delta_base, delta_detail = core_result
        fused_base = fused_base_c0 + delta_base
        fused_detail = fused_detail_c0 + delta_detail
        output, decoder_feature = self.decoder(optical, fused_base, fused_detail)
        if not return_diagnostics:
            return output, decoder_feature
        diagnostics.update(
            {
                "c0_base": fused_base_c0,
                "c0_detail": fused_detail_c0,
                "fused_base": fused_base,
                "fused_detail": fused_detail,
            }
        )
        return output, decoder_feature, diagnostics

    @classmethod
    def from_c0_checkpoint(
        cls,
        checkpoint_path: PathLike,
        map_location: Union[str, torch.device] = "cpu",
        strict_variant: bool = True,
        **model_kwargs,
    ) -> "PDCFModel":
        # weights_only=False is intentional: legacy checkpoints contain Path objects.
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=map_location, weights_only=False
        )
        variant = checkpoint.get("args", {}).get("variant")
        if strict_variant and variant != "frequency_minimax_baseline":
            raise ValueError(
                "PDCF must initialize from frequency_minimax_baseline, "
                f"got {variant!r}"
            )
        model = cls(**model_kwargs)
        model.encoder.load_state_dict(
            _checkpoint_state(checkpoint, "CDDF_Encoder", "DIDF_Encoder")
        )
        model.decoder.load_state_dict(
            _checkpoint_state(checkpoint, "CDDF_Decoder", "DIDF_Decoder")
        )
        model.base_fuse.load_state_dict(
            _checkpoint_state(checkpoint, "BaseFuseLayer")
        )
        model.detail_fuse.load_state_dict(
            _checkpoint_state(checkpoint, "DetailFuseLayer")
        )
        return model

