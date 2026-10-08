"""C1: frozen C0 and zero-initialized per-channel SAR adaptation checks."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from net import DetailFeatureExtraction
from reliability_modules import ModalityDifferenceAdapter
from train_OS import (
    build_models,
    configure_trainability,
    forward_loss,
)


def test_c1_adapter_initializes_to_exact_identity():
    torch.manual_seed(42)
    adapter = ModalityDifferenceAdapter(channels=64, hidden_channels=32)
    optical = torch.randn(2, 64, 12, 10)
    sar = torch.randn(2, 64, 12, 10)
    adapted, diagnostic = adapter(optical, sar)
    assert torch.equal(adapted, sar)
    assert diagnostic["c1_delta_abs_mean"].item() == 0.0
    assert diagnostic["c1_delta_rms"].item() == 0.0


def test_c1_fused_detail_exact_c0_at_initialization():
    torch.manual_seed(13)
    optical = torch.randn(1, 64, 12, 10)
    sar = torch.randn(1, 64, 12, 10)
    detail_fuse = DetailFeatureExtraction(num_layers=1).eval()
    adapter = ModalityDifferenceAdapter().eval()
    with torch.no_grad():
        c0 = detail_fuse(optical + sar)
        adapted, _ = adapter(optical, sar)
        c1 = detail_fuse(optical + adapted)
    assert torch.equal(c0, c1)


def test_c1_feature_correction_is_bounded_by_sar_rms():
    adapter = ModalityDifferenceAdapter(alpha=0.10)
    optical = torch.randn(2, 64, 8, 8)
    sar = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        adapter.output.weight.fill_(0.2)
        adapter.output.bias.fill_(0.2)
        adapted, diagnostics = adapter(optical, sar)
    rms = sar.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    correction = (adapted - sar).abs()
    assert torch.all(correction <= 0.10 * rms + 2e-6)
    assert torch.isfinite(adapted).all()
    assert diagnostics["c1_delta_abs_mean"].item() > 0.0


def test_c1_adapter_only_freezes_original_c0_modules():
    for variant in ("c1_base_adapter", "c1_detail_adapter"):
        models = build_models(torch.device("cpu"), variant)
        configure_trainability(models, "adapter_only")
        for name in ("encoder", "decoder", "base_fuse", "detail_fuse"):
            assert all(
                not parameter.requires_grad
                for parameter in models[name].parameters()
            )
        assert all(
            parameter.requires_grad
            for parameter in models["c1_adapter"].parameters()
        )


def test_c1_adapter_head_receives_gradients():
    torch.manual_seed(31)
    adapter = ModalityDifferenceAdapter()
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn(1, 64, 8, 8)
    adapted, _ = adapter(optical, sar)
    adapted.square().mean().backward()
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.abs().sum().item() > 0.0
    assert adapter.output.bias.grad is not None


if __name__ == "__main__":
    test_c1_adapter_initializes_to_exact_identity()
    test_c1_fused_detail_exact_c0_at_initialization()
    test_c1_feature_correction_is_bounded_by_sar_rms()
    test_c1_adapter_only_freezes_original_c0_modules()
    test_c1_adapter_head_receives_gradients()
    print("C1 modality adapter unit checks passed.")
