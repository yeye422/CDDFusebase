"""Unit checks for the source-preserving CDDFuse direct-fusion path."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from direct_fusion import (
    CrossModalBaseFusion,
    DirectFusionCDDFuse,
    MultiScaleDetailFusion,
)
from train_direct_fusion import DirectFusionLosses, forward_losses
from test_OS import forward_variant, resolve_variant


def test_cross_modal_base_attention_shapes_and_normalization():
    torch.manual_seed(1)
    module = CrossModalBaseFusion(channels=64, num_heads=8)
    optical = torch.randn(2, 64, 11, 13)
    sar = torch.randn_like(optical)
    output, diagnostics = module(optical, sar, return_diagnostics=True)
    assert output.shape == optical.shape
    assert diagnostics["attention_os"].shape == (2, 8, 8, 8)
    assert diagnostics["attention_so"].shape == (2, 8, 8, 8)
    assert torch.allclose(
        diagnostics["attention_os"].sum(dim=-1),
        torch.ones(2, 8, 8),
        atol=1e-6,
    )
    assert torch.allclose(
        diagnostics["temperature_os"], torch.ones(8, 1, 1), atol=1e-6
    )


def test_detail_fusion_preserves_odd_spatial_shape():
    torch.manual_seed(2)
    module = MultiScaleDetailFusion()
    optical = torch.randn(1, 64, 15, 17)
    sar = torch.randn_like(optical)
    base = torch.randn_like(optical)
    assert module(optical, sar, base).shape == optical.shape


def test_r3_output_and_both_source_gradients():
    torch.manual_seed(3)
    model = DirectFusionCDDFuse("r3").eval()
    optical = torch.rand(1, 1, 16, 16, requires_grad=True)
    sar = torch.rand(1, 1, 16, 16, requires_grad=True)
    output, features = model(optical, sar, return_features=True)
    assert output.shape == optical.shape
    assert set(features) == {
        "optical_base",
        "optical_detail",
        "sar_base",
        "sar_detail",
        "fused_base",
        "fused_detail",
    }
    output.mean().backward()
    assert optical.grad is not None and optical.grad.abs().sum() > 0
    assert sar.grad is not None and sar.grad.abs().sum() > 0
    for name, module in (
        ("base", model.base_fusion),
        ("detail", model.detail_fusion),
        ("decoder", model.decoder),
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name


def test_only_r0_uses_decoder_image_skip():
    for variant in DirectFusionCDDFuse.VARIANTS:
        model = DirectFusionCDDFuse(variant)
        assert model.uses_image_skip == (variant == "r0")


def test_r2_and_r3_fusion_parameter_counts_are_matched():
    r2 = DirectFusionCDDFuse("r2")
    r3 = DirectFusionCDDFuse("r3")
    r2_count = sum(
        parameter.numel()
        for module in (r2.base_fusion, r2.detail_fusion)
        for parameter in module.parameters()
    )
    r3_count = sum(
        parameter.numel()
        for module in (r3.base_fusion, r3.detail_fusion)
        for parameter in module.parameters()
    )
    assert abs(r2_count - r3_count) / r3_count < 0.001


def test_backbone_loading_does_not_copy_fusion_layers():
    torch.manual_seed(4)
    source = DirectFusionCDDFuse("r0")
    checkpoint = {
        "CDDF_Encoder": source.encoder.state_dict(),
        "CDDF_Decoder": source.decoder.state_dict(),
    }
    target = DirectFusionCDDFuse("r3")
    before = target.base_fusion.compress.weight.detach().clone()
    target.load_backbone_checkpoint(checkpoint)
    assert torch.equal(
        target.encoder.patch_embed.proj.weight,
        source.encoder.patch_embed.proj.weight,
    )
    assert torch.equal(
        target.decoder.reduce_channel.weight,
        source.decoder.reduce_channel.weight,
    )
    assert torch.equal(target.base_fusion.compress.weight, before)


def test_t0_and_t1_gradient_contracts():
    torch.manual_seed(5)
    model = DirectFusionCDDFuse("r3")
    model.freeze_encoder()
    losses = DirectFusionLosses()
    args = SimpleNamespace(gradient_weight=10.0, reconstruction_weight=0.1)
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand_like(optical)
    valid = torch.ones_like(optical)

    t0, _ = forward_losses(model, losses, optical, sar, valid, "t0", args)
    t0.backward()
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert all(
        parameter.grad is None for parameter in model.base_fusion.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.detail_fusion.parameters()
    )

    model.zero_grad(set_to_none=True)
    t1, _ = forward_losses(model, losses, optical, sar, valid, "t1", args)
    t1.backward()
    for module in (model.decoder, model.base_fusion, model.detail_fusion):
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )


def test_existing_evaluator_dispatches_direct_checkpoint():
    model = DirectFusionCDDFuse("r3").eval()
    checkpoint = {"args": {"variant": "r3"}, "model": model.state_dict()}
    assert resolve_variant(checkpoint, "auto") == "r3"
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand_like(optical)
    with torch.no_grad():
        expected = model(optical, sar)
        actual, diagnostics = forward_variant(
            {"direct": model}, "r3", optical, sar
        )
    assert torch.equal(actual, expected)
    assert diagnostics == {}


def test_losses_accept_amp_style_mixed_dtypes():
    losses = DirectFusionLosses()
    source = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    reconstructed = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    valid = torch.ones_like(source)
    reconstruction = losses.reconstruction(source, reconstructed, valid)
    fusion, intensity, gradient = losses.fusion(
        source, source, reconstructed, valid, 10.0
    )
    for value in (reconstruction, fusion, intensity, gradient):
        assert value.dtype == torch.float32
        assert torch.isfinite(value)


if __name__ == "__main__":
    test_cross_modal_base_attention_shapes_and_normalization()
    test_detail_fusion_preserves_odd_spatial_shape()
    test_r3_output_and_both_source_gradients()
    test_only_r0_uses_decoder_image_skip()
    test_r2_and_r3_fusion_parameter_counts_are_matched()
    test_backbone_loading_does_not_copy_fusion_layers()
    test_t0_and_t1_gradient_contracts()
    test_existing_evaluator_dispatches_direct_checkpoint()
    test_losses_accept_amp_style_mixed_dtypes()
    print("Direct-fusion unit checks passed.")
