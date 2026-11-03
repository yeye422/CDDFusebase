"""Unit checks for the LWPD ablation family and model integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from direct_fusion import DirectFusionCDDFuse
from lwpd import (
    HaarDWT2D,
    LearnableLiftingDWT2D,
    LearnableWaveletPriorDecomposition,
)
from train_lwpd import configure_stage, make_optimizer
from test_OS import resolve_variant


def test_haar_round_trip_even_and_odd_shapes():
    torch.manual_seed(11)
    haar = HaarDWT2D()
    for shape in ((2, 3, 16, 18), (1, 2, 15, 17)):
        source = torch.randn(*shape)
        bands, output_size = haar(source)
        restored = haar.inverse(bands, output_size)
        assert restored.shape == source.shape
        assert torch.allclose(restored, source, atol=1e-6)


def test_fixed_lwpd_is_an_exact_base_detail_partition():
    torch.manual_seed(12)
    module = LearnableWaveletPriorDecomposition(channels=8, mode="fixed")
    optical = torch.randn(1, 8, 15, 17)
    sar = torch.randn_like(optical)
    optical_base, optical_detail, sar_base, sar_detail = module(optical, sar)
    assert torch.allclose(optical_base + optical_detail, optical, atol=1e-6)
    assert torch.allclose(sar_base + sar_detail, sar, atol=1e-6)


def test_lifting_starts_as_haar_and_remains_perfectly_reconstructing():
    torch.manual_seed(121)
    haar = HaarDWT2D()
    lifting = LearnableLiftingDWT2D()
    for shape in ((2, 3, 16, 18), (1, 2, 15, 17)):
        source = torch.randn(*shape)
        haar_bands, haar_size = haar(source)
        lifting_bands, lifting_size = lifting(source)
        assert lifting_size == haar_size
        for expected, actual in zip(haar_bands, lifting_bands):
            assert torch.allclose(actual, expected, atol=1e-6)
        restored = lifting.inverse(lifting_bands, lifting_size)
        assert torch.allclose(restored, source, atol=1e-6)

    with torch.no_grad():
        for parameter in lifting.parameters():
            parameter.uniform_(-0.4, 0.4)
    source = torch.randn(1, 4, 15, 17)
    bands, output_size = lifting(source)
    restored = lifting.inverse(bands, output_size)
    assert torch.allclose(restored, source, atol=2e-6)
    kernels = lifting.kernels()
    assert torch.allclose(kernels["predict_horizontal"].sum(), torch.tensor(1.0))
    assert torch.allclose(kernels["predict_vertical"].sum(), torch.tensor(1.0))
    assert torch.allclose(kernels["update_horizontal"].sum(), torch.tensor(0.5))
    assert torch.allclose(kernels["update_vertical"].sum(), torch.tensor(0.5))


def test_prl_lwpd_starts_from_exact_e2_output_and_receives_gradients():
    torch.manual_seed(122)
    e2 = LearnableWaveletPriorDecomposition(
        channels=8,
        hidden_channels=4,
        mode="gated",
    )
    prl = LearnableWaveletPriorDecomposition(
        channels=8,
        hidden_channels=4,
        mode="gated_prl",
    )
    prl_state = prl.state_dict()
    for name, value in e2.state_dict().items():
        if name in prl_state and prl_state[name].shape == value.shape:
            prl_state[name] = value
    prl.load_state_dict(prl_state)
    optical = torch.randn(1, 8, 15, 17)
    sar = torch.randn_like(optical)
    expected = e2(optical, sar)
    actual = prl(optical, sar)
    for expected_feature, actual_feature in zip(expected, actual):
        assert torch.allclose(actual_feature, expected_feature, atol=2e-6)

    loss = sum(feature.square().mean() for feature in actual)
    loss.backward()
    lifting_gradients = [
        parameter.grad for parameter in prl.haar.parameters()
    ]
    assert all(gradient is not None for gradient in lifting_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in lifting_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in lifting_gradients)


def test_lwpd_modes_preserve_shapes_and_coefficient_bounds():
    torch.manual_seed(13)
    optical = torch.randn(1, 8, 16, 16)
    sar = torch.randn_like(optical)
    sar_image = torch.rand(1, 1, 16, 16)
    for mode in LearnableWaveletPriorDecomposition.MODES:
        module = LearnableWaveletPriorDecomposition(
            channels=8,
            hidden_channels=4,
            mode=mode,
        )
        outputs, diagnostics = module(
            optical,
            sar,
            sar_image=sar_image,
            return_diagnostics=True,
        )
        assert all(output.shape == optical.shape for output in outputs)
        for name in ("optical_gates", "sar_gates", "sar_coefficients"):
            values = diagnostics[name]
            assert torch.isfinite(values).all(), (mode, name)
            assert values.min() >= 0.0, (mode, name)
            assert values.max() <= 1.0, (mode, name)
        if mode == "phase":
            phase = diagnostics["phase_symmetry"]
            assert phase.shape == (1, 4, 8, 8)
            assert phase.min() >= 0.0 and phase.max() <= 1.0
        else:
            assert diagnostics["phase_symmetry"] is None


def test_gated_lwpd_receives_gradients():
    torch.manual_seed(14)
    module = LearnableWaveletPriorDecomposition(
        channels=8,
        hidden_channels=4,
        mode="recovery",
    )
    optical = torch.randn(1, 8, 16, 16)
    sar = torch.randn_like(optical)
    loss = sum(output.square().mean() for output in module(optical, sar))
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_lwpd_e4_integrates_with_direct_fusion_model():
    torch.manual_seed(15)
    model = DirectFusionCDDFuse("lwpd_e4").eval()
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand_like(optical)
    output, features = model(optical, sar, return_features=True)
    assert output.shape == optical.shape
    assert features["optical_base"].shape == (1, 64, 16, 16)
    assert features["sar_detail"].shape == (1, 64, 16, 16)
    assert features["lwpd"]["phase_symmetry"].shape == (1, 4, 8, 8)


def test_lwpd_e2_prl_integrates_with_direct_fusion_model():
    torch.manual_seed(151)
    model = DirectFusionCDDFuse("lwpd_e2_prl").eval()
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand_like(optical)
    output, features = model(optical, sar, return_features=True)
    assert output.shape == optical.shape
    assert features["optical_base"].shape == (1, 64, 16, 16)
    assert features["sar_detail"].shape == (1, 64, 16, 16)
    assert features["lwpd"]["phase_symmetry"] is None
    assert features["lwpd"]["wavelet_deviation"].item() == 0.0


def test_progressive_unfreezing_contract():
    model = DirectFusionCDDFuse("lwpd_e2_prl")
    args = SimpleNamespace(
        decoder_lr=2e-5,
        warmup_fusion_lr=5e-5,
        local_fusion_lr=1e-4,
        joint_fusion_lr=5e-5,
        warmup_lwpd_lr=1e-4,
        warmup_wavelet_lr=1e-5,
        local_lwpd_lr=1e-4,
        local_wavelet_lr=1e-5,
        joint_lwpd_lr=5e-5,
        joint_wavelet_lr=5e-6,
        encoder_tail_lr=1e-5,
        encoder_early_lr=2e-6,
        weight_decay=1e-4,
    )
    optimizer = make_optimizer(model, args)
    wavelet_groups = [
        group for group in optimizer.param_groups if group["group_name"] == "wavelet"
    ]
    assert len(wavelet_groups) == 1
    assert len(wavelet_groups[0]["params"]) == 4
    configure_stage(model, optimizer, "warmup", args)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.lwpd.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert wavelet_groups[0]["lr"] == args.warmup_wavelet_lr

    configure_stage(model, optimizer, "local", args)
    assert any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(
        group["lr"] == args.encoder_tail_lr
        for group in optimizer.param_groups
        if group["group_name"] == "encoder_tail"
    )

    configure_stage(model, optimizer, "joint", args)
    trainable_encoder = [
        parameter.requires_grad for parameter in model.encoder.parameters()
    ]
    assert any(trainable_encoder)
    assert not all(trainable_encoder)  # Unused legacy Base/Detail heads stay frozen.
    assert any(
        group["lr"] == args.encoder_early_lr
        for group in optimizer.param_groups
        if group["group_name"] == "encoder_early"
    )


def test_evaluator_detects_lwpd_checkpoint():
    for variant in DirectFusionCDDFuse.LWPD_VARIANTS:
        checkpoint = {"args": {"variant": variant}}
        assert resolve_variant(checkpoint, "auto") == variant


if __name__ == "__main__":
    test_haar_round_trip_even_and_odd_shapes()
    test_fixed_lwpd_is_an_exact_base_detail_partition()
    test_lifting_starts_as_haar_and_remains_perfectly_reconstructing()
    test_prl_lwpd_starts_from_exact_e2_output_and_receives_gradients()
    test_lwpd_modes_preserve_shapes_and_coefficient_bounds()
    test_gated_lwpd_receives_gradients()
    test_lwpd_e4_integrates_with_direct_fusion_model()
    test_lwpd_e2_prl_integrates_with_direct_fusion_model()
    test_progressive_unfreezing_contract()
    test_evaluator_detects_lwpd_checkpoint()
    print("LWPD unit checks passed.")
