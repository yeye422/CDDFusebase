"""Unit and integration checks for the PAC-Mamba module."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from direct_fusion import DirectFusionCDDFuse
from lwpd import DirectionalPhaseSymmetry
from pac_mamba import (
    BiAxialSelectiveStateSpace,
    PACMambaCDDFuse,
    PACMambaDetailFusion,
)
from test_OS import resolve_variant
from train_pac_mamba import configure_stage, make_optimizer


def make_lwpd_checkpoint(model):
    return {
        "variant": "lwpd_e4",
        "args": {"variant": "lwpd_e4"},
        "CDDF_Encoder": model.encoder.state_dict(),
        "CDDF_Decoder": model.decoder.state_dict(),
        "DirectBaseFusion": model.base_fusion.state_dict(),
        "DirectDetailFusion": model.detail_fusion.state_dict(),
        "LWPD": model.lwpd.state_dict(),
    }


def test_directional_phase_returns_symmetry_and_antisymmetry():
    module = DirectionalPhaseSymmetry(
        wavelengths=(2.0, 4.0), orientations=4, reflect_padding=4
    )
    image = torch.zeros(1, 1, 16, 16)
    image[..., 8:] = 1.0
    evidence = module(image, return_both=True)
    assert set(evidence) == {"symmetry", "antisymmetry"}
    assert evidence["symmetry"].shape == (1, 4, 16, 16)
    assert evidence["antisymmetry"].shape == (1, 4, 16, 16)
    for value in evidence.values():
        assert torch.isfinite(value).all()
        assert value.min() >= 0.0 and value.max() <= 1.0


def test_phase_match_tolerates_one_pixel_shift():
    optical = torch.zeros(1, 4, 9, 9)
    sar = torch.zeros_like(optical)
    optical[:, 1, 4, 4] = 1.0
    sar[:, 1, 4, 5] = 1.0
    matched = PACMambaDetailFusion.phase_match(optical, sar, neighborhood=3)
    unmatched = PACMambaDetailFusion.phase_match(optical, sar, neighborhood=1)
    assert matched[0, 0, 4, 4] > 0.0
    assert matched.sum() > unmatched.sum()


def test_biaxial_ssm_preserves_shape_and_backpropagates():
    torch.manual_seed(21)
    module = BiAxialSelectiveStateSpace(channels=8, expansion=1)
    features = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_zero_initialized_pac_update_preserves_keep_path():
    torch.manual_seed(22)
    module = PACMambaDetailFusion(channels=8, hidden_channels=4)
    optical = torch.randn(1, 8, 16, 16)
    sar = torch.randn_like(optical)
    base = torch.randn_like(optical)
    optical_pa = torch.rand(1, 4, 16, 16)
    sar_pa = torch.rand_like(optical_pa)
    fused, diagnostics = module(
        optical,
        sar,
        base,
        optical_pa,
        sar_pa,
        return_diagnostics=True,
    )
    assert torch.equal(fused, diagnostics["keep"])
    assert diagnostics["interaction"].min() >= 0.0
    assert diagnostics["interaction"].max() <= 1.0
    assert torch.allclose(
        diagnostics["update_scale"], torch.tensor(0.1), atol=1e-6
    )
    fused.mean().backward()
    final_projection = module.update_projection[-1]
    assert final_projection.weight.grad is not None
    assert final_projection.weight.grad.abs().sum() > 0


def test_loaded_pac_model_starts_from_exact_lwpd_output():
    torch.manual_seed(23)
    source = DirectFusionCDDFuse("lwpd_e4").eval()
    pac = PACMambaCDDFuse().eval()
    pac.load_lwpd_checkpoint(make_lwpd_checkpoint(source))
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand_like(optical)
    with torch.no_grad():
        expected = source(optical, sar)
        actual, features = pac(optical, sar, return_features=True)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert features["pac"]["interaction"].shape == (1, 1, 16, 16)
    assert features["phase"]["optical_pa"].shape == (1, 4, 16, 16)


def test_evaluator_detects_pac_checkpoint():
    checkpoint = {"args": {"variant": PACMambaCDDFuse.VARIANT}}
    assert resolve_variant(checkpoint, "auto") == PACMambaCDDFuse.VARIANT


def test_pac_progressive_unfreezing_keeps_existing_modules_adaptive():
    model = PACMambaCDDFuse()
    args = SimpleNamespace(
        warmup_pac_lr=1e-4,
        local_pac_lr=1e-4,
        joint_pac_lr=5e-5,
        warmup_existing_lr=1e-5,
        local_existing_lr=2e-5,
        joint_existing_lr=1e-5,
        encoder_tail_lr=5e-6,
        encoder_early_lr=1e-6,
        weight_decay=1e-4,
    )
    optimizer = make_optimizer(model, args)
    configure_stage(model, optimizer, "warmup", args)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    for module in (
        model.lwpd,
        model.base_fusion,
        model.detail_fusion.keep_fusion,
        model.decoder,
    ):
        assert all(parameter.requires_grad for parameter in module.parameters())

    configure_stage(model, optimizer, "local", args)
    assert any(parameter.requires_grad for parameter in model.encoder.parameters())
    configure_stage(model, optimizer, "joint", args)
    assert any(
        group["lr"] == args.encoder_early_lr
        for group in optimizer.param_groups
        if group["group_name"] == "encoder_early"
    )


if __name__ == "__main__":
    test_directional_phase_returns_symmetry_and_antisymmetry()
    test_phase_match_tolerates_one_pixel_shift()
    test_biaxial_ssm_preserves_shape_and_backpropagates()
    test_zero_initialized_pac_update_preserves_keep_path()
    test_loaded_pac_model_starts_from_exact_lwpd_output()
    test_evaluator_detects_pac_checkpoint()
    test_pac_progressive_unfreezing_keeps_existing_modules_adaptive()
    print("PAC-Mamba unit checks passed.")
