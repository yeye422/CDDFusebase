import torch

from net import DetailFeatureExtraction
from reliability_modules import ResidualSpatialModulator
from train_OS import configure_trainability


def test_r1_zero_initialization_is_exact_c0_identity():
    torch.manual_seed(7)
    detail_fuse = DetailFeatureExtraction(num_layers=1).eval()
    modulator = ResidualSpatialModulator(channels=64, hidden_channels=32, alpha=0.25).eval()

    optical = torch.randn(2, 64, 12, 10)
    sar = torch.randn(2, 64, 12, 10)

    with torch.no_grad():
        modulated_sar, diagnostics = modulator(
            optical, sar, return_maps=True
        )
        c0_detail = detail_fuse(optical + sar)
        r1_detail = detail_fuse(optical + modulated_sar)

    assert torch.equal(modulated_sar, sar)
    assert torch.equal(r1_detail, c0_detail)
    assert torch.count_nonzero(diagnostics["r1_modulation_map"]) == 0
    assert diagnostics["r1_modulation_abs_mean"].item() == 0.0


def test_r1_modulation_is_bounded():
    module = ResidualSpatialModulator(
        channels=64, hidden_channels=32, alpha=0.25
    )
    with torch.no_grad():
        module.output.weight.fill_(1.0)
        module.output.bias.fill_(1.0)

    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn(1, 64, 8, 8)
    _, diagnostics = module(optical, sar, return_maps=True)
    modulation = diagnostics["r1_modulation_map"]

    assert modulation.shape == (1, 1, 8, 8)
    assert torch.isfinite(modulation).all()
    assert modulation.amin().item() >= -0.25
    assert modulation.amax().item() <= 0.25


def test_router_only_freezes_c0_and_keeps_modulator_trainable():
    models = {
        "encoder": torch.nn.Conv2d(1, 1, kernel_size=1),
        "decoder": torch.nn.Conv2d(1, 1, kernel_size=1),
        "base_fuse": torch.nn.Conv2d(1, 1, kernel_size=1),
        "detail_fuse": torch.nn.Conv2d(1, 1, kernel_size=1),
        "detail_modulator": ResidualSpatialModulator(
            channels=64, hidden_channels=32, alpha=0.25
        ),
    }

    configure_trainability(models, "router_only")

    for name in ("encoder", "decoder", "base_fuse", "detail_fuse"):
        assert all(
            not parameter.requires_grad
            for parameter in models[name].parameters()
        )
    assert all(
        parameter.requires_grad
        for parameter in models["detail_modulator"].parameters()
    )


def test_r1_gradient_reaches_only_modulator_parameters():
    torch.manual_seed(11)
    detail_fuse = DetailFeatureExtraction(num_layers=1)
    modulator = ResidualSpatialModulator(
        channels=64, hidden_channels=32, alpha=0.25
    )
    models = {
        "encoder": torch.nn.Conv2d(1, 1, kernel_size=1),
        "decoder": torch.nn.Conv2d(1, 1, kernel_size=1),
        "base_fuse": torch.nn.Conv2d(1, 1, kernel_size=1),
        "detail_fuse": detail_fuse,
        "detail_modulator": modulator,
    }
    configure_trainability(models, "router_only")

    optical = torch.randn(2, 64, 8, 8)
    sar = torch.randn(2, 64, 8, 8)
    modulated_sar, _ = modulator(optical, sar)
    fused = detail_fuse(optical + modulated_sar)
    fused.square().mean().backward()

    assert any(
        parameter.grad is not None
        for parameter in modulator.parameters()
    )
    for name in ("encoder", "decoder", "base_fuse", "detail_fuse"):
        assert all(
            parameter.grad is None
            for parameter in models[name].parameters()
        )


if __name__ == "__main__":
    test_r1_zero_initialization_is_exact_c0_identity()
    test_r1_modulation_is_bounded()
    test_router_only_freezes_c0_and_keeps_modulator_trainable()
    test_r1_gradient_reaches_only_modulator_parameters()
    print("R1 spatial modulation checks passed.")
