import torch

from pdcf_predictability import BidirectionalPredictability
from pdcf_single_path import (
    PDCFSinglePathModel,
    SinglePathFusionAdapter,
    stable_rms,
)
from tests.test_pdcf_p0 import build_toy_model


def build_model(input_mode="predictable_residual"):
    c0 = build_toy_model()
    predictor = BidirectionalPredictability(
        channels=4, hidden_channels=8, blocks=1
    )
    return PDCFSinglePathModel(
        c0,
        predictor,
        input_mode=input_mode,
        hidden_channels=8,
        blocks=1,
        max_relative_correction=0.25,
    )


def test_p2_zero_initialization_is_exact_c0_identity_for_both_inputs():
    torch.manual_seed(30)
    optical = torch.rand(2, 1, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    for mode in SinglePathFusionAdapter.INPUT_MODES:
        model = build_model(mode).eval()
        with torch.no_grad():
            expected, _ = model.c0.forward_c0(optical, sar)
            actual, _, diagnostics = model(
                optical, sar, return_diagnostics=True
            )
        assert torch.equal(actual, expected)
        assert diagnostics["relative_base_correction"].item() == 0.0
        assert diagnostics["relative_detail_correction"].item() == 0.0


def test_p2_modes_have_identical_trainable_parameter_counts():
    models = [build_model(mode) for mode in SinglePathFusionAdapter.INPUT_MODES]
    counts = [
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for model in models
    ]
    assert counts[0] == counts[1]


def test_p2_freezes_c0_and_predictor_but_trains_adapter():
    model = build_model()
    assert all(not parameter.requires_grad for parameter in model.c0.parameters())
    assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())


def test_p2_initial_gradient_reaches_only_output_layer():
    torch.manual_seed(31)
    model = build_model().train()
    optical = torch.rand(2, 1, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    generated, _ = model(optical, sar)
    generated.mean().backward()
    assert model.adapter.output.weight.grad is not None
    assert torch.count_nonzero(model.adapter.output.weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.c0.parameters())
    assert all(parameter.grad is None for parameter in model.predictor.parameters())


def test_p2_correction_bound_is_respected():
    model = build_model()
    with torch.no_grad():
        model.adapter.output.weight.fill_(10.0)
        model.adapter.output.bias.fill_(10.0)
    optical = torch.rand(2, 1, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    _, _, diagnostics = model(optical, sar, return_diagnostics=True)
    assert diagnostics["relative_base_correction"].item() <= 0.25 + 1e-6
    assert diagnostics["relative_detail_correction"].item() <= 0.25 + 1e-6


def test_stable_rms_has_exact_zero_and_finite_zero_gradient():
    values = torch.zeros(2, 4, 3, 3, requires_grad=True)
    result = stable_rms(values)
    assert result.item() == 0.0
    result.backward()
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) == 0
