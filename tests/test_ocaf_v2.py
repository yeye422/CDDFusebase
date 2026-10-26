"""Unit checks for independent-gate OCAF-v2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ocaf_v2 import OpticalCanvasFusionV2, evidence_scores, ocaf_v2_losses


def inputs(batch=2, size=32, feature_size=16):
    optical = torch.rand(batch, 1, size, size)
    clean_sar = torch.rand(batch, 1, size, size)
    counterfactual = (clean_sar + 0.01 * torch.randn_like(clean_sar)).clamp(0, 1)
    baseline = torch.rand(batch, 1, size, size)
    raw = torch.randn(batch, 64, feature_size, feature_size)
    clean = raw.clone()
    clean[:, :8, 4:12, 5:11] *= 0.2
    valid = torch.ones(batch, 1, size, size)
    return optical, clean_sar, counterfactual, baseline, raw, clean, valid


def test_v2_starts_as_exact_optical_identity():
    model = OpticalCanvasFusionV2()
    values = inputs()
    output, _ = model(*values, return_maps=True)
    assert torch.equal(output, values[0])


def test_evidence_no_longer_collapses_exclusive_by_many_products():
    optical = torch.zeros(1, 1, 32, 32)
    sar = torch.zeros_like(optical)
    sar[:, :, :, 16:] = torch.linspace(0, 1, 16)[None, None, None, :]
    counterfactual = sar.clone()
    raw = torch.ones(1, 64, 16, 16)
    clean = raw * 0.9
    scores = evidence_scores(optical, sar, counterfactual, raw, clean)
    assert scores["novelty"].mean() > 0
    assert scores["exclusive"].mean() > 0
    assert scores["exclusive"].mean() >= 0.39 * scores["novelty"].mean()


def test_independent_sar_gate_is_not_softmax_starved():
    model = OpticalCanvasFusionV2()
    values = inputs(batch=1)
    _, diagnostics = model(*values, return_maps=True)
    assert diagnostics["ocaf_sar_gate_mean"] > 0.001
    assert diagnostics["ocaf_common_gate_mean"] > 0.001


def test_v2_losses_reach_both_gates_and_three_amplitudes():
    model = OpticalCanvasFusionV2()
    values = inputs()
    output, diagnostics = model(*values, return_maps=True)
    parts = ocaf_v2_losses(output, values[0], values[-1], diagnostics)
    loss = sum(parts.values())
    assert loss.requires_grad
    loss.backward()
    for parameter in (
        model.common_amplitude,
        model.exclusive_mid_amplitude,
        model.exclusive_high_amplitude,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
    assert model.gate_logits.bias.grad is not None
    assert torch.isfinite(model.gate_logits.bias.grad).all()


def test_amplitudes_are_bounded_in_image_units():
    model = OpticalCanvasFusionV2()
    with torch.no_grad():
        model.common_amplitude.fill_(20)
        model.exclusive_mid_amplitude.fill_(20)
        model.exclusive_high_amplitude.fill_(20)
    output, diagnostics = model(*inputs(batch=1), return_maps=True)
    assert torch.isfinite(output).all()
    assert diagnostics["ocaf_common_amplitude"] <= 2.0 / 255.0 + 1e-7
    assert diagnostics["ocaf_exclusive_mid_amplitude"] <= 1.5 / 255.0 + 1e-7
    assert diagnostics["ocaf_exclusive_high_amplitude"] <= 1.0 / 255.0 + 1e-7


if __name__ == "__main__":
    test_v2_starts_as_exact_optical_identity()
    test_evidence_no_longer_collapses_exclusive_by_many_products()
    test_independent_sar_gate_is_not_softmax_starved()
    test_v2_losses_reach_both_gates_and_three_amplitudes()
    test_amplitudes_are_bounded_in_image_units()
    print("OCAF-v2 unit checks passed.")
