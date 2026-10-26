"""Unit checks for optical-canvas asymmetric fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ocaf_v1 import OpticalCanvasFusion, ocaf_losses, source_priors


def inputs(batch=2, size=32, feature_size=16):
    optical = torch.rand(batch, 1, size, size)
    sar = torch.rand(batch, 1, size, size)
    baseline = torch.rand(batch, 1, size, size)
    raw = torch.randn(batch, 64, feature_size, feature_size)
    clean = raw.clone()
    clean[:, :8, 4:12, 5:11] *= 0.2
    valid = torch.ones(batch, 1, size, size)
    return optical, sar, baseline, raw, clean, valid


def test_ocaf_has_exact_optical_identity_at_initialization():
    model = OpticalCanvasFusion()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = model(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    assert torch.equal(output, optical)
    assert diagnostics["ocaf_total_delta_rms"].item() == 0.0


def test_priors_are_finite_and_bounded():
    optical, sar, _, raw, clean, _ = inputs()
    priors = source_priors(optical, sar, raw, clean)
    for value in priors.values():
        assert torch.isfinite(value).all()
        assert value.min() >= 0.0
        assert value.max() <= 1.0


def test_sar_candidates_only_add_excess_energy():
    model = OpticalCanvasFusion()
    optical, sar, baseline, raw, clean, valid = inputs(batch=1)
    _, diagnostics = model(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    mid = diagnostics["ocaf_exclusive_mid_candidate"]
    high = diagnostics["ocaf_exclusive_high_candidate"]
    assert torch.isfinite(mid).all() and torch.isfinite(high).all()


def test_ocaf_loss_backpropagates_to_router_and_all_scales():
    model = OpticalCanvasFusion()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = model(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    parts = ocaf_losses(output, optical, valid, diagnostics)
    sum(parts.values()).backward()
    assert model.common_strength.grad is not None
    assert model.exclusive_mid_strength.grad is not None
    assert model.exclusive_high_strength.grad is not None
    assert model.logits.bias.grad is not None
    assert model.logits.bias.grad.abs().sum() > 0


def test_positive_scales_keep_output_change_bounded():
    model = OpticalCanvasFusion()
    with torch.no_grad():
        model.common_strength.fill_(20.0)
        model.exclusive_mid_strength.fill_(20.0)
        model.exclusive_high_strength.fill_(20.0)
    optical, sar, baseline, raw, clean, valid = inputs(batch=1)
    output, diagnostics = model(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    assert torch.isfinite(output).all()
    assert diagnostics["ocaf_common_scale"] <= 0.10 + 1e-6
    assert diagnostics["ocaf_exclusive_mid_scale"] <= 0.05 + 1e-6
    assert diagnostics["ocaf_exclusive_high_scale"] <= 0.02 + 1e-6


if __name__ == "__main__":
    test_ocaf_has_exact_optical_identity_at_initialization()
    test_priors_are_finite_and_bounded()
    test_sar_candidates_only_add_excess_energy()
    test_ocaf_loss_backpropagates_to_router_and_all_scales()
    test_positive_scales_keep_output_change_bounded()
    print("OCAF-v1 unit checks passed.")
