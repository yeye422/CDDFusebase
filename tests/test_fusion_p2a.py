"""Unit checks for P2-A source-preserving fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from p2a_source_preserving_fusion import (
    PseudoSensingHeads,
    SourcePreservingDetailAdapter,
    weighted_normalized_l1,
)


def test_adapter_is_exact_b0_identity_at_initialization():
    torch.manual_seed(1)
    adapter = SourcePreservingDetailAdapter()
    optical = torch.randn(2, 64, 8, 8)
    sar = torch.randn_like(optical)
    b0 = torch.randn_like(optical)
    fused, diagnostics = adapter(optical, sar, b0)
    assert torch.equal(fused, b0)
    assert diagnostics["p2a_correction_rms"].item() == 0.0


def test_adapter_correction_is_bounded_per_channel():
    torch.manual_seed(2)
    adapter = SourcePreservingDetailAdapter(max_relative_correction=0.03)
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn_like(optical)
    b0 = torch.randn_like(optical)
    with torch.no_grad():
        adapter.delta.weight.fill_(0.1)
        adapter.delta.bias.fill_(0.1)
        fused, _ = adapter(optical, sar, b0)
    rms = b0.square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert torch.all((fused - b0).abs() <= 0.03 * rms + 1e-6)


def test_pseudo_sensing_heads_start_as_identity():
    heads = PseudoSensingHeads()
    features = torch.randn(2, 64, 8, 8)
    optical, sar = heads(features)
    assert torch.equal(optical, features)
    assert torch.equal(sar, features)


def test_weighted_normalized_l1_is_zero_for_match():
    features = torch.randn(1, 64, 8, 8)
    weight = torch.ones(1, 1, 8, 8)
    assert weighted_normalized_l1(features, features, weight).item() == 0.0


def test_zero_initialized_delta_receives_gradient():
    torch.manual_seed(3)
    adapter = SourcePreservingDetailAdapter()
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn_like(optical)
    b0 = torch.randn_like(optical)
    fused, _ = adapter(optical, sar, b0)
    fused.square().mean().backward()
    assert adapter.delta.weight.grad is not None
    assert adapter.delta.weight.grad.abs().sum().item() > 0.0


if __name__ == "__main__":
    test_adapter_is_exact_b0_identity_at_initialization()
    test_adapter_correction_is_bounded_per_channel()
    test_pseudo_sensing_heads_start_as_identity()
    test_weighted_normalized_l1_is_zero_for_match()
    test_zero_initialized_delta_receives_gradient()
    print("P2-A source-preserving fusion unit checks passed.")
