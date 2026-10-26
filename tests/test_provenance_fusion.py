"""Unit checks for exact source-provenance fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from provenance_fusion import ProvenanceAwareFusion, shapley_decompose


class NonlinearFusion(nn.Module):
    def forward(self, features):
        return features + 0.2 * features.square() + 0.1


def test_shapley_parts_exactly_reconstruct_nonlinear_joint():
    fusion = NonlinearFusion()
    optical = torch.randn(2, 4, 8, 8)
    sar = torch.randn(2, 4, 8, 8)
    parts = shapley_decompose(fusion, optical, sar)
    assert torch.allclose(parts["reconstruction"], parts["joint"], atol=1e-6)
    assert torch.allclose(
        parts["interaction"], 0.4 * optical * sar, atol=1e-6
    )


def test_shapley_is_symmetric_under_modality_swap():
    fusion = NonlinearFusion()
    optical = torch.randn(1, 4, 8, 8)
    sar = torch.randn(1, 4, 8, 8)
    forward = shapley_decompose(fusion, optical, sar)
    swapped = shapley_decompose(fusion, sar, optical)
    assert torch.allclose(forward["optical"], swapped["sar"], atol=1e-6)
    assert torch.allclose(forward["sar"], swapped["optical"], atol=1e-6)


def test_router_initialization_is_exact_original_fusion():
    model = ProvenanceAwareFusion(
        NonlinearFusion(), channels=4, hidden_channels=8
    )
    optical = torch.randn(2, 4, 8, 8)
    sar = torch.randn(2, 4, 8, 8)
    output, diagnostics = model(optical, sar, return_maps=True)
    expected = model.fusion(optical + sar)
    assert torch.equal(output, expected)
    assert diagnostics["provenance_feature_deviation_rms"].item() == 0.0
    assert diagnostics["provenance_reconstruction_error"].item() < 1e-6


def test_router_receives_gradient_while_fusion_stays_frozen():
    model = ProvenanceAwareFusion(
        NonlinearFusion(), channels=4, hidden_channels=8
    )
    optical = torch.randn(2, 4, 8, 8)
    sar = torch.randn(2, 4, 8, 8)
    output, _ = model(optical, sar)
    output.square().mean().backward()
    assert model.gate.output.weight.grad is not None
    assert model.gate.output.weight.grad.abs().sum() > 0


def test_source_gain_is_bounded():
    model = ProvenanceAwareFusion(
        NonlinearFusion(), channels=4, hidden_channels=8,
        max_gain_delta=0.10,
    )
    with torch.no_grad():
        model.gate.output.weight.fill_(10.0)
        model.gate.output.bias.fill_(10.0)
    _, diagnostics = model(
        torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8),
        return_maps=True,
    )
    assert diagnostics["provenance_optical_gain_map"].max() <= 1.10 + 1e-6
    assert diagnostics["provenance_optical_gain_map"].min() >= 0.90 - 1e-6
    assert diagnostics["provenance_sar_gain_map"].max() <= 1.10 + 1e-6
    assert diagnostics["provenance_sar_gain_map"].min() >= 0.90 - 1e-6


if __name__ == "__main__":
    test_shapley_parts_exactly_reconstruct_nonlinear_joint()
    test_shapley_is_symmetric_under_modality_swap()
    test_router_initialization_is_exact_original_fusion()
    test_router_receives_gradient_while_fusion_stays_frozen()
    test_source_gain_is_bounded()
    print("Provenance fusion unit checks passed.")
