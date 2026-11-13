"""Numerical checks for the fixed ASPR operators and differentiable solve."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2_aspr import SPEC, fixed_weights, gradient, gradient_adjoint, solve


def test_gradient_adjoint_includes_odd_boundaries():
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(2, 1, 5, 7, generator=rng)
    q = torch.randn(2, 2, 5, 7, generator=rng)
    torch.testing.assert_close((gradient(x) * q).sum(), (x * gradient_adjoint(q)).sum())


def test_pcg_matches_dense_normal_equations_and_handles_zero_residual():
    rng = torch.Generator().manual_seed(51)
    height, width = 5, 6
    size = height * width
    f0, optical, wo, ws = [torch.rand(1, 1, height, width, generator=rng) for _ in range(4)]
    pixels = torch.arange(size).reshape(height, width)
    identity = torch.eye(size)
    lowpass = torch.zeros_like(identity)
    for scale in (2, 4):
        for y in range(0, height - scale + 1, scale):
            for x in range(0, width - scale + 1, scale):
                ids = pixels[y:y + scale, x:x + scale].flatten()
                lowpass[ids[:, None], ids] += SPEC["alpha"] / scale**2
    starts = torch.cat((pixels[:, :-1].flatten(), pixels[:-1].flatten()))
    ends = torch.cat((pixels[:, 1:].flatten(), pixels[1:].flatten()))
    difference = identity[ends] - identity[starts]
    optical_edges = SPEC["beta"] * wo.flatten()[starts]
    sar_edges = SPEC["gamma"] * ws.flatten()[starts]
    optical_operator = lowpass + difference.T @ (optical_edges[:, None] * difference)
    matrix = SPEC["mu"] * identity + optical_operator + difference.T @ (sar_edges[:, None] * difference)
    rhs = optical_operator @ (optical - f0).flatten()
    expected = f0.flatten() + torch.linalg.solve(matrix, rhs)

    initial = torch.cat((f0, optical)).requires_grad_()
    output, residual = solve(initial, optical.expand_as(initial), wo.expand_as(initial), ws.expand_as(initial),
                             steps=20, diagnostics=True)
    torch.testing.assert_close(output[0].flatten(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output[1], optical[0], atol=0, rtol=0)
    assert residual.max() < 1e-6
    output.square().mean().backward()
    assert torch.isfinite(initial.grad).all()


def test_sar_protection_respects_valid_mask_and_requires_existing_structure():
    optical = torch.zeros(1, 1, 16, 16)
    sar = optical.clone()
    sar[..., 8:] = 1
    valid = torch.ones_like(sar)
    _, absent = fixed_weights(optical, sar, optical, valid)
    _, present = fixed_weights(optical, sar, sar, valid)
    _, invalid = fixed_weights(optical, sar, sar, torch.zeros_like(valid))
    assert absent.count_nonzero() == invalid.count_nonzero() == 0
    assert 0 < present.max() <= SPEC["sar_weight"]
