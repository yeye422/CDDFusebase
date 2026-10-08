"""Minimal CPU tests for D0 invariants and D1 gradient-bearing objectives."""

from types import SimpleNamespace
import torch

from audit_d0_c0_decomposition import (
    high_frequency_fraction, pearson_by_channel, relative_change,
)
from train_OS import apply_d1_constraints


def test_d0_feature_metrics():
    base = torch.arange(64.0).reshape(1, 1, 8, 8)
    cc, valid = pearson_by_channel(base, base)
    assert bool(valid.all())
    assert torch.allclose(cc, torch.ones_like(cc), atol=1e-6)
    cc_negative, _ = pearson_by_channel(base, -base)
    assert torch.allclose(cc_negative, -torch.ones_like(cc_negative), atol=1e-6)
    assert relative_change(base, base)[0] == 0.0
    const = torch.ones(1, 2, 8, 8)
    assert high_frequency_fraction(const) < 1e-8
    alternating = torch.ones(1, 1, 8, 8)
    alternating[..., :, ::2] = -1
    assert high_frequency_fraction(alternating) > 0.95


def test_d1_objective_is_noop_when_disabled():
    main = torch.tensor(2.0, requires_grad=True)
    args = SimpleNamespace(d1_decomp_weight=0.0, d1_recon_weight=0.0)
    result = apply_d1_constraints(main, {}, args)
    assert result is main


def test_d1_objectives_contribute_independent_gradients():
    d = torch.tensor(0.5, requires_grad=True)
    r = torch.tensor(0.3, requires_grad=True)
    components = {"decomp": d, "d1_recon_raw": r}
    ref = {"decomp": 0.25, "recon": 0.15}
    args = SimpleNamespace(
        d1_decomp_weight=0.05, d1_recon_weight=0.10,
        d1_reference=ref
    )
    total = apply_d1_constraints(torch.tensor(1.0), components, args)
    total.backward()
    assert torch.allclose(d.grad, torch.tensor(0.20))
    assert torch.allclose(r.grad, torch.tensor(2.0 / 3.0))
    assert torch.allclose(total.detach(), torch.tensor(1.30))
    assert "d1_decomp_normalized" in components
    assert "d1_recon_normalized" in components


if __name__ == "__main__":
    test_d0_feature_metrics()
    test_d1_objective_is_noop_when_disabled()
    test_d1_objectives_contribute_independent_gradients()
    print("D0 and D1 CPU invariants passed.")
