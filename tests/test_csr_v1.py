"""Unit checks for SPD-guided Competitive Structure Routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from competitive_structure_routing import (
    CompetitiveStructureRouter,
    competitive_structure_losses,
    spd_evidence_maps,
)


def inputs(batch=2, size=32, feature_size=16):
    optical = torch.rand(batch, 1, size, size)
    sar = torch.rand(batch, 1, size, size)
    baseline = torch.rand(batch, 1, size, size)
    raw = torch.randn(batch, 64, feature_size, feature_size)
    clean = raw * 0.8
    valid = torch.ones(batch, 1, size, size)
    return optical, sar, baseline, raw, clean, valid


def test_csr_starts_as_exact_baseline_identity():
    router = CompetitiveStructureRouter()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    assert torch.equal(output, baseline)
    assert diagnostics["csr_correction_rms"].item() == 0.0


def test_spd_removed_evidence_reduces_reliability():
    raw = torch.ones(1, 64, 8, 8)
    clean_stable = raw * 0.95
    clean_removed = raw * 0.05
    _, stable_reliability = spd_evidence_maps(raw, clean_stable, (16, 16))
    _, removed_reliability = spd_evidence_maps(raw, clean_removed, (16, 16))
    assert stable_reliability.mean() > removed_reliability.mean()


def test_csr_losses_backpropagate_to_scale_and_router():
    router = CompetitiveStructureRouter()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    parts = competitive_structure_losses(
        output, baseline, optical, valid, diagnostics
    )
    loss = sum(parts.values())
    assert loss.requires_grad
    loss.backward()
    assert router.mid_strength.grad is not None
    assert router.high_strength.grad is not None
    assert router.logits.bias.grad is not None


def test_output_replacement_is_bounded():
    router = CompetitiveStructureRouter(
        max_mid_replacement=0.10, max_high_replacement=0.03
    )
    with torch.no_grad():
        router.mid_strength.fill_(20.0)
        router.high_strength.fill_(20.0)
    optical, sar, baseline, raw, clean, valid = inputs(batch=1)
    output, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    assert torch.isfinite(output).all()
    assert diagnostics["csr_mid_scale"] <= 0.10 + 1e-6
    assert diagnostics["csr_high_scale"] <= 0.03 + 1e-6


if __name__ == "__main__":
    test_csr_starts_as_exact_baseline_identity()
    test_spd_removed_evidence_reduces_reliability()
    test_csr_losses_backpropagate_to_scale_and_router()
    test_output_replacement_is_bounded()
    print("CSR-v1 unit checks passed.")
