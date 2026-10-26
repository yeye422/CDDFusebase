"""Unit checks for calibrated CSR-v2 routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from competitive_structure_routing_v2 import (
    CompetitiveStructureRouterV2,
    competitive_structure_losses_v2,
    spd_evidence_maps_v2,
)


def inputs(batch=2, size=32, feature_size=16):
    optical = torch.rand(batch, 1, size, size)
    sar = torch.rand(batch, 1, size, size)
    baseline = torch.rand(batch, 1, size, size)
    raw = torch.randn(batch, 64, feature_size, feature_size)
    clean = raw.clone()
    clean[:, :8, 4:12, 5:11] *= 0.1
    valid = torch.ones(batch, 1, size, size)
    return optical, sar, baseline, raw, clean, valid


def test_v2_starts_as_exact_anchor():
    router = CompetitiveStructureRouterV2()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    assert torch.equal(output, baseline)
    assert diagnostics["csr_correction_rms"].item() == 0.0


def test_topk_removed_map_preserves_local_deletion():
    raw = torch.ones(1, 64, 16, 16)
    clean = raw.clone()
    clean[:, :8, 5:11, 6:10] = 0.0
    removed, _ = spd_evidence_maps_v2(raw, clean, (16, 16))
    inside = removed[:, :, 6:10, 7:9].mean()
    outside = removed[:, :, :3, :3].mean()
    assert inside > outside + 0.20


def test_logit_priors_prefer_optical_where_missing_and_removed():
    router = CompetitiveStructureRouterV2()
    optical = torch.zeros(1, 1, 32, 32)
    optical[:, :, :, 16:] = 1.0
    baseline = torch.zeros_like(optical)
    sar = torch.zeros_like(optical)
    raw = torch.ones(1, 64, 16, 16)
    clean = raw.clone()
    clean[:, :8, :, 7:10] = 0.0
    valid = torch.ones_like(optical)
    _, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    gate = diagnostics["csr_optical_gate_map"]
    edge = gate[:, :, :, 15:18].mean()
    flat = gate[:, :, :, :5].mean()
    assert edge > flat


def test_v2_loss_reaches_router_and_scales():
    router = CompetitiveStructureRouterV2()
    optical, sar, baseline, raw, clean, valid = inputs()
    output, diagnostics = router(
        optical, sar, baseline, raw, clean, valid, return_maps=True
    )
    parts = competitive_structure_losses_v2(
        output, baseline, optical, valid, diagnostics
    )
    sum(parts.values()).backward()
    assert router.mid_strength.grad is not None
    assert router.high_strength.grad is not None
    assert router.logits.bias.grad is not None
    assert router.logits.bias.grad.abs().sum() > 0


if __name__ == "__main__":
    test_v2_starts_as_exact_anchor()
    test_topk_removed_map_preserves_local_deletion()
    test_logit_priors_prefer_optical_where_missing_and_removed()
    test_v2_loss_reaches_router_and_scales()
    print("CSR-v2 unit checks passed.")
