"""Unit checks for the P2-A residual scale sweep."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sweep_fusion_p2a_scales import scale_id, scaled_detail


def test_scale_zero_is_exact_identity_and_one_is_full_correction():
    base = torch.randn(2, 64, 8, 8)
    correction = torch.randn_like(base)
    assert torch.equal(scaled_detail(base, correction, 0.0), base)
    assert torch.equal(scaled_detail(base, correction, 1.0), base + correction)


def test_scale_ids_are_stable():
    assert scale_id(0.0) == "a0p000"
    assert scale_id(0.25) == "a0p250"
    assert scale_id(1.0) == "a1p000"


if __name__ == "__main__":
    test_scale_zero_is_exact_identity_and_one_is_full_correction()
    test_scale_ids_are_stable()
    print("P2-A scale sweep unit checks passed.")
