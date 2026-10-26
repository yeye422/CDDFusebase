"""Unit tests for the frozen-feature P0 audit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from audit_fusion_feature_p0 import (
    local_cosine_maps,
    normalized_sensitivity,
    structure_regions,
)


def test_local_cosine_recovers_small_translation():
    features = torch.zeros(1, 2, 9, 9)
    features[:, 0, 3:6, 3:6] = 1.0
    features[:, 1, 1:4, 5:8] = 0.5
    shifted = torch.roll(features, shifts=(1, -1), dims=(-2, -1))
    same, best, _, valid = local_cosine_maps(features, shifted, radius=2)
    mask = valid > 0.5
    assert best[mask].mean() > same[mask].mean()
    assert torch.all(best[mask] <= 1.0 + 1e-6)


def test_structure_regions_are_mutually_exclusive():
    optical = torch.linspace(0, 1, 256).reshape(1, 1, 16, 16)
    sar = optical.flip(-1)
    valid = torch.ones_like(optical)
    regions, _, _ = structure_regions(optical, sar, valid)
    names = list(regions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            assert not torch.any(regions[first] & regions[second])


def test_normalized_sensitivity_is_zero_for_identical_features():
    features = torch.randn(2, 64, 8, 8)
    sensitivity = normalized_sensitivity(features, features)
    assert torch.equal(sensitivity, torch.zeros_like(sensitivity))


if __name__ == "__main__":
    test_local_cosine_recovers_small_translation()
    test_structure_regions_are_mutually_exclusive()
    test_normalized_sensitivity_is_zero_for_identical_features()
    print("P0 fusion feature audit unit checks passed.")
