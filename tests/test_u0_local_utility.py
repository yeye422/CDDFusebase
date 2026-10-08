import numpy as np
import torch

from audit_u0_local_utility import (
    beneficial_direction,
    cosine_edge_weights,
    direction,
    pareto_direction,
    quality_gain,
    signed_contrast,
    stability,
    summarize,
    tile_mask,
    tiles,
)


def scores(q=0.7, nab=0.04, fmi=0.65, vif=0.32, ssim=0.52):
    return {"Qabf": q, "Nabf": nab, "FMI": fmi, "VIF": vif, "SSIM": ssim}


def test_tiles_cover_image_without_overlap():
    result = tiles(256, 256, 4, 48)
    assert len(result) == 16
    coverage = np.zeros((256, 256), dtype=np.int32)
    for _, _, top, bottom, left, right in result:
        coverage[top:bottom, left:right] += 1
    assert np.all(coverage == 1)


def test_cosine_tile_mask_is_local_and_smooth():
    box = (1, 2, 64, 128, 128, 192)
    mask = tile_mask((256, 256), box, 8, "cpu").squeeze().numpy()
    assert mask.shape == (256, 256)
    assert mask.max() == 1.0
    assert mask.min() == 0.0
    assert np.count_nonzero(mask[:64, :]) == 0
    assert np.count_nonzero(mask[128:, :]) == 0
    assert mask[64, 150] == 0.0
    assert mask[90, 160] == 1.0


def test_relative_direction_does_not_imply_a_benefit_over_c0():
    baseline = scores()
    plus = scores(q=0.68)
    minus = scores(q=0.67)
    contrast = signed_contrast(plus, minus, "Qabf")
    assert direction(contrast, 0.0002) == 1
    assert beneficial_direction(plus, minus, baseline, 0.0002) == 0
    assert pareto_direction(plus, minus, baseline, 0.0002) == 0


def test_beneficial_direction_and_pareto_guard():
    baseline = scores()
    plus = scores(q=0.705, nab=0.039, fmi=0.652, vif=0.322)
    minus = scores(q=0.699)
    assert beneficial_direction(plus, minus, baseline, 0.0002) == 1
    assert pareto_direction(plus, minus, baseline, 0.0002) == 1

    # A better Qabf score is not enough when artifacts or VIF deteriorate.
    bad_plus = scores(q=0.705, nab=0.045, fmi=0.652, vif=0.30)
    assert beneficial_direction(bad_plus, minus, baseline, 0.0002) == 1
    assert pareto_direction(bad_plus, minus, baseline, 0.0002) == 0


def test_nabf_gain_and_direction_increase_when_artifacts_decrease():
    baseline = scores(nab=0.04)
    plus = scores(nab=0.03)
    minus = scores(nab=0.05)
    assert quality_gain(plus, baseline, "Nabf") > 0
    assert signed_contrast(plus, minus, "Nabf") > 0


def test_stability_requires_non_neutral_benefit_at_both_strengths():
    records = [
        {
            "image": "a", "row": 0, "col": 0, "eps": 0.10,
            "q_beneficial_direction": 1, "contrast_Qabf": 0.01,
        },
        {
            "image": "a", "row": 0, "col": 0, "eps": 0.20,
            "q_beneficial_direction": 1, "contrast_Qabf": 0.02,
        },
        {
            "image": "a", "row": 0, "col": 1, "eps": 0.10,
            "q_beneficial_direction": 0, "contrast_Qabf": 0.0,
        },
        {
            "image": "a", "row": 0, "col": 1, "eps": 0.20,
            "q_beneficial_direction": -1, "contrast_Qabf": -0.02,
        },
    ]
    result = stability(records)
    assert len(result) == 2
    assert result[0]["direction_agrees"] == 1
    assert result[1]["both_non_neutral"] == 0


if __name__ == "__main__":
    test_tiles_cover_image_without_overlap()
    test_cosine_tile_mask_is_local_and_smooth()
    test_relative_direction_does_not_imply_a_benefit_over_c0()
    test_beneficial_direction_and_pareto_guard()
    test_nabf_gain_and_direction_increase_when_artifacts_decrease()
    test_stability_requires_non_neutral_benefit_at_both_strengths()
    print("U0 intervention invariant checks passed.")
