from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_pdcf_signed_controls import (
    compare_methods,
    detail_gain_output,
    next_calibration_beta,
    output_rms,
    unsharp_output,
    validate_args,
)
from test_OS import PAPER_METRIC_NAMES
from tests.test_pdcf_p0 import build_toy_model
from train_OS import OpticalSARLosses


def test_zero_strength_controls_are_exact_identity():
    torch.manual_seed(50)
    c0 = build_toy_model().eval()
    losses = OpticalSARLosses()
    optical = torch.rand(1, 1, 16, 16)
    sar = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        features = c0.extract_features(optical, sar)
        base, detail = c0.c0_features(*features)
        baseline, _ = c0.decoder(optical, base, detail)
        unsharp = unsharp_output(losses, baseline, beta=0.0, sigma=1.0)
        gained = detail_gain_output(c0, optical, base, detail, beta=0.0)
    assert torch.equal(unsharp, baseline)
    assert torch.equal(gained, baseline)
    assert output_rms(unsharp, baseline) == 0.0
    assert output_rms(gained, baseline) == 0.0


def test_calibration_beta_update_matches_linear_response():
    assert next_calibration_beta(0.1, target_rms=0.02, measured_rms=0.01) == 0.2


def test_pairwise_comparison_uses_candidate_minus_reference():
    rows = []
    for index in range(4):
        row = {}
        for metric_index, metric in enumerate(PAPER_METRIC_NAMES):
            row[f"reference_{metric}"] = float(metric_index)
            row[f"candidate_{metric}"] = float(metric_index + 0.1)
        rows.append(row)
    comparison = compare_methods(
        rows, "candidate", "reference", repetitions=100, seed=3
    )
    assert np.isclose(comparison["Qabf"]["mean_delta"], 0.1)
    assert comparison["Qabf"]["improved_fraction"] == 1.0
    assert comparison["Nabf"]["improved_fraction"] == 0.0


def test_protocol_rejects_overlapping_calibration_and_holdout():
    args = SimpleNamespace(
        output_dir=Path("definitely_missing_control_test_output"),
        signed_scale=-0.05,
        unsharp_sigma=1.0,
        calibration_initial_beta=0.01,
        calibration_iterations=4,
        calibration_tolerance=0.01,
        bootstrap_repetitions=100,
        local_block_size=32,
        local_gradient_budget=0.05,
        calibration_offset=0,
        calibration_limit=200,
        evaluation_offset=100,
        evaluation_limit=800,
    )
    try:
        validate_args(args, split_size=1000)
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping subsets were accepted")
