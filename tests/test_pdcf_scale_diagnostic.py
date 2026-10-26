import numpy as np
import torch

from sweep_pdcf_p2_scales import (
    average_ranks,
    local_change_diagnostics,
    paired_bootstrap_mean_ci,
    safe_correlation,
    scale_id,
    summarize,
)
from test_OS import PAPER_METRIC_NAMES


def test_signed_scale_ids_are_stable_and_unambiguous():
    assert scale_id(-0.05) == "n0p050"
    assert scale_id(0.0) == "0p000"
    assert scale_id(0.05) == "0p050"


def test_paired_bootstrap_is_deterministic_and_contains_constant_mean():
    values = np.full(8, -0.25)
    first = paired_bootstrap_mean_ci(values, 100, np.random.default_rng(7))
    second = paired_bootstrap_mean_ci(values, 100, np.random.default_rng(7))
    assert first == second
    assert first["low"] == -0.25
    assert first["high"] == -0.25


def test_average_ranks_handles_ties_and_correlations_handle_constants():
    np.testing.assert_array_equal(
        average_ranks([3.0, 1.0, 1.0, 2.0]),
        np.asarray([3.0, 0.5, 0.5, 2.0]),
    )
    assert safe_correlation([1, 2, 3], [2, 4, 6]) == 1.0
    assert safe_correlation([1, 1, 1], [2, 4, 6]) is None


def test_local_change_diagnostics_are_zero_for_identity():
    baseline = torch.rand(1, 1, 16, 16)
    result = local_change_diagnostics(
        baseline.clone(), baseline, block_size=8, gradient_budget=0.05
    )
    assert all(value == 0.0 for value in result.values())


def test_local_change_diagnostics_detect_local_change():
    baseline = torch.zeros(1, 1, 16, 16)
    candidate = baseline.clone()
    candidate[:, :, :8, :8] = 1.0
    result = local_change_diagnostics(
        candidate, baseline, block_size=8, gradient_budget=0.05
    )
    assert result["local_output_rms_max"] == 1.0
    assert result["local_output_rms_p95"] > 0.0
    assert result["local_gradient_ratio_max"] > 0.0
    assert result["local_gradient_budget_exceeded_fraction"] > 0.0


def test_summary_persists_bootstrap_and_proxy_metric_correlation():
    methods = ("c0", "predictable_sn0p010")
    diagnostics = (
        "proxy_minimax", "proxy_optical", "proxy_structure",
        "local_output_rms_p95", "local_output_rms_max",
        "local_gradient_ratio_mean", "local_gradient_ratio_p95",
        "local_gradient_ratio_max", "local_gradient_budget_exceeded_fraction",
    )
    rows = []
    for image_index in range(3):
        row = {"image": str(image_index)}
        for method_index, method in enumerate(methods):
            for metric_index, metric in enumerate(PAPER_METRIC_NAMES):
                row[f"{method}_{metric}"] = float(
                    image_index + metric_index + 0.1 * method_index
                )
            row[f"{method}_output_deviation_rms"] = 0.01 * method_index
            for diagnostic in diagnostics:
                row[f"{method}_{diagnostic}"] = float(
                    image_index - 0.1 * method_index
                )
        rows.append(row)
    result = summarize(rows, methods, bootstrap_repetitions=100, bootstrap_seed=7)
    comparison = result["comparison_to_c0"]["predictable_sn0p010"]
    assert comparison["Qabf"]["paired_bootstrap_mean_delta_ci"]["low"] > 0.0
    assert comparison["proxy_minimax"]["paired_bootstrap_mean_delta_ci"]["high"] < 0.0
    assert comparison["Qabf"]["proxy_delta_pearson"] is None
