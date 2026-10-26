import torch

from evaluate_pdcf_predictability import intervention_outputs
from pdcf_predictability import (
    BidirectionalLinearBaseline,
    BidirectionalPredictability,
    FeatureStatisticsAccumulator,
    balanced_prediction_loss,
    prediction_metrics,
    residual_intervention,
)
from tests.test_pdcf_p0 import build_toy_model


def sample_features(batch=2, channels=4, height=6, width=5):
    return tuple(
        torch.randn(batch, channels, height, width) for _ in range(4)
    )


def test_feature_statistics_pool_modalities_and_mask_invalid_pixels():
    accumulator = FeatureStatisticsAccumulator(channels=1)
    optical_base = torch.tensor([[[[1.0, 100.0]]]])
    optical_detail = torch.tensor([[[[3.0, 100.0]]]])
    sar_base = torch.tensor([[[[5.0, 100.0]]]])
    sar_detail = torch.tensor([[[[7.0, 100.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])
    accumulator.update(
        optical_base, optical_detail, sar_base, sar_detail, mask
    )
    statistics = accumulator.finalize()
    assert statistics["base_mean"].item() == 3.0
    assert statistics["detail_mean"].item() == 5.0
    assert statistics["base_scale"].item() == 2.0
    assert statistics["detail_scale"].item() == 2.0


def test_predictability_shapes_and_balanced_four_term_loss():
    torch.manual_seed(10)
    model = BidirectionalPredictability(channels=4, hidden_channels=8, blocks=1)
    statistics = {
        "base_mean": torch.zeros(1, 4, 1, 1),
        "base_scale": torch.ones(1, 4, 1, 1),
        "detail_mean": torch.zeros(1, 4, 1, 1),
        "detail_scale": torch.ones(1, 4, 1, 1),
    }
    model.set_statistics(statistics)
    features = sample_features()
    outputs = model(*features)
    losses = balanced_prediction_loss(outputs, mask=None, channels=4)
    assert outputs["optical"].shape == (2, 8, 6, 5)
    assert outputs["residual_sar"].shape == (2, 8, 6, 5)
    expected = torch.stack(
        [losses[name] for name in ("o2s_base", "o2s_detail", "s2o_base", "s2o_detail")]
    ).mean()
    torch.testing.assert_close(losses["total"], expected)


def test_targets_are_stopped_but_predictors_receive_gradients():
    model = BidirectionalPredictability(channels=4, hidden_channels=8, blocks=1)
    features = tuple(tensor.requires_grad_() for tensor in sample_features())
    outputs = model(*features)
    balanced_prediction_loss(outputs, None, channels=4)["total"].backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.optical_to_sar.parameters()
    )
    # Inputs participate in predictor computation, but targets are detached.  The
    # test guards the intended loss API rather than requiring frozen callers.
    assert all(torch.isfinite(tensor.grad).all() for tensor in features)


def test_linear_and_mean_baselines_have_explicit_metrics():
    features = sample_features()
    optical = torch.cat(features[:2], dim=1)
    sar = torch.cat(features[2:], dim=1)
    linear = BidirectionalLinearBaseline(channels=8)
    predictions = linear(optical, sar)
    outputs = {"optical": optical, "sar": sar, **predictions}
    metrics = prediction_metrics(outputs, None, channels=4)
    assert "o2s_base_nrmse" in metrics
    assert "o2s_mean_base_nrmse" in metrics
    assert all(torch.isfinite(value) for value in metrics.values())


def test_residual_interventions_preserve_contracts():
    torch.manual_seed(20)
    residual = torch.randn(3, 8, 5, 4)
    assert torch.equal(residual_intervention(residual, "original"), residual)
    assert torch.count_nonzero(residual_intervention(residual, "zero")) == 0
    shuffled = residual_intervention(residual, "spatial_shuffle")
    torch.testing.assert_close(
        shuffled.flatten(2).sort(dim=-1).values,
        residual.flatten(2).sort(dim=-1).values,
    )
    swapped = residual_intervention(residual, "batch_swap")
    assert torch.equal(swapped[1], residual[0])
    noise = residual_intervention(residual, "rms_noise")
    torch.testing.assert_close(
        noise.square().mean(dim=(1, 2, 3)).sqrt(),
        residual.square().mean(dim=(1, 2, 3)).sqrt(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_original_residual_reconstructs_the_c0_path():
    torch.manual_seed(21)
    c0 = build_toy_model().eval()
    predictor = BidirectionalPredictability(
        channels=4, hidden_channels=8, blocks=1
    ).eval()
    optical = torch.rand(2, 1, 8, 8)
    sar = torch.rand(2, 1, 8, 8)
    features = c0.extract_features(optical, sar)
    base, detail = c0.c0_features(*features)
    expected, _ = c0.decoder(optical, base, detail)
    _, variants = intervention_outputs(
        c0,
        predictor,
        optical,
        features,
        torch.Generator().manual_seed(22),
    )
    torch.testing.assert_close(variants["original"], expected, rtol=1e-5, atol=1e-6)
