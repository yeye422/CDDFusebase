"""CPU synthetic checks for frozen E2 RICF bottleneck preflight.

Does not load QXS data/checkpoints. Run from repository root:
    python -m unittest -v test_ricf_bottleneck_preflight
"""
import unittest

import torch

from diagnose_ricf_bottleneck_preflight import (
    PROBE_NAMES, build_inputs, edge_residual_proxy, flatten_grid,
    make_transform, matrix_targets, pca_basis, random_basis,
    ridge_fit, ridge_predict, standardize, standardizer, summarize,
    target_error_rows, train_probes,
)


def mock_sample(rows=128, seed=1):
    gen = torch.Generator().manual_seed(seed)
    optical = torch.randn(rows, 64, generator=gen, dtype=torch.float64)
    sar = torch.randn(rows, 64, generator=gen, dtype=torch.float64)
    base = 0.4 * optical + 0.3 * sar
    fused = optical + 0.5 * sar
    return dict(scene=str(seed), base=base, fused=fused,
                optical=optical, sar=sar,
                valid_sar=torch.ones(rows, dtype=torch.bool),
                edge_optical=torch.arange(rows, dtype=torch.float64),
                edge_sar=torch.arange(rows, dtype=torch.float64))


class RICFPreflightTests(unittest.TestCase):
    def test_ridge_generalizes_linear_mapping(self):
        torch.manual_seed(4)
        w = torch.randn(6, 4, dtype=torch.float64)
        x = torch.randn(256, 6, dtype=torch.float64)
        holdout = torch.randn(128, 6, dtype=torch.float64)
        probe = ridge_fit(x, x @ w, ridge=1e-6)
        err = (ridge_predict(holdout, probe) - holdout @ w).square().mean()
        self.assertLess(float(err), 1e-8)

    def test_standardizer_avoids_constant_channel_explosion(self):
        x = torch.tensor([[1., 10.], [1., 14.], [1., 16.]], dtype=torch.float64)
        stats = standardizer(x)
        z = standardize(x, stats)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.allclose(z[:, 0], torch.zeros(3, dtype=torch.float64)))

    def test_pca_rank64_and_random_basis_are_deterministic(self):
        torch.manual_seed(2)
        x = torch.randn(256, 128, dtype=torch.float64)
        mean, pca, explained = pca_basis(x, 64)
        self.assertEqual(tuple(pca.shape), (128, 64))
        self.assertEqual(tuple(mean.shape), (1, 128))
        self.assertGreater(explained, 0.5)
        self.assertLessEqual(explained, 1.0 + 1e-9)
        q1 = random_basis(128, 64, 42)
        q2 = random_basis(128, 64, 42)
        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(q1.T @ q1, torch.eye(64, dtype=torch.float64),
                                   atol=1e-10, rtol=1e-10)

    def test_context_retained_with_equal_64_channel_controls(self):
        training = [mock_sample(256, 11), mock_sample(256, 12)]
        transforms = make_transform(training)
        sample = mock_sample(32, 21)
        probes = build_inputs(sample["base"], sample["fused"],
                              sample["optical"], sample["sar"], transforms)
        self.assertEqual(set(probes), set(PROBE_NAMES))
        self.assertEqual(tuple(probes["base_only"].shape), (32, 64))
        for name in ("e2_fused", "pca64", "random64"):
            self.assertEqual(tuple(probes[name].shape), (32, 128))
            torch.testing.assert_close(probes[name][:, :64], probes["base_only"])
        self.assertEqual(tuple(probes["concat128"].shape), (32, 192))

    def test_linear_probe_training_and_holdout_paths(self):
        training = [mock_sample(256, 3), mock_sample(256, 5)]
        transforms = make_transform(training)
        probes = train_probes(training, transforms)
        self.assertEqual(set(probes), set(PROBE_NAMES))
        val = mock_sample(128, 19)
        x = build_inputs(val["base"], val["fused"], val["optical"], val["sar"],
                         transforms)
        y = matrix_targets(val, transforms)
        errors = []
        for method in PROBE_NAMES:
            prediction = ridge_predict(x[method], probes[method])
            self.assertEqual(prediction.shape, y.shape)
            errors.extend(target_error_rows(
                val["scene"], method, y, prediction, val["valid_sar"]))
        result = summarize(errors, transforms)
        self.assertEqual(set(result["optical"]), set(PROBE_NAMES))
        self.assertGreater(
            result["optical"]["base_only"]["mean_nmse"],
            result["optical"]["concat128"]["mean_nmse"])
        self.assertGreater(
            result["sar"]["base_only"]["mean_nmse"],
            result["sar"]["concat128"]["mean_nmse"])

    def test_edge_proxy_is_not_used_as_supervision(self):
        points = torch.arange(100, dtype=torch.float64)
        truth = torch.zeros(100, 3, dtype=torch.float64)
        pred = (points / 100).unsqueeze(-1).expand(-1, 3)
        result = edge_residual_proxy("test", "method", truth, pred, points, None)
        self.assertGreater(result["high_to_low_residual"], 1)
        self.assertEqual(result["target"], "optical")

    def test_feature_grid_shape(self):
        feature = torch.arange(1*64*16*16, dtype=torch.float32).reshape(1,64,16,16)
        value = flatten_grid(feature, stride=4)
        self.assertEqual(tuple(value.shape), (16,64))
        self.assertEqual(value.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
