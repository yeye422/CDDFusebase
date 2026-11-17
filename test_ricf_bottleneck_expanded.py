"""CPU-only unit tests of streamed expanded RICF statistics (no QXS images)."""

import unittest

import torch

from diagnose_ricf_bottleneck_expanded import (
    CrossMoments, RunningMoments, comparative_summary, pick_scenes
)
from diagnose_ricf_bottleneck_preflight import pca_basis, ridge_fit, ridge_predict, standardizer


class ExpandedRICFTests(unittest.TestCase):

    def test_stream_standardizer_matches_batch(self):
        torch.manual_seed(21)
        x = torch.randn(97, 9, dtype=torch.float64)
        x[:, 3] = 0.5
        accumulator = RunningMoments()
        for chunk in torch.tensor_split(x, 7):
            accumulator.update(chunk)
        mean, scale = accumulator.get_stats()
        want_mean, want_scale = standardizer(x)
        torch.testing.assert_close(mean, want_mean, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(scale, want_scale, atol=1e-12, rtol=1e-12)

    def test_stream_pca_matches_batch_projector(self):
        torch.manual_seed(22)
        x = torch.randn(512, 12, dtype=torch.float64)
        x[:, 8:] += x[:, :4]
        accumulator = CrossMoments()
        for chunk in torch.tensor_split(x, 13):
            accumulator.update(chunk)
        avg, basis, ratio = accumulator.pca(6)
        ref_avg, ref_basis, ref_ratio = pca_basis(x, 6)
        torch.testing.assert_close(avg, ref_avg, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            basis @ basis.T, ref_basis @ ref_basis.T,
            atol=1e-9, rtol=1e-9)
        self.assertAlmostEqual(ratio, ref_ratio, places=10)

    def test_stream_ridge_matches_batch(self):
        torch.manual_seed(23)
        x = torch.randn(555, 13, dtype=torch.float64)
        y = torch.randn(555, 7, dtype=torch.float64) + x[:, :7] * 0.7
        accumulator = CrossMoments()
        for xx, yy in zip(torch.tensor_split(x, 9),
                          torch.tensor_split(y, 9)):
            accumulator.update(xx, yy)
        result = ridge_predict(x[:21], accumulator.ridge(ridge=0.027))
        reference = ridge_predict(x[:21], ridge_fit(x, y, ridge=0.027))
        torch.testing.assert_close(result, reference, atol=1e-10, rtol=1e-10)

    def test_selection_is_deterministic_and_spread(self):
        ids = list(range(1024))
        a = pick_scenes(ids, 128, "even")
        b = pick_scenes(ids, 128, "even")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 128)
        self.assertEqual(len(set(a)), 128)
        self.assertGreater(int(a[-1]), 1000)
        self.assertEqual(pick_scenes(ids, 8, "first"), [str(i) for i in range(8)])
        with self.assertRaises(ValueError):
            pick_scenes(ids, 1025, "even")

    def test_paired_comparisons(self):
        rows = []
        for scene in ("1", "2", "3"):
            for target in ("optical", "sar"):
                for method, mse in (
                    ("base_only", 1.0), ("e2_fused", 0.3),
                    ("pca64", 0.03), ("random64", 0.1),
                    ("concat128", 0.01)):
                    rows.append({"scene":scene,"target":target,
                                 "method":method,"nmse":mse})
        report = comparative_summary(rows)
        self.assertEqual(report["optical"]["pca64_better_scene_count"], 3)
        self.assertAlmostEqual(
            report["sar"]["mean_e2_to_pca_nmse_ratio"],10,places=8)


if __name__ == "__main__":
    unittest.main()
