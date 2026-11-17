"""CPU invariants for frozen E2 OSCF / original-loss gradient diagnosis.

Requires ordinary project dependencies, but never loads a QXS scene/checkpoint.
"""
import unittest

import torch

from diagnose_e2_subspace_gradients import (
    cca_spectrum, grouped_gradient_stats, loss_partition, roll_grid,
)
from train_direct_fusion import DirectFusionLosses


class SubspaceGradientDiagnosticTests(unittest.TestCase):
    def test_correspondence_has_higher_cca_than_independent(self):
        torch.manual_seed(42)
        x = torch.randn(512, 8, dtype=torch.double)
        aligned = x + 0.05 * torch.randn_like(x)
        wrong = torch.randn_like(x)
        correct = cca_spectrum(x, aligned)
        shifted = cca_spectrum(x, wrong)
        self.assertEqual(len(correct), 8)
        self.assertGreater(float(correct[:4].mean()), 0.9)
        self.assertLess(float(shifted[:4].mean()), 0.4)

    def test_roll_control_preserves_feature_histogram(self):
        original = torch.arange(8 * 16).reshape(8 * 16, 1).double()
        wrong = roll_grid(original, 8, 16)
        self.assertEqual(set(original.flatten().tolist()), set(wrong.flatten().tolist()))
        self.assertGreater((original - wrong).abs().sum(), 0.0)

    def test_subspace_invalid_shapes_fail(self):
        with self.assertRaises(ValueError):
            cca_spectrum(torch.randn(64, 8), torch.randn(64, 7))
        with self.assertRaises(ValueError):
            cca_spectrum(torch.randn(16, 8), torch.randn(16, 8))

    def test_source_loss_partition_equals_archived_hard_max(self):
        torch.manual_seed(123)
        losses = DirectFusionLosses()
        optical = torch.rand(1, 1, 32, 32)
        sar = torch.rand_like(optical)
        sar[..., :8, :8] = 0
        valid = (sar > 0).float()
        fused = torch.nn.Parameter(torch.rand_like(optical))
        left, right, info = loss_partition(
            optical, sar, fused, valid, losses, gradient_weight=10.0)
        original = losses.fusion(optical, sar, fused, valid, 10.0)[0]
        torch.testing.assert_close(left + right, original, atol=1e-6, rtol=1e-6)
        self.assertGreater(info["sar_gradient_winner_fraction"], 0)
        self.assertLess(info["sar_gradient_winner_fraction"], 1)
        left_grad = torch.autograd.grad(left, fused, retain_graph=True)[0]
        right_grad = torch.autograd.grad(right, fused, retain_graph=True)[0]
        combined = torch.autograd.grad(original, fused)[0]
        torch.testing.assert_close(left_grad + right_grad, combined,
                                   atol=1e-5, rtol=1e-5)

    def test_grouped_gradient_conflict_computes_correct_cosine(self):
        p = torch.nn.Parameter(torch.ones(2))
        q = torch.nn.Parameter(torch.ones(3))
        named = [("output.a", p), ("optical_projection.b", q)]
        optical = [torch.tensor([1.0, 2.0]), torch.zeros(3)]
        sar = [torch.tensor([-1.0, -2.0]), torch.zeros(3)]
        row = grouped_gradient_stats(named, optical, sar, "all_detail")
        self.assertAlmostEqual(row["grad_cosine"], -1.0)
        self.assertTrue(row["grad_conflict"])
        self.assertEqual(row["active_tensors"], 2)


if __name__ == "__main__":
    unittest.main()
