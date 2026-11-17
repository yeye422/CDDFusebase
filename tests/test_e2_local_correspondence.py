"""Regression checks for local matching and the source-preserving fusion path."""

import unittest

import torch

from e2_local_correspondence import LocalCorrespondenceDetailFusion, OFFSETS, reciprocal_weights, reverse_edges


class TestLocalCorrespondence(unittest.TestCase):
    def test_unique_offset_has_matching_inverse(self):
        optical = torch.zeros(1, 2, 3, 4)
        sar = torch.zeros_like(optical)
        optical[:, 0, 1, 1] = 1
        sar[:, 0, 1, 2] = 1
        sar[:, 1, 1, 1] = 1
        valid = torch.ones_like(sar[:, :1], dtype=torch.bool)
        forward, backward = reciprocal_weights(optical, sar, valid)
        self.assertGreater(float(forward[0, OFFSETS.index((0, 1)), 1, 1]), 0.9)
        self.assertEqual(int(torch.count_nonzero(forward)), 1)
        torch.testing.assert_close(backward, reverse_edges(forward, 0.0))
        valid[..., 1, 2] = False
        for weights in reciprocal_weights(optical, sar, valid):
            self.assertFalse(weights.any())

    def test_ambiguous_neighbors_keep_less_exchange_mass(self):
        optical = sar = torch.ones(1, 1, 5, 5)
        valid = torch.ones_like(sar, dtype=torch.bool)
        forward, _ = reciprocal_weights(optical, sar, valid)
        center_mass = forward.sum(1)[0, 2, 2]
        self.assertLess(float(center_mass), 0.12)
        self.assertTrue((forward.sum(1) <= 1).all())

    def test_reject_does_not_gate_original_values(self):
        model = LocalCorrespondenceDetailFusion().eval()
        optical, sar, base = (torch.randn(2, 64, 15, 17) for _ in range(3))
        valid = torch.zeros_like(sar[:, :1], dtype=torch.bool)
        output_inputs = []
        handle = model.output.register_forward_pre_hook(lambda _, args: output_inputs.append(args[0]))
        try:
            fused, stats = model(optical, sar, base, sar_valid=valid, return_diagnostics=True)
        finally:
            handle.remove()
        self.assertEqual(fused.shape, optical.shape)
        aggregate = output_inputs[0]
        torch.testing.assert_close(aggregate[:, :32], model.optical_projection(optical))
        torch.testing.assert_close(aggregate[:, 32:64], model.sar_projection(sar))
        self.assertFalse(aggregate[:, 128:160].any())
        self.assertEqual(float(stats["match_mass"]), 0)

    def test_descriptors_keep_native_high_band_signal(self):
        model = LocalCorrespondenceDetailFusion()
        zero = torch.zeros(1, 64, 8, 8)
        detail = model.haar.inverse((zero, torch.ones_like(zero), zero, zero))
        pooled = torch.nn.functional.avg_pool2d(detail, 2)
        self.assertFalse(pooled.any())
        *_, (forward, _) = model._describe(detail, detail, None)
        self.assertTrue(forward.any())


if __name__ == "__main__":
    unittest.main()
