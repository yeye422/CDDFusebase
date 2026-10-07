import unittest

import torch

from phase_antisymmetry import MonogenicPhaseAntisymmetry, local_phase_bridge


class PhaseAntisymmetryTests(unittest.TestCase):
    def setUp(self):
        self.model = MonogenicPhaseAntisymmetry(reflect_padding=16)

    def test_constant_image_has_negligible_response(self):
        image = torch.full((1, 1, 64, 64), 0.5)
        result = self.model(image)
        self.assertLess(float(result["fa_mean"].max()), 1e-4)

    def test_step_edge_is_stronger_than_flat_region(self):
        image = torch.full((1, 1, 64, 64), 0.2)
        image[:, :, :, 32:] = 0.8
        result = self.model(image)
        edge = result["fa_mean"][:, :, 8:-8, 29:35].mean()
        flat = torch.cat(
            (
                result["fa_mean"][:, :, 8:-8, 8:20].flatten(),
                result["fa_mean"][:, :, 8:-8, 44:56].flatten(),
            )
        ).mean()
        self.assertGreater(float(edge), float(flat) + 0.1)

    def test_invalid_region_is_zero(self):
        torch.manual_seed(41)
        image = torch.rand(1, 1, 64, 64)
        valid = torch.ones_like(image)
        valid[:, :, :, :12] = 0.0
        result = self.model(image, valid)
        self.assertEqual(float(result["fa_mean"][:, :, :, :12].max()), 0.0)

    def test_outputs_are_finite_and_bounded(self):
        torch.manual_seed(43)
        result = self.model(torch.rand(2, 1, 48, 48))
        self.assertTrue(torch.all(torch.isfinite(result["fa_mean"])))
        self.assertGreaterEqual(float(result["fa_mean"].min()), 0.0)
        self.assertLessEqual(float(result["fa_mean"].max()), 1.0)

    def test_local_bridge_cannot_create_remote_phase_structure(self):
        roa = torch.zeros(1, 1, 17, 17)
        roa[:, :, 8, 8] = 0.8
        phase = torch.zeros_like(roa)
        phase[:, :, 8, 10] = 1.0
        phase[:, :, 2, 2] = 1.0
        bridge, support = local_phase_bridge(roa, phase, kernel_size=5)
        self.assertGreater(float(bridge[:, :, 8, 10]), 0.0)
        self.assertEqual(float(bridge[:, :, 2, 2]), 0.0)
        self.assertEqual(float(support[:, :, 2, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
