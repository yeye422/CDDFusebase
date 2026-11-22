import unittest

import numpy as np
import torch

from diagnose_e2_primitive_dose import (
    BUDGET_TOLERANCE, MAX_PIXEL_EDIT, read_probe, reflect_profile, render_support,
)
from diagnose_e2_primitive_render import pixel_grid, sampling_grid, spot


class PrimitiveDoseTests(unittest.TestCase):
    def test_rounding_and_pixel_cap_apply_to_the_actual_edit(self):
        base = np.tile(np.arange(235, 255, dtype=np.uint8), (10, 1)).ravel()
        residual = np.tile(np.linspace(-1, 3, 20, dtype=np.float32), 10)
        image, stats = render_support(base, residual, target=4)
        actual = np.abs(image.astype(float) - base)
        self.assertTrue(stats["matched"])
        self.assertLessEqual(abs(actual.mean() - 4), BUDGET_TOLERANCE)
        self.assertLessEqual(actual.max(), MAX_PIXEL_EDIT)
        self.assertEqual(actual.max(), stats["max_pixel_edit"])

    def test_unreachable_budget_is_reported(self):
        base = np.full(20, 253, np.uint8)
        _, stats = render_support(base, np.ones(20), target=4)
        self.assertFalse(stats["matched"])
        self.assertEqual(stats["achieved"], 2)

    def test_empirical_profile_tracks_polarity_and_preserves_symmetry(self):
        image = np.full((64, 64), 70, np.uint8)
        image[35:] = 170
        offsets = np.arange(-6, 7)
        segment = np.array([[10, 35], [54, 35]])
        residual = reflect_profile(image, offsets, segment, False)
        np.testing.assert_allclose(residual, -residual[::-1])
        self.assertLess(residual[0], 0)
        self.assertEqual(residual[6], 0)
        self.assertGreater(residual[-1], 0)
        self.assertIsNone(reflect_profile(np.zeros_like(image), offsets, segment, True))

    def test_independent_parallel_edge_readout_detects_an_erasure_near_a_point(self):
        grid = pixel_grid((64, 64))
        texture = spot(grid, torch.tensor([27., 37.]), torch.tensor([1., 1.]), torch.tensor(0.), 30.).numpy()
        base = np.full((64, 64), 60.0) + texture
        base[31:38] += 80
        erased = base.copy()
        erased[38:] += 80
        _, neighbor = sampling_grid(np.array([[10, 38], [54, 38]]))
        before = read_probe(np.rint(base).astype(np.uint8), neighbor, -1)
        after = read_probe(np.rint(erased).astype(np.uint8), neighbor, -1)
        self.assertGreater(before["center_jump_gray"], 70)
        self.assertLess(abs(after["center_jump_gray"]), 1)
        self.assertGreater(before["center_coverage"], after["center_coverage"])


if __name__ == "__main__":
    unittest.main()
