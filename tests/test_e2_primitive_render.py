import unittest

import numpy as np
import torch

from diagnose_e2_primitive_render import (
    BUDGET_TOLERANCE, audit_scene, boundary, matched_render, pixel_grid, spot,
)


class PrimitiveRenderTests(unittest.TestCase):
    def test_boundary_orientation_and_support(self):
        grid = pixel_grid((40, 40))
        start, end = torch.tensor([5., 20.]), torch.tensor([35., 20.])
        residual, window, _ = boundary(grid, start, end, 40)
        self.assertGreater(residual[22, 20].item(), 0)
        self.assertLess(residual[18, 20].item(), 0)
        self.assertEqual(residual[20, 20].item(), 0)
        self.assertEqual(residual[28, 20].item(), 0)
        self.assertEqual(window[20, 4].item(), 0)

    def test_boundary_location_and_contrast_have_finite_gradients(self):
        grid = torch.tensor([[[12.3, 16.4], [15.2, 21.8]]], dtype=torch.float64)
        start = torch.tensor([4.7, 19.2], dtype=torch.float64, requires_grad=True)
        end = torch.tensor([31.6, 20.1], dtype=torch.float64, requires_grad=True)
        contrast = torch.tensor(25., dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b, c: boundary(grid, a, b, c)[0], (start, end, contrast)))

    def test_point_is_compact_and_location_is_differentiable(self):
        grid = pixel_grid((35, 35))
        center = torch.tensor([17., 18.], requires_grad=True)
        values = spot(grid, center, torch.tensor([2., 3.]), torch.tensor(.25), 30.)
        self.assertEqual(values[18, 17].item(), 30.)
        self.assertLess(values[0, 0].item(), .01)
        values[19, 19].backward()
        self.assertTrue(torch.isfinite(center.grad).all())
        self.assertGreater(center.grad.abs().sum().item(), 0)

    def test_quantized_budget_matching_handles_saturation(self):
        base = np.tile(np.arange(240, 256, dtype=np.uint8), (16, 1))
        mask = np.zeros(base.shape, bool)
        mask[3:-3, 2:-2] = True
        residuals = {"dense": np.where(mask, 11., 0.), "primitive": np.where(mask, -18., 0.)}
        images, budgets, target = matched_render(base, residuals, mask)
        self.assertGreater(target, 0)
        for key, image in images.items():
            np.testing.assert_array_equal(image[~mask], base[~mask])
            self.assertAlmostEqual(
                np.abs(image[mask].astype(float) - base[mask]).mean(), budgets[key]["mae"])
            self.assertLessEqual(abs(budgets[key]["mae"] - target), BUDGET_TOLERANCE)

    def test_negative_annotation_is_never_drawn(self):
        source = np.full((80, 80), 60, np.uint8)
        source[36:] = 180
        background = np.full_like(source, 100)
        rows = [
            dict(scene="test", id="A", label="connected", x1=12, y1=36, x2=64, y2=36),
            dict(scene="test", id="B", label="disconnected", x1=16, y1=57, x2=59, y2=57),
        ]
        images, budgets, probes = audit_scene((source, source), background, rows, {})
        self.assertTrue(any(row["injected"] and row["probe"] == "A" for row in probes))
        self.assertFalse(any(row["injected"] and row["probe"] == "B" for row in probes))
        self.assertTrue(all(row["outside_pixels_changed"] == 0 for row in budgets))
        self.assertGreater(int(images["primitive"][38, 40]), int(images["primitive"][34, 40]))
        np.testing.assert_array_equal(images["primitive"][49:], background[49:])


if __name__ == "__main__":
    unittest.main()
