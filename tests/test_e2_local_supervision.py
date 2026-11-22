import unittest

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_e2_local_supervision import objective, signed_profile, source_shortfall, spot_contrast
from diagnose_e2_primitive_render import CENTER, sampling_grid
from train_direct_fusion import DirectFusionLosses


class LocalSupervisionTests(unittest.TestCase):
    def test_legacy_term_matches_existing_training_loss(self):
        generator = torch.Generator().manual_seed(17)
        optical, sar, image = (torch.rand((1, 1, 24, 24), generator=generator) for _ in range(3))
        valid = sar > 0.4
        _, coordinates = sampling_grid(np.array([[6, 12], [18, 12]]))
        center = coordinates[:, CENTER]
        result = objective(image, (optical, sar), valid, valid & False,
                           (center, 1, .15), (center, .03))
        reference = DirectFusionLosses().fusion(optical, sar, image, valid, 10.0)[0]
        torch.testing.assert_close(result["legacy"], reference)

    def test_replacement_prefers_correct_location_and_sign(self):
        shape = (1, 1, 64, 64)
        source = torch.full(shape, .25)
        source[..., 26:, :] = .68
        sar = torch.full(shape, .25)
        mask = torch.ones(shape, dtype=torch.bool)
        _, true_coords = sampling_grid(np.array([[7, 26], [55, 26]]))
        _, false_coords = sampling_grid(np.array([[7, 42], [55, 42]]))
        target, guard = (true_coords[:, CENTER], 1, .20), (false_coords[:, CENTER], .03)
        restored = source.clone().requires_grad_()
        shifted = sar.clone()
        shifted[..., 42:, :] = .68
        reversed_image = .93 - source
        scores = [objective(x, (source, sar), mask, mask, target, guard)
                  for x in (restored, shifted, reversed_image)]
        self.assertLess(scores[0]["replacement"].item(), scores[1]["replacement"].item())
        self.assertGreater(scores[1]["guard"].item(), 0)
        self.assertGreater(scores[2]["source_target"].item(), .4)
        signed_profile(restored, true_coords[:, CENTER]).mean().backward()
        self.assertTrue(torch.isfinite(restored.grad).all())
        self.assertGreater(restored.grad.abs().sum().item(), 0)

    def test_excluding_sobel_halo_preserves_the_outside_term(self):
        image = torch.full((1, 1, 48, 48), .45)
        support = torch.zeros_like(image)
        support[..., 17:24, 7:32] = 1
        edited = image + support * .2
        region = F.max_pool2d(support, 3, stride=1, padding=1) > 0
        _, coordinates = sampling_grid(np.array([[7, 20], [32, 20]]))
        center = coordinates[:, CENTER]
        target, guard = (center, 1, .1), (center, .01)
        scores = [objective(x, (image, image), image > 0, region, target, guard) for x in (image, edited)]
        torch.testing.assert_close(scores[0]["legacy_outside"], scores[1]["legacy_outside"])

    def test_known_sar_spot_is_preferred_to_equal_brightness_at_wrong_location(self):
        y, x = torch.meshgrid(torch.arange(40), torch.arange(40), indexing="ij")
        radius = (x-13).square() + (y-18).square()
        core, ring = radius <= 4, (radius >= 9) & (radius <= 25)
        blank = torch.full((1, 1, 40, 40), .2)
        retained = blank + core * .5
        shifted = blank + core.roll(14, dims=1) * .5
        self.assertAlmostEqual((retained - blank).abs().sum().item(), (shifted - blank).abs().sum().item())
        floor = .25
        self.assertLess(source_shortfall(spot_contrast(retained, core, ring), 1, floor),
                        source_shortfall(spot_contrast(shifted, core, ring), 1, floor))
        self.assertLess(source_shortfall(spot_contrast(retained, core, ring), 1, floor),
                        source_shortfall(spot_contrast(blank, core, ring), 1, floor))


if __name__ == "__main__":
    unittest.main()
