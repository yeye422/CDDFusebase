import unittest
from unittest.mock import patch

import kornia
import torch

from train_direct_fusion import DirectFusionLosses
from utils.legacy_dssim import LegacyDSSIM


class LegacyDSSIMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_negative_similarity_uses_historical_half_unit_ceiling(self):
        torch.manual_seed(42)
        source = torch.rand(1, 1, 17, 19)
        image = (1 - source).requires_grad_(True)
        value = LegacyDSSIM()(source, image)
        self.assertEqual(value.shape, source.shape)
        self.assertTrue((value <= 0.5).all())
        self.assertGreater(float(value.mean()), 0.49)
        value.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_matches_legacy_kornia_values_and_gradients_when_available(self):
        original = getattr(kornia.losses, "SSIM", None)
        if original is None:
            self.skipTest("Historical Kornia SSIM is absent in this environment")
        torch.manual_seed(42)
        source = torch.rand(1, 2, 17, 19)
        image = torch.rand_like(source).requires_grad_(True)
        before = original(11, reduction="none")(source, image)
        after = LegacyDSSIM()(source, image)
        torch.testing.assert_close(before, after, rtol=2e-5, atol=1e-6)
        old_gradient = torch.autograd.grad(before.mean(), image)[0]
        new_gradient = torch.autograd.grad(after.mean(), image)[0]
        torch.testing.assert_close(old_gradient, new_gradient, rtol=2e-5, atol=1e-7)

    def test_missing_old_api_uses_compatible_loss_without_installing_packages(self):
        with patch.object(kornia.losses, "SSIM", None, create=True):
            losses = DirectFusionLosses()
        self.assertIsInstance(losses.ssim, LegacyDSSIM)
        image = torch.rand(1, 1, 17, 19)
        torch.testing.assert_close(losses.ssim(image, image), torch.zeros_like(image), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
