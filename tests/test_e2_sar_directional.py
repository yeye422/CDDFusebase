import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_sar_directional import INITIAL_ATTENUATION, SARDirectionalE2, SARDirectionalShrinkage
from lwpd import HaarDWT2D
from test_OS import forward_variant, load_variant_models
from train_direct_fusion import DirectFusionLosses
from train_lwpd import forward_losses


class SARDirectionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_directional_support_is_phase_robust_and_requires_spatial_coverage(self):
        module = SARDirectionalShrinkage()
        haar = HaarDWT2D()
        for edge_row in (12, 13):
            edge = torch.full((1, 1, 25, 27), 0.25)
            edge[..., edge_row:, :] = 0.75
            impulse = torch.full_like(edge, 0.25)
            impulse[..., edge_row, 13] = 0.75
            support, eligible = module.support(edge, haar)
            isolated, _ = module.support(impulse, haar)
            self.assertEqual(support.shape, (1, 3, 13, 14))
            self.assertTrue(eligible.all())
            self.assertGreater(support[0, 0, edge_row // 2, 5].item(), 0.99)
            self.assertLess(isolated[0, 0].max().item(), 0.5)
            self.assertEqual(support[0, 1].max().item(), 0.0)

    def test_shrinkage_preserves_low_band_and_mean_control_matches_budget(self):
        module = SARDirectionalShrinkage()
        haar = HaarDWT2D()
        torch.manual_seed(42)
        detail = torch.randn(2, 4, 18, 20)
        sar = torch.rand(2, 1, 18, 20)
        sar[:, :, :4, :] = 0.0
        original, _ = haar(detail)
        spatial, stats = module(sar, detail, haar)
        uniform, uniform_stats = module(sar, detail, haar, "mean")
        corrected, _ = haar(spatial)
        torch.testing.assert_close(corrected[0], original[0])
        for before, after in zip(original[1:], corrected[1:]):
            self.assertTrue((after.abs() <= before.abs() + 1e-6).all())
            self.assertTrue((after.abs() >= (1.0 - INITIAL_ATTENUATION) * before.abs() - 1e-6).all())
        torch.testing.assert_close(spatial[:, :, :4], detail[:, :, :4], rtol=0, atol=0)
        torch.testing.assert_close(stats["sar_attenuation_mean"], uniform_stats["sar_attenuation_mean"])
        self.assertFalse(torch.equal(spatial, uniform))
        self.assertIs(module(sar, detail, haar, "off")[0], detail)
        self.assertTrue(torch.equal(module(torch.zeros_like(sar), detail, haar)[0], detail))

    def test_source_reconstruction_and_fusion_training(self):
        torch.manual_seed(42)
        baseline = OpticalBaseAnchoredE2()
        torch.manual_seed(42)
        model = SARDirectionalE2()
        optical = torch.rand(1, 1, 17, 19)
        sar = torch.rand_like(optical)
        with torch.no_grad():
            for original, candidate in zip(baseline.encode_pair(optical, sar), model.encode_pair(optical, sar)):
                self.assertTrue(torch.equal(original, candidate))
            model.sar_detail_mode = "off"
            self.assertTrue(torch.equal(baseline(optical, sar), model(optical, sar)))
        model.sar_detail_mode = "spatial"
        losses = DirectFusionLosses()
        args = SimpleNamespace(gradient_weight=10.0, reconstruction_weight=0.1)
        total, parts = forward_losses(model, losses, optical, sar, torch.ones_like(sar), args)
        total.backward()
        gradient = model.sar_directional.strength_logits.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertGreater(parts["sar_attenuation_mean"].item(), 0.0)

    def test_checkpoint_uses_normal_inference_path(self):
        model = SARDirectionalE2().eval()
        optical = torch.rand(1, 1, 17, 19)
        sar = torch.rand_like(optical)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "e2.pth"
            torch.save({"args": {"variant": model.variant}, "model": model.state_dict()}, checkpoint)
            models, variant = load_variant_models(checkpoint, "cpu")
            self.assertEqual(variant, SARDirectionalE2.VARIANT)
            with torch.no_grad():
                actual, _ = forward_variant(models, variant, optical, sar)
                self.assertTrue(torch.equal(actual, model(optical, sar)))


if __name__ == "__main__":
    unittest.main()
