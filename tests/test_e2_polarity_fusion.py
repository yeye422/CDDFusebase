import unittest
from types import SimpleNamespace

import torch

from e2_polarity_fusion import (
    GradientReconstruction,
    PolarityFusionE2,
    boundary_evidence,
    coarse_adjoint,
    coarse_pool,
    compose_field,
    forward_losses,
    gradient,
    gradient_adjoint,
)
from train_direct_fusion import DirectFusionLosses


class PolarityFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reversed_common_boundary_retains_optical_sign(self):
        optical = torch.full((1, 1, 32, 32), 0.2)
        optical[..., 15:] = 0.8
        sar = 1 - optical
        evidence = boundary_evidence(optical, sar, torch.ones_like(sar))
        reversed_mask = evidence["common_reversed"]
        self.assertGreater(int(reversed_mask.sum()), 0)
        self.assertFalse((reversed_mask & evidence["sar_private"]).any())
        logits = torch.zeros(1, 3, 32, 32)
        aligned = compose_field(evidence, logits)[0]
        signed_mix = compose_field(evidence, logits, "signed_mix")[0]
        torch.testing.assert_close(aligned[:, :1][reversed_mask], torch.full_like(aligned[:, :1][reversed_mask], 0.6))
        torch.testing.assert_close(signed_mix[:, :1][reversed_mask], torch.zeros_like(signed_mix[:, :1][reversed_mask]))

    def test_sar_private_boundary_has_an_independent_path(self):
        optical = torch.full((1, 1, 32, 32), 0.4)
        sar = torch.full_like(optical, 0.8)
        sar[..., 15:] = 0.2
        evidence = boundary_evidence(optical, sar, torch.ones_like(sar))
        mask = evidence["sar_private"]
        self.assertGreater(int(mask.sum()), 0)
        logits = torch.zeros(1, 3, 32, 32)
        field = compose_field(evidence, logits)[0]
        disabled = compose_field(evidence, logits, "no_sar_private")[0]
        self.assertTrue((field[:, :1][mask] < 0).all())
        self.assertTrue((disabled[:, :1][mask] == 0).all())

    def test_poisson_operators_and_implicit_derivative(self):
        torch.manual_seed(42)
        x = torch.rand(1, 1, 16, 20, dtype=torch.float64)
        field = gradient(x)
        coarse = coarse_pool(x)
        probe = torch.randn_like(x)
        dv, dl = torch.randn_like(field), torch.randn_like(coarse)
        torch.testing.assert_close((gradient(probe) * dv).sum(), (probe * gradient_adjoint(dv)).sum())
        torch.testing.assert_close((coarse_pool(probe) * dl).sum(), (probe * coarse_adjoint(dl)).sum())
        field.requires_grad_()
        coarse.requires_grad_()
        recovered, residual = GradientReconstruction.apply(field, coarse, 4.0, 80)
        self.assertLess(residual.item(), 2e-7)
        torch.testing.assert_close(recovered, x, rtol=2e-6, atol=2e-6)
        dv_gradient, dl_gradient = torch.autograd.grad((recovered * probe).sum(), (field, coarse))
        implicit = (dv_gradient * dv).sum() + (dl_gradient * dl).sum()
        with torch.no_grad():
            step = 1e-4
            plus = GradientReconstruction.apply(field + step * dv, coarse + step * dl, 4.0, 80)[0]
            minus = GradientReconstruction.apply(field - step * dv, coarse - step * dl, 4.0, 80)[0]
            finite_difference = ((plus - minus) * probe).sum() / (2 * step)
        torch.testing.assert_close(implicit, finite_difference, rtol=2e-3, atol=2e-3)

    def test_invalid_sar_and_odd_shape(self):
        model = PolarityFusionE2().eval()
        optical = torch.full((1, 1, 17, 19), 0.45)
        sar = torch.full_like(optical, float("nan"))
        with torch.no_grad():
            fused, data = model(optical, sar, return_features=True)
        self.assertEqual(fused.shape, optical.shape)
        torch.testing.assert_close(fused, optical)
        self.assertEqual(data["stats"]["sar_private_fraction"].item(), 0)

    def test_full_fusion_loss_backpropagates(self):
        torch.manual_seed(42)
        model = PolarityFusionE2()
        optical = torch.full((1, 1, 32, 36), 0.2)
        optical[..., 15:] = 0.8
        sar = 1 - optical
        sar[..., 26:] = 0.6
        loss, stats = forward_losses(model, DirectFusionLosses(), optical, sar,
                                     torch.ones_like(sar), SimpleNamespace(reconstruction_weight=0.1))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats["common_reversed_fraction"].item(), 0)
        self.assertGreater(stats["sar_private_fraction"].item(), 0)
        loss.backward()
        for weight in (model.detail_fusion.amplitude[-1].weight, model.detail_fusion.coarse[-1].weight):
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
