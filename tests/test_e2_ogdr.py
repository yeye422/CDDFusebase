"""Fast CPU checks for OGDR identity, locality, controls and E2 training hooks."""

import unittest

import torch

from direct_fusion import MultiScaleDetailFusion
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_ogdr import (MODES, OGDRE2, OpticalGuidedDetailReconstruction,
                     GROUPS, MAX_RESIDUAL_SCALE)


class OGDRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2026)

    def _features(self, h=11, w=13):
        return [torch.randn(1, 64, h, w) for _ in range(3)]

    def test_zero_start_preserves_original_fusion_exactly(self):
        original = MultiScaleDetailFusion().eval()
        inputs = self._features()
        expected = original(*inputs)
        for mode in MODES:
            module = OpticalGuidedDetailReconstruction(original, mode).eval()
            torch.testing.assert_close(module(*inputs), expected, rtol=0, atol=0)
            result, stats = module(*inputs, return_diagnostics=True)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(float(stats["update_rms"]), 0.0)
            off = module(*inputs, condition="off")
            torch.testing.assert_close(off, expected, rtol=0, atol=0)

    def test_constant_feature_has_zero_local_difference(self):
        module = OpticalGuidedDetailReconstruction(MultiScaleDetailFusion())
        for h, w in ((1, 1), (1, 3), (3, 1), (7, 9)):
            feature = torch.full((1, 64, h, w), 0.25)
            kernels = torch.full((1, GROUPS, 9, h, w), 1.0 / 9)
            gates = torch.ones((1, GROUPS, 1, h, w))
            delta = module._local_difference(feature, kernels, gates)
            self.assertEqual(tuple(delta.shape), tuple(feature.shape))
            self.assertFalse(delta.any())

    def test_optical_evidence_is_finite_and_keeps_signed_haar_activity(self):
        module = OpticalGuidedDetailReconstruction(MultiScaleDetailFusion())
        detail = torch.ones((1, 64, 8, 10))
        detail[..., :, 1::2] *= -1
        guide = module._optical_structure(detail)
        self.assertTrue(torch.isfinite(guide).all())
        self.assertEqual(guide.shape, (1, 3, 8, 10))
        self.assertGreaterEqual(float(guide[:, 2:].min()), 0)

    def test_dynamic_and_guided_parameters_get_gradients(self):
        for mode in ("dynamic", "guided", "shifted", "conv"):
            module = OpticalGuidedDetailReconstruction(MultiScaleDetailFusion(), mode)
            inputs = self._features()
            optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
            output = module(*inputs)
            output.square().mean().backward()
            grad = module.output_projection.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
            optimizer.step()
            optimizer.zero_grad()
            module(*inputs).square().mean().backward()
            self.assertIsNotNone(module.kernel_predictor[-1].weight.grad)
            self.assertTrue(torch.isfinite(module.kernel_predictor[-1].weight.grad).all())
            if mode in ("guided", "shifted"):
                self.assertIsNotNone(module.optical_projection.weight.grad)
                self.assertGreater(float(module.optical_projection.weight.grad.abs().sum()), 0)

    def test_alignment_control_changes_only_new_branch(self):
        module = OpticalGuidedDetailReconstruction(MultiScaleDetailFusion(), "guided").eval()
        torch.nn.init.normal_(module.output_projection.weight, std=0.02)
        inputs = self._features(9, 13)
        reference = module.keep_fusion(*inputs)
        off = module(*inputs, condition="off")
        aligned = module(*inputs, condition="aligned")
        shifted = module(*inputs, condition="shifted")
        torch.testing.assert_close(reference, off, rtol=0, atol=0)
        self.assertGreater(float((aligned - shifted).abs().sum()), 1e-8)
        self.assertTrue(torch.isfinite(aligned).all())
        self.assertLessEqual(MAX_RESIDUAL_SCALE, 0.1)

    def test_full_model_and_original_train_hooks(self):
        import run_e2_scratch_twostage as stages
        import train_lwpd as trainer
        from run_e2_ogdr import experiment

        torch.manual_seed(77)
        baseline = OpticalBaseAnchoredE2().eval()
        torch.manual_seed(77)
        candidate = OGDRE2("guided").eval()
        for name, value in baseline.state_dict().items():
            target = name.replace("detail_fusion.", "detail_fusion.keep_fusion.", 1) if name.startswith("detail_fusion.") else name
            torch.testing.assert_close(candidate.state_dict()[target], value, rtol=0, atol=0)

        optical = torch.rand(1, 1, 9, 11)
        sar = torch.rand(1, 1, 9, 11)
        torch.testing.assert_close(candidate(optical, sar), baseline(optical, sar), rtol=0, atol=0)

        pilot = experiment("guided", "pilot")
        args = stages.training_args(pilot)
        self.assertEqual(pilot.variant, candidate.variant)
        optimizer = trainer.make_optimizer(candidate, args)
        group = next(g for g in optimizer.param_groups if g["group_name"] == "detail_fusion")
        self.assertEqual(sum(id(p) == id(candidate.detail_fusion.output_projection.weight)
                             for p in group["params"]), 1)

        stages.configure(candidate, optimizer, args, reconstruction=True)
        self.assertFalse(any(p.requires_grad for p in candidate.detail_fusion.parameters()))
        stages.configure(candidate, optimizer, args, reconstruction=False)
        self.assertTrue(all(p.requires_grad for p in candidate.detail_fusion.parameters()))
        losses = trainer.DirectFusionLosses()
        loss, _ = trainer.forward_losses(candidate, losses, optical, sar,
                                         torch.ones_like(sar), args)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(candidate.detail_fusion.output_projection.weight.grad)
        self.assertGreater(float(candidate.detail_fusion.output_projection.weight.grad.abs().sum()), 0)

        copy = OGDRE2("guided").eval()
        copy.load_state_dict(candidate.state_dict(), strict=True)
        torch.testing.assert_close(copy(optical, sar), candidate(optical, sar), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
