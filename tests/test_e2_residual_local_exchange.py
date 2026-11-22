import io
import unittest

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_local_correspondence import LocalCorrespondenceE2
from e2_residual_local_exchange import ResidualLocalExchangeE2
from run_e2_local_correspondence_full import FULL, RESIDUAL
from test_OS import load_variant_models


class ResidualLocalExchangeTests(unittest.TestCase):
    def test_seeded_model_matches_e2_exactly(self):
        torch.manual_seed(42)
        baseline = OpticalBaseAnchoredE2().eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        model = RESIDUAL.build_model().eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        head = model.detail_fusion
        self.assertEqual([b[0].dilation for b in head.scale_branches], [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(head.residual_projection.weight.count_nonzero().item(), 0)
        o, s = torch.rand(1, 1, 19, 21), torch.rand(1, 1, 19, 21)
        s[..., :6] = 0
        with torch.no_grad():
            torch.testing.assert_close(model(o, s), baseline(o, s), rtol=0, atol=0)

    def test_projection_changes_only_third_branch_and_respects_no_match(self):
        torch.manual_seed(37)
        head = ResidualLocalExchangeE2().detail_fusion
        features = torch.randn(1, 64, 13, 15)
        base = torch.randn_like(features)
        valid = torch.ones_like(features[:, :1], dtype=torch.bool)
        with torch.no_grad():
            head.residual_projection.weight[:, :, 0, 0].copy_(torch.eye(32))
        aggregate_inputs = []
        hook = head.output.register_forward_pre_hook(lambda _, args: aggregate_inputs.append(args[0]))
        try:
            out, stats = head(features, features, base, sar_valid=valid, return_diagnostics=True)
            head(features, features, base, sar_valid=~valid)
        finally:
            hook.remove()
        active, rejected = aggregate_inputs
        width = head.residual_projection.out_channels
        start, end = 4 * width, 5 * width  # optical, SAR, dilation 1, dilation 2, then dilation 3
        torch.testing.assert_close(active[:, :start], rejected[:, :start], rtol=0, atol=0)
        torch.testing.assert_close(active[:, end:], rejected[:, end:], rtol=0, atol=0)
        o, s, _ = head._project_inputs(features, features, base)
        third = head.scale_branches[2](torch.cat((o, s), dim=1))
        torch.testing.assert_close(rejected[:, start:end], third, rtol=0, atol=0)
        self.assertGreater(float(stats["residual_rms"]), 0)
        self.assertGreater(float((active[:, start:end] - third).abs().sum()), 0)
        out.square().mean().backward()
        for projection in (head.sar_value, head.optical_value, head.residual_projection):
            grad = projection.weight.grad
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)

    def test_evaluator_recognizes_both_checkpoint_layouts(self):
        for exp, num_branches, expected in ((FULL, 2, LocalCorrespondenceE2), (RESIDUAL, 3, ResidualLocalExchangeE2)):
            model = exp.build_model().eval()
            file = io.BytesIO()
            torch.save(dict(args=dict(variant=exp.variant), model=model.state_dict()), file)
            file.seek(0)
            models, variant = load_variant_models(file, "cpu")
            self.assertIs(type(models["direct"]), expected)
            self.assertEqual(variant, exp.variant)
            self.assertEqual(len(models["direct"].detail_fusion.scale_branches), num_branches)
