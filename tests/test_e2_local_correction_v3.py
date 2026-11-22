import unittest
from unittest.mock import patch

import torch

from e2_local_correction_v3 import LocalCorrectionV3, LocalFusionObjective, conflict_mask
import run_e2_local_correction_v3 as run
from test_OS import DIRECT_MODEL_TYPES
from train_direct_fusion import DirectFusionLosses


class V3ConflictTests(unittest.TestCase):
    def setUp(self):
        self.optical = torch.full((1, 1, 39, 47), .2)
        self.optical[..., 20:] = .8
        self.sar = torch.full_like(self.optical, .8)
        self.sar[..., 24:] = .2
        self.valid = torch.ones_like(self.optical, dtype=torch.bool)

    def test_mask_requires_both_sources_opposite_and_separated(self):
        o, s, valid = self.optical, self.sar, self.valid
        mask = conflict_mask(o, s, valid)
        self.assertTrue(mask[..., 18:23, 20:24].any())
        self.assertFalse(mask[..., :10].any())
        for other in (1 - s, 1 - o, torch.ones_like(s)):
            self.assertFalse(conflict_mask(o, other, valid).any())
        torch.testing.assert_close(mask.transpose(-1, -2), conflict_mask(o.transpose(-1, -2),
                                   s.transpose(-1, -2), valid.transpose(-1, -2)))
        # Nodata adjacent to an otherwise valid conflict must not create a peak.
        valid[..., 21:23] = False
        self.assertFalse(conflict_mask(o, s * valid, valid).any())

    def test_actual_weighting_and_identity_on_nonconflict(self):
        criterion = DirectFusionLosses()
        for s in (self.optical, self.sar):
            objective = LocalFusionObjective(criterion, self.optical, s, self.valid)
            im = torch.rand_like(s, requires_grad=True)
            out, intensity, grad = objective.fusion(self.optical, s, im, self.valid, 10)
            base, reference_i, reference_g = criterion.fusion(self.optical, s, im, self.valid, 10)
            torch.testing.assert_close(intensity, reference_i)
            if objective.mask.any():
                removed = (criterion.sobel(im) - objective.gradient_target).abs() * objective.mask
                torch.testing.assert_close(grad, reference_g - removed.mean())
            else:
                torch.testing.assert_close(out, base)
            out.backward()
            self.assertTrue(torch.isfinite(im.grad).all())
            self.assertFalse(objective.weight.requires_grad)

    def test_shared_rule_is_used_for_both_outputs(self):
        import argparse
        import e2_local_correction_v3 as candidate
        o, s = self.optical, self.sar
        model = LocalCorrectionV3().eval()
        calls = []
        original = LocalFusionObjective.fusion
        def record(obj, *args):
            calls.append(obj)
            return original(obj, *args)
        with patch.object(LocalFusionObjective, "fusion", record):
            loss, stats = candidate.fusion_forward(model, DirectFusionLosses(), o, s, self.valid.float(),
                                                   argparse.Namespace(gradient_weight=10., reconstruction_weight=.1))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], calls[1])
        self.assertGreater(float(stats["sobel_conflict_fraction"]), 0)
        self.assertIs(DIRECT_MODEL_TYPES[run.PILOT.variant], LocalCorrectionV3)

    def test_pair_registration_has_dedicated_group_and_paths(self):
        entries = [dict(checkpoint=exp.final.as_posix(), result_dir=(run.PILOT.result / name).as_posix())
                   for name, exp in (("e2", run.previous.BASE), ("candidate", run.PILOT))]
        with patch.object(run, "register_many") as save:
            run.register_v3_pair(entries)
        pair, = save.call_args.args
        self.assertEqual([e["id"] for e in pair], [run.GROUP+"-e2", run.GROUP+"-candidate"])
        self.assertTrue(all(e["comparable_group"] == run.GROUP for e in pair))
