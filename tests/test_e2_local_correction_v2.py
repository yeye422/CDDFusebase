import unittest

import numpy as np
import torch

from e2_local_correction_v2 import LocalCorrectionV2, blur_tiles, candidate_loss, source_terms
from run_e2_local_correction_v2 import PILOT, scene_probes
from test_OS import DIRECT_MODEL_TYPES, resolve_variant


class SourceLoss:
    @staticmethod
    def reconstruction(source, output, mask):
        return ((source - output).square() * mask).mean()


class CorrectionV2Tests(unittest.TestCase):
    def test_contrast_and_intact_support(self):
        source = torch.zeros(1, 1, 32, 32)
        source[..., 7:12] = 0.6
        valid = torch.ones_like(source)
        initial, active = blur_tiles(source, valid, False)
        # This particular edge crosses both the degraded and intact tiles.
        exact, keep = candidate_loss(source, source, source, valid, active)
        missing, _ = candidate_loss(initial, source, source, valid, active)
        offset, keep_offset = candidate_loss(initial + 0.125, source, source, valid, active)
        self.assertEqual(float(exact + keep), 0)
        self.assertGreater(float(missing), 0)
        torch.testing.assert_close(missing, offset)
        torch.testing.assert_close(keep_offset, torch.tensor(.125))
        flipped, _ = candidate_loss(1 - source, source, source, valid, active)
        self.assertGreater(float(flipped), float(missing))

    def test_invalid_radar_does_not_contaminate_loss_or_blur(self):
        valid = torch.ones(1, 1, 29, 35)
        valid[..., 7:12] = 0
        original = torch.rand_like(valid)
        modified = torch.where(valid.bool(), original, original + 50)
        a, mask = blur_tiles(original, valid, False)
        b, _ = blur_tiles(modified, valid, False)
        torch.testing.assert_close(a * valid, b * valid)
        for x, y in zip(candidate_loss(a, original, original, valid, mask),
                        candidate_loss(a, modified, modified, valid, mask)):
            torch.testing.assert_close(x, y)
        for value in candidate_loss(a, original, original, valid * 0, mask):
            self.assertEqual(float(value), 0.)

    def test_direct_gradient_reaches_designated_head_in_both_stages(self):
        torch.manual_seed(11)
        model = LocalCorrectionV2().eval()
        optical, sar = [torch.rand(1, 1, 23, 25) for _ in range(2)]
        block = model.decoder.correction
        params = [block.optical_head[-1].weight, block.sar_head[-1].weight, block.score[-1].weight,
                  model.decoder.reduce_channel.weight]
        for reconstruction in (True, False):
            _, terms = source_terms(model, SourceLoss(), optical, sar, torch.ones_like(sar),
                                    model.encode_pair(optical, sar), reconstruction=reconstruction)
            for i, key in enumerate(("optical_contrast", "sar_contrast")):
                grads = torch.autograd.grad(terms[key], params, allow_unused=True, retain_graph=True)
                self.assertGreater(float(grads[i].abs().sum()), 0)
                self.assertTrue(grads[1-i] is None or grads[1-i].count_nonzero() == 0)
                self.assertIsNone(grads[2])
                if not reconstruction:
                    self.assertIsNone(grads[3])

    def test_evaluator_and_fixed_line_diagnostics(self):
        variant = resolve_variant({"args": {"variant": PILOT.variant}}, "auto")
        model = DIRECT_MODEL_TYPES[variant]().eval()
        self.assertIsInstance(model, LocalCorrectionV2)
        source = np.zeros((48, 48), dtype=np.uint8)
        source[:, 24:] = 180
        line = np.array([[24, 12], [24, 36]])
        rows = scene_probes(model, source, source, [("B", line), ("parallel_negative", line - [9, 0])])
        self.assertEqual(len(rows), 8)
        for group in (rows[:4], rows[4:]):
            self.assertEqual({r["method"] for r in group}, {"initial", "optical_candidate", "sar_candidate", "final"})
            self.assertEqual(len({r["hits"] for r in group}), 1)
        self.assertNotEqual(rows[0]["sign"], 0)
        self.assertIsNone(rows[-1]["center_jump_gray"])
