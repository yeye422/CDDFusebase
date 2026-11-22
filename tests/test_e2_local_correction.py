import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from e2_local_correction import LocalCorrection, LocalCorrectionE2, local_error
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize
from test_OS import load_variant_models


class LocalCorrectionTests(unittest.TestCase):
    def test_sar_mask_and_local_convex_bound(self):
        block = LocalCorrection(2)
        with torch.no_grad():
            block.optical_head[-1].bias.fill_(-0.3)
            block.sar_head[-1].bias.fill_(0.5)
        image = torch.rand(2, 1, 13, 17)
        sar = torch.rand_like(image)
        sar[..., :5] = 0
        features = torch.rand(2, 2, 13, 17)
        output, alternatives, scores = block(image, sar, image, features, features, features)
        self.assertEqual(scores.softmax(1)[:, 2:, :, :5].count_nonzero().item(), 0)
        self.assertTrue((output >= alternatives.amin(1, keepdim=True) - 1e-7).all())
        self.assertTrue((output <= alternatives.amax(1, keepdim=True) + 1e-7).all())
        self.assertTrue(torch.isfinite(local_error(alternatives, image, torch.zeros_like(sar))).all())

    def test_checkpoint_auto_load_and_feature_path(self):
        model = LocalCorrectionE2().eval()
        with torch.no_grad():
            model.decoder.correction.optical_head[-1].bias.fill_(0.4)
        image = torch.rand(1, 1, 17, 19)
        with tempfile.TemporaryDirectory() as folder, torch.no_grad():
            file = Path(folder) / "candidate.pth"
            torch.save(dict(args=dict(variant=model.variant), model=model.state_dict()), file)
            models, variant = load_variant_models(file, "cpu")
            self.assertEqual(variant, model.variant)
            out, features = models["direct"](image, image, return_features=True)
            torch.testing.assert_close(out, model(image, image))
            self.assertGreater((out - features["coarse_image"]).abs().max().item(), 0)

    def test_paired_export_preserves_common_finite_counts(self):
        original = np.ones(len(METRICS))
        candidate = original.copy()
        candidate[METRICS.index("MS_SSIM")] = np.nan
        results = {k: [(v, np.ones(len(SOURCE_METRICS)), ())] for k, v in
                   (("e2", original), ("candidate", candidate))}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for key in results:
                (folder / key).mkdir()
            summarize(folder, ["211"], results, [("e2", "candidate")], diagnostic_fields=())
            with (folder / "paired_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = {r["metric"]: r for r in csv.DictReader(handle)}
            self.assertEqual(rows["MS_SSIM"]["common_count"], "0")
            self.assertEqual(rows["Qabf"]["common_count"], "1")
