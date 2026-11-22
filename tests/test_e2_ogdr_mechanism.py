"""Synthetic, no-QXS-data tests for the OGDR frozen mechanism probe."""
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_e2_ogdr_mechanism import (
    FIELDS, analyze, edge_enrichment, normalized_mae255,
    optical_edge_mask, probe_scene, training_history,
)
from e2_ogdr import OGDRE2


class OGDRMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_optical_edge_proxy_constant_image_is_invalid_not_full_mask(self):
        mask = optical_edge_mask(np.zeros((16, 16), dtype=np.uint8))
        self.assertFalse(mask.any())
        self.assertTrue(math.isnan(edge_enrichment(np.ones((16, 16)), mask)))

    def test_aggregate_summary_preserves_finite_values(self):
        row = {name: float(i) for i, name in enumerate(FIELDS[1:], 1)}
        row["scene"] = "211"
        summary = analyze([row])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["means"]["kernel_entropy"], row["kernel_entropy"])

    def test_float_output_mae_scaled_to_8bit(self):
        x = torch.zeros((1, 1, 2, 2))
        y = torch.ones_like(x)
        self.assertEqual(normalized_mae255(x, y), 255.0)

    def test_missing_history_is_reported_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(training_history(Path(directory) / "history.csv"))

    def test_real_frozen_model_path_on_synthetic_image_pair(self):
        torch.manual_seed(42)
        model = OGDRE2("guided").eval()
        with tempfile.TemporaryDirectory() as directory:
            optical = np.full((19, 21, 3), 100, dtype=np.uint8)
            optical[5:14, 8:12] = (230, 190, 180)
            sar = np.full((19, 21), 90, dtype=np.uint8)
            sar[4:16, 10:13] = 160
            o = Path(directory) / "211.png"
            s = Path(directory) / "sar" / "211.png"
            s.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(o), optical))
            self.assertTrue(cv2.imwrite(str(s), sar))
            row, arrays = probe_scene(model, (o, s), torch.device("cpu"))
            self.assertEqual(row["scene"], "211")
            self.assertEqual(len(arrays), 5)
            self.assertEqual(row["off_aligned_changed_fraction"], 0.0)
            self.assertEqual(row["aligned_shifted_changed_fraction"], 0.0)
            self.assertLessEqual(row["normalized_kernel_entropy"], 1.0 + 1e-6)
            self.assertTrue(np.isfinite(row["aligned_shifted_kernel_mae"]))


if __name__ == "__main__":
    unittest.main()
