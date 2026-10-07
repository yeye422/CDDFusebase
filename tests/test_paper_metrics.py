import unittest

import numpy as np

from utils.Evaluator import Evaluator, PaperEvaluator


class PaperEvaluatorTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[0:64, 0:64]
        self.source_a = ((3 * x + 2 * y) % 256).astype(np.float32)
        self.source_b = ((255 - 2 * x + y) % 256).astype(np.float32)
        self.fused = ((self.source_a + self.source_b) / 2.0).astype(np.float32)

    def test_complete_metric_vector_is_finite(self):
        values = PaperEvaluator.metric_vector(
            self.fused, self.source_a, self.source_b
        )
        self.assertEqual(len(PaperEvaluator.METRIC_NAMES), 14)
        self.assertEqual(values.shape, (14,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_identical_images_use_mean_based_ssim(self):
        paper_value = PaperEvaluator.SSIM(
            self.source_a, self.source_a, self.source_a
        )
        legacy_value = Evaluator.SSIM(
            self.source_a, self.source_a, self.source_a
        )
        self.assertAlmostEqual(paper_value, 1.0, places=6)
        self.assertAlmostEqual(legacy_value, 2.0, places=6)

    def test_mutual_information_uses_base_two(self):
        entropy = PaperEvaluator.EN(self.source_a)
        mutual_information = PaperEvaluator._mutual_information_bits(
            self.source_a, self.source_a
        )
        self.assertAlmostEqual(mutual_information, entropy, places=10)

    def test_psnr_uses_fixed_eight_bit_peak(self):
        source = np.zeros((16, 16), dtype=np.float32)
        fused = np.full((16, 16), 10.0, dtype=np.float32)
        expected = 10.0 * np.log10((255.0 ** 2) / 100.0)
        self.assertAlmostEqual(
            PaperEvaluator.PSNR(fused, source, source), expected, places=10
        )

    def test_flat_images_do_not_divide_by_zero(self):
        flat = np.zeros((64, 64), dtype=np.float32)
        self.assertEqual(PaperEvaluator.Qabf(flat, flat, flat), 0.0)
        self.assertEqual(PaperEvaluator.Nabf(flat, flat, flat), 0.0)
        self.assertEqual(PaperEvaluator.CC(flat, flat, flat), 1.0)


if __name__ == "__main__":
    unittest.main()
