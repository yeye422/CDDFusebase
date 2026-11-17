"""Synthetic controls for FSC RMS calibration, no QXS data needed."""
import unittest
import torch
from fsc_amplitude_matching import calibrate_gain, response_stats
from e2_fsc_mamba import LLStateTransfer
from e2_fsc_local_refresh import LocalRefreshTransfer
from pac_mamba import BiAxialSelectiveStateSpace

class AmplitudeMatchingTests(unittest.TestCase):
    def test_linear_response_calibration(self):
        solution = calibrate_gain(lambda g: 0.002*g, 0.14)
        self.assertTrue(solution["matched"])
        self.assertAlmostEqual(solution["gain"], 70., places=4)
        self.assertAlmostEqual(solution["relative_error"], 0., places=5)

    def test_bounded_nonlinear_response(self):
        solution = calibrate_gain(lambda g: 0.001*g + 0.000001*g*g, 0.12)
        self.assertTrue(solution["matched"])
        self.assertLess(solution["relative_error"], 0.05)

    def test_rejects_degenerate_reference_and_unmatchable_gain(self):
        with self.assertRaises(ValueError):
            calibrate_gain(lambda _: 0.0, 0.2)
        with self.assertRaises(ValueError):
            calibrate_gain(lambda g: g, 0.0)
        result = calibrate_gain(lambda g: 0.0001*g, 2.0, max_gain=16)
        self.assertFalse(result["matched"])
        self.assertEqual(result["gain"], 16)

    def test_spatial_counterfactual_uses_same_detail(self):
        torch.manual_seed(42)
        previous = LLStateTransfer(BiAxialSelectiveStateSpace(8),"state")
        ssm = LocalRefreshTransfer(previous,"local_refresh",chunk_size=4).eval()
        joint = torch.randn(1,8,8,8)
        low = torch.randn(1,8,8,8)
        with torch.inference_mode():
            correct = ssm(joint,low,low)
            wrong = ssm(joint,low.roll(4,-2),low.roll(4,-1))
            no_ll = ssm(joint,torch.zeros_like(low),torch.zeros_like(low))
            reference = response_stats(correct,wrong,no_ll)
        self.assertGreater(reference["correct_vs_zero_rms"], 0)
        self.assertGreater(reference["correct_vs_shuffled_rms"], 0)
        x = torch.zeros(1,8,8,8)
        stats = response_stats(joint,low,x)
        self.assertGreater(stats["effect_over_zero"], 0)

    def test_correct_vs_wrong_not_automatically_good(self):
        zero = torch.zeros(2,2)
        a = torch.ones(2,2)
        b = -torch.ones(2,2)
        stats = response_stats(a,b,zero)
        self.assertAlmostEqual(stats["mismatch_over_effect"],2)

if __name__=="__main__":
    unittest.main()
