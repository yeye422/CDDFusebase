"""CPU invariants for frozen FSC state reachability probes (no QXS data)."""

import unittest

import torch

from diagnose_fsc_state_reachability import (
    initial_state, make_profiles, make_spatial, parse_values, segment_rms,
    ssm_with_low, state_effect,
)
from e2_fsc_mamba import LLStateTransfer, scan_with_state
from pac_mamba import BiAxialSelectiveStateSpace, SelectiveScan1D


class FSCReachabilityTests(unittest.TestCase):
    def test_state_response_is_linear_in_initial_scale(self):
        torch.manual_seed(13)
        scan = SelectiveScan1D(8).eval()
        x = torch.randn(2, 12, 8)
        h = torch.randn(2, scan.inner, scan.state_channels)
        with torch.no_grad():
            zero = scan_with_state(scan, x, torch.zeros_like(h))
            d1 = state_effect(scan, x, 0.1 * h, zero)
            d3 = state_effect(scan, x, 0.3 * h, zero)
        torch.testing.assert_close(d3, 3 * d1, atol=2e-6, rtol=2e-3)
        self.assertEqual(len(segment_rms(d1)), 3)

    def test_2d_probe_matches_native_fsc_state_scan(self):
        torch.manual_seed(14)
        ssm = LLStateTransfer(BiAxialSelectiveStateSpace(8), "state").eval()
        joint = torch.randn(1, 8, 8, 8)
        optical = torch.randn_like(joint)
        sar = torch.randn_like(joint)
        with torch.no_grad():
            low = ssm.ll_projection(torch.cat((optical, sar), dim=1)).tanh()
            expected = ssm(joint, optical, sar)
            observed = ssm_with_low(ssm, joint, low)
            spatial = make_spatial(ssm, joint, low)
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
        self.assertGreater(spatial["correct_vs_zero_rms"], 0.0)
        self.assertGreater(spatial["correct_vs_shuffled_rms"], 0.0)

    def test_profile_and_matching_conditions(self):
        torch.manual_seed(15)
        ssm = LLStateTransfer(BiAxialSelectiveStateSpace(8), "state").eval()
        joint = torch.randn(1, 8, 8, 8)
        low = torch.randn_like(joint)
        with torch.no_grad():
            profiles, conditions = make_profiles(ssm, joint, low, [4, 8], [0, 0.1], 2)
        self.assertEqual(len(profiles), 2 * 2 * 2 * 2)
        self.assertEqual(len(conditions), 2 * 2 * 2)
        for row in profiles:
            if row["scale"] == 0:
                self.assertEqual(row["full_delta_rms"], 0.0)
        for row in conditions:
            self.assertGreaterEqual(row["correct_vs_shuffled_rms"], 0.0)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            parse_values("32,32", int)
        with self.assertRaises(ValueError):
            parse_values("", float)


if __name__ == "__main__":
    unittest.main()
