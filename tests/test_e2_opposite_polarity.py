import unittest

import numpy as np

from diagnose_e2_opposite_polarity import OFFSETS, decide_polarity, strongest_offsets


def edge_profiles(offset, amplitude):
    profiles = np.zeros((10, len(OFFSETS)), dtype=np.float32)
    profiles[:, np.flatnonzero(OFFSETS == offset)[0]] = amplitude
    return profiles


class OppositePolarityTests(unittest.TestCase):
    def test_coincident_opposite_profiles_use_optical_polarity_despite_stronger_sar(self):
        report = decide_polarity(edge_profiles(0, -.5), edge_profiles(0, .9), -6., 18.)
        self.assertEqual(report["rule"], "optical_display_polarity")
        self.assertEqual(report["shared_near_fraction"], 1.)

    def test_displaced_strong_sar_is_not_considered_the_same_edge(self):
        report = decide_polarity(edge_profiles(0, -.6), edge_profiles(6, .9), -6., 18.)
        self.assertEqual(report["rule"], "abstain_unconfirmed_correspondence")
        self.assertEqual(report["sar_peak_offset_median_px"], 6.)

    def test_independent_center_coverage_does_not_imply_joint_support(self):
        optical, sar = edge_profiles(0, -.8), edge_profiles(0, .9)
        optical[6:] = 0
        sar[:4] = 0
        report = decide_polarity(optical, sar, -6., 18.)
        self.assertEqual(report["optical_near_fraction"], .6)
        self.assertEqual(report["sar_near_fraction"], .6)
        self.assertEqual(report["shared_near_fraction"], .2)
        self.assertEqual(report["rule"], "abstain_unconfirmed_correspondence")

    def test_peak_sign_must_support_the_fixed_center_polarity(self):
        report = decide_polarity(edge_profiles(0, -.6), edge_profiles(0, -.9), -6., 18.)
        self.assertEqual(report["rule"], "abstain_unconfirmed_correspondence")
        self.assertEqual(report["sar_near_fraction"], 0.)

    def test_same_polarity_is_outside_this_rule(self):
        report = decide_polarity(edge_profiles(0, .6), edge_profiles(0, .8), 10., 18.)
        self.assertEqual(report["rule"], "not_screened_as_opposite")

    def test_tied_zero_profiles_do_not_suggest_a_displacement(self):
        offset, strength = strongest_offsets(edge_profiles(6, 0))
        np.testing.assert_array_equal(offset, np.zeros(10))
        np.testing.assert_array_equal(strength, np.zeros(10))


if __name__ == "__main__":
    unittest.main()
