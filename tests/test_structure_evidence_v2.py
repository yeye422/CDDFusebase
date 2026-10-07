import unittest

import torch

from structure_evidence_v2 import FixedSARLineEvidenceV2


def calibration(scales=(3, 5, 9), threshold=0.10, transition=0.05):
    result = {
        "scales": list(scales),
        "responses": {
            f"roa_raw_s{scale}": {
                "threshold": threshold,
                "transition": transition,
            }
            for scale in scales
        },
    }
    result["responses"]["line_score"] = {
        "threshold": 0.20,
        "transition": 0.05,
    }
    return result


class FixedStructureEvidenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.model = FixedSARLineEvidenceV2(calibration=calibration())

    def test_constant_scene_has_no_line_evidence(self):
        sar = torch.full((1, 1, 48, 48), 0.5)
        result = self.model(sar, torch.ones_like(sar))
        self.assertLess(float(result["r_line"].max()), 0.05)
        self.assertEqual(float(result["line_high_mask"].max()), 0.0)

    def test_step_edge_is_stronger_than_flat_regions(self):
        sar = torch.full((1, 1, 48, 48), 0.2)
        sar[:, :, :, 24:] = 0.8
        result = self.model(sar, torch.ones_like(sar))
        edge = result["r_line"][:, :, 6:-6, 21:27].mean()
        flat = torch.cat(
            (
                result["r_line"][:, :, 6:-6, 6:14].flatten(),
                result["r_line"][:, :, 6:-6, 34:42].flatten(),
            )
        ).mean()
        self.assertGreater(float(edge), float(flat) + 0.2)

    def test_shared_calibration_does_not_force_a_fixed_high_fraction(self):
        flat = torch.full((1, 1, 48, 48), 0.3)
        edge = flat.clone()
        edge[:, :, :, 24:] = 0.9
        flat_map = self.model(flat, torch.ones_like(flat))["line_high_mask"]
        edge_map = self.model(edge, torch.ones_like(edge))["line_high_mask"]
        self.assertGreater(float(edge_map.mean()), float(flat_map.mean()) + 0.01)

    def test_invalid_region_is_zero(self):
        torch.manual_seed(37)
        sar = torch.rand(1, 1, 48, 48)
        valid = torch.ones_like(sar)
        valid[:, :, :, :10] = 0.0
        result = self.model(sar, valid)
        for name in (
            "sar_coarse",
            "coherence",
            "direction_consistency",
            "scale_persistence",
            "r_line",
            "line_high_mask",
        ):
            self.assertEqual(float(result[name][:, :, :, :10].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
