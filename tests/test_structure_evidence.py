import unittest

import torch

from structure_evidence import CALIBRATION_VERSION, FixedSARStructureEvidence


def calibration(scales=(3, 5, 9)):
    names = {
        *(f"roa_s{scale}" for scale in scales),
        *(f"point_s{scale}" for scale in scales),
        "tensor_energy",
    }
    return {
        "version": CALIBRATION_VERSION,
        "scales": list(scales),
        "thresholds": {name: 0.1 for name in names},
        "temperatures": {name: 0.05 for name in names},
    }


class FixedStructureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.model = FixedSARStructureEvidence(
            scales=(3, 5, 9), calibration=calibration()
        )

    def test_constant_image_has_no_combined_evidence(self):
        sar = torch.full((1, 1, 32, 32), 0.5)
        valid = torch.ones_like(sar)
        result = self.model(sar, valid)
        self.assertLess(float(result["r_fixed"].max()), 1e-2)
        for values in result.values():
            self.assertTrue(torch.all(torch.isfinite(values)))

    def test_step_edge_produces_more_line_evidence_than_flat_region(self):
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
        self.assertGreater(float(edge), float(flat) + 0.1)

    def test_bright_point_produces_local_point_evidence(self):
        sar = torch.full((1, 1, 33, 33), 0.2)
        sar[:, :, 15:18, 15:18] = 1.0
        result = self.model(sar, torch.ones_like(sar))
        center = result["r_point"][0, 0, 16, 16]
        background = result["r_point"][0, 0, 4:12, 4:12].mean()
        self.assertGreater(float(center), float(background) + 0.5)

    def test_invalid_region_is_zero_in_every_output(self):
        torch.manual_seed(31)
        sar = torch.rand(1, 1, 32, 32)
        valid = torch.ones_like(sar)
        valid[:, :, :, :8] = 0.0
        result = self.model(sar, valid)
        for values in result.values():
            self.assertEqual(float(values[:, :, :, :8].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
