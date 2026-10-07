"""Frozen line-evidence providers used by C3a/C3b training."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from phase_antisymmetry import MonogenicPhaseAntisymmetry, local_phase_bridge
from structure_evidence_v2 import FixedSARLineEvidenceV2, load_calibration


class FixedLineEvidence(nn.Module):
    def __init__(self, fse_calibration, phase_calibration=None):
        super().__init__()
        self.fse_calibration = fse_calibration
        self.phase_calibration = phase_calibration
        self.fse = FixedSARLineEvidenceV2(
            calibration=fse_calibration,
            scales=tuple(fse_calibration["scales"]),
        )
        self.phase = None
        if phase_calibration is not None:
            if phase_calibration["fse_calibration_sha256"] != fse_calibration["sha256"]:
                raise ValueError("Phase and FSE calibrations do not match")
            if "bridge_score" not in phase_calibration["responses"]:
                raise ValueError("Phase calibration has no local bridge response")
            self.phase = MonogenicPhaseAntisymmetry(
                wavelengths=phase_calibration["wavelengths"],
                sigma_on_f=phase_calibration["sigma_on_f"],
            )

    @classmethod
    def from_paths(cls, fse_path, phase_path=None):
        fse = load_calibration(fse_path)
        phase = None
        if phase_path:
            with Path(phase_path).open(encoding="utf-8") as handle:
                phase = json.load(handle)
        return cls(fse, phase)

    @staticmethod
    def _probability(values, entry):
        return torch.sigmoid(
            (values - float(entry["threshold"]))
            / max(float(entry["transition"]), 1e-6)
        )

    def forward(self, sar, valid):
        fse = self.fse(sar, valid)
        if self.phase is None:
            return {
                "reliability": fse["r_line"].detach(),
                "roa_reliability": fse["r_line"].detach(),
            }
        phase = self.phase(sar, valid)
        fa_probability = self._probability(
            phase["fa_mean"], self.phase_calibration["responses"]["fa_mean"]
        ) * valid
        bridge_score, bridge_support = local_phase_bridge(
            fse["r_line"], fa_probability, kernel_size=5, seed_threshold=0.5
        )
        bridge_probability = self._probability(
            bridge_score,
            self.phase_calibration["responses"]["bridge_score"],
        ) * valid
        return {
            "reliability": bridge_probability.detach(),
            "roa_reliability": fse["r_line"].detach(),
            "fa_probability": fa_probability.detach(),
            "bridge_support": bridge_support.detach(),
        }
