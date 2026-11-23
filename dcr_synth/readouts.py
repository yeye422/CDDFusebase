"""Registered signed boundary readouts in floating-point gray levels."""

import torch
import torch.nn.functional as F

from diagnose_e2_boundary_supervision import BoundaryProbe


class Readouts:
    def __init__(self, optical, sar, baseline, case, settings):
        self.target = BoundaryProbe(optical, case["target"])
        self.protections = [
            (item["role"], BoundaryProbe(sar if item["role"] == "sar_only" else optical,
                                         item["line"]), item["tolerance_gl"])
            for item in case["protections"]
        ]
        self.region = (self.target.taper > 0) & (self.target.across.abs() <= 7)
        self.protected = torch.zeros_like(self.region)
        for _, probe, _ in self.protections:
            self.protected |= (probe.taper > 0) & (probe.across.abs() <= 6)
        self.far = ~(self.region | self.protected)
        self.original = self.target.score(baseline)
        self.initial = self.target_values(255 * baseline)
        desired = torch.full_like(self.initial, case["desired_contrast_gl"])
        self.b = (desired - self.initial).clamp_min(0)
        if case["role"] in ("preserve", "negative"):
            self.b.zero_()
        self.tau = baseline.new_tensor([tau for _, _, tau in self.protections]).repeat_interleave(17)
        self.hit = settings["hit_gl"]
        self.offset_limit = settings["offset_limit_px"]
        self.base_protection = self.protection_values(255 * baseline)

    def target_values(self, gray):
        return self.target.sign * self.target.contrast(gray)

    def protection_values(self, gray):
        parts = [probe.contrast(gray) for _, probe, _ in self.protections]
        return torch.cat(parts) if parts else gray.new_empty(0)

    def offsets(self, gray):
        offsets = torch.arange(-6, 7, device=gray.device, dtype=gray.dtype)
        points = self.target.center[None] + offsets[:, None, None] * self.target.normal
        step = 0.5 * self.target.normal
        plus = 2 * (points + step) / self.target.scale - 1
        minus = 2 * (points - step) / self.target.scale - 1
        response = (F.grid_sample(gray, plus[None], align_corners=True)
                    - F.grid_sample(gray, minus[None], align_corners=True))[0, 0]
        return offsets[response.abs().argmax(dim=0)]

    def points(self, gray):
        contrasts = self.target_values(gray)
        offsets = self.offsets(gray)
        hits = ((contrasts >= self.hit) & (offsets.abs() <= self.offset_limit)).tolist()
        longest = current = 0
        for hit in hits:
            current = current + 1 if hit else 0
            longest = max(longest, current)
        return dict(contrast_gl=float(contrasts.mean()), aligned=int((contrasts > 0).sum()),
                    points_gl=contrasts.tolist(), peak_offsets_px=offsets.tolist(),
                    valid_points=sum(hits), longest=longest)
