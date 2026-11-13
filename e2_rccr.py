"""Relation-guided correction restricted to the output's second Haar lowpass."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2_base_anchor import OpticalBaseAnchoredE2
from train_lwpd import forward_losses as e2_forward_losses


SPEC = dict(max_gate=0.4, first_fusion_max_gate=0.1, contrast_threshold=0.05,
            patch_epsilon=1e-4, deviation_weight=0.01)
STATS = ("rccr_gate_mean", "rccr_deviation", "rccr_outside_range")
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def normalized_patches(image):
    height, width = image.shape[-2:]
    patches = F.unfold(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernel_size=3)
    centered = patches.reshape(-1, 9, height, width)
    centered = centered - centered.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    return centered / (norm + SPEC["patch_epsilon"]), norm / 3


def relation_evidence(initial, optical):
    initial_patches, _ = normalized_patches(initial)
    optical_patches, optical_std = normalized_patches(optical)
    weight = (optical_std / SPEC["contrast_threshold"]).clamp(max=1.0)
    height, width = optical.shape[-2:]
    neighbors = F.pad(torch.cat((initial_patches, optical_patches, weight), dim=1),
                      (1, 1, 1, 1), mode="replicate")
    evidence = torch.zeros_like(weight)
    for dy, dx in OFFSETS:
        y, x = dy + 1, dx + 1
        shifted = neighbors[..., y:y + height, x:x + width]
        initial_neighbor, optical_neighbor, neighbor_weight = shifted.split((9, 9, 1), dim=1)
        initial_relation = (initial_patches * initial_neighbor).sum(dim=1, keepdim=True)
        optical_relation = (optical_patches * optical_neighbor).sum(dim=1, keepdim=True)
        evidence = evidence + torch.minimum(weight, neighbor_weight) * (initial_relation - optical_relation).abs()
    return evidence / len(OFFSETS), weight


class RCCRL2(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -3.0)

    def forward(self, initial, optical, max_gate=SPEC["max_gate"]):
        height, width = initial.shape[-2:]
        padding = (0, -width % 4, 0, -height % 4)
        with torch.autocast(initial.device.type, enabled=False):
            sources = torch.cat((initial.float(), optical.float()), dim=1)
            if any(padding):
                sources = F.pad(sources, padding, mode="replicate")
            initial_2, optical_2 = F.avg_pool2d(sources, 2).split(1, dim=1)
            initial_4, optical_4 = F.avg_pool2d(sources, 4).split(1, dim=1)
            evidence_2, _ = relation_evidence(initial_2, optical_2)
            evidence_4, weight_4 = relation_evidence(initial_4, optical_4)
            difference = optical_4 - initial_4
            # Orthonormal Haar L2 equals four times the 4x4 block mean.
            features = torch.cat((4 * initial_4, 4 * optical_4, 4 * difference,
                                  evidence_4, F.avg_pool2d(evidence_2, 2), weight_4), dim=1)
            gate = max_gate * torch.sigmoid(self.head(features))
            correction = (gate * difference).repeat_interleave(4, -2).repeat_interleave(4, -1)
            return initial.float() + correction[..., :height, :width], gate


class RCCRL2E2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_rccr_l2_v1"

    def __init__(self):
        super().__init__(variant=self.VARIANT)
        self.rccr = RCCRL2()

    def forward_pair(self, optical, sar):
        initial = super().forward(optical, sar)
        fused, gate = self.rccr(initial, optical)
        return initial, fused, gate

    def forward(self, optical, sar, return_features=False):
        if return_features:
            initial, features = super().forward(optical, sar, return_features=True)
            fused, gate = self.rccr(initial, optical)
            return fused, {**features, "initial_fusion": initial, "rccr_gate": gate}
        return self.forward_pair(optical, sar)[1]


def forward_losses(model, losses, optical, sar, valid, args, *, max_gate):
    stats = {}

    def refine(optical, _sar, initial, _valid):
        fused, gate = model.rccr(initial, optical, max_gate)
        stats["rccr_deviation"] = (fused - initial.float()).square().mean()
        stats["rccr_gate_mean"] = gate.detach().mean()
        stats["rccr_outside_range"] = ((fused < 0) | (fused > 1)).float().mean()
        return fused

    total, components = e2_forward_losses(model, losses, optical, sar, valid, args, fusion_transform=refine)
    total = total + SPEC["deviation_weight"] * stats["rccr_deviation"]
    return total, {**components, **stats}
