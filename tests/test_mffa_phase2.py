"""Unit checks for MFFA v1 Phase-2 trainability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mffa_v1 import MFFAV1
from train_mffa_v1_phase2 import configure_phase2


def test_phase2_only_trains_fusion_modules():
    model = MFFAV1()
    configure_phase2(model)
    for name, module in model.named_children():
        expected = name in {"low_fusion", "mid_fusion", "high_fusion"}
        assert all(parameter.requires_grad == expected for parameter in module.parameters())


def test_phase2_high_sar_output_receives_gradient():
    torch.manual_seed(1)
    model = MFFAV1()
    configure_phase2(model)
    optical = torch.rand(1, 1, 32, 32)
    sar = torch.rand_like(optical)
    fused, _ = model.fuse(optical, sar)
    fused.mean().backward()
    gradient = model.high_fusion.sar_delta.weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0.0


if __name__ == "__main__":
    test_phase2_only_trains_fusion_modules()
    test_phase2_high_sar_output_receives_gradient()
    print("MFFA v1 Phase-2 unit checks passed.")
