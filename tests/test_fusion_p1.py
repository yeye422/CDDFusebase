"""Unit checks for P1 stable-SAR Detail fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from p1_stable_detail_fusion import StableSARDetailRouter, stability_teacher


def test_router_initializes_close_to_joint_b0_expert():
    torch.manual_seed(1)
    router = StableSARDetailRouter(initial_gate=0.995)
    optical = torch.randn(2, 64, 8, 8)
    sar = torch.randn_like(optical)
    optical_expert = torch.randn_like(optical)
    joint_expert = torch.randn_like(optical)
    fused, diagnostics = router(optical, sar, optical_expert, joint_expert)
    assert abs(diagnostics["p1_gate_mean"].item() - 0.995) < 1e-5
    assert torch.allclose(
        fused,
        optical_expert + 0.995 * (joint_expert - optical_expert),
        atol=1e-6,
    )


def test_stability_teacher_rewards_lower_perturbation_sensitivity():
    first = torch.ones(1, 4, 8, 8)
    second = first.clone()
    second[:, :, :, 4:] += 0.5
    teacher = stability_teacher(first, second, smooth_kernel=1)
    assert teacher[:, :, :, :3].mean() > teacher[:, :, :, 5:].mean()


def test_router_gate_receives_gradient_at_initialization():
    torch.manual_seed(2)
    router = StableSARDetailRouter()
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn_like(optical)
    optical_expert = torch.randn_like(optical)
    joint_expert = torch.randn_like(optical)
    fused, _ = router(optical, sar, optical_expert, joint_expert)
    fused.square().mean().backward()
    assert router.spatial_gate.weight.grad is not None
    assert router.spatial_gate.weight.grad.abs().sum().item() > 0.0


if __name__ == "__main__":
    test_router_initializes_close_to_joint_b0_expert()
    test_stability_teacher_rewards_lower_perturbation_sensitivity()
    test_router_gate_receives_gradient_at_initialization()
    print("P1 stable Detail fusion unit checks passed.")
