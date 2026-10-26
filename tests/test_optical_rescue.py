"""Unit checks for the Optical Rescue module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optical_rescue import OpticalRescueAdapter, RescueProbeHeads, normalized_l1
from train_optical_rescue import set_active


def test_optical_rescue_is_exact_identity_at_initialization():
    adapter = OpticalRescueAdapter()
    tensors = [torch.randn(2, 64, 8, 8) for _ in range(4)]
    rescued, diagnostics = adapter(*tensors)
    assert torch.equal(rescued, tensors[2])
    assert diagnostics["rescue_rms"].item() == 0.0


def test_optical_rescue_is_bounded_per_channel():
    adapter = OpticalRescueAdapter(max_relative_correction=0.005)
    tensors = [torch.randn(1, 64, 8, 8) for _ in range(4)]
    with torch.no_grad():
        adapter.delta.weight.fill_(0.1); adapter.delta.bias.fill_(0.1)
        rescued, _ = adapter(*tensors)
    rms = tensors[2].square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert torch.all((rescued - tensors[2]).abs() <= 0.005 * rms + 1e-6)


def test_probe_heads_start_as_identity():
    heads = RescueProbeHeads()
    features = torch.randn(1, 64, 8, 8)
    optical, sar = heads(features)
    assert torch.equal(optical, features)
    assert torch.equal(sar, features)


def test_zero_delta_receives_gradient():
    adapter = OpticalRescueAdapter()
    tensors = [torch.randn(1, 64, 8, 8) for _ in range(4)]
    rescued, _ = adapter(*tensors)
    rescued.square().mean().backward()
    assert adapter.delta.weight.grad is not None
    assert adapter.delta.weight.grad.abs().sum().item() > 0


def test_both_training_stages_keep_the_expected_gradient_graph():
    adapter = OpticalRescueAdapter()
    heads = RescueProbeHeads()
    optical, sar, fused, error = [torch.randn(2, 64, 8, 8) for _ in range(4)]

    set_active(adapter, heads, "heads", True)
    predicted_optical, predicted_sar = heads(fused.detach())
    probe_loss = normalized_l1(predicted_optical, optical) + normalized_l1(
        predicted_sar, sar
    )
    assert probe_loss.requires_grad and probe_loss.grad_fn is not None
    probe_loss.backward()
    assert any(p.grad is not None for p in heads.parameters())

    heads.zero_grad(set_to_none=True)
    set_active(adapter, heads, "adapter", True)
    rescued, _ = adapter(optical, sar, fused, error)
    predicted_optical, predicted_sar = heads(rescued)
    adapter_loss = normalized_l1(predicted_optical, optical) + normalized_l1(
        predicted_sar, sar
    )
    assert adapter_loss.requires_grad and adapter_loss.grad_fn is not None
    adapter_loss.backward()
    assert adapter.delta.weight.grad is not None
    assert adapter.delta.weight.grad.abs().sum().item() > 0


def test_validation_does_not_destroy_next_stage_trainability():
    adapter = OpticalRescueAdapter()
    heads = RescueProbeHeads()
    set_active(adapter, heads, "heads", False)
    assert all(p.requires_grad for p in heads.parameters())
    set_active(adapter, heads, "adapter", True)
    assert all(p.requires_grad for p in adapter.parameters())
    assert not any(p.requires_grad for p in heads.parameters())


if __name__ == "__main__":
    test_optical_rescue_is_exact_identity_at_initialization()
    test_optical_rescue_is_bounded_per_channel()
    test_probe_heads_start_as_identity()
    test_zero_delta_receives_gradient()
    test_both_training_stages_keep_the_expected_gradient_graph()
    test_validation_does_not_destroy_next_stage_trainability()
    print("Optical Rescue unit checks passed.")
