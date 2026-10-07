import torch
import torch.nn as nn
from types import SimpleNamespace

from net import DetailFeatureExtraction
from reliability_modules import LocalCrossModalEvidence, SRRDetailFusion
from train_OS import (
    make_optimizers,
    srr_optimizer_parameters,
    srr_reliability_anchor_loss,
)


def test_local_cross_modal_evidence_has_bounded_shape():
    module = LocalCrossModalEvidence(channels=64, embedding_channels=8)
    optical = torch.randn(2, 64, 12, 10)
    sar = torch.randn(2, 64, 12, 10)
    evidence, certainty = module(optical, sar)
    assert evidence.shape == (2, 1, 12, 10)
    assert certainty.shape == evidence.shape
    assert torch.isfinite(evidence).all()
    assert torch.all((0.0 <= evidence) & (evidence <= 1.0))
    assert torch.all((0.0 <= certainty) & (certainty <= 1.0))


def test_srr_detail_fusion_shapes_maps_and_gradients():
    module = SRRDetailFusion(channels=64, hidden_channels=8)
    optical = torch.randn(2, 64, 16, 16, requires_grad=True)
    sar = torch.randn(2, 64, 16, 16, requires_grad=True)
    prior = torch.rand(2, 1, 16, 16)
    valid = torch.ones_like(prior)
    fused, diagnostics = module(
        optical, sar, prior, valid=valid, return_maps=True
    )
    assert fused.shape == optical.shape
    assert torch.isfinite(fused).all()
    for name in (
        "srr_reliability_map",
        "srr_cross_attribute_map",
        "srr_complementary_attribute_map",
        "srr_cross_evidence_map",
        "srr_match_certainty_map",
        "srr_structure_prior_map",
    ):
        value = diagnostics[name]
        assert value.shape == prior.shape
        assert torch.all((0.0 <= value) & (value <= 1.0))
    fused.mean().backward()
    assert optical.grad is not None
    assert sar.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_zero_forced_reliability_removes_routed_sar():
    module = SRRDetailFusion(channels=64, hidden_channels=8)
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn(1, 64, 8, 8)
    prior = torch.ones(1, 1, 8, 8)
    _, diagnostics = module(
        optical,
        sar,
        prior,
        return_maps=True,
        force_reliability=0.0,
    )
    assert torch.count_nonzero(diagnostics["srr_reliability_map"]) == 0
    assert diagnostics["srr_routed_sar_norm"].item() == 0.0


def test_legacy_detail_weights_initialize_terminal_transform():
    legacy = DetailFeatureExtraction(num_layers=1)
    module = SRRDetailFusion(channels=64, hidden_channels=8)
    module.load_legacy_detail_fuse(legacy.state_dict())
    for expected, actual in zip(
        legacy.state_dict().values(),
        module.detail_transform.state_dict().values(),
    ):
        assert torch.equal(expected, actual)


def test_sparse_reliability_anchor_loss_is_balanced_and_differentiable():
    reliability = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    prior = torch.zeros_like(reliability)
    prior[:, :, 0, 0] = 0.9
    cross = torch.linspace(0.0, 1.0, 16).view(1, 1, 4, 4)
    valid = torch.ones_like(reliability)
    (
        loss,
        bce_loss,
        rank_loss,
        positive_fraction,
        negative_fraction,
        positive_mean,
        negative_mean,
        gap,
    ) = srr_reliability_anchor_loss(
        reliability, prior, cross, valid, ranking_margin=0.25
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(bce_loss)
    assert rank_loss.item() == 0.25
    assert positive_mean.item() == negative_mean.item() == 0.5
    assert gap.item() == 0.0
    assert positive_fraction.item() > 0.0
    assert negative_fraction.item() > 0.0
    loss.backward()
    assert reliability.grad is not None
    assert reliability.grad[0, 0, 0, 0].item() < 0.0
    assert torch.any(reliability.grad > 0.0)


def test_srr_router_uses_independent_optimizer_scale():
    models = {
        "base_fuse": nn.Conv2d(1, 1, kernel_size=1),
        "detail_router": SRRDetailFusion(channels=64, hidden_channels=8),
    }
    args = SimpleNamespace(
        lr=1e-5,
        srr_router_lr=1e-4,
        weight_decay=0.0,
        clip_grad=0.01,
        srr_router_clip_grad=0.1,
    )
    optimizers = make_optimizers(models, args)
    assert optimizers["base_fuse"].param_groups[0]["lr"] == 1e-5
    router_groups = {
        group["group_name"]: group
        for group in optimizers["detail_router"].param_groups
    }
    assert router_groups["fusion_slow"]["lr"] == 1e-5
    assert router_groups["evidence_router_fast"]["lr"] == 1e-4
    fast, slow = srr_optimizer_parameters(models["detail_router"])
    fast_ids = {id(parameter) for parameter in fast}
    slow_ids = {id(parameter) for parameter in slow}
    assert fast_ids
    assert slow_ids
    assert fast_ids.isdisjoint(slow_ids)
    evidence_ids = {
        id(parameter)
        for parameter in models["detail_router"].cross_evidence.parameters()
    }
    transform_ids = {
        id(parameter)
        for parameter in models["detail_router"].detail_transform.parameters()
    }
    assert evidence_ids <= fast_ids
    assert transform_ids <= slow_ids


if __name__ == "__main__":
    test_local_cross_modal_evidence_has_bounded_shape()
    test_srr_detail_fusion_shapes_maps_and_gradients()
    test_zero_forced_reliability_removes_routed_sar()
    test_legacy_detail_weights_initialize_terminal_transform()
    test_sparse_reliability_anchor_loss_is_balanced_and_differentiable()
    test_srr_router_uses_independent_optimizer_scale()
    print("SRR detail fusion checks passed.")
