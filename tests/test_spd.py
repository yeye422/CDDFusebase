"""Unit checks for the standalone V2.1 SPD implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from spd_modules import (
    BoundedFeatureCorrection,
    DetailOnlyStructurePreservingDenoiser,
    SelectiveDetailCorrection,
    SelectiveDetailStructurePreservingDenoiser,
    SPDDenoisingHead,
    SPDModel,
    StructurePreservingDenoiser,
    build_spd_from_checkpoint_args,
)
from train_spd import (
    SPDLoss,
    controlled_multiplicative_perturbation,
    normalized_feature_distillation,
    split_official_validation,
)
from evaluate_spd import tile_starts
from compare_spd_fusion import forward_three_paths
from train_spd_fusion_adapt import configure_trainability
from prepare_spd_line_masks import high_confidence_line_mask, joint_line_mask
from sweep_spd_line_gated_detail import continuity_filter


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(1, 64, 1)

    def forward(self, image):
        features = self.projection(image)
        return features, features * 0.5, features


class TinyDecoder(nn.Module):
    def forward(self, optical, base, detail):
        residual = (base + detail).mean(dim=1, keepdim=True) * 0.01
        return torch.sigmoid(torch.logit(optical.clamp(0.01, 0.99)) + residual), base


def test_spd_feature_corrections_are_exact_identity_at_initialization():
    torch.manual_seed(1)
    module = StructurePreservingDenoiser()
    base = torch.randn(2, 64, 8, 8)
    detail = torch.randn_like(base)
    clean_base, clean_detail, diagnostics = module(base, detail)
    assert torch.equal(clean_base, base)
    assert torch.equal(clean_detail, detail)
    assert diagnostics["base_correction_abs_mean"].item() == 0.0
    assert diagnostics["detail_correction_abs_mean"].item() == 0.0


def test_detail_only_spd_has_no_base_parameters_and_preserves_base():
    torch.manual_seed(17)
    module = DetailOnlyStructurePreservingDenoiser()
    base = torch.randn(2, 64, 8, 8)
    detail = torch.randn_like(base)
    clean_base, clean_detail, diagnostics = module(base, detail)
    assert torch.equal(clean_base, base)
    assert torch.equal(clean_detail, detail)
    assert not any(name.startswith("base.") for name, _ in module.named_parameters())
    assert diagnostics["base_correction_rms"].item() == 0.0


def test_selective_detail_spd_is_identity_initialized_and_bypasses_base():
    torch.manual_seed(18)
    module = SelectiveDetailStructurePreservingDenoiser()
    base = torch.randn(2, 64, 9, 11)
    detail = torch.randn_like(base)
    clean_base, clean_detail, diagnostics = module(base, detail)
    assert torch.equal(clean_base, base)
    assert torch.equal(clean_detail, detail)
    assert not any(name.startswith("base.") for name, _ in module.named_parameters())
    assert diagnostics["base_correction_rms"].item() == 0.0
    assert diagnostics["detail_correction_rms"].item() == 0.0


def test_selective_detail_correction_is_bounded_per_channel():
    torch.manual_seed(19)
    module = SelectiveDetailCorrection(max_relative_correction=0.10)
    features = torch.randn(2, 64, 9, 11)
    with torch.no_grad():
        module.delta.weight.fill_(0.1)
        module.delta.bias.fill_(0.1)
        module.spatial_confidence.bias.fill_(5.0)
        module.channel_confidence[-1].bias.fill_(5.0)
        cleaned, diagnostics = module(features, return_maps=True)
    channel_rms = features.square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert torch.all(
        (cleaned - features).abs() <= 0.10 * channel_rms + 1e-6
    )
    assert diagnostics["gate_map"].shape == features.shape
    assert diagnostics["spatial_confidence_map"].shape == (2, 1, 9, 11)
    assert diagnostics["channel_confidence_map"].shape == (2, 64, 1, 1)


def test_selective_detail_checkpoint_metadata_rebuilds_architecture():
    module = build_spd_from_checkpoint_args(
        {
            "spd_variant": "selective_detail",
            "max_relative_correction": 0.07,
            "selective_scale_kernels": [3, 7],
            "selective_correction_highpass_kernel": 7,
        }
    )
    assert isinstance(module, SelectiveDetailStructurePreservingDenoiser)
    assert module.detail.max_relative_correction == 0.07
    assert module.detail.scale_kernels == (3, 7)
    assert module.detail.correction_highpass_kernel == 7


def test_spd_correction_is_rms_bounded():
    torch.manual_seed(2)
    module = BoundedFeatureCorrection(max_relative_correction=0.10)
    features = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        module.delta.weight.fill_(0.1)
        module.delta.bias.fill_(0.1)
        module.gate.bias.fill_(5.0)
        cleaned, _ = module(features)
    rms = features.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    assert torch.all((cleaned - features).abs() <= 0.10 * rms + 1e-6)


def test_zero_initialized_spd_output_head_receives_gradient():
    torch.manual_seed(3)
    module = BoundedFeatureCorrection()
    features = torch.randn(1, 64, 8, 8)
    cleaned, _ = module(features)
    cleaned.square().mean().backward()
    assert module.delta.weight.grad is not None
    assert module.delta.weight.grad.abs().sum().item() > 0.0


def test_denoising_head_is_image_identity_at_initialization():
    torch.manual_seed(4)
    head = SPDDenoisingHead()
    noisy = torch.rand(1, 1, 8, 8).clamp(1.0 / 255.0, 254.0 / 255.0)
    base = torch.randn(1, 64, 8, 8)
    detail = torch.randn_like(base)
    denoised = head(noisy, base, detail)
    assert torch.allclose(denoised, noisy, atol=1e-7)


def test_spd_wrapper_keeps_encoder_frozen_and_in_eval_mode():
    model = SPDModel(TinyEncoder())
    model.train()
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    output = model(torch.rand(1, 1, 8, 8))
    assert output["denoised"].shape == (1, 1, 8, 8)
    assert output["base"].shape == (1, 64, 8, 8)


def test_spd_loss_is_finite_and_masked():
    criterion = SPDLoss()
    reference = torch.rand(1, 1, 16, 16)
    denoised = reference.clone().requires_grad_(True)
    valid = torch.ones_like(reference)
    line = torch.ones_like(reference)
    point = torch.ones_like(reference)
    total, components = criterion(
        denoised,
        reference,
        valid,
        valid,
        line,
        point,
        ssim_weight=0.5,
        line_weight=1.0,
        point_weight=1.0,
    )
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())
    assert denoised.grad is not None


def test_controlled_perturbation_preserves_invalid_pixels():
    torch.manual_seed(5)
    sar = torch.full((1, 1, 8, 8), 0.5)
    valid = torch.ones_like(sar)
    valid[:, :, :2] = 0.0
    perturbed = controlled_multiplicative_perturbation(sar, valid, 0.1)
    assert torch.equal(perturbed[:, :, :2], sar[:, :, :2])
    assert not torch.equal(perturbed[:, :, 2:], sar[:, :, 2:])


def test_feature_distillation_is_zero_for_matching_features():
    features = torch.randn(2, 64, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    loss = normalized_feature_distillation(features, features, mask)
    assert loss.item() == 0.0


def test_official_validation_split_keeps_coordinate_groups_intact():
    from pathlib import Path

    pairs = [
        (Path(f"{x}_{y}.tiff"), Path(f"{x}_{y}.tiff"))
        for x in (5120, 5632)
        for y in (0, 512, 1024)
    ]
    validation, test, val_groups, test_groups = split_official_validation(
        pairs, 0.5
    )
    assert val_groups == ["5120"]
    assert test_groups == ["5632"]
    assert {pair[0].stem.split("_")[0] for pair in validation} == {"5120"}
    assert {pair[0].stem.split("_")[0] for pair in test} == {"5632"}


def test_tiled_evaluation_covers_last_boundary_once():
    starts = tile_starts(512, 128, 32)
    assert starts[0] == 0
    assert starts[-1] == 384
    assert len(starts) == len(set(starts))


def test_zero_spd_makes_image_and_feature_fusion_paths_identical():
    models = {
        "encoder": TinyEncoder(),
        "decoder": TinyDecoder(),
        "base_fuse": nn.Identity(),
        "detail_fuse": nn.Identity(),
        "spd": StructurePreservingDenoiser(),
        "denoising_head": SPDDenoisingHead(),
    }
    optical = torch.rand(1, 1, 8, 8).clamp(0.01, 0.99)
    sar = torch.rand(1, 1, 8, 8).clamp(0.01, 0.99)
    outputs, _, _ = forward_three_paths(models, optical, sar)
    assert torch.allclose(outputs["c0_raw"], outputs["e1_image"], atol=1e-7)
    assert torch.allclose(outputs["c0_raw"], outputs["e2_feature"], atol=1e-7)


def test_e2_adapt_only_trains_fusion_modules():
    models = {
        "encoder": TinyEncoder(),
        "decoder": TinyDecoder(),
        "base_fuse": nn.Conv2d(64, 64, 1),
        "detail_fuse": nn.Conv2d(64, 64, 1),
        "spd": StructurePreservingDenoiser(),
        "denoising_head": SPDDenoisingHead(),
    }
    configure_trainability(models)
    assert all(p.requires_grad for p in models["base_fuse"].parameters())
    assert all(p.requires_grad for p in models["detail_fuse"].parameters())
    for name in ("encoder", "decoder", "spd", "denoising_head"):
        assert not any(p.requires_grad for p in models[name].parameters())


def test_joint_line_mask_requires_support_from_both_images():
    noisy = torch.tensor([[[[1.0, 1.0, 0.25]]]])
    reference = torch.tensor([[[[1.0, 0.0, 0.25]]]])
    joint = joint_line_mask(noisy, reference)
    expected = torch.tensor([[[[1.0, 0.0, 0.25]]]])
    assert torch.allclose(joint, expected)


def test_high_confidence_line_mask_is_continuous_above_floor():
    mask = torch.tensor([0.25, 0.50, 0.75, 1.00])
    mapped = high_confidence_line_mask(mask, floor=0.5)
    assert torch.allclose(mapped, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_continuity_filter_removes_fragments_and_keeps_long_lines():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 2, 2:4] = 0.8
    mask[0, 0, 10, 2:13] = 0.6
    filtered, counts = continuity_filter(
        mask, min_area=5, min_extent=9, min_elongation=2.0, large_area=20
    )
    assert torch.count_nonzero(filtered[0, 0, 2]) == 0
    assert torch.allclose(filtered[0, 0, 10, 2:13], mask[0, 0, 10, 2:13])
    assert counts == [1]


if __name__ == "__main__":
    test_spd_feature_corrections_are_exact_identity_at_initialization()
    test_detail_only_spd_has_no_base_parameters_and_preserves_base()
    test_selective_detail_spd_is_identity_initialized_and_bypasses_base()
    test_selective_detail_correction_is_bounded_per_channel()
    test_selective_detail_checkpoint_metadata_rebuilds_architecture()
    test_spd_correction_is_rms_bounded()
    test_zero_initialized_spd_output_head_receives_gradient()
    test_denoising_head_is_image_identity_at_initialization()
    test_spd_wrapper_keeps_encoder_frozen_and_in_eval_mode()
    test_spd_loss_is_finite_and_masked()
    test_controlled_perturbation_preserves_invalid_pixels()
    test_feature_distillation_is_zero_for_matching_features()
    test_official_validation_split_keeps_coordinate_groups_intact()
    test_tiled_evaluation_covers_last_boundary_once()
    test_zero_spd_makes_image_and_feature_fusion_paths_identical()
    test_e2_adapt_only_trains_fusion_modules()
    test_joint_line_mask_requires_support_from_both_images()
    test_high_confidence_line_mask_is_continuous_above_floor()
    test_continuity_filter_removes_fragments_and_keeps_long_lines()
    print("SPD unit checks passed.")
