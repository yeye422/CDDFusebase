"""E1: exact C0 initialization, separate stems, frozen shared network."""

import torch

from e1_encoder import E1DualStemEncoder
from types import SimpleNamespace

from train_OS import (
    build_models,
    configure_trainability,
    load_model_weights,
    make_optimizers,
)


def synthetic_c0_checkpoint(models):
    return {
        "args": {"variant": "frequency_minimax_baseline"},
        "CDDF_Encoder": models["encoder"].state_dict(),
        "CDDF_Decoder": models["decoder"].state_dict(),
        "BaseFuseLayer": models["base_fuse"].state_dict(),
        "DetailFuseLayer": models["detail_fuse"].state_dict(),
    }


def test_c0_initialization_and_modality_independence():
    torch.manual_seed(123)
    baseline = build_models(torch.device("cpu"), "frequency_minimax_baseline")
    variant = build_models(torch.device("cpu"), "e1_dual_stem")
    load_model_weights(variant, synthetic_c0_checkpoint(baseline))

    base_encoder = baseline["encoder"].eval()
    encoder = variant["encoder"].eval()
    optical = torch.randn(1, 1, 8, 8)
    sar = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        ref_o = base_encoder(optical)
        ref_s = base_encoder(sar)
        got_o = encoder(optical, "optical")
        got_s = encoder(sar, "sar")
    for ref, actual in zip(ref_o, got_o):
        assert torch.equal(ref, actual), "Optical E1 initialization differs from C0"
    for ref, actual in zip(ref_s, got_s):
        assert torch.equal(ref, actual), "SAR E1 initialization differs from C0"

    assert encoder.optical_stem is not encoder.sar_stem
    assert (
        encoder.optical_stem.proj.weight.data_ptr()
        != encoder.sar_stem.proj.weight.data_ptr()
    )
    assert torch.equal(
        encoder.optical_stem.proj.weight, encoder.sar_stem.proj.weight
    )


def test_freeze_and_gradient_flow():
    torch.manual_seed(123)
    encoder = E1DualStemEncoder()
    models = {
        "encoder": encoder,
        "decoder": torch.nn.Conv2d(1, 1, 1),
        "base_fuse": torch.nn.Conv2d(1, 1, 1),
        "detail_fuse": torch.nn.Conv2d(1, 1, 1),
    }
    configure_trainability(models, "stem_only")
    for name, module in models.items():
        for path, param in module.named_parameters():
            expect_trainable = (
                name == "encoder"
                and path.startswith(("optical_stem.", "sar_stem."))
            )
            assert param.requires_grad == expect_trainable, (name, path)

    optical = torch.randn(1, 1, 8, 8)
    sar = torch.randn(1, 1, 8, 8)
    output_o = encoder(optical, "optical")[0]
    output_s = encoder(sar, "sar")[1]
    (output_o.square().mean() + output_s.square().mean()).backward()
    assert encoder.optical_stem.proj.weight.grad is not None
    assert encoder.sar_stem.proj.weight.grad is not None
    assert encoder.optical_stem.proj.weight.grad.abs().sum() > 0
    assert encoder.sar_stem.proj.weight.grad.abs().sum() > 0
    for name, param in encoder.named_parameters():
        if not name.startswith(("optical_stem.", "sar_stem.")):
            assert param.grad is None, name


def test_unfrozen_encoder_and_end2end_parameter_policies():
    models = {
        "encoder": E1DualStemEncoder(),
        "decoder": torch.nn.Conv2d(1, 1, 1),
        "base_fuse": torch.nn.Conv2d(1, 1, 1),
        "detail_fuse": torch.nn.Conv2d(1, 1, 1),
    }
    configure_trainability(models, "encoder_only")
    for name, module in models.items():
        for path, param in module.named_parameters():
            expected = name == "encoder" and not path.startswith("patch_embed.")
            assert param.requires_grad == expected, (name, path)

    args = SimpleNamespace(
        variant="e1_dual_stem",
        lr=1e-5,
        e1_stem_lr=1e-4,
        e1_fusion_lr=1e-6,
        weight_decay=0.0,
    )
    optimizers = make_optimizers(models, args)
    assert set(optimizers) == {"encoder"}
    groups = {
        group["group_name"]: group["lr"]
        for group in optimizers["encoder"].param_groups
    }
    assert groups == {"modality_stems": 1e-4, "shared_encoder": 1e-5}

    configure_trainability(models, "all")
    for name, module in models.items():
        for path, param in module.named_parameters():
            expected = not (name == "encoder" and path.startswith("patch_embed."))
            assert param.requires_grad == expected, (name, path)
    optimizers = make_optimizers(models, args)
    assert set(optimizers) == set(models)
    assert optimizers["base_fuse"].param_groups[0]["lr"] == 1e-6
    assert optimizers["detail_fuse"].param_groups[0]["lr"] == 1e-6
    assert optimizers["decoder"].param_groups[0]["lr"] == 1e-6


def test_invalid_modality_is_rejected():
    encoder = E1DualStemEncoder()
    try:
        encoder(torch.randn(1, 1, 8, 8), "undefined")
    except ValueError:
        return
    raise AssertionError("Unknown modality must raise ValueError")


if __name__ == "__main__":
    test_c0_initialization_and_modality_independence()
    test_freeze_and_gradient_flow()
    test_invalid_modality_is_rejected()
    test_unfrozen_encoder_and_end2end_parameter_policies()
    print("E1 dual stem invariants passed.")
