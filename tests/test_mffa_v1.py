"""Unit checks for the first MFFA architecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mffa_v1 import FixedFrequencyAnalysis, MFFAV1, OpticalAnchoredHighFusion


def test_fixed_analysis_synthesis_is_exact_on_valid_pixels():
    torch.manual_seed(1)
    analysis = FixedFrequencyAnalysis()
    image = torch.rand(2, 1, 32, 32).clamp(1.0 / 255.0, 254.0 / 255.0)
    bands = analysis(image)
    reconstructed, _ = analysis.synthesize(bands["low"], bands["mid"], bands["high"])
    assert torch.allclose(reconstructed, image, atol=1e-6)


def test_high_fusion_is_optical_identity_at_initialization():
    torch.manual_seed(2)
    fusion = OpticalAnchoredHighFusion()
    optical = torch.randn(2, 64, 16, 16)
    sar = torch.randn_like(optical)
    fused = fusion(optical, sar)
    assert torch.equal(fused, optical)


def test_high_sar_residual_is_bounded_per_channel():
    torch.manual_seed(3)
    fusion = OpticalAnchoredHighFusion(max_relative_sar=0.05)
    optical = torch.randn(1, 64, 8, 8)
    sar = torch.randn_like(optical)
    with torch.no_grad():
        fusion.sar_delta.weight.fill_(0.1)
        fusion.sar_delta.bias.fill_(0.1)
        fused = fusion(optical, sar)
    rms = optical.square().mean(dim=(2, 3), keepdim=True).sqrt()
    assert torch.all((fused - optical).abs() <= 0.05 * rms + 1e-6)


def test_optical_and_sar_detail_encoders_do_not_share_parameters():
    model = MFFAV1()
    optical_ids = {id(parameter) for parameter in model.optical_detail_encoder.parameters()}
    sar_ids = {id(parameter) for parameter in model.sar_detail_encoder.parameters()}
    assert optical_ids.isdisjoint(sar_ids)


def test_mffa_forward_shapes_and_no_raw_image_decoder_argument():
    torch.manual_seed(4)
    model = MFFAV1()
    optical = torch.rand(1, 1, 32, 32)
    sar = torch.rand_like(optical)
    fused, outputs = model.fuse(optical, sar, return_diagnostics=True)
    assert fused.shape == optical.shape
    assert outputs["fused"]["low"].shape[-2:] == (16, 16)
    assert outputs["fused"]["mid"].shape[-2:] == (32, 32)
    assert outputs["fused"]["high"].shape[-2:] == (32, 32)
    assert outputs["diagnostics"]["high_sar_residual_rms"].item() == 0.0


if __name__ == "__main__":
    test_fixed_analysis_synthesis_is_exact_on_valid_pixels()
    test_high_fusion_is_optical_identity_at_initialization()
    test_high_sar_residual_is_bounded_per_channel()
    test_optical_and_sar_detail_encoders_do_not_share_parameters()
    test_mffa_forward_shapes_and_no_raw_image_decoder_argument()
    print("MFFA v1 unit checks passed.")
