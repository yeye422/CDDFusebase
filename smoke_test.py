"""Minimal CUDA/CPU forward-backward check for both CDDFuse phases."""

import torch
import torch.nn as nn
import kornia

from net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)
from utils.loss import Fusionloss, cc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Restormer_Encoder().to(device)
    decoder = Restormer_Decoder().to(device)
    base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)

    visible = torch.rand(1, 1, 128, 128, device=device)
    infrared = torch.rand(1, 1, 128, 128, device=device)
    mse = nn.MSELoss()
    ssim = kornia.losses.SSIM(11, reduction="mean")

    # Phase I: reconstruction and decomposition.
    visible_base, visible_detail, _ = encoder(visible)
    infrared_base, infrared_detail, _ = encoder(infrared)
    visible_hat, _ = decoder(visible, visible_base, visible_detail)
    infrared_hat, _ = decoder(infrared, infrared_base, infrared_detail)
    decomposition = cc(visible_detail, infrared_detail) ** 2 / (
        1.01 + cc(visible_base, infrared_base)
    )
    phase1_loss = (
        5 * ssim(visible, visible_hat)
        + mse(visible, visible_hat)
        + 5 * ssim(infrared, infrared_hat)
        + mse(infrared, infrared_hat)
        + 2 * decomposition
    )
    phase1_loss.backward()

    for module in (encoder, decoder, base_fuse, detail_fuse):
        module.zero_grad(set_to_none=True)

    # Phase II: feature fusion and fusion loss.
    visible_base, visible_detail, _ = encoder(visible)
    infrared_base, infrared_detail, _ = encoder(infrared)
    fused_base = base_fuse(visible_base + infrared_base)
    fused_detail = detail_fuse(visible_detail + infrared_detail)
    fused, _ = decoder(visible, fused_base, fused_detail)
    fusion_loss, _, _ = Fusionloss().to(device)(visible, infrared, fused)
    decomposition = cc(visible_detail, infrared_detail) ** 2 / (
        1.01 + cc(visible_base, infrared_base)
    )
    phase2_loss = fusion_loss + 2 * decomposition
    phase2_loss.backward()

    print(
        "smoke test passed:",
        f"device={device}",
        f"phase1_loss={phase1_loss.item():.6f}",
        f"phase2_loss={phase2_loss.item():.6f}",
    )


if __name__ == "__main__":
    main()

