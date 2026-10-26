"""Regression checks for MFFA Phase-1 losses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from train_mffa_v1_phase1 import Phase1Loss


def test_image_loss_accepts_mixed_prediction_and_target_dtypes():
    criterion = Phase1Loss()
    prediction = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    target = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    valid = torch.ones_like(target)
    total, reconstruction, ssim = criterion.image_loss(
        prediction, target, valid, ssim_weight=0.5
    )
    assert total.dtype == torch.float32
    assert torch.isfinite(total)
    assert torch.isfinite(reconstruction)
    assert torch.isfinite(ssim)


def test_band_loss_accepts_mixed_prediction_and_target_dtypes():
    criterion = Phase1Loss()
    prediction = torch.rand(1, 1, 8, 8, dtype=torch.float64)
    target = torch.rand(1, 1, 8, 8, dtype=torch.float32)
    valid = torch.ones_like(target)
    loss = criterion.normalized_band_l1(prediction, target, valid)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_image_loss_accepts_mixed_prediction_and_target_dtypes()
    test_band_loss_accepts_mixed_prediction_and_target_dtypes()
    print("MFFA Phase-1 loss regression checks passed.")
