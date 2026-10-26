"""Regression checks for MFFA Phase-2 evaluation conversion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluate_mffa_v1_phase2 import as_u8


def test_as_u8_detaches_tensor_with_autograd_history():
    source = torch.rand(1, 1, 8, 8, requires_grad=True)
    prediction = torch.sigmoid(source * 2.0)
    converted = as_u8(prediction)
    assert converted.dtype == np.uint8
    assert converted.shape == (8, 8)


if __name__ == "__main__":
    test_as_u8_detaches_tensor_with_autograd_history()
    print("MFFA Phase-2 evaluation regression checks passed.")
