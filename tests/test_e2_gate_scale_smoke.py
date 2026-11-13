"""Synthetic-only checks of Haar partition, gate survival and Detail replay.

No QXS data or E2 checkpoint is loaded. Run from repo root:
    .venv\Scripts\python.exe tests\test_e2_gate_scale_smoke.py
"""
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnose_e2_gate_scale_v1 import (
    ortho_scale_energy, directional_gate_audit, compile_detail_stages,
)
from direct_fusion import MultiScaleDetailFusion
from lwpd import HaarDWT2D


torch.manual_seed(42)
torch.set_num_threads(2)
haar = HaarDWT2D()
tensor = torch.randn(1, 64, 32, 32)
metric = ortho_scale_energy(tensor, haar)
assert abs(sum(metric[key] for key in (
    "fine_fraction", "middle_fraction", "coarse_spatial_fraction",
    "dc_fraction")) - 1.) < 1e-5

constant = torch.ones(1, 64, 32, 32) * 0.7
metric_constant = ortho_scale_energy(constant, haar)
assert metric_constant["dc_fraction"] > 0.99999
assert metric_constant["fine_fraction"] < 1e-7

raw_bands, size = haar(tensor)
gate = torch.full((1, 64, 3, 16, 16), 0.5)
gstats = directional_gate_audit(raw_bands, gate)
assert abs(gstats["total"]["energy_survival"] - .25) < 1e-6

# Stage-wise replay exactly matches MultiScaleDetailFusion's actual forward.
detail_layer = MultiScaleDetailFusion(64).eval()
model = types.SimpleNamespace(detail_fusion=detail_layer)
o_base = tensor
o_detail = tensor * .2
s_base = tensor * .7
s_detail = tensor * .4
base = o_base + .1 * s_base
with torch.inference_mode():
    fused_detail = detail_layer(o_detail, s_detail, base)
    stages = compile_detail_stages(
        model, (o_base, o_detail, s_base, s_detail), base, fused_detail
    )
    assert torch.allclose(stages["fused_detail"], fused_detail)
    for name, value in stages.items():
        stats = ortho_scale_energy(value, haar)
        assert abs(sum(stats[key] for key in (
            "fine_fraction", "middle_fraction", "coarse_spatial_fraction",
            "dc_fraction")) - 1.) < 1e-5, name

print("PASS synthetic Haar Parseval/DC, gate energy, Detail fusion replay")
