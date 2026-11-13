r"""Frozen E2 gate energy and Detail scale/DC audit on fixed QXS validation scenes.

Windows CMD:
  .venv\Scripts\python.exe diagnose_e2_gate_scale_v1.py ^
    --scene-ids 4 44 1267 13273 ^
    --output-dir test_result\e2_gate_scale_v1\fixed4_20261113
Read-only: no training, no test split, no registration modifications.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_e2_information_path import (
    ROOT, DEFAULT_CHECKPOINT, DEFAULT_DATA_ROOT, DEFAULT_MANIFEST,
    as_finite, image_u8, locate_scene, read_pair, scalar_rms, sha256, to_tensor,
)
from e2_base_anchor import OpticalBaseAnchoredE2
from utils.dataset_os import validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator

DIRECTIONS = ("LH", "HL", "HH")


def ortho_scale_energy(x: torch.Tensor, haar) -> dict:
    """Exact two-level orthogonal Haar energy; DC separated from coarse spatial."""
    if x.ndim != 4 or any(n % 4 for n in x.shape[-2:]):
        raise ValueError("Two-level Haar requires feature H,W divisible by 4")
    first, _ = haar(x)
    second, _ = haar(first[0])
    total = x.float().square().sum().double()
    fine = sum(b.float().square().sum().double() for b in first[1:])
    middle = sum(b.float().square().sum().double() for b in second[1:])
    ll = second[0].float()
    average_ll = ll.mean(dim=(-1, -2), keepdim=True)
    dc = average_ll.square().sum().double() * (ll.shape[-1] * ll.shape[-2])
    spatial = (ll - average_ll).square().sum().double()
    if total > 1e-20 and abs(float((fine + middle + spatial + dc) / total) - 1) > 5e-5:
        raise AssertionError("Haar Parseval energy identity violated")
    def fraction(value):
        return as_finite(value / total) if total > 1e-20 else None
    avg = x.float().mean(dim=(-1, -2), keepdim=True)
    return {
        "rms": scalar_rms(x),
        "spatial_mean_rms": scalar_rms(avg),
        "spatial_centered_rms": scalar_rms(x.float() - avg),
        "fine_fraction": fraction(fine),
        "middle_fraction": fraction(middle),
        "coarse_spatial_fraction": fraction(spatial),
        "dc_fraction": fraction(dc),
        "coarse_plus_dc_fraction": fraction(spatial + dc),
    }


def directional_gate_audit(raw_bands, gates: torch.Tensor) -> dict:
    """Raw/gated high-band energy; no claim about genuine SAR structures."""
    if len(raw_bands) != 4 or gates.ndim != 5 or gates.shape[2] != 3:
        raise ValueError("Require LL/LH/HL/HH and B,C,3,h,w gates")
    out = {}
    for i, name in enumerate(DIRECTIONS):
        high = raw_bands[i + 1].float()
        gate = gates[:, :, i].float()
        if high.shape != gate.shape:
            raise ValueError(f"Gate shape mismatch in {name}")
        unfiltered = high.square().sum().double()
        filtered = (gate * high).square().sum().double()
        out[name] = {
            "gate_mean": as_finite(gate.mean()),
            "gate_median": as_finite(gate.median()),
            "raw_rms": scalar_rms(high),
            "gated_rms": scalar_rms(gate * high),
            "energy_survival": as_finite(filtered/unfiltered) if unfiltered > 1e-20 else None,
            "rms_survival": as_finite((filtered/unfiltered).sqrt()) if unfiltered > 1e-20 else None,
            "energy_weighted_gate": as_finite((gate * high.square()).sum()/unfiltered) if unfiltered > 1e-20 else None,
        }
    raw_total = sum(b.float().square().sum().double() for b in raw_bands[1:])
    post_total = sum((b.float()*gates[:, :, i].float()).square().sum().double()
                     for i, b in enumerate(raw_bands[1:]))
    out["total"] = {
        "gate_mean": as_finite(gates.float().mean()),
        "energy_survival": as_finite(post_total/raw_total) if raw_total > 1e-20 else None,
        "rms_survival": as_finite((post_total/raw_total).sqrt()) if raw_total > 1e-20 else None,
        "raw_energy": as_finite(raw_total),
        "gated_energy": as_finite(post_total),
    }
    return out


def compile_detail_stages(model, encoded, fused_base, fused_detail):
    """Replay source projections, dilation branches, Base context, and output."""
    _, optical_detail, _, sar_detail = encoded
    layer = model.detail_fusion
    opt = layer.optical_projection(optical_detail)
    sar = layer.sar_projection(sar_detail)
    joint = torch.cat((opt, sar), dim=1)
    branches = {
        f"dilation_{d}": branch(joint)
        for d, branch in zip((1, 2, 3), layer.scale_branches)
    }
    base_context = layer.base_projection(fused_base)
    cat = torch.cat((opt, sar, *branches.values(), base_context), dim=1)
    pre = layer.output[0](cat)
    activated = layer.output[1](pre)
    reconstructed = layer.output[2](activated)
    torch.testing.assert_close(reconstructed, fused_detail, rtol=0, atol=1e-5)
    return {
        "optical_projection": opt,
        "sar_projection": sar,
        **branches,
        "base_context": base_context,
        "concatenated": cat,
        "pre_activation": pre,
        "post_activation": activated,
        "fused_detail": fused_detail,
    }


def run_scene(model, optical_u8, sar_u8):
    """Report both source attenuation and post-fusion scale/DC composition."""
    dev = next(model.parameters()).device
    optical = to_tensor(optical_u8, dev)
    sar = to_tensor(sar_u8, dev)
    with torch.inference_mode():
        encoded, diagnostic = model.encode_pair(optical, sar, return_diagnostics=True)
        shared = {
            "optical": model.encoder.encoder_level1(model.encoder.patch_embed(optical)),
            "sar": model.encoder.encoder_level1(model.encoder.patch_embed(sar)),
        }
        gate_audit, band_inputs, raw_and_gated = {}, {}, {}
        for source, index in (("optical", 1), ("sar", 3)):
            bands, shape = model.lwpd.haar(shared[source])
            gates = diagnostic[source + "_gates"]
            band_inputs[source] = (bands, shape, gates)
            gate_audit[source] = directional_gate_audit(bands, gates)
            zero = torch.zeros_like(bands[0])
            raw = model.lwpd.haar.inverse((zero, *bands[1:]), shape)
            gated = model.lwpd.haar.inverse(
                (zero, *(bands[i+1]*gates[:, :, i] for i in range(3))), shape
            )
            torch.testing.assert_close(gated, encoded[index], rtol=0, atol=1e-5)
            raw_and_gated[source + "_raw_detail"] = raw
            raw_and_gated[source + "_gated_detail"] = gated
        fb, fd = model.fuse_features(*encoded)
        reference = model.reconstruct(optical, fb, fd)
        torch.testing.assert_close(reference, model(optical, sar), rtol=0, atol=1e-6)
        stages = compile_detail_stages(model, encoded, fb, fd)
        features = {
            **raw_and_gated,
            **stages,
            "fused_base": fb,
            "decoder_reduced": model.decoder.reduce_channel(torch.cat((fb, fd), dim=1)),
        }
        spectra = {
            name: ortho_scale_energy(value, model.lwpd.haar)
            for name, value in features.items()
        }
        ref_gray = image_u8(reference)
        relaxed = {}
        for source, index in (("optical", 1), ("sar", 3)):
            bands, shape, gates = band_inputs[source]
            zero = torch.zeros_like(bands[0])
            # Move gates 10% toward unity, holding upstream features fixed.
            restoration = model.lwpd.haar.inverse(
                (zero, *(bands[i+1]*(1-gates[:, :, i]) for i in range(3))), shape
            )
            modified = list(encoded)
            modified[index] = modified[index] + 0.1*restoration
            new_fb, new_fd = model.fuse_features(*modified)
            new_y = model.reconstruct(optical, new_fb, new_fd)
            new_gray = image_u8(new_y)
            def ssim_delta(src):
                return as_finite(
                    FusionEvaluator._ssim_pair(new_gray, src)
                    - FusionEvaluator._ssim_pair(ref_gray, src)
                )
            relaxed[source] = {
                "gate_relaxation_fraction": 0.1,
                "added_detail_rms": scalar_rms(0.1*restoration),
                "output_rms": scalar_rms(new_y - reference),
                "output_mae_8bit_float": as_finite((new_y-reference).abs().mean()*255),
                "output_ssim_delta_optical": ssim_delta(optical_u8),
                "output_ssim_delta_sar": ssim_delta(sar_u8),
            }
        return {
            "gate_audit": gate_audit,
            "scale_energy": spectra,
            "frozen_gate_relaxation": relaxed,
            "baseline_ssim_optical": as_finite(FusionEvaluator._ssim_pair(ref_gray, optical_u8)),
            "baseline_ssim_sar": as_finite(FusionEvaluator._ssim_pair(ref_gray, sar_u8)),
        }


def aggregate(rows):
    values = {}
    def flatten(prefix, item):
        if isinstance(item, dict):
            for key, value in item.items():
                flatten(f"{prefix}.{key}" if prefix else key, value)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            values.setdefault(prefix, []).append(float(item))
    for item in rows:
        flatten("", item["diagnostics"])
    return {
        name: {"n": len(v), "mean": float(np.mean(v)),
               "min": float(np.min(v)), "max": float(np.max(v))}
        for name, v in sorted(values.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    if len(args.scene_ids) > 12 or len(set(args.scene_ids)) != len(args.scene_ids):
        parser.error("Choose 1-12 distinct scene IDs")
    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest_splits(manifest)
    for scene in args.scene_ids:
        if str(scene) not in manifest["splits"]["val"]:
            parser.error(f"Scene {scene} must be in the fixed validation split")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output directory exists: {output}")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    local_inputs = {
        str(scene): locate_scene(args.data_root, manifest, str(scene))
        for scene in args.scene_ids
    }
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != OpticalBaseAnchoredE2.VARIANT:
        raise ValueError("Require optical Base E2 checkpoint")
    model = OpticalBaseAnchoredE2(checkpoint["variant"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scene in args.scene_ids:
        paths = local_inputs[str(scene)]
        o, s = read_pair(paths)
        d = run_scene(model, o, s)
        rows.append({
            "scene_id": str(scene),
            "inputs_sha256": {"optical": sha256(paths[0]), "sar": sha256(paths[1])},
            "diagnostics": d,
        })
        print(f"val {scene}: optical gate energy survival "
              f"{d['gate_audit']['optical']['total']['energy_survival']:.4f}, "
              f"SAR {d['gate_audit']['sar']['total']['energy_survival']:.4f}; "
              f"fused Detail DC {d['scale_energy']['fused_detail']['dc_fraction']:.4f}", flush=True)
    data = {
        "protocol": "e2_gate_and_orthogonal_scale_v1",
        "status": "diagnostic_fixed_val_subset",
        "split": "val",
        "scene_ids": args.scene_ids,
        "precision": "fp32_no_tf32",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(Path(__file__)),
        "imported_diagnostic_sha256": sha256(ROOT / "diagnose_e2_information_path.py"),
        "metric_profile": DEFAULT_PROFILE,
        "scenes": rows,
        "aggregates": aggregate(rows),
        "limitations": [
            "Gate energy survival is not retention of true/usable information",
            "Frozen gate relaxation is not an estimate of retraining gain",
            "No clean SAR, semantic labels, or guaranteed pixel-perfect alignment",
            "Four preselected development scenes cannot establish population statistics",
            "Haar statistics operate on encoder features, not physical SAR intensities",
        ],
    }
    with (output / "diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id", "gate_optical_energy_survival", "gate_sar_energy_survival",
            "raw_detail_optical_rms", "gated_detail_optical_rms",
            "raw_detail_sar_rms", "gated_detail_sar_rms",
            "fused_detail_dc_fraction", "fused_detail_fine_fraction",
            "fused_detail_middle_fraction", "fused_detail_coarse_spatial_fraction",
            "optical_projection_dc_fraction", "sar_projection_dc_fraction",
            "detail_pre_activation_dc_fraction", "detail_post_activation_dc_fraction",
            "relax_optical_output_rms", "relax_sar_output_rms",
            "relax_optical_ssimO_delta", "relax_sar_ssimO_delta",
        ])
        for row in rows:
            d = row["diagnostics"]
            gate, energy, relax = d["gate_audit"], d["scale_energy"], d["frozen_gate_relaxation"]
            writer.writerow([
                row["scene_id"], gate["optical"]["total"]["energy_survival"],
                gate["sar"]["total"]["energy_survival"],
                energy["optical_raw_detail"]["rms"],
                energy["optical_gated_detail"]["rms"],
                energy["sar_raw_detail"]["rms"],
                energy["sar_gated_detail"]["rms"],
                *[energy["fused_detail"][k] for k in (
                    "dc_fraction", "fine_fraction", "middle_fraction",
                    "coarse_spatial_fraction")],
                energy["optical_projection"]["dc_fraction"],
                energy["sar_projection"]["dc_fraction"],
                energy["pre_activation"]["dc_fraction"],
                energy["post_activation"]["dc_fraction"],
                relax["optical"]["output_rms"], relax["sar"]["output_rms"],
                relax["optical"]["output_ssim_delta_optical"],
                relax["sar"]["output_ssim_delta_optical"],
            ])
    print("Saved", output / "diagnosis.json", "and", output / "summary.csv")


if __name__ == "__main__":
    main()
