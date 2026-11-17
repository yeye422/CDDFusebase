r"""Frozen optical-Base E2 information-path localization on fixed QXS validation IDs.

Run this script from the CDDFusebase repository root.

Does not train, edit the checkpoint, use test data, or assume SAR speckle labels.
The centered perturbations estimate local response sensitivities, not causal
attribution to an independently retrained component nor a measure of utility.

Example (Windows CMD, repository root)::

    .venv\Scripts\python.exe diagnose_e2_information_path.py ^
      --scene-ids 4 44 1267 13273 ^
      --output-dir test_result\e2_information_path_v1\fixed4_20261113
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from e2_base_anchor import OpticalBaseAnchoredE2
from utils.dataset_os import validate_manifest_splits
from utils.fusion_metrics import FusionEvaluator, DEFAULT_PROFILE

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path("../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json")
DEFAULT_DATA_ROOT = Path("../datasets/QXSLAB_SAROPT")
DEFAULT_CHECKPOINT = Path("models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth")
SOURCE_NAMES = ("optical_base", "optical_detail", "sar_base", "sar_detail")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def as_finite(x):
    x = float(x)
    return x if math.isfinite(x) else None


def image_u8(tensor):
    return (tensor.detach().clamp(0, 1)[0, 0].cpu().numpy() * 255).round().astype(np.uint8)


def scalar_rms(tensor):
    return as_finite(tensor.float().square().mean().sqrt())


def split_lmh(tensor):
    """Algebraically exact feature pyramid: fine + mid + coarse == original."""
    h, w = tensor.shape[-2:]
    if min(h, w) < 4:
        raise ValueError("Need at least 4x4 feature maps")
    p2 = torch_f.interpolate(torch_f.avg_pool2d(tensor, 2), size=(h, w), mode="bilinear", align_corners=False)
    p4 = torch_f.interpolate(torch_f.avg_pool2d(tensor, 4), size=(h, w), mode="bilinear", align_corners=False)
    return {"fine": tensor - p2, "middle": p2 - p4, "coarse": p4}


def gaussian_decompose(image, sigma=2):
    image = np.asarray(image, dtype=np.float64)
    low = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return low, image - low


def safe_pearson(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-15:
        return None
    return as_finite(np.dot(a, b) / denom)


def compare_to_sources(gray, optical, sar):
    """Always compare image content on a shared full-image support.

    Invalid SAR pixels, if present, remain part of the original source image
    for the SSIM/highpass diagnostics, so those values must be interpreted
    alongside the reported SAR-valid fraction.
    """
    l_f, h_f = gaussian_decompose(gray)
    l_o, h_o = gaussian_decompose(optical)
    l_s, h_s = gaussian_decompose(sar)
    return {
        "ssim_optical": as_finite(FusionEvaluator._ssim_pair(gray, optical)),
        "ssim_sar": as_finite(FusionEvaluator._ssim_pair(gray, sar)),
        "low_corr_optical": safe_pearson(l_f, l_o),
        "low_corr_sar": safe_pearson(l_f, l_s),
        "high_corr_optical": safe_pearson(h_f, h_o),
        "high_corr_sar": safe_pearson(h_f, h_s),
        "low_mae_optical": as_finite(np.abs(l_f-l_o).mean()/255),
        "low_mae_sar": as_finite(np.abs(l_f-l_s).mean()/255),
    }


def source_recon_metrics(original, reconstruction):
    ref = original.astype(np.float64) / 255
    rec = reconstruction.astype(np.float64) / 255
    _, hr = gaussian_decompose(original)
    _, hc = gaussian_decompose(reconstruction)
    return {
        "mae": as_finite(np.abs(ref-rec).mean()),
        "ssim": as_finite(FusionEvaluator._ssim_pair(reconstruction, original)),
        "highpass_corr": safe_pearson(hr, hc),
        "highpass_energy_ratio": as_finite(np.mean(hc**2) / max(np.mean(hr**2), 1e-12)),
    }


def locate_scene(data_root, manifest, scene_id):
    files = []
    for key in ("optical_dir", "sar_dir"):
        folder = (data_root / manifest[key]).resolve()
        matches = [p for p in folder.glob(f"{scene_id}.*") if p.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {scene_id} image in {folder}; got {len(matches)}")
        files.append(matches[0])
    return files


def read_pair(paths):
    opt_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    sar = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
    if opt_bgr is None or sar is None:
        raise OSError(f"Cannot read input pair: {paths}")
    opt = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    if opt.shape != sar.shape or opt.shape != (256, 256):
        raise ValueError(f"Expected aligned 256x256 image pairs, got {opt.shape}, {sar.shape}")
    return opt, sar


def to_tensor(image, device):
    return torch.from_numpy(image.astype(np.float32)[None, None] / 255).to(device)


def get_decoder_output(model, base, detail, optical_tensor):
    return model.reconstruct(optical_tensor, base, detail)


def response_stats(positive, negative, reference):
    """Finite difference derivative in output intensity, not information contribution."""
    derivative = (positive - negative) / 2
    # Caller has already offset +/- by eps; response measured per +/- displacement.
    d = derivative[0, 0].detach().cpu().numpy().astype(np.float64)
    image = reference[0, 0].detach().cpu().numpy().astype(np.float64)
    low, high = gaussian_decompose(d, sigma=2)
    return {
        "output_half_difference_rms_0to1": as_finite(np.sqrt(np.mean(d**2))),
        "output_half_difference_mae_8bit": as_finite(np.mean(np.abs(d))*255),
        "response_low_rms": as_finite(np.sqrt(np.mean(low**2))),
        "response_high_rms": as_finite(np.sqrt(np.mean(high**2))),
        "response_high_fraction_energy": as_finite(np.mean(high**2)/max(np.mean(d**2),1e-15)),
        "perturbation_vs_output_std": as_finite(np.sqrt(np.mean(d**2))/max(np.std(image),1e-8)),
    }


def probe_one(model, optical, sar, delta):
    device = next(model.parameters()).device
    o, s = (to_tensor(img, device) for img in (optical, sar))
    with torch.inference_mode():
        encoded, gates = model.encode_pair(o, s, return_diagnostics=True)
        bo, do, bs, ds = encoded
        bf, df = model.fuse_features(*encoded)
        y = get_decoder_output(model, bf, df, o)
        # Identity check guards feature ordering, skips and decoder path.
        torch.testing.assert_close(y, model(o, s), rtol=0, atol=1e-6)

        opt_recon = image_u8(model.reconstruct(o, bo, do))
        sar_recon = image_u8(model.reconstruct(s, bs, ds))
        y8 = image_u8(y)
        quality = compare_to_sources(y8, optical, sar)
        # Baseline uses quantized outputs to match the project's evaluation convention.
        result = {
            "baseline": quality,
            "sar_valid_fraction": as_finite(np.mean(sar > 0)),
            "source_reconstruction": {
                "optical": source_recon_metrics(optical, opt_recon),
                "sar": source_recon_metrics(sar, sar_recon),
            },
            "features": {
                name: {"rms": scalar_rms(tensor)}
                for name, tensor in zip(SOURCE_NAMES, encoded)
            },
            "base_correction_rms": scalar_rms(bf - bo),
            "base_correction_to_optical_base_rms": as_finite(
                scalar_rms(bf-bo) / max(scalar_rms(bo), 1e-12)),
            "decoder_input_rms": {"fused_base": scalar_rms(bf), "fused_detail": scalar_rms(df)},
            "source_path_sensitivity": {},
            "decoder_sensitivity": {},
            "decoder_detail_scale_sensitivity": {},
        }
        for name in ("optical_gates", "sar_gates"):
            tensor = gates.get(name)
            if tensor is not None:
                result["features"][name] = {
                    "mean": as_finite(tensor.float().mean()),
                    "per_direction_mean": [as_finite(x) for x in tensor.float().mean(dim=(0, 1, 3, 4))],
                }
        # (1) Fusion+decoder response to an infinitesimal feature-amplitude change.
        # Every intervention recomputes Base, Detail and decoder with the other
        # three input features held exactly fixed.
        for index, name in enumerate(SOURCE_NAMES):
            altered = []
            for sign in (1., -1.):
                x = list(encoded)
                x[index] = encoded[index] * (1. + sign*delta)
                new_b, new_d = model.fuse_features(*x)
                altered.append(get_decoder_output(model, new_b, new_d, o))
            stat = response_stats(altered[0], altered[1], y)
            stat["response_derivative_rms_per_unit_gain"] = as_finite(
                stat["output_half_difference_rms_0to1"] / delta)
            result["source_path_sensitivity"][name] = stat
        # (2) Decoder-only sensitivity: fused features held apart.
        for name, base, detail in (
            ("fused_base", bf, df),
            ("fused_detail", bf, df),
        ):
            plus_base, minus_base = bf, bf
            plus_detail, minus_detail = df, df
            if name == "fused_base":
                plus_base, minus_base = bf*(1+delta), bf*(1-delta)
            else:
                plus_detail, minus_detail = df*(1+delta), df*(1-delta)
            yp = get_decoder_output(model, plus_base, plus_detail, o)
            ym = get_decoder_output(model, minus_base, minus_detail, o)
            stat = response_stats(yp, ym, y)
            stat["response_derivative_rms_per_unit_gain"] = as_finite(
                stat["output_half_difference_rms_0to1"] / delta)
            result["decoder_sensitivity"][name] = stat
        # (3) Scale-selective Detail injection at the decoder input.
        bands = split_lmh(df)
        torch.testing.assert_close(sum(bands.values()), df, rtol=0, atol=1e-5)
        for name, band in bands.items():
            yp = get_decoder_output(model, bf, df + delta*band, o)
            ym = get_decoder_output(model, bf, df - delta*band, o)
            stat = response_stats(yp, ym, y)
            stat["response_derivative_rms_per_unit_gain"] = as_finite(
                stat["output_half_difference_rms_0to1"] / delta)
            stat["feature_band_energy_fraction"] = as_finite(
                (band.float().square().mean()/df.float().square().mean().clamp_min(1e-12)))
            result["decoder_detail_scale_sensitivity"][name] = stat
        # (4) The optical anchor is checked with both downstream paths.
        # These are *large OOD removal* controls; only for sanity, never causal effect.
        y_anchor_direct = get_decoder_output(model, bo, df, o)
        y_anchor_full = get_decoder_output(model, bo, model.detail_fusion(do, ds, bo), o)
        result["base_correction_ablation_noncausal"] = {
            "decoder_direct_only_mae_8bit": as_finite((y_anchor_direct-y).abs().mean()*255),
            "full_downstream_mae_8bit": as_finite((y_anchor_full-y).abs().mean()*255),
        }
    return result


def aggregate(rows):
    """Numeric leaves summarized across scenes, excluding the scene ID."""
    def flatten(prefix, item, out):
        if isinstance(item, dict):
            for k, value in item.items():
                flatten(f"{prefix}.{k}" if prefix else str(k), value, out)
        elif isinstance(item, list):
            for i, value in enumerate(item):
                flatten(f"{prefix}.{i}", value, out)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            out[prefix] = float(item)
    values = {}
    for row in rows:
        flat = {}; flatten("", row["diagnostics"], flat)
        for key, number in flat.items():
            values.setdefault(key, []).append(number)
    return {key: {"n": len(data), "mean": float(np.mean(data)),
                  "median": float(np.median(data)), "min": min(data), "max": max(data)}
            for key, data in sorted(values.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not 0 < args.delta <= 0.1:
        parser.error("delta must be in (0, 0.1]")
    if len(args.scene_ids) > 12 or len(set(args.scene_ids)) != len(args.scene_ids):
        parser.error("Choose 1-12 distinct scene IDs")
    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest_splits(manifest)
    for scene in args.scene_ids:
        if str(scene) not in manifest["splits"]["val"]:
            parser.error(f"Scene {scene} is not in the fixed validation split")
    output = args.output_dir.resolve()
    # Refuse silent overwrites of results; no new filesystem mutation until validated.
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != OpticalBaseAnchoredE2.VARIANT:
        raise ValueError("This protocol requires lwpd_e2_base_anchor_v1 checkpoint")
    model = OpticalBaseAnchoredE2(checkpoint["variant"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    # Check every input before creating a new output directory.
    scene_paths = {str(scene): locate_scene(args.data_root, manifest, str(scene))
                   for scene in args.scene_ids}
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    for scene in args.scene_ids:
        paths = scene_paths[str(scene)]
        optical, sar = read_pair(paths)
        diagnostic = probe_one(model, optical, sar, args.delta)
        rows.append({"scene_id": str(scene), "input_sha256": {"optical": sha256(paths[0]), "sar": sha256(paths[1])},
                     "diagnostics": diagnostic})
        print(f"Completed val scene {scene}: optical_SSIM={diagnostic['baseline']['ssim_optical']:.5f}, "
              f"SAR_SSIM={diagnostic['baseline']['ssim_sar']:.5f}", flush=True)
    payload = {"protocol": "frozen_e2_information_path_v1", "status": "diagnostic_fixed_subset",
               "split": "val", "scene_ids": args.scene_ids, "count": len(rows),
               "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
               "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
               "script_sha256": sha256(Path(__file__)), "metric_profile": DEFAULT_PROFILE,
               "precision": "fp32_no_tf32", "delta": args.delta, "device": args.device,
               "limits": ["finite-difference response is not information utility", 
                          "source reconstruction tests final trained decoder, not Stage-I checkpoint",
                          "no clean SAR, manual structure label, or pixel-perfect registration assumption",
                          "no independent retraining, no test split, no checkpoint selection"],
               "scenes": rows, "aggregates": aggregate(rows)}
    with (output / "diagnosis.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "SSIM_O", "SSIM_S", "recon_O_SSIM", "recon_S_SSIM",
                         "sens_O_Base", "sens_O_Detail", "sens_S_Base", "sens_S_Detail",
                         "decoder_Base", "decoder_Detail", "decoder_Detail_fine",
                         "decoder_Detail_middle", "decoder_Detail_coarse"])
        for row in rows:
            d = row["diagnostics"]
            def sensitivity(section, key):
                return d[section][key]["response_derivative_rms_per_unit_gain"]
            writer.writerow([row["scene_id"], d["baseline"]["ssim_optical"], d["baseline"]["ssim_sar"],
                             d["source_reconstruction"]["optical"]["ssim"], d["source_reconstruction"]["sar"]["ssim"],
                             *[sensitivity("source_path_sensitivity", k) for k in SOURCE_NAMES],
                             *[sensitivity("decoder_sensitivity", k) for k in ("fused_base", "fused_detail")],
                             *[sensitivity("decoder_detail_scale_sensitivity", k) for k in ("fine", "middle", "coarse")]])
    print(f"Saved {output / 'diagnosis.json'} and {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
