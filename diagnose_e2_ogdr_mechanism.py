"""Frozen OGDR mechanism probe on a fixed subset of the QXS val64 scenes.

Recomputes aligned / shifted / off from the *same trained checkpoint*, plus
kernel/gate sensitivity and float-image deltas. This is diagnostics, not a new
quality metric or a new training experiment; optical gradient masks are weak
unlabelled proxies, not ground-truth structures.
"""

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

from diagnose_e2_information_path import read_pair, to_tensor
from e2_ogdr import OGDRE2
from run_qxs_reference_test1000 import DATA
from run_e2_ogdr import experiment
from utils.dataset_os import manifest_pairs


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "scene", "update_rms", "relative_update_rms", "kernel_entropy",
    "normalized_kernel_entropy", "kernel_center_mass", "kernel_spatial_std",
    "aligned_shifted_kernel_mae", "aligned_shifted_gate_mae",
    "off_aligned_float_mae255", "off_shifted_float_mae255",
    "aligned_shifted_float_mae255", "off_aligned_changed_fraction",
    "aligned_shifted_changed_fraction",
    "aligned_optical_edge_enrichment", "shifted_optical_edge_enrichment",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gray_u8(tensor):
    return tensor[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()


def normalized_mae255(first, second):
    return float((first.float() - second.float()).abs().mean() * 255)


def optical_edge_mask(image):
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    threshold = np.quantile(magnitude, 0.8)
    return magnitude > threshold  # Strict: reject constant all-zero images.


def edge_enrichment(delta, edge_mask):
    if not edge_mask.any() or edge_mask.all():
        return float("nan")
    delta = delta.astype(np.float32)
    edge = float(delta[edge_mask].mean())
    elsewhere = float(delta[~edge_mask].mean())
    return edge / (elsewhere + 1e-6)


def probe_scene(model, pair, device):
    optical, sar = read_pair(pair)
    o, s = to_tensor(optical, device), to_tensor(sar, device)
    detail_head = model.detail_fusion
    with torch.inference_mode():
        encoded = model.encode_pair(o, s)
        base, off_detail = model.fuse_features(*encoded, condition="off")
        _, (on_detail, stats) = model.fuse_features(
            *encoded, condition="aligned", return_diagnostics=True)
        _, shifted_detail = model.fuse_features(*encoded, condition="shifted")
        # Reuse the exact same old fusion output and Base to isolate the
        # kernel-condition perturbation from the rest of the full model.
        optical_detail = encoded[1]
        aligned_context, _ = detail_head._condition(
            optical_detail, off_detail, base, "guided")
        shifted_context, _ = detail_head._condition(
            optical_detail, off_detail, base, "shifted")
        batch, height, width = off_detail.shape[0], *off_detail.shape[-2:]
        logits_a = detail_head.kernel_predictor(aligned_context)
        logits_s = detail_head.kernel_predictor(shifted_context)
        kernels_a = logits_a.reshape(batch, 4, 9, height, width).softmax(dim=2)
        kernels_s = logits_s.reshape(batch, 4, 9, height, width).softmax(dim=2)
        gate_a = detail_head.gate_predictor(aligned_context).sigmoid()
        gate_s = detail_head.gate_predictor(shifted_context).sigmoid()

        out_off = model.reconstruct(o, base, off_detail)
        out_on = model.reconstruct(o, base, on_detail)
        out_shift = model.reconstruct(o, base, shifted_detail)
        gray_off, gray_on, gray_shift = (gray_u8(t) for t in (out_off, out_on, out_shift))
        changed_off_on = np.abs(gray_on.astype(np.int16) - gray_off.astype(np.int16))
        changed_off_shift = np.abs(gray_shift.astype(np.int16) - gray_off.astype(np.int16))
        mask = optical_edge_mask(optical)

        entropy = float(stats["kernel_entropy"])
        row = dict(
            scene=pair[0].stem,
            update_rms=float(stats["update_rms"]),
            relative_update_rms=float(stats["relative_update_rms"]),
            kernel_entropy=entropy,
            normalized_kernel_entropy=entropy / math.log(9),
            kernel_center_mass=float(kernels_a[:, :, 4].mean()),
            kernel_spatial_std=float(kernels_a.std(dim=(-2, -1)).mean()),
            aligned_shifted_kernel_mae=float((kernels_a-kernels_s).abs().mean()),
            aligned_shifted_gate_mae=float((gate_a-gate_s).abs().mean()),
            off_aligned_float_mae255=normalized_mae255(out_off, out_on),
            off_shifted_float_mae255=normalized_mae255(out_off, out_shift),
            aligned_shifted_float_mae255=normalized_mae255(out_on, out_shift),
            off_aligned_changed_fraction=float(np.mean(gray_on != gray_off)),
            aligned_shifted_changed_fraction=float(np.mean(gray_on != gray_shift)),
            aligned_optical_edge_enrichment=edge_enrichment(changed_off_on, mask),
            shifted_optical_edge_enrichment=edge_enrichment(changed_off_shift, mask),
        )
        return row, (gray_off, gray_on, gray_shift, changed_off_on, changed_off_shift)


def write_panel(output, scene, arrays):
    off, on, shifted, delta, shift_delta = arrays
    # Common fixed visualization scale; avoid per-scene auto-contrast.
    diff = lambda x: np.minimum(x.astype(np.float32) * 8, 255).astype(np.uint8)
    panel = np.concatenate((off, on, shifted, diff(delta), diff(shift_delta)), axis=1)
    if not cv2.imwrite(str(output / (scene + ".png")), panel):
        raise OSError("Could not save diagnostic panel")


def training_history(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stages = [r for r in rows if r.get("stage") == "fusion"]
    if not stages:
        return {"fusion_epochs": 0}
    fields = ["train_loss", "val_loss", "train_fusion", "val_fusion",
              "train_gradient", "val_gradient", "train_intensity", "val_intensity"]
    result = {"fusion_epochs": len(stages)}
    for name in fields:
        values = [float(r[name]) for r in stages if r.get(name) not in (None, "")]
        if values:
            result[name] = dict(first=values[0], last=values[-1],
                                min=min(values), max=max(values))
    return result


def analyze(rows):
    report = {"count": len(rows), "means": {}, "ranges": {}}
    for key in FIELDS[1:]:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        report["means"][key] = float(finite.mean()) if finite.size else None
        report["ranges"][key] = [float(finite.min()), float(finite.max())] if finite.size else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=8,
                        help="Uniformly spaced val64 scenes; use 64 for complete validation set.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("test_result/e2_ogdr_guided_mechanism_probe_v1/val8_seed42"))
    parser.add_argument("--save-panels", type=int, default=8)
    args = parser.parse_args()
    os.chdir(ROOT)
    exp = experiment("guided", "pilot")
    manifest = json.loads(exp.manifest.read_text(encoding="utf-8"))
    ids = manifest["splits"]["val"]
    if len(ids) != 64 or args.scenes < 1 or args.scenes > 64:
        raise ValueError("Expected existing OGDR pilot val64 and 1..64 requested scenes")
    selected = [ids[index] for index in np.linspace(0, 63, args.scenes, dtype=int)]
    if len(set(selected)) != len(selected):
        raise ValueError("Repeated deterministic validation scene selection")
    pairs, _ = manifest_pairs(DATA, exp.manifest)
    available = {pair[0].stem: pair for pair in pairs["val"]}
    if set(selected) - set(available):
        raise ValueError("Requested scene ID is missing from manifest_pairs")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserve existing mechanism probe: {args.output_dir}")
    saved = torch.load(exp.final, map_location="cpu", weights_only=False)
    if saved["variant"] != exp.variant or saved["epoch"] != 15 or saved["stage"] != "fusion":
        raise ValueError("Expected trained OGDR guided epoch_016 checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OGDRE2("guided").to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    args.output_dir.mkdir(parents=True)
    panel_dir = args.output_dir / "panels"
    panel_dir.mkdir()
    results = []
    for index, scene in enumerate(selected):
        row, panel = probe_scene(model, available[scene], device)
        results.append(row)
        if index < args.save_panels:
            write_panel(panel_dir, scene, panel)
        print(f"OGDR frozen mechanism {index + 1}/{len(selected)} scene={scene}", flush=True)
    with (args.output_dir / "scene_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    summary = analyze(results)
    summary.update(
        protocol=dict(dataset="QXS-SAROPT", split="val", sample_ids=selected,
                      metric_profile="diagnostic_float_and_uint8_not_quality_evaluation",
                      source_checkpoint=exp.final.as_posix(),
                      checkpoint_sha256=sha256_file(exp.final),
                      manifest_sha256=sha256_file(exp.manifest),
                      device=str(device),
                      control="same trained checkpoint, aligned / shifted / off"),
        history=training_history(exp.run_dir / "history.csv"),
        caveat="Optical-image Sobel alignment is an unlabeled weak proxy, not structure truth."
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print("Probe written:", args.output_dir / "summary.json")
    print("Kernel entropy ratio:", summary["means"]["normalized_kernel_entropy"])
    print("Same-checkpoint float output aligned-vs-shifted MAE (0..255):",
          summary["means"]["aligned_shifted_float_mae255"])


if __name__ == "__main__":
    main()
