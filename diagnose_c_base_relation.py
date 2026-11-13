"""Frozen E2 Base interaction probe. Processes one explicit validation scene."""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from e2_base_anchor import OpticalBaseAnchoredE2
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, MANIFEST, read_image, read_json, sha256_file, write_json
from utils.dataset_os import validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path("models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth")
METHODS = ("global", "local", "centered_local", "balanced_local")
WINDOW, STRIDE, PRIOR_WEIGHT = 32, 16, 0.5
ROI = (slice(112, 144), slice(112, 144))
REMOTE_ROWS = (slice(0, 64), slice(192, 256))


def region(tensor, box):
    return tensor[..., box[0], box[1]]


def windows(stride):
    for y in range(0, 256 - WINDOW + 1, stride):
        for x in range(0, 256 - WINDOW + 1, stride):
            yield slice(y, y + WINDOW), slice(x, x + WINDOW)


def relation(q, k, mask, centered, epsilon):
    q, k = (tensor.flatten(-2) for tensor in (q, k))
    mask = mask.flatten()
    count = mask.sum()
    if centered:
        q = q - (q * mask).sum(-1, keepdim=True) / count.clamp_min(1)
        k = k - (k * mask).sum(-1, keepdim=True) / count.clamp_min(1)
    q, k = q * mask, k * mask
    q_norm, k_norm = (tensor.norm(dim=-1, keepdim=True) for tensor in (q, k))
    valid = (q_norm > epsilon) & (k_norm.transpose(-2, -1) > epsilon)
    valid = valid & (count >= (2 if centered else 1))
    logits = (q / q_norm.clamp_min(epsilon)) @ (k / k_norm.clamp_min(epsilon)).transpose(-2, -1)
    return logits, valid, count


def balanced_prior(q, k, mask, epsilon):
    shape = (*q.shape[:-2], k.shape[-3])
    numerator = q.new_zeros(shape)
    denominator = torch.zeros_like(numerator)
    for box in windows(WINDOW):
        logits, valid, area = relation(region(q, box), region(k, box), region(mask, box), True, epsilon)
        weight = area * valid
        numerator += logits * weight
        denominator += weight
    return numerator / denominator.clamp_min(1), denominator > 0


def rms(value):
    return float(value.square().mean().sqrt())


def regional_interaction(q, k, v, mask, temperature, epsilon, global_attention, collect):
    prior, prior_valid = balanced_prior(q, k, mask, epsilon)
    prior = torch.where(prior_valid, prior, 0)
    ramp = 1 - (2 * (torch.arange(WINDOW, device=q.device) + 0.5) / WINDOW - 1).abs()
    taper = ramp[:, None] * ramp[None, :]
    total = torch.zeros_like(mask, dtype=v.dtype)
    outputs = {method: torch.zeros_like(v) for method in METHODS[1:]}
    target_attentions = {method: [] for method in outputs}
    stats = []
    for box in windows(STRIDE):
        q_window, k_window, local_mask = region(q, box), region(k, box), region(mask, box)
        local, _, _ = relation(q_window, k_window, local_mask, False, epsilon)
        centered, valid, count = relation(q_window, k_window, local_mask, True, epsilon)
        balanced = torch.where(valid, centered.lerp(prior, PRIOR_WEIGHT), prior)
        # Valid local pairs keep their estimate when no prior pair exists.
        balanced = torch.where(valid & ~prior_valid, centered, balanced)
        centered = torch.where(valid, centered, 0)
        value = region(v, box).flatten(-2)
        target_overlap = all(w.start < r.stop and r.start < w.stop for w, r in zip(box, ROI))
        for method, logits in zip(outputs, (local, centered, balanced)):
            attention = (logits / temperature).softmax(-1)
            region(outputs[method], box).add_((attention @ value).unflatten(-1, (WINDOW, WINDOW)) * taper)
            if target_overlap:
                target_attentions[method].append(attention)
            if collect and method == "centered_local":
                stats.append(dict(y=box[0].start, x=box[1].start, valid_pixels=int(count),
                                  valid_pairs=float(valid.float().mean()), q_rms=rms(q_window),
                                  k_rms=rms(k_window), attention_vs_global_rms=rms(attention - global_attention),
                                  centered_vs_prior_rms=rms(centered - prior)))
        region(total, box).add_(taper)
    return ({name: output / total for name, output in outputs.items()},
            {name: torch.stack(items) for name, items in target_attentions.items()}, stats)


def remote_sar(tensor, mask, kind):
    changed = tensor.clone()
    for box in windows(WINDOW):
        if not any(rows.start <= box[0].start and box[0].stop <= rows.stop for rows in REMOTE_ROWS):
            continue
        tile = region(tensor, box)
        if kind == "affine":
            local_mask = region(mask, box)
            energy = (tile.square() * local_mask).sum((-2, -1), keepdim=True)
            scale = (energy / local_mask.sum().clamp_min(1)).sqrt()
            region(changed, box).copy_(4 * tile + 2 * scale)
        else:
            # Permute valid SAR Q/K positions together, preserving the invalid mask
            # and within-tile SAR marginal statistics while disrupting registration.
            indices = region(mask, box).flatten().nonzero().flatten()
            values = tile.flatten(-2)
            shuffled = values.clone()
            shuffled[..., indices] = values[..., indices.roll(indices.numel() // 2)]
            region(changed, box).copy_(shuffled.reshape_as(tile))
    return changed


def quality(gray, optical, sar):
    metrics = dict(zip(FusionEvaluator.METRIC_NAMES, FusionEvaluator.metric_vector(gray, optical, sar)))
    for name, source in (("O", optical), ("S", sar)):
        metrics[f"SSIM_{name}"] = FusionEvaluator._ssim_pair(gray, source)
        metrics[f"ROI_SSIM_{name}"] = FusionEvaluator._ssim_pair(region(gray, ROI), region(source, ROI))
    return {name: float(value) if math.isfinite(value) else None for name, value in metrics.items()}


def compare(probed, original):
    base_delta = region(probed["base"] - original["base"], ROI)
    image_delta = region(probed["image"] - original["image"], ROI)
    correction_delta = region(probed["correction"] - original["correction"], ROI)
    return dict(
        attention_rms={name: rms(attn - original["attention"][name]) for name, attn in probed["attention"].items()},
        correction_rms=rms(correction_delta),
        relative_correction_rms=rms(correction_delta) / max(rms(region(original["correction"], ROI)), 1e-12),
        base_rms=rms(base_delta),
        relative_base_rms=rms(base_delta) / max(rms(region(original["base"], ROI)), 1e-12),
        image_roi_mae_8bit=float(image_delta.abs().mean() * 255),
        image_roi_max_8bit=float(image_delta.abs().amax() * 255),
    )


@torch.inference_mode()
def probe(optical, sar, device, output):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = OpticalBaseAnchoredE2(checkpoint["variant"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    optical_t, sar_t = [torch.from_numpy(a.astype(np.float32) / 255)[None, None].to(device)
                        for a in (optical, sar)]
    bo, do, bs, ds = model.encode_pair(optical_t, sar_t)
    layer = model.base_fusion
    oq, ok = layer.optical_qk(layer.optical_norm(bo)).chunk(2, dim=1)
    sq, sk = layer.sar_qk(layer.sar_norm(bs)).chunk(2, dim=1)
    vo, vs = layer.optical_value(bo), layer.sar_value(bs)
    mask = sar_t[0, 0] > 0

    def heads(tensor):
        return layer._heads(tensor).unflatten(-1, optical.shape)

    runs = {}
    diagnostics = {}
    for kind in ("clean", "affine", "shuffle"):
        sq_probe, sk_probe = sq, sk
        if kind != "clean":
            sq_probe, sk_probe = (remote_sar(tensor, mask, kind) for tensor in (sq, sk))
            for changed, original in ((sq_probe, sq), (sk_probe, sk)):
                torch.testing.assert_close(changed[..., 64:192, :], original[..., 64:192, :], rtol=0, atol=0)
        mixed, matrices = {}, {}
        for name, q, k, v, raw_temp in (
            ("os", oq, sk_probe, vs, layer.raw_temperature_os),
            ("so", sq_probe, ok, vo, layer.raw_temperature_so),
        ):
            global_value, attention = layer._interaction(q, k, v, raw_temp)
            temperature = F.softplus(raw_temp) + layer.epsilon
            local, local_attention, stats = regional_interaction(
                heads(q), heads(k), heads(v), mask, temperature, layer.epsilon, attention, kind == "clean"
            )
            mixed[name] = {"global": global_value.reshape_as(v),
                           **{key: value.reshape_as(v) for key, value in local.items()}}
            matrices[name] = {"global": attention, **local_attention}
            if kind == "clean":
                diagnostics[name] = stats
        runs[kind] = {}
        for method in METHODS:
            correction = layer.transform(layer.compress(torch.cat((vo, vs, mixed["os"][method], mixed["so"][method]), 1)))
            base = bo + correction
            detail = model.detail_fusion(do, ds, base)
            image = model.reconstruct(optical_t, base, detail)
            runs[kind][method] = dict(correction=correction, base=base, image=image,
                                      attention={name: mats[method] for name, mats in matrices.items()})
    expected, _ = layer(bo, bs, return_diagnostics=True)
    torch.testing.assert_close(runs["clean"]["global"]["correction"], expected, rtol=0, atol=0)
    torch.testing.assert_close(runs["clean"]["global"]["image"], model(optical_t, sar_t), rtol=0, atol=0)
    summary = {"valid_fraction": float(mask.float().mean()), "baseline_exact": True,
               "clean_quality": {}, "clean_vs_global": {}, "perturbations": {}}
    for method, result in runs["clean"].items():
        gray = np.clip(np.rint(result["image"][0, 0].cpu().numpy() * 255), 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(output / f"{method}.png"), gray):
            raise OSError(f"Cannot save {method} preview")
        summary["clean_quality"][method] = quality(gray, optical, sar)
        if method != "global":
            compared = compare(result, runs["clean"]["global"])
            del compared["attention_rms"]  # Windowed versus global attention is logged by region.
            summary["clean_vs_global"][method] = compared
    for kind in ("affine", "shuffle"):
        summary["perturbations"][kind] = {method: compare(runs[kind][method], runs["clean"][method]) for method in METHODS}
    write_json(output / "windows.json", diagnostics)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output_dir.resolve().relative_to(ROOT)
    manifest = read_json(MANIFEST)
    validate_manifest_splits(manifest)
    if args.scene_id not in manifest["splits"]["val"]:
        parser.error("scene-id must be in the fixed validation split")
    offset = manifest["splits"]["val"].index(args.scene_id)
    paths = []
    for folder in ("optical_dir", "sar_dir"):
        matches = list((DATA / manifest[folder]).glob(f"{args.scene_id}.*"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one input in {folder} for scene {args.scene_id}")
        paths.append(matches[0])
    optical = cv2.cvtColor(read_image(paths[0], cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)[..., 0]
    sar = read_image(paths[1], cv2.IMREAD_GRAYSCALE)
    if optical.shape != (256, 256) or sar.shape != optical.shape:
        raise ValueError("This fixed-geometry probe requires two 256 × 256 images")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    summary = probe(optical, sar, args.device, output)
    metadata = dict(scene_id=args.scene_id, device=args.device, precision="fp32_no_tf32",
                    split="val", offset=offset, sample_count=1, window=WINDOW, stride=STRIDE,
                    prior_weight=PRIOR_WEIGHT, roi=[[s.start, s.stop] for s in ROI],
                    remote_rows=[[s.start, s.stop] for s in REMOTE_ROWS], sar_valid="SAR uint8 > 0",
                    checkpoint=CHECKPOINT.as_posix(), checkpoint_sha256=sha256_file(CHECKPOINT),
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=sha256_file(MANIFEST),
                    input_sha256={path.as_posix(): sha256_file(path) for path in paths},
                    source_sha256={name: sha256_file(Path(name)) for name in
                                   (Path(__file__).name, "direct_fusion.py", "e2_base_anchor.py", "net.py", "lwpd.py")},
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_round_uint8",
                    torch_version=torch.__version__)
    summary_path = output / "summary.json"
    metadata_path = output / "run_metadata.json"
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)
    metrics = {}
    metric_paths = {}
    for kind, by_method in summary["perturbations"].items():
        for method, values in by_method.items():
            key = f"{kind}_{method}_base_rms"
            metrics[key] = values["base_rms"]
            metric_paths[key] = ["perturbations", kind, method, "base_rms"]
    eid = f"QXS-E2-C-DIAG-{args.scene_id}-{sha256_file(metadata_path)[:12]}"
    register_many([dict(
        id=eid, status="diagnostic_single_scene", task="engineering", dataset="QXS-SAROPT",
        method="E2 光学 Base 固定权重的区域通道关系干预", comparable_group=eid,
        **{key: metadata[key] for key in ("split", "offset", "sample_count", "split_manifest",
                                         "split_manifest_sha256", "checkpoint", "checkpoint_sha256",
                                         "metric_profile", "output_policy")},
        result_dir=output.as_posix(), metadata_path=metadata_path.as_posix(), metrics=metrics,
        metric_source=dict(type="json_paths", path=summary_path.as_posix(), paths=metric_paths),
        notes="单场景，无训练。冻结基线 Q/K/V 投影，在关系估计中干预远端 SAR Q/K。",
    )])
    print(f"Scene {args.scene_id}: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
