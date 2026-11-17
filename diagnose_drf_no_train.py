"""冻结光学 Base E2，在固定小样本上检验 Detail 分离。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnose_e2_information_path import image_u8, read_pair, to_tensor
from diagnose_stg_no_train import DEFAULT_LABELS, endpoints, load_inputs, load_visibility, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
REMOVAL_RMS = 0.10
METHODS = ("e2", "uniform", "optical_base_variance", "detail_coherence", "detail_and_fused_base")
METRICS = (*FusionEvaluator.METRIC_NAMES, "background_hp_rms")


def local_mean(x, kernel=5):
    return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def base_variance(base, haar):
    low = haar(base)[0][0]
    variance = (local_mean(low.square()) - local_mean(low).square()).clamp_min(0)
    return variance.mean(dim=1, keepdim=True)


def direction_coherence(bands):
    horizontal = bands[0].new_ones(1, 5) / 5
    diagonal = torch.eye(5, dtype=horizontal.dtype, device=horizontal.device) / 5
    filters = ((horizontal,), (horizontal.T,), (diagonal, diagonal.flip(1)))
    result = []
    for band, directions in zip(bands, filters):
        b, c, h, w = band.shape
        coeff = band.reshape(b*c, 1, h, w)
        scores = []
        for kernel in directions:
            kernel = kernel[None, None]
            padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            mean = F.conv2d(coeff, kernel, padding=padding).reshape(b, c, h, w)
            power = F.conv2d(coeff.square(), kernel, padding=padding).reshape(b, c, h, w)
            scores.append(mean.square().mean(dim=1, keepdim=True) / power.mean(dim=1, keepdim=True).clamp_min(1e-10))
        result.append(torch.stack(scores).amax(0).clamp(0, 1))
    return result


def split_texture(detail, optical_base, fused_base, haar):
    (low, *high), size = haar(detail)
    optical_var = base_variance(optical_base, haar)
    fixed = (optical_var >= optical_var.median()).to(detail.dtype)
    fused_var = base_variance(fused_base, haar)
    context = fused_var / (fused_var + fused_var.median().clamp_min(1e-10))
    coherence = direction_coherence(high)
    structure_scores = {
        "optical_base_variance": [fixed] * 3,
        "detail_coherence": coherence,
        "detail_and_fused_base": [(score + context) / 2 for score in coherence],
    }
    textures = {"uniform": detail}
    for name, scores in structure_scores.items():
        structure = haar.inverse((low, *(band * score for band, score in zip(high, scores))), size)
        texture = haar.inverse((torch.zeros_like(low), *(band * (1 - score) for band, score in zip(high, scores))), size)
        torch.testing.assert_close(structure + texture, detail, atol=2e-6, rtol=1e-5)
        textures[name] = texture
    return textures


def matched_removal(detail, texture):
    ratio = torch.linalg.vector_norm(texture) / torch.linalg.vector_norm(detail)
    fraction = REMOVAL_RMS / float(ratio)
    if not 0 < fraction <= 1:
        raise ValueError("The candidate texture cannot be removed at the common RMS without over-subtraction.")
    return detail - fraction * texture, fraction


def background_mask(optical, sar, labels):
    valid = cv2.erode((sar > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    for row in labels:
        a, b = endpoints(row)
        cv2.line(valid, tuple(a), tuple(b), 0, 13)
    valid = valid.astype(bool)
    # Select source-flat regions using only low-frequency image gradients.
    optical_strength, sar_strength = map(lambda image: np.hypot(*gradients(image, 3)), (optical, sar))
    flat = (optical_strength <= np.quantile(optical_strength[valid], .35)) & (sar_strength <= np.quantile(sar_strength[valid], .35))
    return valid & flat


def gradients(image, sigma):
    filtered = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    return [cv2.Sobel(filtered, cv2.CV_32F, dx, dy, ksize=3, scale=1/8) for dx, dy in ((1, 0), (0, 1))]


def edge_contrast(dx, dy, row):
    a, b = endpoints(row).astype(np.float32)
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    line = a + np.linspace(.15, .85, 33, dtype=np.float32)[:, None] * (b - a)
    corridor = line[None] + np.arange(-4, 5, dtype=np.float32)[:, None, None] * normal
    response = cv2.remap(dx * normal[0] + dy * normal[1], corridor[..., 0], corridor[..., 1], cv2.INTER_LINEAR)
    return float(np.abs(response.mean(axis=1)).max())


@torch.inference_mode()
def probe_scene(model, paths, labels, visibility, scene, output, device):
    optical, sar = read_pair(paths[:2])
    o, s = (to_tensor(image, device) for image in (optical, sar))
    bo, do, bs, ds = model.encode_pair(o, s)
    base, detail = model.fuse_features(bo, do, bs, ds)
    original = model.reconstruct(o, base, detail)
    archived = cv2.imread(str(paths[2]), cv2.IMREAD_GRAYSCALE)
    original_gray = image_u8(original)
    archive_diff = int(np.abs(archived.astype(int) - original_gray).max())
    if archive_diff > 1:
        raise ValueError(f"Scene {scene}: frozen E2 output disagrees with archive by {archive_diff} levels.")
    textures = split_texture(ds, bo, base, model.lwpd.haar)
    flat = background_mask(optical, sar, labels)
    if not flat.any():
        raise ValueError(f"Scene {scene}: no common source-flat background.")
    rows, edges = [], []
    for method in METHODS:
        gray, fraction = original_gray, 0.0
        if method != "e2":
            removed, fraction = matched_removal(ds, textures[method])
            fused = model.detail_fusion(do, removed, base)
            gray = image_u8(model.reconstruct(o, base, fused))
        directory = output / "gray" / method
        directory.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(directory / f"{scene}.png"), gray):
            raise OSError(f"Cannot save the output for {scene} ({method}).")
        metric_values = FusionEvaluator.metric_vector(gray, optical, sar)
        highpass = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.25)
        rows.append(dict(scene=scene, method=method, texture_fraction_removed=fraction, flat_pixels=int(flat.sum()),
                         archive_max_u8_difference=archive_diff, background_hp_rms=float(np.sqrt(np.mean(highpass[flat]**2))),
                         **dict(zip(FusionEvaluator.METRIC_NAMES, map(float, metric_values)))))
        dx, dy = gradients(gray, 1.25)
        for label in labels:
            if label["label"] == "connected":
                e2_visibility = visibility[(scene, label["id"])]
                edges.append(dict(scene=scene, id=label["id"], visibility=e2_visibility, method=method,
                                  normal_contrast=edge_contrast(dx, dy, label)))
    return rows, edges


def paired(rows, keys, metrics):
    grouped = {name: [row for row in rows if row["method"] == name] for name in METHODS}
    ids = [[tuple(row[key] for key in keys) for row in group] for group in grouped.values()]
    if any(order != ids[0] for order in ids):
        raise ValueError("Mismatched comparisons.")
    summary = []
    for metric in metrics:
        values = np.asarray([[row[metric] for row in group] for group in grouped.values()])
        common = np.isfinite(values).all(axis=0)
        if not common.any():
            continue
        means = values[:, common].mean(axis=1)
        for name, mean in zip(METHODS, means):
            summary.append(dict(metric=metric, method=name, common_count=int(common.sum()), mean=float(mean),
                                delta_e2=float(mean-means[0]), delta_uniform=float(mean-means[1])))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", help="Run just one member of the fixed validation sample.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    archive, sample, labels, paths = load_inputs(DEFAULT_LABELS, ROOT.parent / "datasets/QXSLAB_SAROPT")
    visibility_file = DEFAULT_LABELS / "e2_visibility.csv"
    visibility = load_visibility(visibility_file, labels)
    if args.scene_id:
        if args.scene_id not in paths:
            parser.error("The selected scene is outside the fixed sample.")
        paths = {args.scene_id: paths[args.scene_id]}
    checkpoint = archive["checkpoints"]["e2"]
    digest = sha256_file(ROOT / checkpoint)
    if digest != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Checkpoint does not match the frozen archive.")
    state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, edges = [], []
    for scene, files in paths.items():
        scene_labels = [label for label in labels if label["scene"] == scene]
        values, line_values = probe_scene(model, files, scene_labels, visibility, scene, args.output_dir, args.device)
        rows.extend(values)
        edges.extend(line_values)
        print(f"completed scene {scene}", flush=True)
    write_csv(args.output_dir / "per_image.csv", rows)
    write_csv(args.output_dir / "per_segment.csv", edges)
    comparisons = paired(rows, ("scene",), METRICS)
    for subset in ("all", "missing", "present"):
        selected = [row for row in edges if subset == "all" or row["visibility"] == subset]
        comparisons.extend(dict(row, metric=f"{subset}_normal_contrast")
                           for row in paired(selected, ("scene", "id"), ("normal_contrast",)))
    write_csv(args.output_dir / "paired_summary.csv", comparisons)
    metadata = dict(**sample, evaluated_scene_ids=list(paths), checkpoint=checkpoint, checkpoint_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8", removal_rms=REMOVAL_RMS,
                    device=args.device, evaluation_script_sha256=sha256_file(Path(__file__)),
                    source_labels_sha256=sha256_file(DEFAULT_LABELS / "source_labels.csv"),
                    visibility_sha256=sha256_file(visibility_file),
                    inputs_sha256={scene: [sha256_file(path) for path in files] for scene, files in paths.items()})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
