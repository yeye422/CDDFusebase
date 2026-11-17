"""Equal-RMS gradient interventions on frozen E2 fusion Detail."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from diagnose_drf_no_train import gradients
from diagnose_e2_information_path import image_u8, read_pair, to_tensor
from diagnose_stg_no_train import DEFAULT_LABELS, endpoints, load_inputs, load_visibility, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from train_direct_fusion import DirectFusionLosses
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
SAR_LABELS = ROOT / "test_result/okd_frozen_gradient_v1/sar_source_candidates.csv"
METHODS = ("e2", "original", "aligned", "shifted")
DEFAULT_STEP = 0.002
TARGET_FRACTION = 0.8
GUARD = 8


def direction_response(image):
    axis = torch.arange(-4, 5, dtype=image.dtype, device=image.device)
    gaussian = torch.exp(-0.5 * (axis / 1.25).square())
    gaussian = gaussian / gaussian.sum()
    smooth = F.conv2d(F.pad(image, (4,) * 4, mode="reflect"), torch.outer(gaussian, gaussian)[None, None])
    sobel = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    return F.conv2d(F.pad(smooth, (1,) * 4, mode="reflect"), torch.stack((sobel, sobel.T))[:, None])


def optical_teacher(optical):
    g = direction_response(optical)
    magnitude = g.norm(dim=1, keepdim=True)
    dx, dy = g[:, :1], g[:, 1:]
    xx, yy, xy = (F.avg_pool2d(x, 9, stride=1, padding=4) for x in (dx.square(), dy.square(), dx * dy))
    coherence = torch.sqrt((xx - yy).square() + 4 * xy.square()) / (xx + yy).clamp_min(1e-8)
    limits = torch.quantile(magnitude[..., GUARD:-GUARD, GUARD:-GUARD], magnitude.new_tensor([0.60, 0.95]))
    active = (magnitude >= limits[0]) & (magnitude <= limits[1]) & (coherence >= 0.6)
    active[..., :GUARD, :] = active[..., -GUARD:, :] = False
    active[..., :, :GUARD] = active[..., :, -GUARD:] = False
    if not active.any():
        raise ValueError("No eligible optical teacher samples.")
    return g / magnitude.clamp_min(1e-7), magnitude.clamp_min(1e-4), active


def shifted_teacher(teacher):
    def shift_interior(x):
        result = x.clone()
        interior = x[..., GUARD:-GUARD, GUARD:-GUARD]
        h, w = interior.shape[-2:]
        result[..., GUARD:-GUARD, GUARD:-GUARD] = interior.roll((h//3, w//4), dims=(-2, -1))
        return result
    return tuple(shift_interior(x) for x in teacher)


def optical_structure_loss(response, teacher):
    normal, target, mask = teacher
    # Keep signed directional filter responses. Only enforce a magnitude floor,
    # so an edge with source-supported reverse polarity remains admissible.
    projection = (response * normal).sum(dim=1, keepdim=True).abs()
    shortfall = (TARGET_FRACTION - projection / target).relu().square()
    return shortfall.masked_select(mask).mean()


def rms(x):
    return x.float().square().mean().sqrt()


def line_geometry(row):
    a, b = endpoints(row).astype(np.float32)
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    line = a + np.linspace(.15, .85, 33, dtype=np.float32)[:, None] * (b - a)
    corridor = line[None] + np.arange(-6, 7, dtype=np.float32)[:, None, None] * normal
    return normal, corridor


def line_response(gradient, normal, samples):
    projected = gradient[0] * normal[0] + gradient[1] * normal[1]
    return cv2.remap(projected, samples[..., 0], samples[..., 1], cv2.INTER_LINEAR)


def prepare_probes(paths, labels, old_visibility, sar_candidates):
    selected = {}
    assessment = []
    for scene, files in paths.items():
        optical, sar = read_pair(files[:2])
        optical_g, sar_g = gradients(optical, 1.25), gradients(sar, 1.25)
        current = [dict(row, cohort="optical_weak") for row in labels
                   if row["scene"] == scene and old_visibility.get((scene, row["id"])) == "missing"]
        current.extend(dict(row, cohort="sar_dominant") for row in sar_candidates if row["scene"] == scene)
        selected[scene] = []
        for row in current:
            normal, corridor = line_geometry(row)
            opt, radar = (line_response(g, normal, corridor).mean(axis=1) for g in (optical_g, sar_g))
            target = opt if row["cohort"] == "optical_weak" else radar
            best = int(np.argmax(np.abs(target)))
            opt_strength, sar_strength = float(np.abs(opt).max()), float(np.abs(radar).max())
            accepted = row["cohort"] == "optical_weak" or (sar_strength >= 2 and sar_strength >= 1.5 * opt_strength)
            assessment.append(dict(**row, optical_strength=opt_strength, sar_strength=sar_strength, accepted=accepted,
                                   offset=best-6, reference_signed=float(target[best])))
            if accepted:
                selected[scene].append((row, normal, corridor[best:best+1], float(target[best])))
    return selected, assessment


def compare(rows, keys, metrics, reference):
    groups = {name: {tuple(row[k] for k in keys): row for row in rows if row["method"] == name}
              for name in METHODS}
    shared = sorted(set.intersection(*(set(group) for group in groups.values())))
    result = []
    if not shared:
        return result
    for metric in metrics:
        values = np.asarray([[group[key][metric] for key in shared] for group in groups.values()])
        good = np.isfinite(values).all(0)
        if not good.any():
            continue
        control = values[METHODS.index(reference), good]
        for name, value in zip(METHODS, values[:, good]):
            delta = value - control
            result.append(dict(metric=metric, method=name, reference=reference, common_count=int(good.sum()),
                               mean=float(value.mean()), delta=float(delta.mean()),
                               higher=int((delta > 0).sum()), lower=int((delta < 0).sum())))
    return result


def probe_scene(model, losses, files, scene, probes, output, device, step):
    optical, sar = read_pair(files[:2])
    o, s = (to_tensor(image, device) for image in (optical, sar))
    valid = (s > 0).float()
    with torch.no_grad():
        base, detail = model.fuse_features(*model.encode_pair(o, s))
        teacher = optical_teacher(o)
        shifted = shifted_teacher(teacher)
    d = detail.detach().requires_grad_(True)
    baseline = model.reconstruct(o, base, d)
    structure = direction_response(baseline)
    original_loss = losses.fusion(o, s, baseline, valid, gradient_weight=10)[0]
    guide_loss = optical_structure_loss(structure, teacher)
    shuffled_loss = optical_structure_loss(structure, shifted)
    gradients_d = [torch.autograd.grad(loss, d, retain_graph=index < 2)[0].detach()
                   for index, loss in enumerate((original_loss, guide_loss, shuffled_loss))]
    scales = [rms(x) for x in gradients_d]
    if any(not bool(torch.isfinite(g).all()) or float(scale) <= 0 for g, scale in zip(gradients_d, scales)):
        raise FloatingPointError(f"Degenerate intervention gradient on scene {scene}.")
    original, optical_guide, spatial_control = (g / scale for g, scale in zip(gradients_d, scales))
    directions = dict(original=original, aligned=original+optical_guide, shifted=original+spatial_control)
    detail_rms = rms(detail)
    rows, line_rows = [], []
    with torch.no_grad():
        base_gray = image_u8(baseline)
        archive_gray = cv2.imread(str(files[2]), cv2.IMREAD_GRAYSCALE)
        if archive_gray is None or np.abs(base_gray.astype(int) - archive_gray).max() > 1:
            raise ValueError(f"Frozen E2 does not match the archive on {scene}.")
        for method in METHODS:
            if method == "e2":
                fused, actual = baseline.detach(), 0.0
            else:
                g = directions[method]
                perturbed = detail - (step * detail_rms / rms(g)) * g
                actual = float(rms(perturbed-detail) / detail_rms)
                if abs(actual - step) > 1e-6:
                    raise AssertionError("Detail steps must use the same RMS budget.")
                fused = model.reconstruct(o, base, perturbed)
            directory = output / "gray" / method
            directory.mkdir(parents=True, exist_ok=True)
            gray = image_u8(fused)
            if not cv2.imwrite(str(directory / f"{scene}.png"), gray):
                raise OSError(f"Cannot save {method} image {scene}.")
            metrics = dict(zip(FusionEvaluator.METRIC_NAMES, map(float, FusionEvaluator.metric_vector(gray, optical, sar))))
            row = dict(scene=scene, method=method, feature_step_rms_ratio=actual,
                       original_loss=float(losses.fusion(o, s, fused, valid, gradient_weight=10)[0]),
                       aligned_loss=float(optical_structure_loss(direction_response(fused), teacher)),
                       **metrics)
            rows.append(row)
            g_f = gradients(fused[0, 0].cpu().numpy() * 255, 1.25)
            support = teacher[2][0, 0].cpu().numpy().astype(np.uint8)
            for label, normal, samples, target in probes:
                along = line_response(g_f, normal, samples) * np.sign(target)
                active = cv2.remap(support, samples[..., 0], samples[..., 1], cv2.INTER_NEAREST)
                line_rows.append(dict(scene=scene, id=label["id"], cohort=label["cohort"], method=method,
                                      signed_response=float(along.mean()),
                                      continuity=float((along >= 0.5 * abs(target)).mean()),
                                      teacher_support=float(active.mean())))
    diag = dict(scene=scene, teacher_coverage=float(teacher[2].float().mean()), detail_rms=float(detail_rms),
                original_gradient_rms=float(scales[0]), aligned_gradient_rms=float(scales[1]),
                shifted_gradient_rms=float(scales[2]), original_aligned_cosine=float((original * optical_guide).mean()),
                original_shifted_cosine=float((original * spatial_control).mean()))
    return rows, line_rows, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--scene-id", help="Only run one of the eight fixed scenes.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--step-rms", type=float, default=DEFAULT_STEP)
    args = parser.parse_args()
    if not 0 < args.step_rms <= 0.05:
        parser.error("--step-rms must be in (0, 0.05].")
    archive, sample, labels, paths = load_inputs(DEFAULT_LABELS, ROOT.parent / "datasets/QXSLAB_SAROPT")
    if args.scene_id:
        if args.scene_id not in paths:
            parser.error("The scene must belong to the fixed val8 subset.")
        paths = {args.scene_id: paths[args.scene_id]}
    visibility_path = DEFAULT_LABELS / "e2_visibility.csv"
    visibility = load_visibility(visibility_path, labels)
    with SAR_LABELS.open(encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    probes, assessment = prepare_probes(paths, labels, visibility, candidates)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "source_probes.csv", assessment)
    if args.sources_only:
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = archive["checkpoints"]["e2"]
    digest = sha256_file(ROOT / checkpoint)
    if digest != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Archived E2 checkpoint hash mismatch.")
    state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    losses = DirectFusionLosses().to(args.device).eval().requires_grad_(False)
    rows, line_rows, diagnostics = [], [], []
    for scene, files in paths.items():
        image_rows, scene_lines, diag = probe_scene(
            model, losses, files, scene, probes[scene], args.output_dir, args.device, args.step_rms,
        )
        rows.extend(image_rows)
        line_rows.extend(scene_lines)
        diagnostics.append(diag)
        print(f"{scene}: {len(scene_lines)//len(METHODS)} retained probes", flush=True)
    write_csv(args.output_dir / "per_image.csv", rows)
    write_csv(args.output_dir / "per_probe.csv", line_rows)
    write_csv(args.output_dir / "gradients.csv", diagnostics)
    comparisons = []
    for control in ("e2", "original", "shifted"):
        comparisons.extend(dict(row, cohort="image") for row in
                           compare(rows, ("scene",), FusionEvaluator.METRIC_NAMES, control))
        for cohort in ("optical_weak", "sar_dominant"):
            selected = [row for row in line_rows if row["cohort"] == cohort]
            comparisons.extend(dict(row, cohort=cohort) for row in
                               compare(selected, ("scene", "id"), ("signed_response", "continuity"), control))
    write_csv(args.output_dir / "paired_summary.csv", comparisons)
    metadata = dict(**sample, evaluated_scene_ids=list(paths), checkpoint=checkpoint, checkpoint_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8", device=args.device,
                    detail_step_relative_rms=args.step_rms, target_response_fraction=TARGET_FRACTION,
                    script_sha256=sha256_file(Path(__file__)), sar_labels_sha256=sha256_file(SAR_LABELS),
                    source_labels_sha256=sha256_file(DEFAULT_LABELS / "source_labels.csv"),
                    visibility_sha256=sha256_file(visibility_path),
                    inputs_sha256={scene: [sha256_file(path) for path in files] for scene, files in paths.items()})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
