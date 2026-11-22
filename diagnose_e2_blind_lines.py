"""Source-only line selection followed by an independent geometry audit on archived E2."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from diagnose_e2_primitive_dose import reflect_profile, render_support
from diagnose_e2_primitive_render import (
    CENTER, DATASET, LABELS, ROOT, boundary, dump_json, endpoints, load_inputs,
    load_visibility, measure_line, pixel_grid, sampling_grid, source_contrast, write_csv,
)
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


METHODS = ("e2", "smooth", "source_profile", "primitive")
MAX_LINES = 8
MIN_LENGTH = 24
MAE = 2


def detect_lines(primary, other, primary_is_sar):
    """Rank source-native straight segments without access to annotations or fused pixels."""
    smooth = cv2.GaussianBlur(primary, (0, 0), 1.0)
    edges = cv2.Canny(smooth, 40, 100, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=MIN_LENGTH, maxLineGap=3)
    accepted = []
    for line in ([] if lines is None else lines[:, 0]):
        segment = line.reshape(2, 2).astype(np.float32)
        length = np.linalg.norm(segment[1] - segment[0])
        if length < MIN_LENGTH or np.any(segment < 10) or np.any(segment > np.array(primary.shape[::-1]) - 11):
            continue
        _, coords = sampling_grid(segment)
        contrast = source_contrast(primary, coords, primary_is_sar)
        other_contrast = source_contrast(other, coords, not primary_is_sar)
        if not contrast or other_contrast * contrast < 0 or (primary_is_sar and other_contrast != 0):
            continue
        accepted.append((float(length * abs(contrast)), segment.copy(), contrast, other_contrast))
    return sorted(accepted, key=lambda item: -item[0])


def edits(optical, background, proposals):
    grid = pixel_grid(optical.shape)
    residuals = {name: np.zeros(optical.shape, np.float32) for name in METHODS[1:]}
    blur_delta = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 2) - background
    support = np.zeros(optical.shape, bool)
    for _, segment, contrast, _ in proposals:
        start, end = torch.as_tensor(segment, dtype=grid.dtype)
        primitive, window, across = boundary(grid, start, end, contrast)
        weight = window.numpy()
        empirical = reflect_profile(optical, across.numpy(), segment, False)
        if empirical is None:
            raise ValueError("Source profile undefined on an accepted optical segment")
        residuals["primitive"] += primitive.numpy()
        residuals["source_profile"] += weight * empirical
        residuals["smooth"] += weight * blur_delta
        support |= weight > 0
    images = {"e2": background}
    budgets = {}
    for method, residual in residuals.items():
        image = background.copy()
        if support.any():
            image[support], budgets[method] = render_support(background[support], residual[support], MAE)
        else:
            budgets[method] = dict(matched=False, achieved=0.0)
        images[method] = image
    return images, budgets, support


def geometry_coverage(segment, proposals):
    """Fraction of annotation samples within 2 px of a similarly oriented finite proposal."""
    centers = sampling_grid(segment)[0][:, CENTER]
    direction = segment[1] - segment[0]
    direction = direction / np.linalg.norm(direction)
    hit = np.zeros(len(centers), bool)
    for _, candidate, _, _ in proposals:
        vector = candidate[1] - candidate[0]
        length = np.linalg.norm(vector)
        tangent = vector / length
        if abs(direction @ tangent) < np.cos(np.deg2rad(15)):
            continue
        along = np.clip((centers - candidate[0]) @ tangent, 0, length)
        closest = candidate[0] + along[:, None] * tangent
        hit |= np.linalg.norm(centers - closest, axis=1) <= 2
    return float(hit.mean())


def score_probe(scene, probe, label, visibility, segment, sign, images, proposals, matched):
    _, coords = sampling_grid(segment)
    rows = []
    for method, image in images.items():
        response = profile(image, coords)[:, CENTER]
        evidence = sign * response if sign else np.abs(response)
        active = evidence >= MIN_SUPPORT
        bounds = np.flatnonzero(np.diff(np.r_[False, active, False]))
        lengths = bounds[1::2] - bounds[::2]
        rows.append(dict(scene=scene, probe=probe, label=label, visibility=visibility,
                         expected_sign=int(sign), method=method, budgets_matched=matched,
                         geometry_coverage=geometry_coverage(segment, proposals),
                         longest_fraction=float(lengths.max(initial=0) / len(active)),
                         **measure_line(image, coords, sign)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "remaining7"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    relative = output.relative_to(ROOT).as_posix()
    archive, sample, labels, paths = load_inputs(LABELS, DATASET)
    visibility = load_visibility(LABELS / "e2_visibility.csv", labels)
    scenes = sample["scene_ids"][:1] if args.mode == "smoke" else sample["scene_ids"][1:]
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Archived checkpoint mismatch")
    output.mkdir(parents=True, exist_ok=False)
    results = {name: [] for name in METHODS}
    for name in METHODS:
        (output / name / "gray").mkdir(parents=True)
    scores, summary = [], {}
    for scene in scenes:
        op, sp, bg = paths[scene]
        with Image.open(op) as image:
            optical = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2YCrCb)[..., 0]
        with Image.open(sp) as image:
            sar = np.array(image)
        proposals = detect_lines(optical, sar, False)[:MAX_LINES]
        radar = detect_lines(sar, optical, True)[:1]
        def record(item):
            score, segment, contrast, other = item
            return dict(rank_score=score, xy=segment.tolist(), contrast=contrast, other_contrast=other)
        dump_json(output / f"selection_{scene}.json",
                  dict(optical=list(map(record, proposals)), sar_probe=list(map(record, radar))))
        with Image.open(bg) as image:
            background = np.array(image)
        if any(x.shape != optical.shape or x.dtype != np.uint8 for x in (sar, background)):
            raise ValueError(f"Invalid source or archived image: {scene}")
        images, budgets, mask = edits(optical, background, proposals)
        matched = all(b["matched"] for b in budgets.values())
        rows = [r for r in labels if r["scene"] == scene]
        for row in rows:
            segment = endpoints(row)
            _, coords = sampling_grid(segment)
            signed = source_contrast(optical, coords, False) if row["label"] == "connected" else 0
            scores.extend(score_probe(scene, row["id"], row["label"],
                                      visibility.get((scene, row["id"]), ""), segment,
                                      np.sign(signed), images, proposals, matched))
        for _, segment, contrast, _ in radar:
            scores.extend(score_probe(scene, "SAR-proxy", "source_response", "", segment,
                                      np.sign(contrast), images, proposals, matched))
        for name, image in images.items():
            Image.fromarray(image).save(output / name / "gray" / f"{scene}.png")
            results[name].append((FusionEvaluator.metric_vector(image, optical, sar), (), ()))
        overlay = Image.fromarray(optical).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for _, line, _, _ in proposals:
            draw.line(list(map(tuple, line.tolist())), fill="cyan", width=1)
        for row in rows:
            draw.line(list(map(tuple, endpoints(row).tolist())),
                      fill="yellow" if row["label"] == "connected" else "magenta", width=1)
        overlay.save(output / f"geometry_{scene}.png")
        summary[scene] = dict(proposals=len(proposals), support_pixels=int(mask.sum()),
                              sar_probes=len(radar), budgets=budgets, all_matched=matched,
                              input_sha256={p.name + "_" + k: sha256_file(p)
                                            for p, k in zip(paths[scene], ("optical", "sar", "e2"))})
    contrasts = [(name, "primitive") for name in METHODS if name != "primitive"]
    summarize(output, scenes, results, contrasts, source_metrics=(), diagnostic_fields=())
    write_csv(output / "probes.csv", scores)
    counts = dict(scenes=len(scenes), matched_scenes=sum(r["all_matched"] for r in summary.values()),
                  confirmed_sar_points=0)
    dump_json(output / "summary.json", dict(**counts, by_scene=summary))
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, scene_ids=scenes, checkpoint=archive["checkpoints"]["e2"],
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], archive_sha256=sample["archive_metadata_sha256"],
        metric_profile=DEFAULT_PROFILE, max_lines=MAX_LINES, min_length_px=MIN_LENGTH, support_mae=MAE,
        geometry_tolerance_px=2, geometry_angle_degrees=15, implementation_sha256=sha256_file(Path(__file__)),
        labels_sha256={f: sha256_file(LABELS / f) for f in ("source_labels.csv", "e2_visibility.csv")}))
    group = "QXS-E2-BLIND-LINES-" + hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    register_many([dict(id=group, comparable_group=group, task="engineering", status="diagnostic_fixed_subset",
                        method="Source-only line selection and equal-budget archived image edits",
                        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(scenes),
                        selected_scene_ids=scenes, checkpoint=archive["checkpoints"]["e2"],
                        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
                        split_manifest_sha256=sample["source_manifest_sha256"], metric_profile=DEFAULT_PROFILE,
                        output_policy="clamp_then_uint8", result_dir=relative,
                        metadata_path=relative + "/run_metadata.json", metrics=counts,
                        metric_source=dict(type="json_paths", path=relative + "/summary.json",
                                           paths={k: [k] for k in counts}),
                        notes="Source-only optical proposals. Archived human probes scored after selection. "
                              "SAR source response is a proxy. See probes.csv and paired_comparison.csv.")])


if __name__ == "__main__":
    main()
