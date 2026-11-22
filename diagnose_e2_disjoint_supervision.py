"""Audit probe overlap and test local supervision on separate archived validation scenes."""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from diagnose_e2_local_supervision import (
    MIN_SAR_CONTRAST, evaluate, generate_edits, preview, summarize_probe,
)
from diagnose_e2_primitive_dose import reflect_profile, render_support
from diagnose_e2_primitive_render import (
    CENTER, DATASET, LABELS, ROOT, boundary, dump_json, endpoints, load_inputs,
    load_visibility, pixel_grid, sampling_grid, source_contrast, write_csv,
)
from diagnose_stg_no_train import DEFAULT_LABELS
from maintain_experiment_index import sha256_file
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE


PREVIOUS = ROOT / "test_result/e2_local_supervision_v1/missing/summary.json"
INDEPENDENT_PROBES = (("1276", "B"), ("3513", "A"), ("8456", "A"))
OFFSETS = tuple(sign * extent for extent in range(12, 41, 2) for sign in (1, -1))
RESTORE_MAE = 8


@dataclass
class Region:
    segment: np.ndarray
    coordinates: np.ndarray
    read: np.ndarray
    write: np.ndarray
    coverage: np.ndarray
    across: np.ndarray


def bilinear_footprint(coordinates, shape):
    """Return all pixels with nonzero bilinear interpolation weights."""
    points = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > np.array(shape[::-1]) - 1):
        raise ValueError("Probe reads outside the image.")
    # Use the same float32 coordinate normalization and sampler as signed_profile.
    with torch.enable_grad():
        image = torch.zeros((1, 1, *shape), dtype=torch.float32, requires_grad=True)
        scale = image.new_tensor([max(n - 1, 1) for n in shape[::-1]])
        grid = (image.new_tensor(points) / scale * 2 - 1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(image, grid, align_corners=True, padding_mode="border")
        weights, = torch.autograd.grad(sampled.sum(), image)
    return weights[0, 0].numpy() > 0


def region(segment, grid):
    _, coordinates = sampling_grid(segment)
    start, end = torch.as_tensor(segment, dtype=grid.dtype)
    _, coverage, across = boundary(grid, start, end, 1)
    coverage = coverage.numpy()
    shape = coverage.shape
    return Region(segment, coordinates, bilinear_footprint(coordinates[:, CENTER], shape),
                  coverage > 0, coverage, across.numpy())


def overlap(left, right):
    return dict(read_read=int(np.count_nonzero(left.read & right.read)),
                left_edit_right_read=int(np.count_nonzero(left.write & right.read)),
                right_edit_left_read=int(np.count_nonzero(right.write & left.read)))


def separated(left, right):
    return not any(overlap(left, right).values())


def source_evidence(optical, sar, coordinates):
    return [source_contrast(image, coordinates, index == 1)
            for index, image in enumerate((optical, sar))]


def select_regions(target, optical, sar, grid):
    direction = target.segment[1] - target.segment[0]
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    negatives, radar_lines = [], []
    for offset in OFFSETS:
        translation = np.rint(offset * normal).astype(int)
        moved = target.segment + translation
        if np.any(moved < 7) or np.any(moved >= np.array(optical.shape[::-1]) - 7):
            continue
        candidate = region(moved, grid)
        if not separated(target, candidate):
            continue
        evidence = source_evidence(optical, sar, candidate.coordinates)
        if evidence == [0.0, 0.0]:
            negatives.append((candidate, translation))
        elif evidence[0] == 0 and abs(evidence[1]) >= MIN_SAR_CONTRAST:
            radar_lines.append((candidate, evidence[1]))
    for neg, displacement in negatives:
        for radar, contrast in radar_lines:
            if separated(neg, radar):
                return (neg, displacement), (radar, contrast)
    return (negatives[0] if negatives else None), None


def source_only_preview(path, optical, sar, target, negative, radar):
    lines = [(target.segment, "#59d5ec"), (negative.segment, "#ffa773")]
    if radar is not None:
        lines.append((radar.segment, "#c5a2ff"))
    height, width = optical.shape
    sheet = Image.new("RGB", (2 * width + 30, height + 54), "#151a20")
    draw = ImageDraw.Draw(sheet)
    for j, (name, values) in enumerate((("Optical Y", optical), ("SAR", sar))):
        x = 10 + j * (width + 10)
        draw.text((x, 8), name, fill="white")
        sheet.paste(Image.fromarray(values), (x, 25))
        for line, color in lines:
            a, b = line
            draw.line((x + int(a[0]), 25 + int(a[1]), x + int(b[0]), 25 + int(b[1])), fill=color)
    draw.text((10, height + 32), "Cyan: annotation | Orange: negative | Purple: SAR source evidence",
              fill="#becbdb")
    sheet.save(path)


def cross_edit_counts(data, primary, negative, radar):
    reference = data["edits"]["e2"]
    reads = dict(target=primary.read, guard=negative.read)
    own = dict(restored="target", smoothed="target", shifted="guard")
    if radar is not None:
        reads["sar"] = radar.read
        own.update(sar_retained="sar", sar_erased="sar")
    return {
        method: {name: int(np.count_nonzero((data["edits"][method] != reference) & mask))
                 for name, mask in reads.items() if name != owner}
        for method, owner in own.items()
    }


def check_legacy_geometry():
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    _, _, labels, _ = load_inputs(LABELS, DATASET)
    segments = {f"{row['scene']}-{row['id']}": endpoints(row) for row in labels}
    grid = pixel_grid((256, 256))
    for key, decision in prior["decisions"].items():
        target = region(segments[key], grid)
        negative = region(segments[key] + np.asarray(decision["false_translation"]), grid)
        radar = region(np.asarray(decision["sar_segment"]), grid)
        print(key, json.dumps(dict(target_guard=overlap(target, negative), target_sar=overlap(target, radar),
                                   guard_sar=overlap(negative, radar))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "smoke", "independent"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    if args.mode == "audit":
        check_legacy_geometry()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required when rendering edits")
    output = (ROOT / args.output_dir).resolve()
    location = output.relative_to(ROOT).as_posix()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    archive, sample, labels, paths = load_inputs(DEFAULT_LABELS, DATASET)
    visible = load_visibility(DEFAULT_LABELS / "e2_visibility.csv", labels)
    targets = INDEPENDENT_PROBES[:1] if args.mode == "smoke" else INDEPENDENT_PROBES
    selected = {(row["scene"], row["id"]): row for row in labels}
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if any(f"{scene}-{key}" in previous["decisions"] or visible.get((scene, key)) != "missing"
           for scene, key in targets):
        raise ValueError("Expected separately annotated missing boundaries outside the previous ranking.")
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    checkpoint_hash = sha256_file(checkpoint)
    if (checkpoint_hash != archive["checkpoint_hashes"]["e2"] or archive["metric_profile"] != DEFAULT_PROFILE
            or archive["output_policy"] != "clamp_then_uint8"):
        raise ValueError("Unexpected E2 archive protocol.")

    output.mkdir(parents=True, exist_ok=False)
    sources, decisions, values = {}, {}, []
    for scene, key in targets:
        label = selected[(scene, key)]
        name = f"{scene}-{key}"
        optical_path, sar_path, baseline_path = paths[scene]
        optical_color = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        optical = cv2.cvtColor(optical_color, cv2.COLOR_BGR2YCrCb)[..., 0]
        sar = cv2.imread(str(sar_path), cv2.IMREAD_UNCHANGED)
        if sar is None or sar.dtype != np.uint8 or sar.shape != optical.shape:
            raise ValueError(f"Invalid SAR source: {scene}")
        sources[name] = dict(optical=sha256_file(optical_path), sar=sha256_file(sar_path))
        grid = pixel_grid(optical.shape)
        target = region(endpoints(label), grid)
        evidence = source_evidence(optical, sar, target.coordinates)
        if not any(evidence):
            decisions[name] = dict(skipped="no_evidence_at_fixed_center", source_evidence=evidence)
            continue
        false, radar = select_regions(target, optical, sar, grid)
        if false is None:
            decisions[name] = dict(skipped="no_spatially_separate_negative", source_evidence=evidence)
            continue
        negative, shift = false
        radar_region, radar_contrast = radar if radar is not None else (None, None)
        sector = output / name
        sector.mkdir()
        source_only_preview(sector / "source_probes.png", optical, sar, target, negative, radar_region)
        decision = dict(
            target_source_evidence=evidence, target_segment=target.segment.tolist(),
            guard_segment=negative.segment.tolist(), guard_source_evidence=source_evidence(optical, sar, negative.coordinates),
            sar_source_evidence=radar_contrast, sar_segment=radar_region.segment.tolist() if radar_region else None,
            target_guard=overlap(target, negative))
        if radar_region:
            decision.update(target_sar=overlap(target, radar_region), guard_sar=overlap(negative, radar_region))
        decisions[name] = decision
        dump_json(sector / "source_probes.json", decision)

        # Geometry and source screening are complete before reading the fusion output.
        baseline = cv2.imread(str(baseline_path), cv2.IMREAD_UNCHANGED)
        if baseline is None or baseline.dtype != np.uint8 or baseline.shape != optical.shape:
            raise ValueError(f"Invalid E2 background: {scene}")
        sources[name]["archived_e2"] = sha256_file(baseline_path)
        source_index = int(np.argmax(np.abs(evidence)))
        source = (optical, sar)[source_index]
        empirical = reflect_profile(source, target.across, target.segment, source_index == 1)
        if empirical is None:
            decision["skipped"] = "no_valid_profile"
            continue
        mask = target.write & ((sar > 0) if source_index == 1 else True)
        pixels, budget = render_support(baseline[mask], (target.coverage * empirical)[mask], RESTORE_MAE)
        if not budget["matched"]:
            decision.update(skipped="source_profile_budget_unreachable", restoration_budget=budget)
            continue
        restored = baseline.copy()
        restored[mask] = pixels

        def fixed_selection(segment, optical_image, sar_image):
            radar_probe = ((radar_region.segment, radar_region.coordinates, radar_contrast)
                           if radar_region is not None else None)
            return (shift, negative.coordinates), radar_probe

        data, reason = generate_edits(optical, sar, baseline, restored, label, select_probes=fixed_selection)
        if data is None:
            decision["skipped"] = reason
            continue
        cross = cross_edit_counts(data, target, negative, radar_region)
        if any(pixel_count for row in cross.values() for pixel_count in row.values()):
            raise ValueError(f"An edit changed another probe's readout: {name}: {cross}")
        rows = evaluate(optical, sar, data)
        decision.update(actual_cross_edit_pixels=cross, restoration_budget=budget, **summarize_probe(rows))
        values.extend(dict(probe=name, **row) for row in rows)
        for method, gray in data["edits"].items():
            Image.fromarray(gray).save(sector / f"{method}.png")
        preview(sector / "preview.png", label, optical, sar, data)
        print(name, json.dumps(decision, ensure_ascii=False), flush=True)
    if values:
        write_csv(output / "scores.csv", values)
    counts = dict(selected_probes=len(targets), primary_matched=sum(
        d.get("primary_matched", False) for d in decisions.values()), restored_wins=sum(
        d.get("restored_wins", False) for d in decisions.values()), sar_matched=sum(
        d.get("sar_matched", False) for d in decisions.values()), sar_retention_wins=sum(
        d.get("sar_retention_wins", False) for d in decisions.values()), real_confirmed_sar_points=0)
    dump_json(output / "summary.json", dict(**counts, decisions=decisions))
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, source_hashes=sources, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], scene_ids=[scene for scene, _ in targets],
        metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", search_offsets=OFFSETS,
        optical_or_sar_restore_mae=RESTORE_MAE, previous_summary_sha256=sha256_file(PREVIOUS),
        labels_sha256={name: sha256_file(DEFAULT_LABELS / name)
                       for name in ("sample.json", "source_labels.csv", "e2_visibility.csv")},
        implementation_sha256={name: sha256_file(ROOT / name) for name in
                               ("diagnose_e2_disjoint_supervision.py", "diagnose_e2_local_supervision.py",
                                "diagnose_e2_primitive_render.py", "diagnose_e2_primitive_dose.py")}))
    experiment = "QXS-E2-DISJOINT-SUPERVISION-" + hashlib.sha256(location.encode()).hexdigest()[:12].upper()
    register_many([dict(
        id=experiment, comparable_group=experiment, task="engineering", status="diagnostic_fixed_subset",
        method="Local supervision with independent read footprints and disjoint edits",
        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(targets),
        selected_scene_ids=[scene for scene, _ in targets],
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=checkpoint_hash,
        split_manifest=sample["source_manifest"], split_manifest_sha256=sample["source_manifest_sha256"],
        metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", result_dir=location,
        metadata_path=location + "/run_metadata.json", metrics=counts,
        metric_source=dict(type="json_paths", path=location + "/summary.json",
                           paths={name: [name] for name in counts}),
        notes="Separately annotated missing edges. SAR line support only. No training or inference."
    )])


if __name__ == "__main__":
    main()
