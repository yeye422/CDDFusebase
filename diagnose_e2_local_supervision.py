"""Rank equal-budget edits of archived fusion images under local replacement supervision."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from diagnose_e2_primitive_dose import reflect_profile, render_support
from diagnose_e2_primitive_render import (
    CENTER, DATASET, DENSE_SIGMA, LABELS, ROOT, boundary, dump_json, endpoints,
    load_inputs, load_visibility, measure_line, pixel_grid, sampling_grid, source_contrast, write_csv,
)
from maintain_experiment_index import sha256_file
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator
from utils.loss import Sobelxy


DOSE_ARCHIVE = ROOT / "test_result/e2_primitive_dose_v1/missing4"
PEAK_FRACTION = 0.5
GUARD_SLACK = 2.0 / 255
PARALLEL_OFFSETS = (12, -12, 14, -14, 16, -16, 18, -18, 20, -20, 22, -22, 24, -24)
MIN_SAR_CONTRAST = 10.0
SCORES = ("e2", "restored", "shifted", "smoothed")
METRICS = FusionEvaluator.METRIC_NAMES


def signed_profile(image, coordinates):
    """Differentiable median contrast on fixed normal strips, in normalized luminance."""
    height, width = image.shape[-2:]
    scale = image.new_tensor([max(width - 1, 1), max(height - 1, 1)])
    grid = image.new_tensor(coordinates) / scale * 2 - 1
    pixels = F.grid_sample(image, grid.unsqueeze(0), align_corners=True,
                           padding_mode="border")[0, 0]
    positive, negative = pixels.chunk(2, dim=-1)
    return positive.median(dim=-1).values - negative.median(dim=-1).values


def source_shortfall(signal, sign, threshold):
    return F.relu(threshold - sign * signal).mean()


def spot_contrast(image, core, ring):
    """Contrast for a separately annotated compact SAR target."""
    return image[..., core].median() - image[..., ring].median()


def objective(image, sources, sar_valid, edited_region, target, guard, sar_probe=None):
    sobel = Sobelxy()
    optical, sar = sources
    intensity_target = torch.where(sar_valid, torch.maximum(optical, sar), optical)
    gradient_target = torch.where(sar_valid, torch.maximum(sobel(optical), sobel(sar)), sobel(optical))
    intensity = (image - intensity_target).abs()
    gradient = (sobel(image) - gradient_target).abs()
    legacy = intensity + 10 * gradient
    outside = legacy.masked_fill(edited_region, 0).mean()
    coordinates, sign, floor = target
    target_loss = source_shortfall(signed_profile(image, coordinates), sign, floor)
    false_coordinates, guard_ceiling = guard
    false_loss = F.relu(signed_profile(image, false_coordinates).abs() - guard_ceiling).mean()
    sar_loss = image.new_zeros(())
    if sar_probe is not None:
        location, sar_sign, sar_floor = sar_probe
        sar_loss = source_shortfall(signed_profile(image, location), sar_sign, sar_floor)
    fraction = edited_region.float().mean()
    return dict(legacy=legacy.mean(), legacy_intensity=intensity.mean(),
                legacy_gradient=gradient.mean(), legacy_outside=outside, source_target=target_loss,
                guard=false_loss, sar=sar_loss,
                replacement_without_guard=outside + fraction * (target_loss + sar_loss),
                replacement=outside + fraction * (target_loss + sar_loss + false_loss))


def parallel_probes(segment, optical, sar):
    direction = segment[1] - segment[0]
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    height, width = optical.shape
    false = radar = None
    for offset in PARALLEL_OFFSETS:
        translation = np.rint(offset * normal).astype(int)
        moved = segment + translation
        if np.any(moved < 7) or np.any(moved >= (width - 7, height - 7)):
            continue
        _, coordinates = sampling_grid(moved)
        evidence = [source_contrast(image, coordinates, j == 1)
                    for j, image in enumerate((optical, sar))]
        if false is None and evidence == [0.0, 0.0]:
            false = (translation, coordinates)
        if radar is None and evidence[0] == 0 and abs(evidence[1]) >= MIN_SAR_CONTRAST:
            radar = (moved, coordinates, evidence[1])
    return false, radar


def generate_edits(optical, sar, baseline, restored, annotation, *, select_probes=parallel_probes):
    segment = endpoints(annotation)
    _, coordinates = sampling_grid(segment)
    source_support = [source_contrast(image, coordinates, j == 1)
                      for j, image in enumerate((optical, sar))]
    selected = int(np.argmax(np.abs(source_support)))
    contrast = source_support[selected]
    false, radar = select_probes(segment, optical, sar)
    if not contrast or false is None:
        return None, "source_or_unverified_parallel_probe"
    translation, false_coordinates = false
    grid = pixel_grid(baseline.shape)
    start, end = torch.as_tensor(segment, dtype=grid.dtype)
    _, window, across = boundary(grid, start, end, contrast)
    weight = window.numpy()
    diff = restored.astype(np.float32) - baseline
    translated = cv2.warpAffine(diff, np.float32([[1, 0, translation[0]], [0, 1, translation[1]]]),
                                (baseline.shape[1], baseline.shape[0]), flags=cv2.INTER_NEAREST)
    smooth = cv2.GaussianBlur(baseline.astype(np.float32), (0, 0), DENSE_SIGMA) - baseline
    smooth *= weight * (np.abs(across.numpy()) >= 3)
    if selected == 1:
        smooth *= sar > 0
    residuals = dict(shifted=translated, smoothed=smooth)
    sar_probe = radar_segment = None
    if radar is not None:
        radar_segment, radar_coordinates, radar_contrast = radar
        a, b = torch.as_tensor(radar_segment, dtype=grid.dtype)
        radar_delta, radar_window, normal = boundary(grid, a, b, radar_contrast)
        empirical = reflect_profile(sar, normal.numpy(), radar_segment, True)
        signed_delta = radar_delta.numpy() if empirical is None else empirical * radar_window.numpy()
        signed_delta *= sar > 0
        residuals.update(sar_retained=signed_delta, sar_erased=-signed_delta)
        sar_probe = (radar_coordinates[:, CENTER], np.sign(radar_contrast),
                     PEAK_FRACTION * abs(radar_contrast) / 255)
    support = np.logical_or.reduce([diff != 0, *(delta != 0 for delta in residuals.values())])
    if not np.any(diff):
        return None, "unchanged_positive_control"
    target_mae = float(np.abs(diff[support]).mean())
    edits = {"e2": baseline, "restored": restored}
    budgets = {"e2": dict(achieved=0.0, matched=True),
               "restored": dict(achieved=target_mae, matched=target_mae > 0)}
    for key, residual in residuals.items():
        edit, budgets[key] = render_support(baseline[support], residual[support], target_mae)
        edits[key] = baseline.copy()
        edits[key][support] = edit
    return dict(edits=edits, budgets=budgets, target_mae=target_mae,
                support=support, target_coordinates=coordinates, false_coordinates=false_coordinates,
                target=(coordinates[:, CENTER], np.sign(contrast), PEAK_FRACTION * abs(contrast) / 255),
                sar_probe=sar_probe, radar_segment=radar_segment, selected=selected, translation=translation), ""


def evaluate(optical, sar, data):
    sources = tuple(torch.from_numpy(img.astype(np.float32) / 255)[None, None] for img in (optical, sar))
    valid = (sources[1] > 0)
    mask = torch.from_numpy(data["support"])[None, None]
    # Include every Sobel centre affected by any intervention.
    edited_region = F.max_pool2d(mask.float(), 3, stride=1, padding=1) > 0
    coords = data["false_coordinates"][:, CENTER]
    guard_ceiling = torch.maximum(*(signed_profile(image, coords).abs() for image in sources)) + GUARD_SLACK
    rows = []
    for key, image in data["edits"].items():
        value = torch.from_numpy(image.astype(np.float32) / 255)[None, None]
        scores = objective(value, sources, valid, edited_region, data["target"],
                           (coords, guard_ceiling), data["sar_probe"])
        edge = measure_line(image, data["target_coordinates"], data["target"][1])
        sar_signal = (signed_profile(value, data["sar_probe"][0]).median().item() * data["sar_probe"][1]
                      if data["sar_probe"] is not None else None)
        rows.append(dict(method=key, target_mae=data["target_mae"], support_pixels=int(mask.sum()),
                         selected_source=("optical", "sar")[data["selected"]],
                         **data["budgets"][key], **{name: v.item() for name, v in scores.items()},
                         **edge, sar_signed_contrast=sar_signal,
                         **dict(zip(METRICS, FusionEvaluator.metric_vector(image, optical, sar)))))
    return rows


def preview(path, annotation, optical, sar, data):
    segment = endpoints(annotation)
    lines = [(segment, "#59d5ec"), (segment + data["translation"], "#ffa773")]
    if data["radar_segment"] is not None:
        lines.append((data["radar_segment"], "#c5a2ff"))
    box_points = np.vstack([line for line, _ in lines])
    x0, y0 = np.maximum(box_points.min(axis=0) - 16, 0)
    x1, y1 = np.minimum(box_points.max(axis=0) + 17, (optical.shape[1], optical.shape[0]))
    panels = {"Optical": optical, "SAR": sar, **data["edits"]}
    width, height = (x1 - x0) * 2, (y1 - y0) * 2
    sheet = Image.new("RGB", (len(panels) * (width + 8) + 8, height + 47), "#151a20")
    draw = ImageDraw.Draw(sheet)
    for j, (title, img) in enumerate(panels.items()):
        x = 8 + j * (width + 8)
        crop = Image.fromarray(img[y0:y1, x0:x1]).resize((width, height), Image.Resampling.NEAREST)
        sheet.paste(crop, (x, 23))
        draw.text((x, 7), title.replace("_", " "), fill="white")
        if title in ("Optical", "SAR"):
            for line, color in lines:
                a, b = 2 * (line - (x0, y0))
                draw.line((x+a[0], 23+a[1], x+b[0], 23+b[1]), fill=color)
    draw.text((8, height + 29), "Cyan: labelled boundary | Orange: shifted control | Purple: SAR probe",
              fill="#becbdb")
    sheet.save(path)


def summarize_probe(rows):
    by_name = {row["method"]: row for row in rows}
    fixed = [by_name[name] for name in SCORES[1:]]
    comparable = all(row["matched"] for row in fixed)
    restored = by_name["restored"]
    result = dict(primary_matched=comparable,
                  legacy_rank=[row["method"] for row in sorted(fixed, key=lambda r: r["legacy"])],
                  replacement_rank=[row["method"] for row in sorted(fixed, key=lambda r: r["replacement"])],
                  restored_wins=comparable and all(
                      restored["replacement"] < row["replacement"] - 1e-9 for row in fixed[1:]),
                  restored_keeps_guard=comparable and restored["guard"] <= by_name["e2"]["guard"] + 1e-9)
    if "sar_retained" in by_name:
        positive, negative = (by_name[k] for k in ("sar_retained", "sar_erased"))
        result["sar_matched"] = positive["matched"] and negative["matched"]
        result["sar_retention_wins"] = result["sar_matched"] and positive["sar"] < negative["sar"] - 1e-9
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "missing"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = (ROOT / args.output_dir).resolve()
    location = output.relative_to(ROOT).as_posix()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    archive, sample, labels, paths = load_inputs(LABELS, DATASET)
    visible = load_visibility(LABELS / "e2_visibility.csv", labels)
    dose_metadata = json.loads((DOSE_ARCHIVE / "run_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((DOSE_ARCHIVE / "summary.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    if sha256_file(checkpoint) != dose_metadata["checkpoint_sha256"] or dose_metadata["evaluator"] != DEFAULT_PROFILE:
        raise ValueError("Expected the archived E2 checkpoint and evaluation protocol.")
    if (dose_metadata["split_manifest_sha256"] != sample["source_manifest_sha256"]
            or dose_metadata["output_policy"] != "clamp_then_uint8"):
        raise ValueError("The budget archive uses different inputs or output policy.")
    for name, sha in dose_metadata["labels_sha256"].items():
        if sha256_file(LABELS / name) != sha:
            raise ValueError(f"Annotation SHA differs: {name}")
    targets = [r for r in labels if visible.get((r["scene"], r["id"])) == "missing"
               and f"{r['scene']}-{r['id']}" in summary["matched_probes_by_dose"]["8"]]
    if args.mode == "smoke":
        targets = [r for r in targets if (r["scene"], r["id"]) == ("4880", "B")]
    if not targets:
        raise ValueError("No matching archived positive controls.")
    output.mkdir(parents=True, exist_ok=False)
    all_rows, decisions, sources = [], {}, {}
    for row in targets:
        scene = row["scene"]
        key = f"{scene}-{row['id']}"
        optical_path, sar_path, e2_path = paths[scene]
        restored_path = DOSE_ARCHIVE / "mae_8/source_profile/gray" / f"{key}.png"
        hashes = dict(optical=sha256_file(optical_path), sar=sha256_file(sar_path), archived_e2=sha256_file(e2_path))
        if hashes != dose_metadata["input_hashes"][scene]:
            raise ValueError(f"Changed source or background: {key}")
        sources[key] = dict(**hashes, restored=sha256_file(restored_path))
        optical = cv2.cvtColor(cv2.imread(str(optical_path)), cv2.COLOR_BGR2YCrCb)[..., 0]
        sar, baseline, restored = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                                   for path in (sar_path, e2_path, restored_path)]
        if any(image is None or image.dtype != np.uint8 or image.shape != optical.shape
               for image in (sar, baseline, restored)):
            raise ValueError(f"Invalid input image: {key}")
        data, reason = generate_edits(optical, sar, baseline, restored, row)
        if data is None:
            decisions[key] = dict(skipped=reason)
            continue
        scores = evaluate(optical, sar, data)
        decisions[key] = dict(false_translation=data["translation"].tolist(),
                              sar_segment=data["radar_segment"].tolist() if data["sar_probe"] is not None else None,
                              **summarize_probe(scores))
        all_rows.extend(dict(probe=key, **r) for r in scores)
        folder = output / key
        folder.mkdir()
        for method, gray in data["edits"].items():
            Image.fromarray(gray).save(folder / f"{method}.png")
        preview(folder / "preview.png", row, optical, sar, data)
        print(key, decisions[key], flush=True)
    if all_rows:
        write_csv(output / "scores.csv", all_rows)
    stats = dict(probes=len(targets), primary_matched=sum(x.get("primary_matched", False) for x in decisions.values()),
                 restored_wins=sum(x.get("restored_wins", False) for x in decisions.values()),
                 sar_matched=sum(x.get("sar_matched", False) for x in decisions.values()),
                 sar_retention_wins=sum(x.get("sar_retention_wins", False) for x in decisions.values()))
    dump_json(output / "summary.json", dict(**stats, decisions=decisions))
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, sources=sources, evaluator=DEFAULT_PROFILE, output_policy="clamp_then_uint8",
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha256_file(checkpoint),
        split_manifest=sample["source_manifest"], split_manifest_sha256=sample["source_manifest_sha256"],
        floor_fraction=PEAK_FRACTION, guard_slack=GUARD_SLACK, parallel_offsets=PARALLEL_OFFSETS,
        dose_metadata_sha256=sha256_file(DOSE_ARCHIVE / "run_metadata.json"),
        implementation_sha256={p.name: sha256_file(p) for p in
                               (Path(__file__), ROOT / "diagnose_e2_primitive_dose.py",
                                ROOT / "diagnose_e2_primitive_render.py")}))
    identity = "QXS-E2-LOCAL-SUPERVISION-" + hashlib.sha256(location.encode()).hexdigest()[:12].upper()
    register_many([dict(
        id=identity, comparable_group=identity, task="engineering", status="diagnostic_fixed_subset",
        method="Equal-budget ranking under local replacement supervision", dataset="QXS-SAROPT", split="val",
        offset=0, sample_count=len({r["scene"] for r in targets}), checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha256_file(checkpoint), split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], metric_profile=DEFAULT_PROFILE,
        output_policy="clamp_then_uint8", result_dir=location, metadata_path=location + "/run_metadata.json",
        metrics=stats, metric_source=dict(type="json_paths", path=location + "/summary.json",
                                          paths={k: [k] for k in stats}),
        notes="Fixed optical annotation and source-screened SAR line. Image edits only, no model inference or training."
    )])


if __name__ == "__main__":
    main()
