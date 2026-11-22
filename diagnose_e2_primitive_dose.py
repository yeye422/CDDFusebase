"""Measure how edit strength affects recovery of annotated missing boundaries."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_e2_primitive_render import (
    BUDGET_TOLERANCE, CENTER, DATASET, DENSE_SIGMA, LABELS, ROOT, boundary, dump_json,
    endpoints, load_inputs, load_visibility, measure_line, pixel_grid, sample_pixels,
    sampling_grid, source_contrast, write_csv,
)
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ARCHIVE = ROOT / "test_result/e2_primitive_render_v1/holdout8"
DOSES = (1, 2, 4, 8)
MAX_PIXEL_EDIT = 24.0
MAX_GAIN = 64.0
METHODS = ("dense", "source_profile", "primitive")
NORMAL_GRID = np.arange(-8.0, 9.0)


def reflect_profile(image, region, segment, is_sar):
    """Get a signed, empirical normal profile without importing tangential texture."""
    start, end = segment.astype(float)
    direction = end - start
    length = np.linalg.norm(direction)
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]])
    centers = start + np.arange(5, length - 5, 2)[:, None] * tangent
    coordinates = centers[:, None] + NORMAL_GRID[None, :, None] * normal
    pixels = sample_pixels(image.astype(np.float32), coordinates)
    valid = np.isfinite(pixels).all(axis=1)
    if is_sar:
        validity = sample_pixels((image > 0).astype(np.uint8), coordinates, cv2.INTER_NEAREST, 0)
        valid &= validity.all(axis=1)
    if not valid.any():
        return None
    median = np.median(pixels[valid], axis=0)
    antisymmetric = 0.5 * (median - median[::-1])
    return np.interp(region, NORMAL_GRID, antisymmetric).astype(np.float32)


def render_support(reference, residual, target):
    """Calibrate in the actual uint8 domain while limiting per-pixel changes."""
    baseline = reference.astype(np.float32)

    def projected(gain):
        edit = np.clip(gain * residual, -MAX_PIXEL_EDIT, MAX_PIXEL_EDIT)
        pixels = np.rint(np.clip(baseline + edit, 0, 255))
        return pixels, float(np.abs(pixels - baseline).mean())

    low, high = 0.0, MAX_GAIN
    _, maximum = projected(high)
    if maximum >= target:
        for _ in range(23):
            mid = (low + high) / 2
            if projected(mid)[1] >= target:
                high = mid
            else:
                low = mid
    else:
        low = high
    gain = min((low, high), key=lambda x: abs(projected(x)[1] - target))
    pixels, achieved = projected(gain)
    return pixels.astype(np.uint8), dict(
        achieved=achieved, gain=gain, attainable_max=maximum,
        matched=abs(achieved - target) <= BUDGET_TOLERANCE,
        max_pixel_edit=float(np.abs(pixels - baseline).max()))


def variants(optical, sar, row):
    sources = (optical, sar)
    segment = endpoints(row)
    _, coordinates = sampling_grid(segment)
    contrasts = [source_contrast(image, coordinates, index == 1) for index, image in enumerate(sources)]
    selected = int(np.argmax(np.abs(contrasts)))
    if contrasts[selected] == 0:
        return None, contrasts, "no_signed_source_at_fixed_geometry"
    image = sources[selected]
    grid = pixel_grid(image.shape)
    start, end = torch.as_tensor(segment, dtype=grid.dtype)
    primitive, window, across = boundary(grid, start, end, contrasts[selected])
    empirical = reflect_profile(image, across.numpy(), segment, selected == 1)
    if empirical is None:
        return None, contrasts, "no_valid_source_profile"
    weight = window.numpy()
    support = (weight > 0) & ((image > 0) if selected == 1 else True)
    highpass = image.astype(np.float32) - cv2.GaussianBlur(image.astype(np.float32), (0, 0), DENSE_SIGMA)
    return dict(selected=selected, coordinates=coordinates, support=support,
                residuals=dict(dense=weight * highpass, source_profile=weight * empirical,
                               primitive=primitive.numpy())), contrasts, ""


def read_probe(image, coordinates, sign):
    result = measure_line(image, coordinates, sign)
    pixels = sample_pixels(image.astype(np.float32), coordinates[:, CENTER])
    first, second = np.split(pixels[np.isfinite(pixels).all(axis=1)], 2, axis=1)
    difference = np.median(first, axis=1) - np.median(second, axis=1) if first.size else []
    result["center_jump_gray"] = float(np.median(sign * difference)) if len(difference) else None
    return result


def preview(path, row, optical, sar, background, edited):
    points = endpoints(row)
    x0, y0 = np.maximum(points.min(axis=0) - 18, 0)
    x1, y1 = np.minimum(points.max(axis=0) + 19, (optical.shape[1], optical.shape[0]))
    cell_w, cell_h = 174, 136
    headers = ("Optical Y", "SAR", "Archived E2", "Dense residual", "Source profile", "Primitive")
    sheet = Image.new("RGB", (len(headers) * cell_w + 100, 32 + len(DOSES) * cell_h), "#171b22")
    draw = ImageDraw.Draw(sheet)

    def paste(image, column, index, match=True):
        crop = Image.fromarray(image[y0:y1, x0:x1]).convert("RGB")
        crop.thumbnail((cell_w - 12, cell_h - 12), Image.Resampling.NEAREST)
        scale = max(1, min((cell_w - 12) // crop.width, (cell_h - 12) // crop.height))
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        sheet.paste(crop, (100 + column * cell_w + (cell_w - crop.width) // 2, 32 + index * cell_h))
        if not match:
            draw.text((105 + column * cell_w, 35 + index * cell_h), "Unmatched budget", fill="#ffac71")

    for column, title in enumerate(headers):
        draw.text((106 + column * cell_w, 10), title, fill="white")
    draw.text((8, 33), f"{row['scene']}-{row['id']}", fill="white")
    for index, dose in enumerate(DOSES):
        draw.text((8, 57 + index * cell_h), f"MAE {dose}", fill="#d1d9e4")
        for column, image in enumerate((optical, sar, background)):
            paste(image, column, index)
        for column, method in enumerate(METHODS, start=3):
            image, info = edited[dose][method]
            paste(image, column, index, info["matched"])
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "missing"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATASET)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = (ROOT / args.output_dir).resolve()
    location = output.relative_to(ROOT).as_posix()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    archive, sample, labels, paths = load_inputs(LABELS, args.data_root)
    visibility = load_visibility(LABELS / "e2_visibility.csv", labels)
    targets = [row for row in labels if visibility.get((row["scene"], row["id"])) == "missing"]
    if args.mode == "smoke":
        targets = [row for row in targets if (row["scene"], row["id"]) == ("4880", "B")]
    if not targets:
        raise ValueError("No annotated missing boundaries found.")
    previous = json.loads((ARCHIVE / "run_metadata.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != archive["checkpoint_hashes"]["e2"] or checkpoint_hash != previous["checkpoint_sha256"]:
        raise ValueError("Archived checkpoint hash differs.")
    if (previous["split_manifest_sha256"] != sample["source_manifest_sha256"]
            or previous["evaluator"] != DEFAULT_PROFILE or previous["output_policy"] != "clamp_then_uint8"):
        raise ValueError("Archived evaluation protocol differs.")
    labels_hashes = {name: sha256_file(LABELS / name) for name in ("source_labels.csv", "e2_visibility.csv")}
    if any(previous["labels_sha256"].get(name) != value for name, value in labels_hashes.items()):
        raise ValueError("Boundary annotations differ from the eight-scene audit.")
    metric_names = FusionEvaluator.METRIC_NAMES
    with (ARCHIVE / "e2/metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        cached = {row["image"]: [float(row[k] or "nan") for k in metric_names]
                  for row in csv.DictReader(handle) if row["image"] in paths}
    output.mkdir(parents=True, exist_ok=False)
    (output / "preview").mkdir()
    metric_rows = {dose: {key: [] for key in ("e2", *METHODS)} for dose in DOSES}
    probe_ids_by_dose = {dose: [] for dose in DOSES}
    measured, decisions, input_hashes = [], [], {}
    for row in targets:
        scene, probe = row["scene"], f"{row['scene']}-{row['id']}"
        source_paths = paths[scene]
        hashes = {key: sha256_file(path) for key, path in zip(("optical", "sar", "archived_e2"), source_paths)}
        if hashes != previous["input_sha256"][scene]:
            raise ValueError(f"Source or E2 background differed from the eight-scene audit: {scene}")
        input_hashes[scene] = hashes
        with Image.open(source_paths[0]) as source:
            optical = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2YCrCb)[..., 0]
        sar, background = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in source_paths[1:]]
        data, contrasts, skipped = variants(optical, sar, row)
        decisions.append(dict(probe=probe, scene=scene, optical_contrast=contrasts[0],
                              sar_contrast=contrasts[1], skipped=skipped))
        if data is None:
            continue
        support, coordinates = data["support"], data["coordinates"]
        sign = int(np.sign(contrasts[data["selected"]]))
        base_readout = read_probe(background, coordinates, sign)
        edited = {}
        for dose in DOSES:
            edited[dose] = {}
            for method, residual in data["residuals"].items():
                pixels, info = render_support(background[support], residual[support], dose)
                image = background.copy()
                image[support] = pixels
                edited[dose][method] = (image, info)
            comparable = all(info["matched"] for _, info in edited[dose].values())
            if comparable:
                probe_ids_by_dose[dose].append(probe)
                metric_rows[dose]["e2"].append((cached[scene], (), ()))
            for method, (image, info) in edited[dose].items():
                readout = read_probe(image, coordinates, sign)
                measured.append(dict(probe=probe, scene=scene, method=method, target_mae=dose,
                                     support_pixels=int(support.sum()), all_methods_matched=comparable,
                                     selected_source=("optical", "sar")[data["selected"]],
                                     baseline_center_strength=base_readout["center_strength"],
                                     baseline_center_coverage=base_readout["center_coverage"],
                                     baseline_center_jump_gray=base_readout["center_jump_gray"],
                                     baseline_flank_strength=base_readout["flank_strength"],
                                     baseline_peak_offset_px=base_readout["peak_offset_px"],
                                     **info, **readout))
                folder = output / f"mae_{dose}" / method
                (folder / "gray").mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(folder / "gray" / f"{probe}.png")
                if comparable:
                    metric_rows[dose][method].append((FusionEvaluator.metric_vector(image, optical, sar), (), ()))
        preview(output / "preview" / f"{probe}.png", row, optical, sar, background, edited)
        print(f"{probe}: {sum(all(info['matched'] for _, info in edits.values()) for edits in edited.values())}"
              f"/{len(DOSES)} matched budgets", flush=True)

    contrasts = (("e2", "dense"), ("e2", "source_profile"), ("e2", "primitive"),
                 ("dense", "primitive"), ("source_profile", "primitive"))
    for dose, ids in probe_ids_by_dose.items():
        if ids:
            directory = output / f"mae_{dose}"
            (directory / "e2").mkdir()
            summarize(directory, ids, metric_rows[dose], contrasts, source_metrics=(), diagnostic_fields=())
    write_csv(output / "probe_selection.csv", decisions)
    if measured:
        write_csv(output / "dose_probes.csv", measured)
    counts = dict(selected_probes=len(targets), edited_probes=sum(not item["skipped"] for item in decisions),
                  matched_probe_doses=sum(len(ids) for ids in probe_ids_by_dose.values()))
    dump_json(output / "summary.json", dict(**counts, matched_probes_by_dose=probe_ids_by_dose))
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, scene_ids=list(input_hashes), input_hashes=input_hashes,
        archived_audit_metadata_sha256=sha256_file(ARCHIVE / "run_metadata.json"),
        cached_e2_metrics_sha256=sha256_file(ARCHIVE / "e2/metrics.csv"),
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=checkpoint_hash,
        split_manifest=sample["source_manifest"], split_manifest_sha256=sample["source_manifest_sha256"],
        evaluator=DEFAULT_PROFILE, output_policy="clamp_then_uint8", doses=DOSES,
        max_pixel_edit=MAX_PIXEL_EDIT, max_gain=MAX_GAIN, budget_tolerance=BUDGET_TOLERANCE,
        implementation_sha256={path.name: sha256_file(path) for path in
                               (Path(__file__), ROOT / "diagnose_e2_primitive_render.py")},
        labels_sha256=labels_hashes, real_point_annotations=0))
    identity = "QXS-E2-PRIMITIVE-DOSE-" + hashlib.sha256(location.encode()).hexdigest()[:12].upper()
    register_many([dict(
        id=identity, comparable_group=identity, task="engineering", status="diagnostic_fixed_subset",
        method="Annotated missing boundary dose response with a source-profile control",
        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(input_hashes),
        selected_scene_ids=list(input_hashes), checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], result_dir=location,
        metadata_path=location + "/run_metadata.json", metric_profile=DEFAULT_PROFILE,
        output_policy="clamp_then_uint8", metrics=counts,
        metric_source=dict(type="json_paths", path=location + "/summary.json", paths={key: [key] for key in counts}),
        notes="Archived single-boundary rendering at matched uint8 edit budgets. No training or model inference."
    )])


if __name__ == "__main__":
    main()
