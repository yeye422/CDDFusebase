"""Compare annotated boundary rendering with matched dense residuals on archived E2 images."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_e2_contour_evidence import MIN_SUPPORT, OFFSETS, dump_json, profile, sampling_grid
from diagnose_stg_no_train import DATASET, ROOT, endpoints, load_inputs, load_visibility, write_csv
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


LABELS = ROOT / "test_result/e2_contour_evidence_v1/holdout_labels"
METHODS = ("e2", "dense", "primitive")
SUPPORT_RADIUS = 8.0
EDGE_WIDTH = 1.0
DENSE_SIGMA = 2.0
TARGET_MAE = 2.0
BUDGET_TOLERANCE = 0.05
MIN_CONTRAST = 3.0
MIN_SIGN_AGREEMENT = 0.60
CENTER = int(np.flatnonzero(OFFSETS == 0)[0])


def boundary(grid, start, end, contrast):
    """Render a bounded signed step with a smooth, finite support window."""
    direction = end - start
    length = torch.linalg.vector_norm(direction)
    tangent = direction / length
    normal = torch.stack((-tangent[1], tangent[0]))
    relative = grid - start
    along, across = relative @ tangent, relative @ normal
    end_distance = torch.minimum(along, length - along)
    end_weight = torch.sin((end_distance / 5).clamp(0, 1) * (torch.pi / 2)).square()
    across_weight = torch.cos((across.abs() / SUPPORT_RADIUS).clamp(0, 1) * (torch.pi / 2)).square()
    window = torch.where(across.abs() < SUPPORT_RADIUS, end_weight * across_weight, 0.0)
    return 0.5 * contrast * torch.tanh(across / EDGE_WIDTH) * window, window, across


def spot(grid, center, axes, angle, amplitude):
    """Render an elliptical point response for the synthetic geometry check."""
    dx, dy = (grid - center).unbind(-1)
    cosine, sine = angle.cos(), angle.sin()
    radius_squared = ((dx * cosine + dy * sine) / axes[0]).square()
    radius_squared = radius_squared + ((-dx * sine + dy * cosine) / axes[1]).square()
    return amplitude * torch.exp(-0.5 * radius_squared)


def pixel_grid(shape):
    y, x = torch.meshgrid(*(torch.arange(size, dtype=torch.float32) for size in shape), indexing="ij")
    return torch.stack((x, y), dim=-1)


def sample_pixels(image, coordinates, interpolation=cv2.INTER_LINEAR, border=float("nan")):
    xy = coordinates.reshape(-1, coordinates.shape[-2], 2).astype(np.float32)
    values = cv2.remap(image, xy[..., 0].copy(), xy[..., 1].copy(), interpolation,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return values.reshape(coordinates.shape[:-1])


def source_contrast(image, coordinates, is_sar):
    center = coordinates[:, CENTER]
    pixels = sample_pixels(image.astype(np.float32), center)
    valid = np.isfinite(pixels).all(axis=1)
    if is_sar:
        support = sample_pixels((image > 0).astype(np.uint8), center, cv2.INTER_NEAREST, 0)
        valid &= support.all(axis=1)
    if valid.mean() < MIN_SIGN_AGREEMENT:
        return 0.0
    left, right = np.split(pixels[valid], 2, axis=1)
    differences = np.median(left, axis=1) - np.median(right, axis=1)
    signed = float(np.median(differences))
    if abs(signed) < MIN_CONTRAST or np.mean(differences * np.sign(signed) > 0) < MIN_SIGN_AGREEMENT:
        return 0.0
    return signed


def quantize(background, residual, gain):
    return np.rint(np.clip(background + gain * residual, 0, 255)).astype(np.uint8)


def matched_render(background, residuals, support):
    """Match the achieved uint8 MAE of both edits on their common annotated support."""
    reference = background[support].astype(np.float32)

    def effect(residual, gain):
        return float(np.abs(quantize(reference, residual[support], gain) - reference).mean())

    maximum = {name: effect(residual, 1.0) for name, residual in residuals.items()}
    target = min(TARGET_MAE, *maximum.values())
    if target <= 0:
        return {name: background.copy() for name in residuals}, {
            name: dict(gain=0.0, mae=0.0, nominal_mae=value) for name, value in maximum.items()
        }, target
    images, budgets = {}, {}
    for name, residual in residuals.items():
        low, high = 0.0, 1.0
        for _ in range(20):
            middle = (low + high) / 2
            if effect(residual, middle) < target:
                low = middle
            else:
                high = middle
        gain = min((low, high), key=lambda value: abs(effect(residual, value) - target))
        images[name] = quantize(background, residual, gain)
        budgets[name] = dict(gain=gain, mae=effect(residual, gain), nominal_mae=maximum[name])
    if any(abs(row["mae"] - target) > BUDGET_TOLERANCE for row in budgets.values()):
        raise ValueError("The rounded edits could not be matched within the MAE tolerance.")
    return images, budgets, target


def measure_line(image, coordinates, sign):
    signed = profile(image, coordinates)
    evidence = np.abs(signed) if sign == 0 else signed * sign
    offset_strength = np.median(evidence, axis=0)
    # Side responses are a proxy for displaced or additional edges in the same strip.
    outer = np.abs(OFFSETS) >= 4
    return dict(center_strength=float(offset_strength[CENTER]),
                center_coverage=float(np.mean(evidence[:, CENTER] >= MIN_SUPPORT)),
                flank_strength=float(np.max(offset_strength[outer])),
                peak_offset_px=int(OFFSETS[np.argmax(offset_strength)]))


def audit_scene(sources, background, rows, visibility):
    grid = pixel_grid(background.shape)
    residuals = {name: np.zeros(background.shape, np.float32) for name in METHODS[1:]}
    highpass = [image.astype(np.float32) - cv2.GaussianBlur(image.astype(np.float32), (0, 0), DENSE_SIGMA)
                for image in sources]
    support = np.zeros(background.shape, bool)
    center_region = np.zeros(background.shape, bool)
    probes = []
    for row in rows:
        segment = endpoints(row)
        _, coordinates = sampling_grid(segment)
        contrasts = [source_contrast(image, coordinates, index == 1)
                     for index, image in enumerate(sources)]
        selected = int(np.argmax(np.abs(contrasts)))
        active = row["label"] == "connected" and contrasts[selected] != 0
        start, end = torch.as_tensor(segment, dtype=grid.dtype)
        residual, window, normal_distance = boundary(grid, start, end, contrasts[selected])
        window = window.numpy()
        region = window > 0
        if active:
            dense = window * highpass[selected]
            if selected == 1:
                dense = np.where(sources[1] > 0, dense, 0)
                residual = residual * torch.as_tensor(sources[1] > 0)
                region &= sources[1] > 0
            residuals["dense"] += dense
            residuals["primitive"] += residual.numpy()
            support |= region
            center_region |= region & (normal_distance.abs().numpy() <= 3)
        probes.append((row, coordinates, region, contrasts, selected if active else None))

    images, budgets, target = matched_render(background, residuals, support) if support.any() else (
        {name: background.copy() for name in residuals},
        {name: dict(gain=0.0, mae=0.0, nominal_mae=0.0) for name in residuals}, 0.0,
    )
    images = {"e2": background, **images}
    flank = support & ~center_region
    background_float = background.astype(np.float32)
    background_hp = background_float - cv2.GaussianBlur(background_float, (0, 0), DENSE_SIGMA)
    budget_rows, measurements = [], []
    for name, image in images.items():
        values = image.astype(np.float32)
        high = values - cv2.GaussianBlur(values, (0, 0), DENSE_SIGMA)
        budget_rows.append(dict(method=name, target_mae=target, support_pixels=int(support.sum()),
                                outside_pixels_changed=int(np.count_nonzero(image[~support] != background[~support])),
                                flank_highpass_mae=float(np.abs(high[flank] - background_hp[flank]).mean())
                                if flank.any() else None, **budgets.get(name, dict(gain=0, mae=0, nominal_mae=0))))
        for row, coordinates, region, contrasts, selected in probes:
            sign = int(np.sign(contrasts[selected])) if selected is not None else 0
            measurements.append(dict(scene=row["scene"], probe=row["id"], label=row["label"],
                                     e2_visibility=visibility.get((row["scene"], row["id"]), ""),
                                     method=name, injected=selected is not None and target > 0,
                                     selected_source=("optical", "sar")[selected] if selected is not None else "",
                                     optical_contrast=contrasts[0], sar_contrast=contrasts[1], expected_sign=sign,
                                     probe_support_overlap=int(np.count_nonzero(region & support)),
                                     **measure_line(image, coordinates, sign)))
    return images, budget_rows, measurements


def save_preview(path, scene, optical, sar, images, rows):
    panels = {"Optical Y": optical, "SAR": sar, "Archived E2": images["e2"],
              "Dense residual": images["dense"], "Boundary render": images["primitive"]}
    height, width = optical.shape
    sheet = Image.new("RGB", (len(panels) * (width + 8) + 8, height + 66), "#171b22")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(panels.items()):
        x = 8 + index * (width + 8)
        draw.text((x, 13), f"{scene} | {label}", fill="white")
        sheet.paste(Image.fromarray(image), (x, 34))
        if index < 2:
            for row in rows:
                a, b = endpoints(row)
                color = "#57d5f1" if row["label"] == "connected" else "#ffac71"
                draw.line((x + a[0], 34 + a[1], x + b[0], 34 + b[1]), fill=color)
                draw.text((x + a[0] + 2, 34 + a[1] - 11), row["id"], fill=color)
    draw.text((8, height + 45), "Cyan: annotated boundary  |  Orange: disconnected probe", fill="#d1d9e4")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "holdout"))
    parser.add_argument("--data-root", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = (ROOT / args.output_dir).resolve()
    relative = output.relative_to(ROOT).as_posix()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    archive, sample, labels, paths = load_inputs(LABELS, args.data_root)
    visibility = load_visibility(LABELS / "e2_visibility.csv", labels)
    scenes = sample["scene_ids"][:1] if args.mode == "smoke" else sample["scene_ids"]
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Archived E2 checkpoint hash differs.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "preview").mkdir()
    results = {key: [] for key in METHODS}
    for key in METHODS:
        (output / key / "gray").mkdir(parents=True)
    all_probes, all_budgets, input_hashes = [], [], {}
    for scene in scenes:
        optical_path, sar_path, background_path = paths[scene]
        with Image.open(optical_path) as image:
            optical = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2YCrCb)[..., 0]
        sar, background = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                           for path in (sar_path, background_path)]
        if any(image is None or image.shape != optical.shape or image.dtype != np.uint8
               for image in (sar, background)):
            raise ValueError(f"Invalid input in scene {scene}")
        input_hashes[scene] = {name: sha256_file(path) for name, path in zip(
            ("optical", "sar", "archived_e2"), paths[scene])}
        rows = [row for row in labels if row["scene"] == scene and row["label"] != "uncertain"]
        images, budgets, probes = audit_scene((optical, sar), background, rows, visibility)
        all_probes.extend(probes)
        all_budgets.extend(dict(scene=scene, **row) for row in budgets)
        for name, image in images.items():
            Image.fromarray(image).save(output / name / "gray" / f"{scene}.png")
            results[name].append((FusionEvaluator.metric_vector(image, optical, sar), (), ()))
        save_preview(output / "preview" / f"{scene}.png", scene, optical, sar, images, rows)
        print(f"Scene {scene}: {sum(row['injected'] for row in probes if row['method'] == 'primitive')} "
              f"rendered probes; target MAE {budgets[0]['target_mae']:.3f}", flush=True)

    paired = (("e2", "dense"), ("e2", "primitive"), ("dense", "primitive"))
    metrics = summarize(output, scenes, results, paired, source_metrics=(), diagnostic_fields=())
    write_csv(output / "probes.csv", all_probes)
    write_csv(output / "scene_budgets.csv", all_budgets)
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, scene_ids=scenes, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], input_sha256=input_hashes,
        archive_metadata_sha256=sample["archive_metadata_sha256"], evaluator=DEFAULT_PROFILE,
        output_policy="clamp_then_uint8", target_mae_limit=TARGET_MAE, budget_tolerance=BUDGET_TOLERANCE,
        support_radius_px=SUPPORT_RADIUS, edge_width_px=EDGE_WIDTH, dense_sigma_px=DENSE_SIGMA,
        min_source_contrast=MIN_CONTRAST, min_sign_agreement=MIN_SIGN_AGREEMENT,
        implementation_sha256=sha256_file(Path(__file__)),
        labels_sha256={key: sha256_file(LABELS / key) for key in
                       ("sample.json", "source_labels.csv", "e2_visibility.csv")},
        real_point_annotations=0))

    group = "QXS-E2-PRIMITIVE-ORACLE-" + hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    register_many([dict(
        id=f"{group}-{name}", comparable_group=group, task="engineering", status="diagnostic_fixed_subset",
        method={"e2": "Archived optical Base E2", "dense": "Matched geometry-window dense residual",
                "primitive": "Annotated boundary primitive rendering"}[name],
        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(scenes), selected_scene_ids=scenes,
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=checkpoint_hash,
        split_manifest=sample["source_manifest"], split_manifest_sha256=sample["source_manifest_sha256"],
        result_dir=f"{relative}/{name}", metadata_path=f"{relative}/run_metadata.json",
        metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", metrics=metrics[name],
        metric_source=dict(type="csv_row", path=f"{relative}/{name}/metrics.csv", match=dict(image="mean"),
                           columns={key: key for key in metrics[name]}),
        notes="Archived image edit with annotated boundary geometry. No training or model inference. "
              "Dense and primitive have matched uint8 support MAE. Points are synthetic only.")
        for name in METHODS])
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
