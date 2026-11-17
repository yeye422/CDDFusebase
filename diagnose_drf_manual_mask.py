"""比较人工背景掩码和等幅均匀衰减，对 E2 不做任何训练。"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_drf_no_train import (
    REMOVAL_RMS, background_mask, edge_contrast, gradients, matched_removal,
)
from diagnose_e2_information_path import image_u8, read_pair, to_tensor
from diagnose_stg_no_train import DEFAULT_LABELS, endpoints, load_inputs, load_visibility, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "test_result/drf_manual_mask_v1/annotations.json"
CONTROL = ROOT / "test_result/drf_fixed_split_v1/val8_20261117"
METHODS = ("e2", "uniform", "manual_background")
LOCAL_METRICS = ("manual_roi_hp_rms", "source_flat_hp_rms", "held_out_flat_hp_rms")
PROTECT_WIDTH = 19


def masks(sar, polygons, scene_labels):
    """Shrink background polygons and mask full Haar blocks away from marked edges."""
    background = np.zeros(sar.shape, dtype=np.uint8)
    protected = np.zeros_like(background)
    for polygon in polygons:
        points = np.asarray(polygon["xy"], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or np.any((points < 0) | (points > 255)):
            raise ValueError("Invalid source background polygon.")
        cv2.fillPoly(background, [points], 1)
    for row in scene_labels:
        if row["label"] == "connected":
            a, b = endpoints(row)
            cv2.line(protected, tuple(a), tuple(b), 1, PROTECT_WIDTH)
    interior = cv2.erode(background, np.ones((5, 5), np.uint8))
    valid = cv2.erode((sar > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    support = interior & valid & (1 - protected)
    h, w = sar.shape
    blocks = support.reshape(h//2, 2, w//2, 2).min(axis=(1, 3))
    distances = cv2.distanceTransform(blocks, cv2.DIST_L2, 5)
    weight = np.minimum(distances / 3, 1).repeat(2, axis=0).repeat(2, axis=1)
    if not np.any(weight >= 2/3) or np.any(weight[protected > 0]):
        raise ValueError("No background interior or positive weight on a marked edge.")
    return weight, protected.astype(bool)


def source_review(photos, output):
    for start in range(0, len(photos), 4):
        sheet = Image.new("RGB", (546, 1104), "#202025")
        draw = ImageDraw.Draw(sheet)
        for index, (scene, files, weight, protected) in enumerate(photos[start:start+4]):
            y = 18 + 276 * index
            for j, title in enumerate(("Optical", "SAR")):
                with Image.open(files[j]) as image:
                    color = np.array(image.convert("RGB"), dtype=np.float32)
                active = weight > 0
                color[active] = color[active] * .69 + np.array([255, 154, 80]) * .31
                color[protected] = color[protected] * .64 + np.array([52, 223, 232]) * .36
                x = 8 + 270 * j
                sheet.paste(Image.fromarray(color.astype(np.uint8)), (x, y))
                draw.text((x, y-16), f"{scene} {title}  cyan: preserve   orange: background", fill="white")
        sheet.save(output / f"source_mask_review_{start//4+1}.png")


def previous_rows(scene_ids, sample, archive):
    metadata_path = CONTROL / "run_metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = dict(evaluated_scene_ids=scene_ids, source_manifest_sha256=sample["source_manifest_sha256"],
                    checkpoint_sha256=archive["checkpoint_hashes"]["e2"], removal_rms=REMOVAL_RMS,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8",
                    source_labels_sha256=sha256_file(DEFAULT_LABELS / "source_labels.csv"),
                    visibility_sha256=sha256_file(DEFAULT_LABELS / "e2_visibility.csv"))
    if any(meta[key] != value for key, value in expected.items()):
        raise ValueError("The archived controls use a different configuration.")
    with (CONTROL / "per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {(row["scene"], row["method"]): row for row in csv.DictReader(handle)}
    return meta, rows


def rms_in(highpass, mask):
    return float(np.sqrt(np.mean(highpass[mask]**2))) if mask.any() else float("nan")


@torch.inference_mode()
def evaluate_scene(model, device, scene, paths, scene_labels, visibility, weight, protected,
                   output, previous):
    optical, sar = read_pair(paths[:2])
    source_flat = background_mask(optical, sar, scene_labels)
    affected_margin = cv2.dilate((weight > 0).astype(np.uint8), np.ones((17, 17), np.uint8))
    regions = (weight >= 2/3, source_flat, source_flat & (affected_margin == 0))
    o, s = (to_tensor(image, device) for image in (optical, sar))
    bo, do, bs, ds = model.encode_pair(o, s)
    base, detail = model.fuse_features(bo, do, bs, ds)
    archived_gray = {method: cv2.imread(str(CONTROL / "gray" / method / f"{scene}.png"), cv2.IMREAD_GRAYSCALE)
                     for method in METHODS[:2]}
    if any(gray is None or gray.shape != sar.shape for gray in archived_gray.values()):
        raise ValueError(f"Archived grayscale image is missing for {scene}.")
    original_gray = image_u8(model.reconstruct(o, base, detail))
    baseline_difference = int(np.abs(archived_gray["e2"].astype(int) - original_gray).max())
    if baseline_difference > 1:
        raise ValueError(f"Frozen E2 image for {scene} disagrees with the archived control.")
    masked = ds * torch.as_tensor(weight, device=device)
    support_rms = float(torch.linalg.vector_norm(masked) / torch.linalg.vector_norm(ds))
    diag = dict(scene=scene, protected_pixels=int(protected.sum()), background_pixels=int((weight > 0).sum()),
                manual_core_pixels=int(regions[0].sum()), source_flat_pixels=int(regions[1].sum()),
                held_out_pixels=int(regions[2].sum()), available_relative_rms=support_rms,
                archived_e2_max_u8_difference=baseline_difference)
    if support_rms < REMOVAL_RMS:
        return [], [], dict(diag, status="insufficient_support_for_budget")
    ds_new, fraction = matched_removal(ds, masked)
    delta = ds - ds_new
    actual_rms = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ds))
    outside = torch.as_tensor(weight == 0, device=device)[None, None]
    if abs(actual_rms - REMOVAL_RMS) > 1e-6 or bool(torch.any(delta.masked_select(outside) != 0)):
        raise AssertionError("Manual mask did not respect the shared budget or background support.")
    edited = image_u8(model.reconstruct(o, base, model.detail_fusion(do, ds_new, base)))
    if not cv2.imwrite(str(output / "gray" / f"{scene}.png"), edited):
        raise OSError(f"Cannot save manually masked output for {scene}.")
    if not cv2.imwrite(str(output / "mask" / f"{scene}.png"), np.rint(weight*255).astype(np.uint8)):
        raise OSError(f"Cannot save background mask for {scene}.")
    gray_by_method = dict(archived_gray, manual_background=edited)
    rows, segments = [], []
    for method, gray in gray_by_method.items():
        if method == "manual_background":
            metrics = dict(zip(FusionEvaluator.METRIC_NAMES, map(float, FusionEvaluator.metric_vector(gray, optical, sar))))
        else:
            metrics = {key: float(previous[scene, method][key]) for key in FusionEvaluator.METRIC_NAMES}
        image = gray.astype(np.float32)
        highpass = image - cv2.GaussianBlur(image, (0, 0), 1.25)
        rows.append(dict(scene=scene, method=method, **metrics,
                         **dict(zip(LOCAL_METRICS, (rms_in(highpass, region) for region in regions)))))
        dx, dy = gradients(gray, 1.25)
        for label in scene_labels:
            if label["label"] == "connected":
                segments.append(dict(scene=scene, id=label["id"], visibility=visibility[scene, label["id"]],
                                     method=method, normal_contrast=edge_contrast(dx, dy, label)))
    return rows, segments, dict(diag, status="paired", removal_fraction=fraction, actual_relative_rms=actual_rms)


def paired_summary(rows, metrics, keys):
    groups = {name: {tuple(row[key] for key in keys): row for row in rows if row["method"] == name}
              for name in METHODS}
    common = sorted(set.intersection(*(set(group) for group in groups.values())))
    result = []
    for metric in metrics:
        values = np.asarray([[group[key][metric] for key in common] for group in groups.values()])
        finite = np.isfinite(values).all(axis=0)
        if finite.any():
            means = values[:, finite].mean(axis=1)
            result.extend(dict(metric=metric, method=method, common_count=int(finite.sum()), mean=float(mean),
                               delta_uniform=float(mean - means[1]))
                          for method, mean in zip(METHODS, means))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    archive, sample, labels, paths = load_inputs(DEFAULT_LABELS, ROOT.parent / "datasets/QXSLAB_SAROPT")
    annotations = json.loads(LABELS.read_text(encoding="utf-8"))
    if ROOT / annotations["scene_labels"] != DEFAULT_LABELS / "source_labels.csv" or set(annotations["background"]) != set(paths):
        raise ValueError("The annotations are not for the fixed validation sample.")
    photos = []
    for scene, files in paths.items():
        scene_labels = [row for row in labels if row["scene"] == scene]
        _, sar = read_pair(files[:2])
        photos.append((scene, files, *masks(sar, annotations["background"][scene], scene_labels)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_review(photos, args.output_dir)
    if args.preview:
        return
    meta, previous = previous_rows(list(paths), sample, archive)
    checkpoint = ROOT / meta["checkpoint"]
    if sha256_file(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("The checkpoint hash does not match the archived controls.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    visibility_path = DEFAULT_LABELS / "e2_visibility.csv"
    visibility = load_visibility(visibility_path, labels)
    for name in ("gray", "mask"):
        (args.output_dir / name).mkdir()
    rows, segments, diagnostics = [], [], []
    for scene, files, weight, protected in photos:
        actual_hashes = [sha256_file(path) for path in files]
        if actual_hashes != meta["inputs_sha256"][scene]:
            raise ValueError(f"Source or E2 archive changed for {scene}.")
        scene_labels = [row for row in labels if row["scene"] == scene]
        scene_rows, scene_segments, diag = evaluate_scene(
            model, args.device, scene, files, scene_labels, visibility, weight, protected, args.output_dir, previous,
        )
        rows.extend(scene_rows)
        segments.extend(scene_segments)
        diagnostics.append(diag)
        print(scene, diag["status"], flush=True)
    write_csv(args.output_dir / "mask_diagnostics.csv", diagnostics)
    if not rows:
        raise ValueError("No masks can match the shared budget.")
    write_csv(args.output_dir / "per_image.csv", rows)
    write_csv(args.output_dir / "per_segment.csv", segments)
    summary = paired_summary(rows, (*FusionEvaluator.METRIC_NAMES, *LOCAL_METRICS), ("scene",))
    for subset in ("all", "missing", "present"):
        selected = [row for row in segments if subset == "all" or row["visibility"] == subset]
        summary.extend(dict(row, metric=f"{subset}_normal_contrast")
                       for row in paired_summary(selected, ("normal_contrast",), ("scene", "id")))
    write_csv(args.output_dir / "paired_summary.csv", summary)
    metadata = dict(**sample, paired_scene_ids=[row["scene"] for row in diagnostics if row["status"] == "paired"],
                    checkpoint=meta["checkpoint"], checkpoint_sha256=meta["checkpoint_sha256"],
                    metric_profile=DEFAULT_PROFILE, output_policy=meta["output_policy"],
                    removal_rms=REMOVAL_RMS, protect_width=PROTECT_WIDTH, device=args.device,
                    evaluation_script_sha256=sha256_file(Path(__file__)), annotation_sha256=sha256_file(LABELS),
                    visibility_sha256=sha256_file(visibility_path),
                    previous_run_sha256=sha256_file(CONTROL / "run_metadata.json"),
                    previous_metrics_sha256=sha256_file(CONTROL / "per_image.csv"),
                    reused_gray_sha256={scene: {name: sha256_file(CONTROL / "gray" / name / f"{scene}.png")
                                               for name in METHODS[:2]} for scene in paths})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
