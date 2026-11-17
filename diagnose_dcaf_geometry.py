"""Frozen E2 under source-only approximate control-point translations."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_drf_no_train import edge_contrast, gradients
from diagnose_e2_information_path import image_u8, read_pair, to_tensor
from diagnose_stg_no_train import DEFAULT_LABELS, load_inputs, write_csv
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "test_result/dcaf_geometry_v1"
CROP = 16


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def shift(x, offset, *, periodic=False):
    dx, dy = map(int, offset)
    if periodic:
        return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    h, w = x.shape[-2:]
    y = (torch.arange(h, device=x.device) - dy).clamp(0, h - 1)
    x_indices = (torch.arange(w, device=x.device) - dx).clamp(0, w - 1)
    return x.index_select(-2, y).index_select(-1, x_indices)


def roi(x):
    return x[..., CROP:-CROP, CROP:-CROP]


def positions(rows):
    points = []
    for row in rows:
        optical = np.array([int(row["opt_x"]), int(row["opt_y"])])
        sar = np.array([int(row["sar_x"]), int(row["sar_y"])])
        points.append((row, optical, sar))
    translation = np.rint(np.median([o-s for r, o, s in points if r["role"] == "fit"], axis=0)).astype(int)
    if np.abs(translation).max() >= CROP:
        raise ValueError("A marked displacement exceeds the common valid crop.")
    readout = [dict(**row, dx=int(delta[0]), dy=int(delta[1]),
                    before=float(np.linalg.norm(delta)), corrected=float(np.linalg.norm(delta-translation)),
                    reversed=float(np.linalg.norm(delta+translation)))
               for row, o, s in points for delta in (o-s,)]
    return translation.tolist(), readout


def preview(scene, paths, points, output):
    sheet = Image.new("RGB", (1056, 546), "#181b21")
    tile = ImageDraw.Draw(sheet)
    for col, prefix in enumerate(("opt", "sar")):
        x, y = 8 + col*528, 25
        with Image.open(paths[col]) as source:
            sheet.paste(source.convert("RGB").resize((512, 512), Image.Resampling.NEAREST), (x, y))
        tile.text((x, 6), f"{scene}  {prefix}  cyan: fit   orange: held out", fill="white")
        for point in points:
            px, py = x+2*int(point[prefix+"_x"]), y+2*int(point[prefix+"_y"])
            color = "#37def4" if point["role"] == "fit" else "#ffa049"
            tile.ellipse((px-7, py-7, px+7, py+7), outline=color, width=2)
            tile.text((px+9, py-16), point["id"], fill=color, stroke_width=1, stroke_fill="black")
    sheet.save(output / f"{scene}_control_points.png")


@torch.inference_mode()
def natural_probe(model, optical, sar, offset, device):
    o, s = (to_tensor(image, device) for image in (optical, sar))
    opposite = [-v for v in offset]
    aligned, reversed_field = (shift(s, delta) for delta in (offset, opposite))
    base_o, detail_o, base_s, detail_s = model.encode_pair(o, s)
    base, late_detail = model.fuse_features(base_o, detail_o, base_s, shift(detail_s, offset))
    images = dict(zero=model(o, s), correct_pre_haar=model(o, aligned), reverse_pre_haar=model(o, reversed_field),
                  correct_post_gate=model.reconstruct(o, base, late_detail))
    images["shift_existing_output"] = shift(images["zero"], offset)
    # Integer round-trip on a periodic test canvas verifies displacement sign and indexing.
    artificial = shift(s, (2, -1), periodic=True)
    restored = shift(artificial, (-2, 1), periodic=True)
    torch.testing.assert_close(restored, s, rtol=0, atol=0)
    torch.testing.assert_close(roi(shift(s, (2, -1))), roi(artificial), rtol=0, atol=0)
    torch.testing.assert_close(model(o, restored), images["zero"], rtol=0, atol=1e-6)
    synthetic = dict(injected_dx=2, injected_dy=-1,
                     output_mae_on_common_crop=float(roi(model(o, artificial)-images["zero"]).abs().mean()),
                     exact_input_roundtrip=True)
    output_control_mae = float(roi(images["correct_pre_haar"] - images["shift_existing_output"]).abs().mean())
    return {key: image_u8(value) for key, value in images.items()}, image_u8(aligned), synthetic, output_control_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    archive, sample, labels, paths = load_inputs(DEFAULT_LABELS, ROOT.parent / "datasets/QXSLAB_SAROPT")
    scene_csv = ANNOTATIONS / "source_scene_review.csv"
    points_csv = ANNOTATIONS / "control_points.csv"
    review, points = csv_rows(scene_csv), csv_rows(points_csv)
    if [row["scene"] for row in review] != list(paths):
        raise ValueError("Source review is outside the fixed eight-scene protocol.")
    scene = "5730"
    if {row["scene"] for row in points} != {scene} or {row["role"] for row in points} != {"fit", "holdout"}:
        raise ValueError("Expected marked fit and holdout points for the exploratory scene.")
    offset, residuals = positions(points)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    preview(scene, paths[scene], points, args.output_dir)
    write_csv(args.output_dir / "point_residuals.csv", residuals)
    checkpoint = archive["checkpoints"]["e2"]
    digest = sha256_file(ROOT / checkpoint)
    if digest != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Archive checkpoint changed.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    model = OpticalBaseAnchoredE2(state["variant"])
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    optical, sar = read_pair(paths[scene][:2])
    images, aligned_sar, synthetic, output_control_mae = natural_probe(model, optical, sar, offset, args.device)
    for name, gray in dict(images, optical_source=optical, sar_source=sar, sar_corrected=aligned_sar).items():
        if not cv2.imwrite(str(args.output_dir / f"{name}.png"), gray):
            raise OSError(f"Cannot save {name}.")
    archived = cv2.imread(str(paths[scene][2]), cv2.IMREAD_GRAYSCALE)
    if archived is None or np.abs(images["zero"].astype(int) - archived).max() > 1:
        raise ValueError("Frozen zero-offset output differs from the archived E2.")
    measurements = []
    for reference, source in (("corrected_sar", aligned_sar), ("raw_sar", sar)):
        for method, gray in images.items():
            metrics = FusionEvaluator.metric_vector(roi(gray), roi(optical), roi(source))
            measurements.append(dict(method=method, reference=reference,
                                     **dict(zip(FusionEvaluator.METRIC_NAMES, map(float, metrics)))))
    write_csv(args.output_dir / "per_image.csv", measurements)
    local = []
    lines = [row for row in labels if row["scene"] == scene and row["label"] == "connected"]
    for name, image in dict(images, optical=optical, raw_sar=sar, corrected_sar=aligned_sar).items():
        g = gradients(image, 1.25)
        local.extend(dict(method=name, probe=row["id"], optical_line_contrast=edge_contrast(*g, row)) for row in lines)
    write_csv(args.output_dir / "per_probe.csv", local)
    metadata = dict(**sample, evaluated_scene_id=scene, translation_xy=offset, crop_per_side=CROP,
                    checkpoint=checkpoint, checkpoint_sha256=digest, device=args.device,
                    metric_profile=DEFAULT_PROFILE, output_policy="clamp_round_uint8_common_inner_crop",
                    script_sha256=sha256_file(Path(__file__)), source_review_sha256=sha256_file(scene_csv),
                    control_points_sha256=sha256_file(points_csv),
                    inputs_sha256=[sha256_file(path) for path in paths[scene]], synthetic=synthetic,
                    pre_haar_vs_shifted_output_mae=output_control_mae)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{scene}: pre-Haar correction {offset}; {residuals[-1]}", flush=True)


if __name__ == "__main__":
    main()
