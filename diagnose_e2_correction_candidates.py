"""One frozen forward on the pre-existing 4880-B probe. No training."""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_dose import read_probe
from diagnose_e2_primitive_render import (
    CENTER, LABELS, ROOT, endpoints, sample_pixels, sampling_grid, source_contrast, write_csv,
)
from e2_local_correction import LocalCorrectionE2
from maintain_experiment_index import sha256_file
from run_e2_local_correction import PILOT
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json, write_json


SCENE = "4880"
OUTPUT = Path("test_result/e2_correction_candidate_diagnosis_v1/scene4880_B")


def longest(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return int((bounds[1::2] - bounds[::2]).max(initial=0))


def gray(tensor):
    return tensor.float().cpu().numpy() * 255


def quantize(values):
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def center_bound(lower, upper, coords, sign):
    """Optimistic normalized response bound for any pixelwise convex mixture.

    Gaussian blur, bilinear sampling and the median preserve pixelwise order.
    Maximize the signed side-to-side contrast, then drop both nonnegative MADs.
    Work on quantized extrema because the registered probe reads uint8 outputs.
    """
    def sides(image):
        smooth = cv2.GaussianBlur(image.astype(np.float32) / 255, (0, 0), 0.8)
        readings = sample_pixels(smooth, coords[:, CENTER])
        return np.median(readings.reshape(len(coords), 2, -1), axis=2)

    lo, hi = sides(quantize(lower)), sides(quantize(upper))
    dmax = hi[:, 0] - lo[:, 1] if sign == 1 else hi[:, 1] - lo[:, 0]
    positive = np.maximum(dmax, 0)
    return positive / (positive + 0.04)


def make_preview(images, segment, output):
    xy0 = segment.min(0) - (20, 9)
    xy1 = segment.max(0) + (21, 10)
    crop = (*xy0, *xy1)
    w, h = (xy1 - xy0) * 2
    sheet = Image.new("RGB", ((w + 10) * len(images), h + 40), "#20242a")
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(images.items()):
        x = 4 + i * (w + 10)
        tile = Image.fromarray(quantize(im)).crop(crop).resize((w, h), Image.Resampling.NEAREST)
        sheet.paste(tile, (x, 27))
        draw.text((x, 5), name, fill="white")
        for px, py in (segment - xy0) * 2:
            draw.line((x + px - 6, 27 + py, x + px - 2, 27 + py), fill="#ffb33f")
    sheet.save(output)


@torch.inference_mode()
def main():
    import os
    os.chdir(ROOT)
    meta = read_json(PILOT.result / "run_metadata.json")
    manifest = read_json(PILOT.manifest)
    pair = locate_scene(DATA, manifest, SCENE)
    if (sha256_file(PILOT.final) != meta["checkpoint_hashes"]["candidate"]
            or sha256_file(PILOT.manifest) != meta["split_manifest_sha256"]
            or [sha256_file(p) for p in pair] != meta["inputs"][SCENE] or SCENE not in meta["scene_ids"]):
        raise ValueError("Mismatch with archived val64 run")
    with (LABELS / "source_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        row, = (r for r in csv.DictReader(handle) if (r["scene"], r["id"]) == (SCENE, "B"))
    segment = endpoints(row)
    _, coords = sampling_grid(segment)
    optical, sar = read_pair(pair)
    sign = int(np.sign(source_contrast(optical, coords, False)))
    if sign != -1:
        raise ValueError("Expected the original negative optical reference at 4880-B")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalCorrectionE2().to(device).eval()
    state = torch.load(PILOT.final, map_location="cpu", weights_only=False)
    assert (state["variant"], state["epoch"], state["stage"]) == (PILOT.variant, 15, "fusion")
    model.load_state_dict(state["model"], strict=True)
    captured = []
    hook = model.decoder.correction.register_forward_hook(lambda module, args, out: captured.append(out))
    try:
        result = model(to_tensor(optical, device), to_tensor(sar, device))
    finally:
        hook.remove()
    (_, options, scores), = captured
    candidates = gray(options[0])
    weights = scores[0].softmax(dim=0).cpu().numpy()
    fused = gray(result[0, 0])
    archived = np.asarray(Image.open(PILOT.result / "candidate/gray" / f"{SCENE}.png"))
    if not np.array_equal(quantize(fused), archived):
        raise ValueError("Frozen output does not reproduce archived val64 PNG")
    # Match the model's invalid-SAR mask for all counterfactuals.
    valid = sar > 0
    eligible = candidates.copy()
    eligible[2] = np.where(valid, eligible[2], eligible[0])
    low, high = eligible.min(0), eligible.max(0)
    uniform = (candidates[0] + candidates[1] + valid * candidates[2]) / (2 + valid.astype(int))
    y, x = np.indices(sar.shape)
    tangent = segment[1] - segment[0]
    # Annotation-assisted action: take extrema of eligible candidates on opposite sides.
    side = ((y - segment[0, 1]) * tangent[0] - (x - segment[0, 0]) * tangent[1])
    oracle = np.where(side * sign >= 0, high, low)
    methods = dict(coarse=candidates[0], optical_head=candidates[1], sar_head=eligible[2],
                   uniform=uniform, final=fused, sided_oracle=oracle)
    sources = dict(optical=optical, sar=sar)
    archived_e2 = np.asarray(Image.open(Path(meta["baseline_archive"]) / "e2/gray" / f"{SCENE}.png"))
    readings, details = [], []
    for name, values in dict(**sources, archived_e2=archived_e2, **methods).items():
        for precision, im in (("uint8", quantize(values)), ("float_gray", values)):
            response = sign * profile(im, coords)[:, CENTER]
            threshold = response >= MIN_SUPPORT
            readings.append(dict(method=name, precision=precision, hits=int(threshold.sum()),
                                 longest=longest(threshold), **read_probe(im, coords, sign)))
            details.extend(dict(method=name, precision=precision, sample=i, signed_center=float(v))
                           for i, v in enumerate(response))
    bound = center_bound(low, high, coords, sign)
    upper_hits = bound >= MIN_SUPPORT - 1e-6
    if any(np.any(sign * profile(quantize(im), coords)[:, CENTER] > bound + 1e-5)
           for im in methods.values()):
        raise ValueError("Envelope bound violated")
    stats = {}
    for i, key in enumerate(("coarse", "optical", "sar")):
        local = sample_pixels(weights[i], coords[:, CENTER])
        stats[key] = dict(mean=float(local.mean()), min=float(local.min()), max=float(local.max()))
    roi_delta = {k: float(np.abs(sample_pixels(v - candidates[0], coords[:, CENTER])).mean())
                 for k, v in methods.items()}
    counts = dict(samples=len(coords), envelope_possible=int(upper_hits.sum()),
                  envelope_longest=longest(upper_hits), archived_png_mismatches=0)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "summary.json", dict(**counts, sampled_weights=stats, sampled_abs_delta_gray=roi_delta))
    write_csv(OUTPUT / "probe.csv", readings)
    write_csv(OUTPUT / "profile.csv", details)
    write_csv(OUTPUT / "bound.csv", [dict(sample=i, signed_response_upper=float(v)) for i, v in enumerate(bound)])
    make_preview(dict(Optical=optical, SAR=sar, E2=archived_e2, Initial=candidates[0],
                      Optical_head=candidates[1], SAR_head=eligible[2], Final=fused,
                      Label_guided=oracle), segment, OUTPUT / "crop.png")
    write_json(OUTPUT / "run_metadata.json", dict(
        scene=SCENE, probe="B", coordinates=segment.tolist(), expected_sign=sign,
        threshold=MIN_SUPPORT, device=str(device), torch=str(torch.__version__),
        checkpoint=PILOT.final.as_posix(), checkpoint_sha256=sha256_file(PILOT.final),
        split_manifest=PILOT.manifest.as_posix(), split_manifest_sha256=sha256_file(PILOT.manifest),
        source_hashes=meta["inputs"][SCENE], val_metadata_sha256=sha256_file(PILOT.result / "run_metadata.json"),
        labels_sha256=sha256_file(LABELS / "source_labels.csv"), script_sha256=sha256_file(Path(__file__))))
    identity = "QXS-E2-LOCAL-CORRECTION-CANDIDATES-V1-SCENE4880-B"
    register_many([dict(id=identity, comparable_group=identity, task="engineering", status="diagnostic_single_scene",
                        method="Frozen local correction candidate readout and convex envelope",
                        dataset="QXS-SAROPT", split="val", offset=meta["scene_ids"].index(SCENE),
                        sample_count=1, selected_scene_ids=[SCENE], checkpoint=PILOT.final.as_posix(),
                        checkpoint_sha256=sha256_file(PILOT.final), split_manifest=PILOT.manifest.as_posix(),
                        split_manifest_sha256=sha256_file(PILOT.manifest), metric_profile=meta["metric_profile"],
                        output_policy=meta["output_policy"], result_dir=OUTPUT.as_posix(),
                        metadata_path=(OUTPUT / "run_metadata.json").as_posix(), metrics=counts,
                        metric_source=dict(type="json_paths", path=(OUTPUT / "summary.json").as_posix(),
                                           paths={k: [k] for k in counts}),
                        notes="Fixed optical-polarity probe; bound only applies to the frozen candidate images.")])


if __name__ == "__main__":
    main()
