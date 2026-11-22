"""Single-scene control for the annotation-assisted candidate envelope at 4880-B."""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diagnose_e2_correction_candidates import OUTPUT as FIRST, SCENE, gray, longest, quantize
from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_dose import read_probe
from diagnose_e2_primitive_render import (
    CENTER, LABELS, ROOT, endpoints, sampling_grid, source_contrast, write_csv,
)
from e2_local_correction import LocalCorrectionE2
from maintain_experiment_index import sha256_file
from run_e2_local_correction import PILOT
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json, write_json


OUTPUT = Path("test_result/e2_correction_brightness_control_v1/scene4880_B")
NEGATIVE = Path("test_result/e2_local_supervision_v1/smoke_4880_B/summary.json")


def compare(segment, name, optical, sar, options, sign, direction):
    _, coords = sampling_grid(segment)
    base = options[0]
    y, x = np.indices(base.shape)
    dx, dy = x - segment[0, 0], y - segment[0, 1]
    along = dx * direction[0] + dy * direction[1]
    normal = -dx * direction[1] + dy * direction[0]
    window = (along >= 0) & (along <= np.linalg.norm(segment[1] - segment[0])) & (abs(normal) <= 8)
    # Both edits use identical geometry, footprint and polarity.
    envelope = np.where(sign * normal >= 0, options.max(0), options.min(0))
    envelope = np.where(window, envelope, base)
    step = np.where(sign * normal >= 0, 1.0, -1.0)
    def render(gain):
        return quantize(np.where(window, np.clip(base + gain * step, 0, 255), base))
    original = quantize(base)
    def mae(image):
        return float(np.abs(image[window].astype(float) - original[window]).mean())
    target = mae(quantize(envelope))
    lo, hi = 0., 255.
    for _ in range(28):
        mid = (lo + hi) / 2
        if mae(render(mid)) < target:
            lo = mid
        else:
            hi = mid
    control = min((render(lo), render(hi)), key=lambda im: abs(mae(im) - target))
    if abs(mae(control) - target) > 0.05:
        raise ValueError("Unmatched uint8 edit budget")
    rows = []
    for method, image in (("initial", original), ("candidate_envelope", quantize(envelope)), ("constant_step", control)):
        active = sign * profile(image, coords)[:, CENTER] >= MIN_SUPPORT
        rows.append(dict(probe=name, method=method, hits=int(active.sum()), longest=longest(active),
                         edit_mae=mae(image), **read_probe(image, coords, sign)))
    stats = {}
    for key, head in zip(("optical", "sar"), options[1:]):
        delta = (head - base)[window]
        stats[key] = dict(mean=float(delta.mean()), std=float(delta.std()),
                          positive_fraction=float((delta > 0).mean()), negative_fraction=float((delta < 0).mean()))
    return rows, dict(segment=segment.tolist(), window_pixels=int(window.sum()),
                      optical_contrast=source_contrast(optical, coords, False),
                      sar_contrast=source_contrast(sar, coords, True), residuals=stats)


@torch.inference_mode()
def main():
    os.chdir(ROOT)
    prior = read_json(FIRST / "run_metadata.json")
    meta = read_json(PILOT.result / "run_metadata.json")
    if sha256_file(PILOT.final) != prior["checkpoint_sha256"] or sha256_file(PILOT.manifest) != prior["split_manifest_sha256"]:
        raise ValueError("Model or split differs from previous single-scene intervention")
    files = locate_scene(DATA, read_json(PILOT.manifest), SCENE)
    if [sha256_file(path) for path in files] != meta["inputs"][SCENE]:
        raise ValueError("Input mismatch")
    optical, sar = read_pair(files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalCorrectionE2().to(device).eval()
    model.load_state_dict(torch.load(PILOT.final, map_location="cpu", weights_only=False)["model"])
    captured = []
    hook = model.decoder.correction.register_forward_hook(lambda module, args, out: captured.append(out[1]))
    try:
        fused = model(to_tensor(optical, device), to_tensor(sar, device))
    finally:
        hook.remove()
    with Image.open(PILOT.result / "candidate/gray/4880.png") as archived:
        if not np.array_equal(np.asarray(archived), quantize(gray(fused[0, 0]))):
            raise ValueError("Saved output mismatch")
    options = gray(captured[0][0])
    options[2] = np.where(sar > 0, options[2], options[0])
    with (LABELS / "source_labels.csv").open(encoding="utf-8-sig") as handle:
        row, = (r for r in csv.DictReader(handle) if (r["scene"], r["id"]) == (SCENE, "B"))
    segment = endpoints(row)
    shift = read_json(NEGATIVE)["decisions"]["4880-B"]["false_translation"]
    direction = (segment[1] - segment[0]) / np.linalg.norm(segment[1] - segment[0])
    rows, regions = [], {}
    for name, line in (("target", segment), ("parallel_negative", segment + shift)):
        measurements, regions[name] = compare(line, name, optical, sar, options, -1, direction)
        rows.extend(measurements)
    stats = {r["probe"] + "_" + r["method"] + "_hits": r["hits"] for r in rows}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT / "comparison.csv", rows)
    write_json(OUTPUT / "summary.json", dict(**stats, regions=regions))
    write_json(OUTPUT / "run_metadata.json", dict(prior, parent_metadata_sha256=sha256_file(FIRST / "run_metadata.json"),
                                                  negative_source_sha256=sha256_file(NEGATIVE),
                                                  script_sha256=sha256_file(Path(__file__))))
    key = "QXS-E2-LOCAL-CORRECTION-BRIGHTNESS-CONTROL-V1-SCENE4880-B"
    register_many([dict(id=key, comparable_group=key, task="engineering", status="diagnostic_single_scene",
                        method="Candidate envelope vs matched constant signed brightness step",
                        dataset="QXS-SAROPT", split="val", offset=meta["scene_ids"].index(SCENE), sample_count=1,
                        selected_scene_ids=[SCENE], checkpoint=PILOT.final.as_posix(), checkpoint_sha256=prior["checkpoint_sha256"],
                        split_manifest=PILOT.manifest.as_posix(), split_manifest_sha256=prior["split_manifest_sha256"],
                        metric_profile=meta["metric_profile"], output_policy=meta["output_policy"], result_dir=OUTPUT.as_posix(),
                        metadata_path=(OUTPUT / "run_metadata.json").as_posix(), metrics=stats,
                        metric_source=dict(type="json_paths", path=(OUTPUT / "summary.json").as_posix(),
                                           paths={k: [k] for k in stats}),
                        notes="Same footprint and uint8 MAE per probe, sign and geometry supplied, no training.")])


if __name__ == "__main__":
    main()
