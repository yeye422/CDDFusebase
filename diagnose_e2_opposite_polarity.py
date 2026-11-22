"""Check source-only geometric agreement before choosing a display polarity."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from diagnose_e2_contour_evidence import MIN_SUPPORT, OFFSETS, profile
from diagnose_e2_disjoint_supervision import region, select_regions, source_evidence
from diagnose_e2_primitive_dose import reflect_profile, render_support
from diagnose_e2_primitive_render import (
    DATASET, DENSE_SIGMA, LABELS, MIN_SIGN_AGREEMENT, ROOT, dump_json, endpoints,
    load_inputs, measure_line, pixel_grid, write_csv,
)
from diagnose_stg_no_train import DEFAULT_LABELS
from maintain_experiment_index import sha256_file
from run_e2_matched_no_pac import register_many
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


CASES = (("3513", "A"), ("4880", "B"), ("10095", "B"))
NEAR_PX = 2
MAX_SOURCE_SEPARATION_PX = 2
EDIT_MAE = 8
METRICS = FusionEvaluator.METRIC_NAMES


def strongest_offsets(signed):
    """Read unsigned strongest locations, choosing a central tie deterministically."""
    strength = np.abs(signed)
    order = np.argsort(np.abs(OFFSETS), kind="stable")
    index = order[np.argmax(strength[:, order], axis=1)]
    return OFFSETS[index], strength[np.arange(len(index)), index]


def decide_polarity(optical_profiles, sar_profiles, optical_contrast, sar_contrast):
    profiles = (optical_profiles, sar_profiles)
    contrasts = (optical_contrast, sar_contrast)
    offsets, strengths = zip(*(strongest_offsets(source) for source in profiles))
    supported = [value >= MIN_SUPPORT for value in strengths]
    near = []
    for values, offset, valid, contrast in zip(profiles, offsets, supported, contrasts):
        indices = np.searchsorted(OFFSETS, offset)
        signed_peak = values[np.arange(len(offset)), indices]
        near.append(valid & (np.abs(offset) <= NEAR_PX) & (signed_peak * contrast > 0))
    shared = near[0] & near[1] & (np.abs(offsets[0] - offsets[1]) <= MAX_SOURCE_SEPARATION_PX)
    fraction = float(shared.mean())
    report = dict(
        optical_contrast=optical_contrast, sar_contrast=sar_contrast,
        optical_near_fraction=float(near[0].mean()), sar_near_fraction=float(near[1].mean()),
        shared_near_fraction=fraction,
        optical_peak_offset_median_px=float(np.median(offsets[0][supported[0]])) if supported[0].any() else None,
        sar_peak_offset_median_px=float(np.median(offsets[1][supported[1]])) if supported[1].any() else None)
    if optical_contrast * sar_contrast >= 0:
        label = "not_screened_as_opposite"
    elif fraction < MIN_SIGN_AGREEMENT:
        label = "abstain_unconfirmed_correspondence"
    else:
        label = "optical_display_polarity"
    return dict(rule=label, **report)


def make_edits(optical, sar, background, target):
    empirical = [reflect_profile(image, target.across, target.segment, j == 1)
                 for j, image in enumerate((optical, sar))]
    if any(value is None for value in empirical):
        return {}, {}, "source_profile_unavailable"
    support = target.write & (sar > 0)
    if not support.any():
        return {}, {}, "no_common_valid_support"
    values = background.astype(np.float32)
    smooth = cv2.GaussianBlur(values, (0, 0), DENSE_SIGMA) - values
    # All variants edit the same valid window at the same achieved uint8 budget.
    residuals = dict(optical=empirical[0], sar=empirical[1], smoothed=smooth)
    edits, achieved = {"e2": background}, {}
    for name, residual in residuals.items():
        pixels, budget = render_support(background[support], (target.coverage * residual)[support], EDIT_MAE)
        image = background.copy()
        image[support] = pixels
        edits[name], achieved[name] = image, budget
    return edits, achieved, ""


def score_image(image, optical, sar, coords, contrasts):
    unsigned = measure_line(image, coords, 0)
    directional = {name: measure_line(image, coords, np.sign(contrast))["center_strength"] if contrast else None
                   for name, contrast in zip(("optical_signed_center", "sar_signed_center"), contrasts)}
    return dict(unsigned_center_strength=unsigned["center_strength"],
                unsigned_center_coverage=unsigned["center_coverage"],
                unsigned_flank_strength=unsigned["flank_strength"],
                unsigned_peak_offset_px=unsigned["peak_offset_px"], **directional,
                **dict(zip(METRICS, FusionEvaluator.metric_vector(image, optical, sar))))


def save_preview(path, optical, sar, edits, target, title):
    x0, y0 = np.maximum(target.segment.min(axis=0) - 19, 0)
    x1, y1 = np.minimum(target.segment.max(axis=0) + 20, np.asarray(optical.shape[::-1]))
    panels = {"Source: optical": optical, "Source: SAR": sar, **edits}
    w, h = (x1 - x0) * 2, (y1 - y0) * 2
    sheet = Image.new("RGB", (len(panels) * (w + 9) + 9, h + 55), "#151a20")
    draw = ImageDraw.Draw(sheet)
    for j, (label, image) in enumerate(panels.items()):
        x = 9 + j * (w + 9)
        draw.text((x, 7), label, fill="white")
        crop = Image.fromarray(image[y0:y1, x0:x1]).resize((w, h), Image.Resampling.NEAREST)
        sheet.paste(crop, (x, 26))
        if j < 2:
            a, b = 2 * (target.segment - (x0, y0))
            draw.line((x + a[0], 26 + a[1], x + b[0], 26 + b[1]), fill="#59d5ec")
    draw.text((9, h + 34), title, fill="#becbdb")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "opposite"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = (ROOT / args.output_dir).resolve()
    location = output.relative_to(ROOT).as_posix()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    archive, sample, development, dev_paths = load_inputs(DEFAULT_LABELS, DATASET)
    holdout_archive, holdout_sample, holdout, holdout_paths = load_inputs(LABELS, DATASET)
    if (sample["source_manifest_sha256"] != holdout_sample["source_manifest_sha256"]
            or archive["checkpoint_hashes"]["e2"] != holdout_archive["checkpoint_hashes"]["e2"]
            or archive["metric_profile"] != DEFAULT_PROFILE or archive["output_policy"] != "clamp_then_uint8"):
        raise ValueError("Unexpected E2 archive protocol.")
    checkpoint = ROOT / archive["checkpoints"]["e2"]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != archive["checkpoint_hashes"]["e2"]:
        raise ValueError("Unexpected E2 checkpoint hash.")
    labels = {(r["scene"], r["id"]): r for r in development + holdout}
    paths = {**dev_paths, **holdout_paths}
    output.mkdir(parents=True, exist_ok=False)
    selected = CASES[:1] if args.mode == "smoke" else CASES
    hashes, decisions, scores, line_rows = {}, {}, [], []
    for scene, key in selected:
        name = f"{scene}-{key}"
        optical_path, sar_path, e2_path = paths[scene]
        rgb = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Missing optical source: {scene}")
        optical = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)[..., 0]
        sar = cv2.imread(str(sar_path), cv2.IMREAD_UNCHANGED)
        if sar is None or sar.dtype != np.uint8 or sar.shape != optical.shape:
            raise ValueError(f"Invalid SAR source: {scene}")
        grid = pixel_grid(optical.shape)
        target = region(endpoints(labels[(scene, key)]), grid)
        contrasts = source_evidence(optical, sar, target.coordinates)
        profiles = [profile(image, target.coordinates, sar=index == 1)
                    for index, image in enumerate((optical, sar))]
        decision = decisions[name] = decide_polarity(*profiles, *contrasts)
        decision["stronger_at_center"] = ("optical", "sar")[int(np.argmax(np.abs(contrasts)))]
        folder = output / name
        folder.mkdir()
        dump_json(folder / "source_decision.json", decision)
        for method, values in zip(("optical", "sar"), profiles):
            offset, strength = strongest_offsets(values)
            for i, (peak, amplitude) in enumerate(zip(offset, strength)):
                line_rows.append(dict(probe=name, source=method, position=i, peak_offset_px=int(peak),
                                      peak_strength=float(amplitude)))
        # Read the old fusion only after committing the source-only rule.
        background = cv2.imread(str(e2_path), cv2.IMREAD_UNCHANGED)
        if background is None or background.dtype != np.uint8 or background.shape != optical.shape:
            raise ValueError(f"Invalid E2 background: {scene}")
        hashes[name] = {k: sha256_file(path) for k, path in zip(
            ("optical", "sar", "archived_e2"), paths[scene])}
        edits, budgets, reason = make_edits(optical, sar, background, target)
        if reason:
            decision["skipped"] = reason
            continue
        matched = all(value["matched"] for value in budgets.values())
        decision["matched"] = matched
        edits["rule"] = edits["optical"] if decision["rule"] == "optical_display_polarity" else background
        probe_scores = {}
        for method, image in edits.items():
            # Reuse already scored E2/optical pixels for the rule output.
            if method == "rule":
                reference_method = "optical" if image is edits["optical"] else "e2"
                scores.append(dict(probe_scores[reference_method], method="rule"))
            else:
                measurements = score_image(image, optical, sar, target.coordinates, contrasts)
                probe_scores[method] = dict(probe=name, method=method, all_edits_matched=matched,
                                            mae=budgets.get(method, {}).get("achieved", 0.0), **measurements)
                scores.append(probe_scores[method])
            Image.fromarray(image).save(folder / f"{method}.png")
        guard, radar = select_regions(target, optical, sar, grid)
        other_reads = {"guard": guard[0].read} if guard else {}
        if radar:
            other_reads["sar"] = radar[0].read
        decision["changed_pixels_in_other_readouts"] = {
            method: {label: int(np.count_nonzero((image != background) & mask))
                     for label, mask in other_reads.items()}
            for method, image in edits.items()}
        if any(value for item in decision["changed_pixels_in_other_readouts"].values() for value in item.values()):
            raise ValueError(f"An edit touched another probe readout: {name}")
        save_preview(folder / "preview.png", optical, sar, edits, target, name + " | " + decision["rule"])
        print(name, json.dumps(decision, ensure_ascii=False), flush=True)

    if scores:
        write_csv(output / "scores.csv", scores)
    write_csv(output / "source_positions.csv", line_rows)
    counts = dict(probes=len(selected), matched=sum(d.get("matched", False) for d in decisions.values()),
                  shared_opposite=sum(d["rule"] == "optical_display_polarity" for d in decisions.values()),
                  abstained=sum(d["rule"] == "abstain_unconfirmed_correspondence" for d in decisions.values()))
    dump_json(output / "summary.json", dict(**counts, decisions=decisions))
    dump_json(output / "run_metadata.json", dict(
        mode=args.mode, source_hashes=hashes, checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], scene_ids=[s for s, _ in selected],
        metric_profile=DEFAULT_PROFILE, output_policy="clamp_then_uint8", edit_mae=EDIT_MAE,
        near_px=NEAR_PX, max_source_separation_px=MAX_SOURCE_SEPARATION_PX,
        min_support=MIN_SUPPORT, min_coincident_fraction=MIN_SIGN_AGREEMENT,
        labels_sha256={x.name: sha256_file(x / "source_labels.csv") for x in (DEFAULT_LABELS, LABELS)},
        implementation_sha256={name: sha256_file(ROOT / name) for name in
                               ("diagnose_e2_opposite_polarity.py", "diagnose_e2_disjoint_supervision.py",
                                "diagnose_e2_contour_evidence.py", "diagnose_e2_primitive_dose.py")}))
    identity = "QXS-E2-OPPOSITE-POLARITY-" + hashlib.sha256(location.encode()).hexdigest()[:12].upper()
    register_many([dict(
        id=identity, comparable_group=identity, task="engineering", status="diagnostic_fixed_subset",
        method="Source-only correspondence screening before opposite-polarity selection",
        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(selected),
        selected_scene_ids=[s for s, _ in selected], checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        checkpoint_sha256=checkpoint_hash, split_manifest=sample["source_manifest"],
        split_manifest_sha256=sample["source_manifest_sha256"], metric_profile=DEFAULT_PROFILE,
        output_policy="clamp_then_uint8", result_dir=location, metadata_path=location + "/run_metadata.json",
        metrics=counts, metric_source=dict(type="json_paths", path=location + "/summary.json",
                                          paths={k: [k] for k in counts}),
        notes="Pretraining archived-image check. Optical polarity is a display convention after geometric screening."
    )])


if __name__ == "__main__":
    main()
