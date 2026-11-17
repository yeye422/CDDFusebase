"""Compare dense contour evidence on a small fixed QXS validation sample."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from diagnose_stg_no_train import (
    DATASET, DEFAULT_LABELS, ROOT, draw_review, endpoints, hits_at_budget, load_inputs,
    load_visibility, write_csv,
)
from maintain_experiment_index import sha256_file


OFFSETS = np.arange(-6, 7, 2)
STEP_PX = 3
MIN_SUPPORT = 0.30
MIN_COVERAGE = 0.75
MAX_GAP = 3
MIN_SOURCE_FRACTION = 0.20
METHODS = ("optical", "sar", "best_single", "union", "free_switch", "linked")
FALSE_BUDGETS = (0, 1, 2, 4)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_holdout(output, data_root):
    development = json.loads((DEFAULT_LABELS / "sample.json").read_text(encoding="utf-8"))
    archive = json.loads((ROOT / development["archive_metadata"]).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / development["source_manifest"]).read_text(encoding="utf-8"))
    indices = list(range(1, 64, 8))
    scenes = [archive["scene_ids"][idx] for idx in indices]
    if set(scenes) & set(development["scene_ids"]) or not set(scenes) <= set(manifest["splits"]["val"]):
        raise ValueError("Invalid holdout membership.")
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "sample.json", dict(development, scene_ids=scenes, sampling_indices_zero_based=indices))
    for start in range(0, len(scenes), 4):
        sheet = Image.new("RGB", (1120, 584), "#202025")
        draw = ImageDraw.Draw(sheet)
        for k, scene in enumerate(scenes[start:start + 4]):
            for j, modality in enumerate(("optical", "sar")):
                x = (k % 2) * 560 + 22 + j * 270
                y = (k // 2) * 292 + 27
                with Image.open(data_root / manifest[f"{modality}_dir"] / f"{scene}.png") as image:
                    if image.size != (256, 256):
                        raise ValueError(f"Unexpected size in scene {scene}.")
                    sheet.paste(image.convert("RGB"), (x, y))
                draw.text((x, y - 25), f"{scene} {modality.upper()}", fill="white")
                for t in range(0, 256, 32):
                    draw.text((x + t, y - 11), str(t), fill="white")
                    if j == 0:
                        draw.text((x - 21, y + t - 5), str(t), fill="white")
                    for u in range(0, 256, 32):
                        draw.line((x + t - 2, y + u, x + t + 2, y + u), fill="#43ffdc")
                        draw.line((x + t, y + u - 2, x + t, y + u + 2), fill="#43ffdc")
        sheet.save(output / f"sources_{start // 4 + 1}.png")


def sampling_grid(segment):
    a, b = np.asarray(segment, dtype=float)
    length = np.linalg.norm(b - a)
    if length < 12:
        raise ValueError("A contour probe must be at least 12 pixels long.")
    tangent = (b - a) / length
    normal = np.array([-tangent[1], tangent[0]])
    distance = np.linspace(0, length, int(np.ceil(length / STEP_PX)) + 1)
    centers = a + distance[:, None, None] * tangent + OFFSETS[None, :, None] * normal
    side = np.array([(n * normal + t * tangent) for n in (2, 3, 4) for t in (-1, 0, 1)])
    return centers, centers[:, :, None, :] + np.concatenate((side, -side))[None, None, :, :]


def profile(gray, coordinates, sar=False):
    """Signed two-sided contrast normalized by within-side variation."""
    smooth = cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), 0.8)
    steps, offsets, width, _ = coordinates.shape
    xy = coordinates.astype(np.float32).reshape(-1, width, 2)
    x, y = xy[:, :, 0].copy(), xy[:, :, 1].copy()
    samples = cv2.remap(smooth, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=float("nan")).reshape(steps, offsets, 2, -1)
    valid = np.isfinite(samples).all(axis=(2, 3))
    if sar:
        mask = cv2.remap((gray > 0).astype(np.uint8), x, y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid &= mask.reshape(steps, offsets, -1).all(axis=2)
    side_median = np.median(samples, axis=3)
    side_mad = np.median(np.abs(samples - side_median[..., None]), axis=3)
    contrast = side_median[:, :, 0] - side_median[:, :, 1]
    strength = contrast / (np.abs(contrast) + side_mad.sum(axis=2) + 0.04)
    return np.where(valid, strength, 0.0)


def best_path(evidence, sources, restrict_switch=False):
    """Run the same spatial search for all methods and backtrack one path."""
    steps, states, offsets = evidence.shape
    offset_ids = np.arange(offsets)
    lateral = np.abs(offset_ids[:, None] - offset_ids[None, :])
    movement = np.where(lateral <= 1, 0.015 * lateral, np.inf)
    center_cost = 0.006 * np.abs(offset_ids - offsets // 2)
    allowed = (sources[:, None] != sources[None, :]) | np.eye(states, dtype=bool)
    changing_source = sources[:, None] != sources[None, :]
    values = evidence[0] - center_cost
    back = np.zeros((steps, states, offsets), dtype=int)
    for step in range(1, steps):
        transitions = np.broadcast_to(allowed[:, None, :, None], (states, offsets, states, offsets))
        if restrict_switch:
            previous = evidence[step - 1] >= MIN_SUPPORT
            current = evidence[step] >= MIN_SUPPORT
            overlap_previous = (previous[:, None, :] & previous[None, :, :]).transpose(0, 2, 1)[..., None]
            overlap_current = (current[:, None, :] & current[None, :, :])[:, None, :, :]
            transitions = transitions & (~changing_source[:, None, :, None] | overlap_previous | overlap_current)
        candidates = values[:, :, None, None] - movement[None, :, None, :]
        candidates = np.where(transitions, candidates, -np.inf).reshape(states * offsets, states, offsets)
        back[step] = np.argmax(candidates, axis=0)
        values = candidates.max(axis=0) + evidence[step] - center_cost
    flat = int(np.argmax(values))
    score = float(values.flat[flat] / steps)
    state_path, offset_path = np.empty(steps, dtype=int), np.empty(steps, dtype=int)
    for step in range(steps - 1, -1, -1):
        state, offset = divmod(flat, offsets)
        state_path[step], offset_path[step] = state, offset
        flat = back[step, state, offset]
    source_path = sources[state_path]
    strength = evidence[np.arange(steps), state_path, offset_path]
    return score, strength, source_path, offset_path


def read_path(result):
    score, strength, source, _ = result
    supported = strength >= MIN_SUPPORT
    gaps = np.diff(np.r_[-1, np.flatnonzero(supported), len(supported)]) - 1
    coverage = float(np.mean(supported))
    gap = int(gaps.max())
    return dict(score=score, coverage=coverage, gap=gap, continuous=coverage >= MIN_COVERAGE and gap <= MAX_GAP,
                optical_fraction=float(np.mean(supported & (source == 0))),
                sar_fraction=float(np.mean(supported & (source == 1))))


def analyze_segment(row, images):
    centers, coordinates = sampling_grid(endpoints(row))
    optical = profile(images[0], coordinates)
    sar = profile(images[1], coordinates, sar=True)
    fused = profile(images[2], coordinates)
    signed = np.maximum(np.stack((optical, -optical, sar, -sar), axis=1), 0)
    sources = np.array([0, 0, 1, 1])
    paths = {
        "optical": best_path(signed[:, :2], sources[:2]),
        "sar": best_path(signed[:, 2:], sources[2:]),
        "union": best_path(signed.max(axis=1, keepdims=True), np.array([-1])),
        "free_switch": best_path(signed, sources),
        "linked": best_path(signed, sources, restrict_switch=True),
        "e2": best_path(np.maximum(np.stack((fused, -fused), axis=1), 0), np.array([-1, -1])),
    }
    stats = {method: read_path(path) for method, path in paths.items()}
    row = dict(row, candidate=True, samples=len(centers), best_single=max(stats["optical"]["score"], stats["sar"]["score"]))
    for method, values in stats.items():
        row[method] = values["score"]
        row.update((f"{method}_{key}", values[key]) for key in ("coverage", "gap", "continuous"))
    linked = stats["linked"]
    row["linked_optical_fraction"] = linked["optical_fraction"]
    row["linked_sar_fraction"] = linked["sar_fraction"]
    row["complementary"] = (
        linked["continuous"] and not any(stats[m]["continuous"] for m in ("optical", "sar"))
        and min(linked["optical_fraction"], linked["sar_fraction"]) >= MIN_SOURCE_FRACTION
    )
    samples = {}
    for method, (_, strength, source, offset_path) in paths.items():
        samples[method] = dict(xy=centers[np.arange(len(centers)), offset_path].round(2).tolist(),
                               strength=strength.round(5).tolist(), source=source.tolist())
    return row, samples


def endpoints_review(paths, details, output):
    cases = [row for row in details if row["e2_visibility"] == "missing" or row["complementary"]]
    if not cases:
        return
    sheet = Image.new("RGB", (828, 146 * len(cases)), "#1c2026")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(cases):
        label = row["scene"], row["id"]
        center = endpoints(row).mean(axis=0).astype(int)
        left = np.clip(center - (64, 57), (0, 0), (128, 142))
        box = tuple(np.r_[left, left + (128, 114)])
        for j, modality in enumerate(("optical", "sar", "e2")):
            x, y = 10 + 276 * j, 22 + 146 * i
            with Image.open(paths[label[0]][j]) as image:
                sheet.paste(image.crop(box).convert("RGB").resize((256, 114)), (x, y))
            draw.text((x, y - 16), f"{label[0]}-{label[1]}  {modality.upper()}", fill="white")
            for where, strength in zip(row["paths"][modality]["xy"], row["paths"][modality]["strength"]):
                px, py = (np.asarray(where) - left) * (2, 1) + (x, y)
                color = "#7df4cd" if strength >= MIN_SUPPORT else "#ef776f"
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    sheet.save(output / "missing_probes.png")


def score(labels_dir, data_root, output):
    archive, sample, labels, paths = load_inputs(labels_dir, data_root)
    checkpoint = archive["checkpoints"]["e2"]
    checkpoint_hash = archive["checkpoint_hashes"]["e2"]
    if sha256_file(ROOT / checkpoint) != checkpoint_hash:
        raise ValueError("The archived E2 checkpoint hash does not match.")
    visibility_path = labels_dir / "e2_visibility.csv"
    visibility = load_visibility(visibility_path, labels)
    rows, details = [], []
    for scene, files in paths.items():
        images = []
        for j, path in enumerate(files):
            with Image.open(path) as image:
                band = image.convert("YCbCr").getchannel("Y") if j == 0 else image.convert("L")
                images.append(np.asarray(band))
        for label in (label for label in labels if label["scene"] == scene):
            row, samples = analyze_segment(label, images)
            row["e2_visibility"] = visibility.get((scene, row["id"]), "")
            rows.append(row)
            details.append(dict(row, paths=samples))
    positive = [r for r in rows if r["label"] == "connected"]
    missing = [r for r in positive if r["e2_visibility"] == "missing"]
    negative = [r for r in rows if r["label"] == "disconnected"]
    budgets = [dict(group=group, budget=budget, method=method, **hits_at_budget(probes + negative, method, budget))
               for group, probes in (("missing", missing), ("all_connected", positive))
               for budget in FALSE_BUDGETS for method in METHODS]
    coverage = {name: {m: sum(r[f"{m}_continuous"] for r in probes) for m in METHODS if m != "best_single"}
                for name, probes in (("missing", missing), ("all_connected", positive), ("negative", negative))}
    summary = dict(scenes=len(paths), connected=len(positive), missing=len(missing), negative=len(negative),
                   uncertain_source=sum(r["label"] == "uncertain" for r in rows),
                   uncertain_e2=sum(r["e2_visibility"] == "uncertain" for r in positive),
                   coverage=coverage, linked_only_missing=sum(r["complementary"] for r in missing),
                   linked_only_negatives=sum(r["complementary"] for r in negative),
                   representative_budget_one={r["method"]: r["hits"] for r in budgets
                                              if r["group"] == "missing" and r["budget"] == 1})
    meta = dict(**sample, checkpoint=checkpoint, checkpoint_sha256=checkpoint_hash,
                source_labels_sha256=sha256_file(labels_dir / "source_labels.csv"),
                e2_visibility_sha256=sha256_file(visibility_path), script_sha256=sha256_file(Path(__file__)),
                offsets=OFFSETS.tolist(), step_px=STEP_PX, min_support=MIN_SUPPORT, min_coverage=MIN_COVERAGE,
                max_gap=MAX_GAP, min_source_fraction=MIN_SOURCE_FRACTION,
                images_sha256={scene: {m: sha256_file(p) for m, p in zip(("optical", "sar", "e2"), files)}
                               for scene, files in paths.items()})
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "segments.csv", rows)
    write_csv(output / "operating_points.csv", budgets)
    dump_json(output / "paths.json", details)
    dump_json(output / "summary.json", summary)
    dump_json(output / "run_metadata.json", meta)
    endpoints_review(paths, details, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "review", "score"))
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--data-root", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_holdout(args.output_dir, args.data_root)
    elif args.action == "review":
        _, _, labels, paths = load_inputs(args.labels_dir, args.data_root)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        draw_review(labels, paths, args.output_dir)
    else:
        score(args.labels_dir, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
