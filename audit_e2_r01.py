"""R0.1 analysis of archived R0 metrics and source-first scene review."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_e2_architecture_r0 import (
    METHODS, MS_WEIGHTS, ROOT, load_inputs, read_image, scale_readouts, sha256, write_csv,
)


R0 = ROOT / "test_result/e2_architecture_r0_v1/val1000_seed42"
PILOT = ROOT / "models/e2_scratch_source_interval_pilot_v1/seed42/pilot_manifest.json"
METHOD_KEYS = ("FMI", "VIF", "Qabf", "Nabf", "SSIM")
PROBES = (*METHOD_KEYS, "source_dominant_pixels_o", "source_dominant_pixels_s",
          "low_source_gradient_pixels", "output_highpass_rms")
SOURCE_LABELS = ("scene", "probe", "source", "x0", "y0", "x1", "y1", "notes")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_archive():
    metadata = read_json(R0 / "run_metadata.json")
    scenes = metadata["selected_scene_ids"]
    records = read_csv(R0 / "per_scene.csv")
    by_key = {(row["scene"], row["method"]): row for row in records}
    if (metadata["split"] != "val" or len(records) != len(by_key)
            or set(by_key) != {(scene, method) for scene in scenes for method in METHODS}):
        raise ValueError("Unexpected R0 archive or incomplete method × scene coverage")
    return scenes, by_key


def components(row, side):
    return np.array([float(row[f"cs_{side}_{scale}x"]) for scale in (1, 2, 4, 8)]
                    + [float(row[f"ssim_{side}_16x"])])


def audit_archive(output, scenes, records):
    values = np.empty((len(scenes), len(METHODS), 2))
    all_terms = np.empty((*values.shape, 5))
    for i, scene in enumerate(scenes):
        for j, method in enumerate(METHODS):
            row = records[scene, method]
            for k, side in enumerate(("o", "s")):
                terms = components(row, side)
                all_terms[i, j, k] = terms
                if not np.isfinite(terms).all():
                    raise ValueError(f"Nonfinite component, {scene}/{method}/{side}")
                value = np.prod(np.maximum(terms, 0) ** MS_WEIGHTS)
                original = float(row[f"ms_ssim_{side}"])
                if (np.any(terms < 0) != np.isnan(original)
                        or (np.isfinite(original) and not np.isclose(value, original, atol=1e-10, rtol=0))):
                    raise ValueError(f"Cached component replay mismatch, {scene}/{method}/{side}")
                values[i, j, k] = value
    negative = all_terms < 0
    valid = ~negative.any(axis=(1, 2, 3))
    scores = values.mean(axis=2)
    ranks, frequencies, cohorts, tradeoffs, per_scene = [], [], [], [], []
    for j, method in enumerate(METHODS):
        official = np.array([float(records[scene, method]["MS_SSIM"]) for scene in scenes])
        if not np.allclose(scores[valid, j], official[valid], atol=1e-10, rtol=0):
            raise ValueError(f"Official score replay mismatch: {method}")
        for scope, mask in (("primary_common", valid), ("clamp_common", valid),
                            ("clamp_all", np.ones(len(scenes), dtype=bool)), ("clamp_excluded", ~valid)):
            series = official if scope == "primary_common" else scores[:, j]
            ranks.append(dict(scope=scope, method=method, count=int(mask.sum()),
                              mean=float(series[mask].mean()),
                              mean_delta_e2=float((series[mask] - scores[mask, 0]).mean())))
        for k, side in enumerate(("optical", "sar")):
            for level, scale in enumerate((1, 2, 4, 8, 16)):
                frequencies.append(dict(method=method, source=side, scale=scale,
                                        term="SSIM" if scale == 16 else "CS",
                                        negative_count=int(negative[:, j, k, level].sum()),
                                        total=len(scenes)))
        for i, scene in enumerate(scenes):
            per_scene.append(dict(scene=scene, method=method, primary_common=int(valid[i]),
                                  primary=records[scene, method]["MS_SSIM"],
                                  clamp_optical=values[i, j, 0], clamp_sar=values[i, j, 1],
                                  clamp_mean=scores[i, j]))
        for key in PROBES:
            series = np.array([float(records[scene, method][key]) for scene in scenes])
            a, b = series[valid], series[~valid]
            pooled = np.sqrt((a.var() + b.var()) / 2)
            cohorts.append(dict(method=method, readout=key, valid_n=len(a), excluded_n=len(b),
                                valid_mean=a.mean(), excluded_mean=b.mean(),
                                excluded_minus_valid_std=(b.mean() - a.mean()) / pooled if pooled else ""))
        if j == 0:
            continue
        delta = values[:, j] - values[:, 0]
        trade = (delta[:, 0] < 0) & (delta[:, 1] > 0)
        for scope, scope_mask in (("common", valid), ("excluded_clamp", ~valid)):
            for kind, mask in (("sar_up_optical_down", scope_mask & trade),
                               ("other", scope_mask & ~trade)):
                for key in METHOD_KEYS:
                    d = np.array([float(records[scene, method][key]) - float(records[scene, "e2"][key])
                                  for scene in scenes])[mask]
                    tradeoffs.append(dict(scope=scope, subset=kind, method=method, metric=key,
                                          count=len(d), mean_delta=d.mean() if len(d) else "",
                                          worse_count=int((d > 0 if key == "Nabf" else d < 0).sum())))
    for name, rows in (("ranking", ranks), ("negative_frequency", frequencies),
                       ("cohort_differences", cohorts), ("source_tradeoff", tradeoffs),
                       ("ms_ssim_per_scene", per_scene)):
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "metadata.json", {
        "analysis": "R0 archived scalars; clip each source's four CS and last SSIM to zero",
        "input_csv": str(R0 / "per_scene.csv"), "input_sha256": sha256(R0 / "per_scene.csv"),
        "scene_count": len(scenes), "primary_common_count": int(valid.sum()),
        "negative_scene_counts": dict(zip(("optical", "sar"), negative.any(axis=(1, 3)).sum(axis=0).tolist())),
        "script_sha256": sha256(Path(__file__)),
    })


def input_paths(scene, manifest_path, manifest):
    dataset = manifest_path.parent.parent
    return (dataset / manifest["optical_dir"] / f"{scene}.png",
            dataset / manifest["sar_dir"] / f"{scene}.png")


def save_png(path, array):
    # imencode + tofile handles Unicode directories on Windows.
    ok, encoded = cv2.imencode(".png", np.rint(np.clip(array, 0, 255)).astype(np.uint8))
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(str(path))


def prepare(output, scenes, records, count, seed):
    _, manifest_path, manifest, _, _, _ = load_inputs()
    pilot_ids = set(map(str, read_json(PILOT)["splits"]["val"]))
    pilot_ids.update(("14015", "12158", "15628", "15481", "15334", "19853", "18763"))
    pools = {"common": [], "excluded": []}
    for scene in scenes:
        if scene not in pilot_ids:
            finite = all(np.isfinite(float(records[scene, method]["MS_SSIM"])) for method in METHODS)
            pools["common" if finite else "excluded"].append(scene)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pools["common"], size=(count + 1) // 2, replace=False).tolist()
    chosen += rng.choice(pools["excluded"], size=count // 2, replace=False).tolist()
    rng.shuffle(chosen)
    # The review packet exposes source pairs and scene IDs only.
    for scene in chosen:
        optical_path, sar_path = input_paths(scene, manifest_path, manifest)
        optical = read_image(optical_path, optical=True)
        sar = read_image(sar_path)
        save_png(output / f"{scene}.png", np.hstack((optical, sar)))
    with (output / "source_labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_LABELS)
        writer.writerows([[scene, "", "", "", "", "", "", ""] for scene in chosen])
    write_json(output / "sample.json", {"scenes": chosen, "seed": seed,
                                        "excluded_pilot_ids": sorted(pilot_ids),
                                        "geographic_block_ids_available": False,
                                        "split_manifest_sha256": sha256(manifest_path)})


def load_annotations(labels, chosen):
    annotations = read_csv(labels)
    if {row["scene"] for row in annotations} != set(chosen):
        raise ValueError("Record each sampled scene, including scenes with no clear source boundary")
    grouped = {scene: [] for scene in chosen}
    probed, no_boundary = set(), set()
    for row in annotations:
        scene = row["scene"]
        if row["source"] == "none":
            if not row["notes"].strip():
                raise ValueError(f"{scene}: document why no clear boundary was found")
            no_boundary.add(scene)
            continue
        if row["source"] not in ("optical", "sar", "both") or not row["probe"]:
            raise ValueError(f"{scene}: add an annotated source and probe ID or mark source=none")
        coords = tuple(int(row[key]) for key in SOURCE_LABELS[3:7])
        if not all(0 <= x < 256 for x in coords) or coords[:2] == coords[2:]:
            raise ValueError(f"{scene}: invalid boundary endpoints")
        probe = scene, row["probe"]
        if probe in probed:
            raise ValueError(f"{scene}: duplicate probe ID")
        probed.add(probe)
        grouped[scene].append(coords)
    if any(grouped[scene] for scene in no_boundary):
        raise ValueError("A scene marked source=none also contains a labeled boundary")
    return grouped


def align_source(source, fused, mask):
    src, target = source[mask], fused[mask]
    if src.size < 32 or src.std() < 1e-8 or target.std() < 1e-8:
        return None
    aligned = source.copy()
    aligned[mask] = np.clip((src - src.mean()) * target.std() / src.std() + target.mean(), 0, 255)
    return aligned


def local_moment_error(fused, source, supported):
    errors = []
    for image in (fused, source):
        blocks = image.reshape(16, 16, 16, 16).transpose(0, 2, 1, 3)[supported]
        errors.append(np.stack((blocks.mean(axis=(1, 2)), blocks.std(axis=(1, 2))), axis=1))
    return np.abs(errors[0] - errors[1]).mean(axis=0) if supported.any() else (np.nan, np.nan)


def inspect(output, packet, labels, expected_hash, records, only_scene):
    if sha256(labels) != expected_hash.lower():
        raise ValueError("Source annotation hash does not match the locked value")
    _, manifest_path, manifest, _, images, _ = load_inputs()
    sample = read_json(packet / "sample.json")
    if sample["split_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Review sample came from another split manifest")
    grouped = load_annotations(labels, sample["scenes"])
    if only_scene is not None:
        grouped = {only_scene: grouped[only_scene]}
    rows = []
    for scene, segments in grouped.items():
        optical_path, sar_path = input_paths(scene, manifest_path, manifest)
        optical, sar = read_image(optical_path, optical=True), read_image(sar_path)
        valid = sar > 0
        supported = valid.reshape(16, 16, 16, 16).all(axis=(1, 3))
        tiles = [optical, sar]
        for method in METHODS:
            fused = read_image(images[method] / f"{scene}.png")
            tiles.append(fused)
            for side, source in (("o", optical), ("s", sar)):
                aligned = align_source(source, fused, valid)
                if aligned is not None:
                    stats, ms = scale_readouts(fused, aligned, side)
                    aligned_mean, aligned_sd = local_moment_error(fused, aligned, supported)
                else:
                    stats, ms = {}, float("nan")
                    aligned_mean, aligned_sd = np.nan, np.nan
                raw_mean, raw_sd = local_moment_error(fused, source, supported)
                original = records[scene, method]
                rows.append({"scene": scene, "method": method, "side": side,
                             "complete_patches": int(supported.sum()),
                             "raw_local_mean_mae": raw_mean, "raw_local_sd_mae": raw_sd,
                             "aligned_local_mean_mae": aligned_mean, "aligned_local_sd_mae": aligned_sd,
                             "raw_ms": original[f"ms_ssim_{side}"], "aligned_ms": ms,
                             "raw_cs_4x": original[f"cs_{side}_4x"], "aligned_cs_4x": stats.get(f"cs_{side}_4x", ""),
                             "raw_cs_8x": original[f"cs_{side}_8x"], "aligned_cs_8x": stats.get(f"cs_{side}_8x", "")})
        panel = np.hstack(tiles)
        save_png(output / f"{scene}_original.png", panel)
        overlay = cv2.cvtColor(panel.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for x0, y0, x1, y1 in segments:
            for col in range(len(tiles)):
                cv2.line(overlay, (x0 + 256 * col, y0), (x1 + 256 * col, y1), (0, 160, 255), 1)
        save_png(output / f"{scene}_overlay.png", overlay)
    write_csv(output / "photometric.csv", rows)
    write_json(output / "review_metadata.json", {"columns": ["optical", "sar", *METHODS],
               "source_labels_sha256": sha256(labels), "sample_sha256": sha256(packet / "sample.json"),
               "input_csv_sha256": sha256(R0 / "per_scene.csv"), "scenes": list(grouped)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("audit")
    prep = subs.add_parser("prepare")
    prep.add_argument("--count", type=int, default=24)
    prep.add_argument("--seed", type=int, default=20261123)
    show = subs.add_parser("inspect")
    show.add_argument("--packet", type=Path, required=True)
    show.add_argument("--labels", type=Path, required=True)
    show.add_argument("--labels-sha256", required=True)
    show.add_argument("--scene")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    scenes, records = load_archive()
    if args.action == "prepare" and args.count < 1:
        parser.error("--count must be positive")
    args.output_dir.mkdir(parents=True)
    if args.action == "audit":
        audit_archive(args.output_dir, scenes, records)
    elif args.action == "prepare":
        prepare(args.output_dir, scenes, records, args.count, args.seed)
    else:
        inspect(args.output_dir, args.packet, args.labels, args.labels_sha256, records, args.scene)


if __name__ == "__main__":
    main()
