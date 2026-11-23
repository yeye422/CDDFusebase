"""Attribute differences in archived QXS-SAROPT validation outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from utils.reference_metrics import ReferenceEvaluator, _gaussian_ssim


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "test_result/e2_quality_audit_v1/val1000_seed42/run_metadata.json"
PROFILE = "reference_v2_candidate_with_paper_fmi"
METHODS = ("e2", "sar_directional", "strict_overshoot", "source_interval")
METRICS = ("SSIM", "MS_SSIM", "FMI", "MI", "VIF", "Qabf", "Nabf")
MS_WEIGHTS = (.0448, .2856, .3001, .2363, .1333)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["split"], audit["metric_profile"], audit["output_policy"]) != (
        "val", PROFILE, "clamp_then_uint8"
    ):
        raise ValueError("Unexpected validation archive protocol")
    manifest_path = ROOT / audit["split_manifest"]
    if sha256(manifest_path) != audit["split_manifest_sha256"]:
        raise ValueError("Split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_ids = audit["selected_scene_ids"]
    if scene_ids != [str(scene) for scene in manifest["splits"]["val"]]:
        raise ValueError("Archive scene order differs from the validation manifest")

    images, metrics = {}, {}
    for method in METHODS:
        source = audit["sources"][method]
        csv_path = ROOT / source["csv"]
        if sha256(csv_path) != source["csv_sha256"]:
            raise ValueError(f"Archived metric file has changed: {csv_path}")
        if method == "e2":
            images[method] = ROOT / "test_result/e2_scratch_base_anchor_v1/val1000/seed42/gray"
        else:
            images[method] = csv_path.parent / "gray"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row["image"] not in ("mean", "std")]
        if [row["image"] for row in rows] != scene_ids:
            raise ValueError(f"Validation scene order mismatch: {csv_path}")
        metrics[method] = {
            row["image"]: {key: float(row[key]) for key in METRICS} for row in rows
        }
    return audit, manifest_path, manifest, scene_ids, images, metrics


def read_image(path, optical=False):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if optical else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[:2] != (256, 256):
        raise ValueError(f"Expected 256x256 QXS-SAROPT tile: {path}")
    if optical:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
    return image.astype(np.float64)


def scale_readouts(fused, source, name):
    stats, terms = {}, []
    for level in range(5):
        ss, cs = _gaussian_ssim(fused, source)
        stats[f"ssim_{name}_{2**level}x"] = ss
        stats[f"cs_{name}_{2**level}x"] = cs
        terms.append(ss if level == 4 else cs)
        if level < 4:
            # On even shapes this is the evaluator's anchored 2x2 filter plus stride 2.
            fused, source = (
                image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2).mean(axis=(1, 3))
                for image in (fused, source)
            )
    stats[f"negative_ms_terms_{name}"] = "|".join(
        f"{2**level}x" for level, term in enumerate(terms) if term < 0
    )
    ms = float("nan") if min(terms) < 0 else float(np.prod(np.power(terms, MS_WEIGHTS)))
    return stats, ms


def source_edges(image):
    smoothed = cv2.GaussianBlur(image, (0, 0), 1)
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, scale=1 / 8)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, scale=1 / 8)
    return np.stack((gx, gy), axis=-1), np.hypot(gx, gy)


def gradient_transfer(fused_gradient, source_gradient, mask):
    if mask.sum() < 32:
        return float("nan")
    fused, source = fused_gradient[mask], source_gradient[mask]
    denom = np.linalg.norm(fused) * np.linalg.norm(source)
    return float((fused * source).sum() / denom) if denom > 0 else float("nan")


def masked_rms(values, mask):
    return float(np.sqrt(np.mean(values[mask]**2))) if mask.sum() >= 32 else float("nan")


def image_readouts(fused, optical, sar, archived):
    stats = dict(archived)
    for name, source in (("o", optical), ("s", sar)):
        pair_stats, stats[f"ms_ssim_{name}"] = scale_readouts(fused, source, name)
        stats.update(pair_stats)
    replay = (
        (stats["ssim_o_1x"] + stats["ssim_s_1x"]) / 2,
        (stats["ms_ssim_o"] + stats["ms_ssim_s"]) / 2,
        ReferenceEvaluator.Nabf(fused, optical, sar),
    )
    expected = tuple(archived[key] for key in ("SSIM", "MS_SSIM", "Nabf"))
    if not np.allclose(replay, expected, atol=1e-8, rtol=0, equal_nan=True):
        raise ValueError(f"Archived image/metric replay mismatch: {replay} vs {expected}")

    fg, _ = source_edges(fused)
    og, om = source_edges(optical)
    sg, sm = source_edges(sar)
    # Keep derivative support away from invalid SAR or image boundaries.
    valid = cv2.erode((sar > 0).astype(np.uint8), np.ones((11, 11), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    hp = fused - cv2.GaussianBlur(fused, (0, 0), 2)
    for name, sg_vector, source_mag, other_mag in (("o", og, om, sm), ("s", sg, sm, om)):
        mask = valid & (source_mag >= 2) & (source_mag >= 2 * other_mag)
        stats[f"source_dominant_pixels_{name}"] = int(mask.sum())
        stats[f"signed_gradient_transfer_{name}"] = gradient_transfer(fg, sg_vector, mask)
        stats[f"highpass_rms_source_dominant_{name}"] = masked_rms(hp, mask)

    stats["output_highpass_rms"] = float(np.sqrt(np.mean(hp*hp)))
    low_source_gradient = valid & (om < 2) & (sm < 2)
    stats["low_source_gradient_pixels"] = int(low_source_gradient.sum())
    stats["highpass_rms_low_source_gradient"] = masked_rms(hp, low_source_gradient)
    return stats


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlations(rows):
    summaries = []
    for left, right in (
        ("output_highpass_rms", "Nabf"),
        ("highpass_rms_source_dominant_o", "Nabf"),
        ("highpass_rms_source_dominant_s", "Nabf"),
        ("highpass_rms_low_source_gradient", "Nabf"),
        ("SSIM", "FMI"), ("MS_SSIM", "VIF"), ("SSIM", "Qabf"),
    ):
        excluded = {row["scene"] for row in rows
                    if not (np.isfinite(row[left]) and np.isfinite(row[right]))}
        for method in METHODS[1:]:
            subset = [row for row in rows
                      if row["method"] == method and row["scene"] not in excluded]
            x = np.array([row[left] for row in subset])
            y = np.array([row[right] for row in subset])
            rho = float(spearmanr(x, y).statistic) if (
                len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            ) else float("nan")
            summaries.append({"method": method, "delta_x": left, "delta_y": right,
                              "count": len(x), "spearman": rho})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", help="One validation tile for a smoke run")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    audit, manifest_path, manifest, scene_ids, images, metrics = load_inputs()
    if args.scene:
        if args.scene not in scene_ids:
            parser.error("--scene must be in the archived validation split")
        scene_ids = [args.scene]
    data_root = manifest_path.parent.parent
    rows = []
    for scene in scene_ids:
        optical = read_image(data_root / manifest["optical_dir"] / f"{scene}.png", optical=True)
        sar = read_image(data_root / manifest["sar_dir"] / f"{scene}.png")
        for method in METHODS:
            try:
                values = image_readouts(
                    read_image(images[method] / f"{scene}.png"), optical, sar, metrics[method][scene]
                )
            except ValueError as exc:
                raise ValueError(f"{method}, scene {scene}: {exc}") from exc
            rows.append({"scene": scene, "method": method, **values})
    numeric = [key for key in rows[0] if key not in ("scene", "method")
               and not key.startswith("negative_ms_terms")]
    by_scene = {(row["scene"], row["method"]): row for row in rows}
    deltas = []
    for scene in scene_ids:
        baseline = by_scene[scene, "e2"]
        for method in METHODS[1:]:
            candidate = by_scene[scene, method]
            deltas.append({"scene": scene, "method": method,
                           **{key: candidate[key] - baseline[key] for key in numeric}})
    summary = []
    for key in numeric:
        common = [scene for scene in scene_ids
                  if all(np.isfinite(by_scene[scene, method][key]) for method in METHODS)]
        for method in METHODS:
            values = np.array([by_scene[scene, method][key] for scene in common])
            paired = values - np.array([by_scene[scene, "e2"][key] for scene in common])
            summary.append({"readout": key, "method": method, "common_count": len(common),
                            "mean": values.mean() if common else "", "std": values.std() if common else "",
                            "mean_delta_e2": paired.mean() if common else ""})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_scene.csv", rows)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    write_csv(args.output_dir / "common_scene_summary.csv", summary)
    write_csv(args.output_dir / "paired_correlations.csv", correlations(deltas))
    metadata = {
        "split": "val", "selected_scene_ids": scene_ids, "metric_profile": PROFILE,
        "audit": str(AUDIT), "audit_sha256": sha256(AUDIT),
        "split_manifest": str(manifest_path.resolve()),
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "sources": {method: audit["sources"][method] for method in METHODS},
        "gray_directories": {method: str(path) for method, path in images.items()},
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Read {len(scene_ids)} validation scenes for {len(METHODS)} archived outputs")


if __name__ == "__main__":
    main()
