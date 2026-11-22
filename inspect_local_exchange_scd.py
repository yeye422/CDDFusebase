"""Read-only decomposition of registered SCD values from archived grayscale images."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from diagnose_e2_information_path import read_pair
from maintain_experiment_index import sha256_file
from run_qxs_reference_test1000 import DATA, read_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import FusionEvaluator


BASE = Path("test_result/e2_local_correction_pilot_v3/val936/seed42/e2")
EXCHANGE = Path("test_result/e2_local_correspondence_val936_v1/seed42/candidate")


def components(folder, scene, optical, sar, original):
    raw = cv2.imread(str(folder / "gray" / f"{scene}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint8 or raw.shape != optical.shape:
        raise ValueError(f"Missing or invalid archived grayscale image: {folder}/{scene}")
    fused = raw.astype(np.float64)
    optical_term = FusionEvaluator._correlation(optical, fused - sar)
    sar_term = FusionEvaluator._correlation(sar, fused - optical)
    if not np.isclose(optical_term + sar_term, original, equal_nan=True, rtol=0, atol=1e-8):
        raise ValueError(f"Decomposition differs from archived SCD: {folder}/{scene}")
    return [optical_term, sar_term]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("first", "val936"))
    mode = parser.parse_args().mode
    os.chdir(Path(__file__).resolve().parent)
    meta = read_json(EXCHANGE.parent / "comparison_vs_e2/run_metadata.json")
    split = Path(meta["split_manifest"])
    if sha256_file(split) != meta["split_manifest_sha256"]:
        raise ValueError("The archived split has changed")
    pairs, _ = manifest_pairs(DATA, split)
    ids = [a.stem for a, _ in pairs["val"]]
    if len(ids) != 936 or ids != meta["scene_ids"]:
        raise ValueError("Unexpected archived validation set")
    tables = {}
    for folder in (BASE, EXCHANGE):
        path = folder / "metrics.csv"
        if sha256_file(path) != meta["input_tables"][str(path)]:
            raise ValueError(f"Changed archived metrics: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["image"] not in ("mean", "std")]
        if [r["image"] for r in rows] != ids:
            raise ValueError(f"Unexpected scene order: {path}")
        tables[folder] = {r["image"]: r for r in rows}
    parts, joint = [], []
    for pair in (pairs["val"][:1] if mode == "first" else pairs["val"]):
        scene = pair[0].stem
        optical, sar = (x.astype(np.float64) for x in read_pair(pair))
        parts.append([components(f, scene, optical, sar, float(tables[f][scene]["SCD"])) for f in (BASE, EXCHANGE)])
        a, b = tables[BASE][scene], tables[EXCHANGE][scene]
        joint.append(float(b["Qabf"]) > float(a["Qabf"]) and float(b["Nabf"]) < float(a["Nabf"])
                     and float(b["VIF"]) > float(a["VIF"]))
    values = np.asarray(parts)
    finite = np.isfinite(values).all(axis=(1, 2))
    scd_fell = values[:, 1].sum(axis=1) < values[:, 0].sum(axis=1)
    lines = []
    for name, subset in (("all", finite), ("SCD fell", finite & scd_fell),
                         ("three improved + SCD fell", finite & joint & scd_fell)):
        v = values[subset]
        if not len(v):
            continue
        lines.append(f"{name}, n={len(v)}")
        for i, label in enumerate(("corr(optical,F-SAR)", "corr(SAR,F-optical)")):
            base, exchange = v[:, :, i].T
            d = exchange - base
            lines.append(f"  {label}: {base.mean():.6f} -> {exchange.mean():.6f}, "
                  f"delta={d.mean():+.6f}, decreased={sum(d < 0)}")
    lines.append(f"Checked {len(parts)} archived pairs; common finite pairs: {int(finite.sum())}.")
    report = "\n".join(lines) + "\n"
    output = EXCHANGE.parent / "comparison_vs_e2" / f"scd_components_{mode}.txt"
    if output.exists() and output.read_text(encoding="utf-8") != report:
        raise FileExistsError(f"Preserving an existing, different report: {output}")
    if not output.exists():
        with output.open("x", encoding="utf-8") as handle:
            handle.write(report)
    print(report, end="")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
