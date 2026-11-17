"""RICF preflight: linear recoverability at the frozen E2 Detail bottleneck.

All neural parameters frozen. Ridge probes and the PCA basis are fitted ONLY
on locked train1024 scenes. Evaluate on separate val64 scenes. No RICF model
and no learned multimodal fusion network is trained by this diagnostic.

A 64-dim PCA input is a trained-on-train-data linear capacity control, NOT
an image fusion output. Lower source-feature readout error alone does not
establish better image fusion or prove RICF's usefulness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_qxs_reference_test1000 import DATA

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("test_result/e2_ricf_bottleneck_preflight_v1/train8_val16_seed42")
SEED = 42
SOURCE_NAMES = ("optical", "sar")
PROBE_NAMES = ("base_only", "e2_fused", "pca64", "random64", "concat128")
FEATURE_STRIDE = 4
RIDGE = 0.01
EPS = 1e-8


def flatten_grid(tensor, stride=FEATURE_STRIDE):
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Expected one BCHW feature map")
    return (tensor[0, :, ::stride, ::stride].permute(1, 2, 0)
            .reshape(-1, tensor.shape[1]).detach().cpu().double())


def standardizer(matrix):
    matrix = matrix.double()
    mean = matrix.mean(0, keepdim=True)
    # Preserve constants rather than amplifying very-low-variance channels.
    stdev = matrix.std(0, keepdim=True, unbiased=False)
    stdev = torch.where(stdev > 1e-6, stdev, torch.ones_like(stdev))
    return mean, stdev


def standardize(matrix, stats):
    return (matrix.double() - stats[0]) / stats[1]


def ridge_fit(x, y, ridge=RIDGE):
    """Train-only multivariate ridge with an unregularized intercept."""
    x, y = x.double(), y.double()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected matched [pixels, features] matrices")
    if x.shape[0] < x.shape[1] or not 0 < ridge < 1:
        raise ValueError("Insufficient samples or invalid ridge")
    xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc / len(x)
    lam = ridge * (torch.trace(gram) / gram.shape[0]).clamp_min(EPS)
    weights = torch.linalg.solve(
        gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype), xc.T @ yc / len(x))
    return xm, ym, weights


def ridge_predict(x, probe):
    xm, ym, weights = probe
    return (x.double() - xm) @ weights + ym


def pca_basis(train, components=64):
    train = train.double()
    if train.ndim != 2 or components <= 0 or components > train.shape[1]:
        raise ValueError("Invalid PCA dimensions")
    mean = train.mean(0, keepdim=True)
    centered = train - mean
    covariance = centered.T @ centered / len(train)
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -components:].flip(1)
    return mean, basis, float(values[-components:].clamp_min(0).sum() /
                               values.clamp_min(0).sum().clamp_min(EPS))


def random_basis(dimension=128, components=64, seed=SEED):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dimension, components, dtype=torch.float64, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced")[0]


def build_inputs(base, fused, optical, sar, transforms):
    """All probes retain the SAME fused Base context for a fair comparison."""
    base_z = standardize(base, transforms["base_stats"])
    source_z = torch.cat((
        standardize(optical, transforms["optical_stats"]),
        standardize(sar, transforms["sar_stats"])), dim=1)
    fused_z = standardize(fused, transforms["fused_stats"])
    pca_mean, pca_vectors, _ = transforms["pca"]
    return {
        "base_only": base_z,
        "e2_fused": torch.cat((base_z, fused_z), dim=1),
        "pca64": torch.cat((base_z, (source_z - pca_mean) @ pca_vectors), dim=1),
        "random64": torch.cat((base_z, source_z @ transforms["random"]), dim=1),
        "concat128": torch.cat((base_z, source_z), dim=1),
    }


def make_transform(train):
    packed = {name: torch.cat([sample[name] for sample in train], dim=0)
              for name in ("base", "fused", *SOURCE_NAMES)}
    transforms = {name + "_stats": standardizer(packed[name])
                  for name in packed}
    source = torch.cat((
        standardize(packed["optical"], transforms["optical_stats"]),
        standardize(packed["sar"], transforms["sar_stats"])), dim=1)
    transforms["pca"] = pca_basis(source, components=packed["optical"].shape[1])
    transforms["random"] = random_basis(
        source.shape[1], packed["optical"].shape[1], seed=SEED)
    return transforms


def matrix_targets(sample, transforms):
    return torch.cat((
        standardize(sample["optical"], transforms["optical_stats"]),
        standardize(sample["sar"], transforms["sar_stats"])), dim=1)


def original_edge_magnitude(image):
    """Image-domain Sobel magnitude; only a residual correlation proxy."""
    import cv2
    image = np.asarray(image, dtype=np.float32) / 255.0
    sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return torch.from_numpy(
        (np.abs(sx) + np.abs(sy))[::FEATURE_STRIDE, ::FEATURE_STRIDE]
        .reshape(-1).astype(np.float64))


def encode_scene(model, manifest, scene, split, device):
    optical_u8, sar_u8 = read_pair(locate_scene(DATA, manifest, scene))
    optical, sar = to_tensor(optical_u8, device), to_tensor(sar_u8, device)
    with torch.inference_mode():
        ob, od, sb, sd = model.encode_pair(optical, sar)
        fb, fd = model.fuse_features(ob, od, sb, sd)
    out = {
        "scene": str(scene), "split": split,
        "base": flatten_grid(fb), "fused": flatten_grid(fd),
        "optical": flatten_grid(od), "sar": flatten_grid(sd),
        "edge_optical": original_edge_magnitude(optical_u8),
        "edge_sar": original_edge_magnitude(sar_u8),
        "valid_sar": torch.from_numpy(
            (sar_u8[::FEATURE_STRIDE, ::FEATURE_STRIDE] > 0).reshape(-1)),
    }
    n = len(out["optical"])
    if any(len(out[name]) != n for name in
           ("base", "fused", "sar", "edge_optical", "edge_sar", "valid_sar")):
        raise ValueError("Input features / source image spatial sampling mismatch")
    if out["optical"].shape[1] != 64 or out["sar"].shape[1] != 64:
        raise ValueError("Expected original E2 Detail to have 64 channels")
    if not torch.isfinite(out["fused"]).all():
        raise ValueError(f"Nonfinite E2 features in {split}:{scene}")
    return out


def target_error_rows(scene, name, actual, pred, valid):
    """Target MSE is in TRAIN-standardized feature units, not physical SAR."""
    if actual.shape != pred.shape or actual.shape[1] != 128:
        raise ValueError("Expected concatenated 64+64 source Detail targets")
    rows = []
    for source, view in (("optical", slice(0, 64)), ("sar", slice(64, 128))):
        keep = torch.ones(len(actual), dtype=torch.bool)
        if source == "sar":
            keep = valid
        if not bool(keep.any()):
            raise ValueError("Empty SAR valid support")
        truth, guess = actual[keep, view], pred[keep, view]
        err = (truth - guess).square()
        nerr = float(err.mean())
        variance = float((truth - truth.mean(dim=0)).square().mean())
        rows.append(dict(scene=str(scene), method=name, target=source,
                         nmse=nerr, heldout_variance=variance,
                         r2=1 - nerr / max(variance, EPS),
                         valid_pixels=int(keep.sum())))
    return rows


def edge_residual_proxy(scene, method, target, prediction, edge, valid):
    """High/low raw-source gradient quartile residual MSE.

    This is not a structure label; SAR high gradients can be speckle.
    """
    squared = (target - prediction).square().mean(1)
    edge, squared = edge.double(), squared.double()
    usable = torch.ones(len(edge), dtype=torch.bool) if valid is None else valid
    if usable.sum() < 32:
        raise ValueError("Insufficient valid pixels")
    thresholds = torch.quantile(edge[usable], torch.tensor([0.2, 0.8], dtype=torch.float64))
    low = usable & (edge <= thresholds[0])
    high = usable & (edge >= thresholds[1])
    low_mse = float(squared[low].mean())
    high_mse = float(squared[high].mean())
    return dict(scene=str(scene), method=method,
                target="sar" if valid is not None else "optical",
                low_gradient_residual_mse=low_mse,
                high_gradient_residual_mse=high_mse,
                high_to_low_residual=high_mse/max(low_mse, EPS),
                high_gradient_pixels=int(high.sum()),
                low_gradient_pixels=int(low.sum()))


def train_probes(samples, transforms):
    concat = {name: [] for name in PROBE_NAMES}
    targets = []
    for s in samples:
        predictors = build_inputs(s["base"], s["fused"], s["optical"], s["sar"],
                                  transforms)
        for name in PROBE_NAMES:
            concat[name].append(predictors[name])
        targets.append(matrix_targets(s, transforms))
    y = torch.cat(targets, dim=0)
    probes = {name: ridge_fit(torch.cat(values, dim=0), y)
              for name, values in concat.items()}
    return probes


def evaluate(samples, transforms, probes):
    errors, edges = [], []
    for sample in samples:
        predictors = build_inputs(
            sample["base"], sample["fused"], sample["optical"], sample["sar"],
            transforms)
        actual = matrix_targets(sample, transforms)
        for name, predictor in predictors.items():
            pred = ridge_predict(predictor, probes[name])
            errors.extend(target_error_rows(
                sample["scene"], name, actual, pred, sample["valid_sar"]))
            for source, view, edge, valid in (
                ("optical", slice(0,64), sample["edge_optical"], None),
                ("sar", slice(64,128), sample["edge_sar"], sample["valid_sar"])):
                edges.append(edge_residual_proxy(
                    sample["scene"], name, actual[:, view], pred[:, view],
                    edge, valid))
    return errors, edges


def summarize(errors, transforms):
    results = {}
    for target in SOURCE_NAMES:
        rows = [row for row in errors if row["target"] == target]
        base_rows = [row for row in rows if row["method"] == "base_only"]
        baseline = sum(row["nmse"] for row in base_rows) / len(base_rows)
        methods = {}
        for method in PROBE_NAMES:
            selected = [row for row in rows if row["method"] == method]
            mean_nmse = sum(row["nmse"] for row in selected) / len(selected)
            methods[method] = dict(
                mean_nmse=mean_nmse,
                mean_r2=sum(row["r2"] for row in selected)/len(selected),
                improvement_vs_base=(baseline-mean_nmse)/max(baseline, EPS),
                scenes_better_than_base=sum(
                    r["nmse"] < next(x["nmse"] for x in rows
                    if x["scene"] == r["scene"] and x["method"] == "base_only")
                    for r in selected))
        results[target] = methods
    results["pca_explained_variance_64"] = transforms["pca"][2]
    return results


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows")
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    if not (2 <= args.train_scenes <= 64 and 2 <= args.val_scenes <= 64):
        raise ValueError("Scene counts must be in [2, 64]")
    if not 0 < args.ridge < 1:
        raise ValueError("Ridge must be in (0,1)")
    global RIDGE
    RIDGE = args.ridge
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    manifest = source_manifest()
    train_ids = [str(x) for x in manifest["splits"]["train"][:args.train_scenes]]
    val_ids = [str(x) for x in manifest["splits"]["val"][:args.val_scenes]]
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation scene IDs overlap")
    output = args.output_dir or DEFAULT_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite previous experiment: {output}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = frozen_checkpoint(device)
    training = []
    for i, scene in enumerate(train_ids):
        training.append(encode_scene(model, manifest, scene, "train", device))
        print(f"feature train {i+1}/{len(train_ids)}: {scene}", flush=True)
    transforms = make_transform(training)
    probes = train_probes(training, transforms)
    validation = []
    for i, scene in enumerate(val_ids):
        validation.append(encode_scene(model, manifest, scene, "val", device))
        print(f"feature val {i+1}/{len(val_ids)}: {scene}", flush=True)
    errors, edge_errors = evaluate(validation, transforms, probes)
    results = summarize(errors, transforms)
    meta = {
        "status": "frozen_e2_linear_probes_only_no_ricf_training",
        "dataset": "QXS-SAROPT", "split": "matched_pilot_train1024_val64",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "script_sha256": sha256_file(Path(__file__)),
        "train_scene_ids": train_ids, "val_scene_ids": val_ids,
        "seed": SEED, "device": str(device),
        "feature_stride": FEATURE_STRIDE, "ridge": RIDGE,
        "conditions": {
            "base_only": "64-dim E2 fused_base",
            "e2_fused": "E2 fused_base + original 64-dim E2 fused_detail",
            "pca64": "E2 fused_base + train-fitted rank64 PCA of concat optical/sar Details",
            "random64": "E2 fused_base + fixed seeded rank64 orthonormal random projection",
            "concat128": "E2 fused_base + raw concatenated 128-dim source Detail oracle",
        },
        "readout": "train-only linear ridge probes; no neural parameter updates",
        "targets": "source E2 optical/sar Detail (train-standardized channels)",
        "caveats": [
            "Feature recoverability != useful image fusion information",
            "PCA and concat controls see source Detail directly, E2 is trained for fusion image output",
            "The probe is linear; nonlinearly recoverable source information may be underestimated",
            "High source-image gradient is NOT a ground-truth structure label",
            "No inference image metrics or RICF network quality are evaluated",
        ],
        "summary": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ricf-preflight-", dir=output.parent) as temp:
        pending = Path(temp) / "result"
        pending.mkdir()
        write_csv(pending / "recoverability.csv", errors)
        write_csv(pending / "edge_residual_proxy.csv", edge_errors)
        (pending / "summary.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        pending.rename(output)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=8)
    parser.add_argument("--val-scenes", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
