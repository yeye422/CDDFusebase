"""Expanded frozen E2 RICF bottleneck diagnostic, memory bounded.

Train-only: deterministic evenly-spaced train1024 samples. Three frozen feature
passes accumulate standardization, PCA and ridge sufficient statistics without
retaining all feature maps. Evaluate the existing five readouts on the complete
locked pilot val64, one scene at a time. No fusion-model training or test data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from statistics import median
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_subspace_gradients import frozen_checkpoint
from diagnose_ricf_bottleneck_preflight import (
    EPS, FEATURE_STRIDE, PROBE_NAMES, RIDGE, ROOT, SEED, SOURCE_NAMES,
    build_inputs, encode_scene, evaluate, make_transform, matrix_targets,
    random_basis, standardize, summarize, write_csv,
)
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from utils.dataset_os import evenly_spaced

DEFAULT_OUTPUT = Path("test_result/e2_ricf_bottleneck_expanded_v1/train128_val64_seed42")


class RunningMoments:
    """Summarize independent feature rows without retaining pixel matrices."""

    def __init__(self):
        self.n = 0
        self.sum_x = None
        self.sum_x2 = None

    def update(self, x):
        x = x.detach().cpu().double()
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Expected finite 2-D features")
        if self.sum_x is None:
            self.sum_x = torch.zeros(x.shape[1], dtype=torch.float64)
            self.sum_x2 = torch.zeros_like(self.sum_x)
        if x.shape[1] != self.sum_x.numel():
            raise ValueError("Feature dimension changed")
        self.n += x.shape[0]
        self.sum_x += x.sum(0)
        self.sum_x2 += x.square().sum(0)

    def get_stats(self):
        if self.n < 2:
            raise ValueError("Not enough observations")
        mean = (self.sum_x / self.n).unsqueeze(0)
        variance = (self.sum_x2 / self.n - mean.flatten().square()).clamp_min(0)
        std = variance.sqrt().unsqueeze(0)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        return mean, std


class CrossMoments:
    """Centered second moments for ridge or PCA, with train-only intercept."""

    def __init__(self):
        self.n = 0
        self.sum_x = None
        self.sum_y = None
        self.xtx = None
        self.xty = None

    def update(self, x, y=None):
        x = x.detach().cpu().double()
        y = x if y is None else y.detach().cpu().double()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("Expected paired 2-D features")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Nonfinite feature samples")
        if self.sum_x is None:
            self.sum_x = torch.zeros(x.shape[1], dtype=torch.float64)
            self.sum_y = torch.zeros(y.shape[1], dtype=torch.float64)
            self.xtx = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
            self.xty = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
        if x.shape[1] != len(self.sum_x) or y.shape[1] != len(self.sum_y):
            raise ValueError("Feature dimensions changed")
        self.n += len(x)
        self.sum_x += x.sum(0)
        self.sum_y += y.sum(0)
        self.xtx += x.T @ x
        self.xty += x.T @ y

    def centered(self):
        if self.n < 2:
            raise ValueError("Not enough samples")
        xm = self.sum_x / self.n
        ym = self.sum_y / self.n
        cov_xx = self.xtx / self.n - torch.outer(xm, xm)
        cov_xy = self.xty / self.n - torch.outer(xm, ym)
        cov_xx = (cov_xx + cov_xx.T) * 0.5
        return xm[None, :], ym[None, :], cov_xx, cov_xy

    def pca(self, components=64):
        xm, _, cov, _ = self.centered()
        if components > cov.shape[0]:
            raise ValueError("PCA rank greater than input dimension")
        eig, vec = torch.linalg.eigh(cov)
        eig = eig.clamp_min(0)
        return xm, vec[:, -components:].flip(1), float(
            eig[-components:].sum() / eig.sum().clamp_min(EPS))

    def ridge(self, ridge=RIDGE):
        xm, ym, gram, xy = self.centered()
        if self.n < gram.shape[0] or not 0 < ridge < 1:
            raise ValueError("Insufficient samples or invalid ridge")
        lam = ridge * (torch.trace(gram)/gram.shape[0]).clamp_min(EPS)
        weight = torch.linalg.solve(
            gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype), xy)
        return xm, ym, weight


def pick_scenes(ids, count, policy):
    if not 2 <= count <= len(ids):
        raise ValueError("Requested sample count is outside the available split")
    if policy == "even":
        picked = evenly_spaced(ids, count)
    elif policy == "first":
        picked = ids[:count]
    else:
        raise ValueError(f"Unsupported sampling policy: {policy}")
    result = [str(x) for x in picked]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate selected IDs")
    return result


def gather_transform(model, manifest, train_ids, device):
    groups = {key: RunningMoments() for key in ("base", "fused", "optical", "sar")}
    for index, scene in enumerate(train_ids):
        features = encode_scene(model, manifest, scene, "train_stats", device)
        for key, stats in groups.items():
            stats.update(features[key])
        if (index + 1) % 16 == 0:
            print(f"stats pass {index+1}/{len(train_ids)}", flush=True)
    transforms = {key + "_stats": item.get_stats() for key, item in groups.items()}
    covariance = CrossMoments()
    for index, scene in enumerate(train_ids):
        features = encode_scene(model, manifest, scene, "train_pca", device)
        standardized = torch.cat((
            standardize(features["optical"], transforms["optical_stats"]),
            standardize(features["sar"], transforms["sar_stats"])), dim=1)
        covariance.update(standardized)
        if (index + 1) % 16 == 0:
            print(f"pca pass {index+1}/{len(train_ids)}", flush=True)
    transforms["pca"] = covariance.pca(components=64)
    transforms["random"] = random_basis(128, 64, seed=SEED)
    return transforms


def fit_stream_probes(model, manifest, train_ids, device, transforms, ridge):
    statistics = {name: CrossMoments() for name in PROBE_NAMES}
    for index, scene in enumerate(train_ids):
        features = encode_scene(model, manifest, scene, "train_ridge", device)
        x = build_inputs(features["base"], features["fused"],
                         features["optical"], features["sar"], transforms)
        y = matrix_targets(features, transforms)
        for name, accumulator in statistics.items():
            accumulator.update(x[name], y)
        if (index + 1) % 16 == 0:
            print(f"ridge pass {index+1}/{len(train_ids)}", flush=True)
    return {name: accumulator.ridge(ridge=ridge) for name, accumulator in statistics.items()}


def comparative_summary(errors):
    comparisons = {}
    for source in SOURCE_NAMES:
        by = {}
        for row in errors:
            if row["target"] == source:
                by.setdefault(row["scene"], {})[row["method"]] = row["nmse"]
        pca_wins = 0
        e2_wins = 0
        ratios = []
        for scene, values in by.items():
            if len(values) != len(PROBE_NAMES):
                raise ValueError(f"Missing method for {scene}")
            pca_wins += values["pca64"] < values["e2_fused"]
            e2_wins += values["e2_fused"] < values["pca64"]
            ratios.append(values["e2_fused"] / max(values["pca64"], EPS))
        sorted_e2 = sorted(values["e2_fused"] for values in by.values())
        trimmed = sorted_e2[:-max(1, len(sorted_e2)//10)] if len(sorted_e2)>=10 else sorted_e2
        comparisons[source] = {
            "pca64_better_scene_count": pca_wins,
            "e2_fused_better_scene_count": e2_wins,
            "mean_e2_to_pca_nmse_ratio": sum(ratios)/len(ratios),
            "median_e2_to_pca_nmse_ratio": float(median(ratios)),
            "e2_trimmed_mean_nmse_remove_largest_10pct": sum(trimmed)/len(trimmed),
            "evaluated_scene_count": len(by),
        }
    return comparisons


def run(args):
    os.chdir(ROOT)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    if not 0 < args.ridge < 1:
        raise ValueError("--ridge must be in (0,1)")
    manifest = source_manifest()
    train_ids = pick_scenes(manifest["splits"]["train"], args.train_scenes, args.train_selection)
    val_ids = pick_scenes(manifest["splits"]["val"], args.val_scenes, "even")
    if set(train_ids).intersection(val_ids):
        raise ValueError("Train and validation scene overlap")
    output = args.output_dir or DEFAULT_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = frozen_checkpoint(device)
    transforms = gather_transform(model, manifest, train_ids, device)
    probes = fit_stream_probes(model, manifest, train_ids, device, transforms, args.ridge)
    errors, edge_errors = [], []
    for index, scene in enumerate(val_ids):
        features = encode_scene(model, manifest, scene, "val", device)
        rows, edge_rows = evaluate([features], transforms, probes)
        errors.extend(rows)
        edge_errors.extend(edge_rows)
        if (index + 1) % 16 == 0:
            print(f"validation {index+1}/{len(val_ids)}", flush=True)
    report = {
        "status": "frozen_e2_streamed_probes_no_ricf_training",
        "dataset": "QXS-SAROPT", "split": "matched_pilot_train1024_val64",
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "script_sha256": sha256_file(Path(__file__)),
        "device": str(device), "seed": SEED,
        "train_scene_count": len(train_ids), "val_scene_count": len(val_ids),
        "train_selection": args.train_selection,
        "train_ids": train_ids, "val_ids": val_ids,
        "ridge": args.ridge, "feature_stride": FEATURE_STRIDE,
        "conditions": ["base_only", "e2_fused", "pca64", "random64", "concat128"],
        "summary": summarize(errors, transforms),
        "paired_comparison": comparative_summary(errors),
        "caveats": [
            "Source Detail linear recoverability is not fused image quality.",
            "PCA is optimized for input variance; E2 was trained for image fusion.",
            "Strong SAR gradients may reflect speckle; residuals are not structure truth.",
            "An invertible 128-to-128 coupling does not ensure final 128-to-64 compression is lossless.",
            "No RICF model or optimizer step was used; test1000 was not accessed."
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ricf-expanded-", dir=output.parent) as temp:
        pending = Path(temp)/"result"
        pending.mkdir()
        write_csv(pending/"recoverability.csv", errors)
        write_csv(pending/"edge_residual_proxy.csv", edge_errors)
        (pending/"summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        pending.rename(output)
    print(json.dumps({
        "summary": report["summary"],
        "paired_comparison": report["paired_comparison"]
    }, indent=2, ensure_ascii=False), flush=True)
    print(f"saved {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=128)
    parser.add_argument("--val-scenes", type=int, default=64)
    parser.add_argument("--train-selection", choices=("even","first"), default="even")
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
