"""冻结已有 SRR，在同一 Detail 路由入口比较 SAR 原位、平移和均值先验。"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_drf_no_train import background_mask
from diagnose_e2_information_path import image_u8, read_pair, to_tensor
from diagnose_rct_predictive_transfer import (
    DATA, LABEL_DIRS, METRICS, ROOT, digest, line_readout, prepare_lines,
    read_json, write_csv, write_json,
)
from maintain_experiment_index import sha256_file
from rct_predictive_transfer import paired_stats
from run_e2_matched_no_pac import register_many
from test_OS import forward_variant, load_variant_models
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ARCHIVE = Path("test_result/reliability_v5/srr_detail_v6_ranking_e010_validation/run_metadata.json")
CHECKPOINT = Path("models/reliability_v5/srr_detail_v6_ranking_pilot/epoch_010.pth")
MANIFEST = Path("../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json")
METHODS = ("aligned", "shifted", "constant")
LOCAL = ("sar_continuity", "false_path_coverage", "flat_hp_rms")


def controls(prior, valid):
    # Half-period translation is its own inverse on this even-sized grid.
    h, w = prior.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError("Expected an even-sized image")

    def translate(x):
        return x.roll((h // 2, w // 2), dims=(-2, -1))

    support = valid & translate(valid)
    if not support.any():
        raise ValueError("No shared valid SAR support")
    shifted = torch.where(support, translate(prior), prior)
    constant = torch.where(support, prior[support].mean(), prior)
    torch.testing.assert_close(prior[support].sort().values, shifted[support].sort().values, atol=0, rtol=0)
    torch.testing.assert_close(constant.mean(), prior.mean(), atol=1e-6, rtol=0)
    return (prior, shifted, constant), support


@torch.inference_mode()
def probe(models, scene, files, labels, device, out):
    optical, sar = read_pair(files)
    o, s = [to_tensor(x, device) for x in (optical, sar)]
    valid = s > 0
    normal, _ = forward_variant(models, "srr_detail", o, s)
    bo, do, _ = models["encoder"](o)
    bs, ds, _ = models["encoder"](s)
    base = models["base_fuse"](bo + bs)
    prior = models["line_evidence"](s, valid.float())["reliability"]
    priors, support = controls(prior, valid)
    flat = background_mask(optical, sar, labels)
    lines, _ = prepare_lines(labels, scene, optical, sar)
    rows, probes = [], []
    for method, condition in zip(METHODS, priors):
        detail, _ = models["detail_router"](do, ds, condition, valid=valid.float())
        fused = models["decoder"](o, base, detail)[0]
        if method == "aligned":
            torch.testing.assert_close(fused, normal, rtol=1e-5, atol=1e-6)
        gray = image_u8(fused)
        if out:
            directory = out / method
            directory.mkdir(exist_ok=True)
            if not cv2.imwrite(str(directory / f"{scene}.png"), gray):
                raise OSError("Cannot save fusion image")
        # Use exactly the displayed, quantized output for all quality readouts.
        samples = line_readout(to_tensor(gray, "cpu"), lines, method)
        probes.extend(samples)
        sar_lines = [p["continuity"] for p in samples if p["cohort"].startswith("sar_")]
        false_lines = [p["path_coverage"] for p in samples if p["cohort"] == "disconnected"]
        g = gray.astype(np.float32)
        high = g - cv2.GaussianBlur(g, (0, 0), 1.25)
        metrics = dict(zip(METRICS, map(float, FusionEvaluator.metric_vector(gray, optical, sar))))
        rows.append(dict(scene=scene, method=method, **metrics,
                         sar_continuity=np.mean(sar_lines) if sar_lines else np.nan,
                         false_path_coverage=np.mean(false_lines) if false_lines else np.nan,
                         flat_hp_rms=np.sqrt(np.mean(high[flat]**2)) if flat.any() else np.nan,
                         sar_line_count=len(sar_lines), flat_pixels=int(flat.sum()),
                         common_fraction=float(support.float().mean()),
                         prior_mae=float((condition-prior).abs().mean()),
                         output_mae_u8=float(np.abs(g-image_u8(normal)).mean())))
    return rows, probes


def compare(rows):
    by = {m: {r["scene"]: r for r in rows if r["method"] == m} for m in METHODS}
    results = []
    for metric in (*LOCAL, *METRICS):
        scenes = [s for s in by["aligned"] if all(np.isfinite(by[m][s][metric]) for m in METHODS)]
        for control in METHODS[1:]:
            stats = paired_stats([by["aligned"][s][metric] for s in scenes],
                                 [by[control][s][metric] for s in scenes],
                                 lower_is_better=metric in ("Nabf", "flat_hp_rms", "false_path_coverage"))
            results.append(dict(metric=metric, control=control, scenes=" ".join(scenes), **stats))
    return results


def position_decision(comparisons):
    """只在两种对照都支持边线连续性增益时报告位置收益。防护指标采用配对均值。"""
    checks = {}
    for control in METHODS[1:]:
        values = {r["metric"]: r for r in comparisons if r["control"] == control}
        edge = values["sar_continuity"]
        checks[control] = dict(
            edge_gain=edge["n"] >= 2 and edge["ci_low"] > 0,
            **{key: values[key]["n"] >= 2 and values[key]["favorable_gain"] >= 0
               for key in ("flat_hp_rms", "false_path_coverage", "Nabf")})
    supported = all(all(check.values()) for check in checks.values())
    return dict(status="本组支持位置收益" if supported else "本组未建立联合收益",
                checks=checks, scope="冻结 SRR，预选验证场景，均匀区和伪连接为代理指标")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "val16"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_result/sar_prior_position_v1/val16"))
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    archive = read_json(ARCHIVE)
    if (sha256_file(CHECKPOINT) != archive["checkpoint_sha256"] or
            sha256_file(MANIFEST) != archive["split_manifest_sha256"]):
        raise ValueError("Unexpected checkpoint or split")
    manifest = read_json(MANIFEST)
    scenes, labels, annotation_hashes = [], [], {}
    for folder in LABEL_DIRS:
        sample = read_json(folder / "sample.json")
        scenes.extend(sample["scene_ids"])
        for name in ("sample.json", "source_labels.csv"):
            annotation_hashes[str((folder / name).relative_to(ROOT))] = sha256_file(folder / name)
        with (folder / "source_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
            labels.extend(dict(row, visibility="all") for row in csv.DictReader(handle))
    if len(scenes) != 16 or len(set(scenes)) != 16 or not set(scenes) <= set(manifest["splits"]["val"]):
        raise ValueError("Expected 16 distinct annotated validation scenes")
    out = args.output_dir if args.action == "val16" else None
    if out:
        out.mkdir(exist_ok=False, parents=True)
    else:
        scenes = scenes[:1]
    models, variant = load_variant_models(CHECKPOINT, device)
    if variant != "srr_detail":
        raise ValueError("Unexpected model variant")
    rows, probes, inputs = [], [], {}
    for scene in scenes:
        files = [DATA / manifest[k] / f"{scene}.png" for k in ("optical_dir", "sar_dir")]
        inputs[scene] = [sha256_file(p) for p in files]
        images, segments = probe(models, scene, files, [x for x in labels if x["scene"] == scene], device, out)
        rows.extend(images)
        probes.extend(segments)
        print(scene, flush=True)
    comparisons = compare(rows)
    if not out:
        for row in rows:
            print(row["method"], {k: round(float(row[k]), 6) for k in (*LOCAL, "Qabf", "Nabf", "output_mae_u8")})
        return
    write_csv(out / "per_image.csv", rows)
    write_csv(out / "per_line.csv", probes)
    write_csv(out / "paired.csv", comparisons)
    decision = position_decision(comparisons)
    write_json(out / "decision.json", decision)
    print(decision["status"], flush=True)
    meta = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256_file(CHECKPOINT),
                split_manifest=str(MANIFEST), split_manifest_sha256=sha256_file(MANIFEST),
                scenes=scenes, inputs=inputs, annotation_hashes=annotation_hashes,
                prior_calibration=models["line_evidence"].fse_calibration["sha256"],
                script_sha256=sha256_file(Path(__file__)), device=device, metric_profile=DEFAULT_PROFILE)
    write_json(out / "run_metadata.json", meta)
    group = "SAR-PRIOR-POSITION-" + digest(dict(meta=meta, directory=str(out.resolve())))[:16].upper()
    summary, entries = {}, []
    for method in METHODS:
        field = "candidate_mean" if method == "aligned" else "reference_mean"
        control = "shifted" if method == "aligned" else method
        means = {r["metric"]: r[field] for r in comparisons if r["control"] == control and r["n"]}
        summary[method] = means
        entries.append(dict(id=group+"-"+method, method="Frozen SRR "+method, task="fusion",
                            status="diagnostic_fixed_subset", comparable_group=group, dataset="QXS-SAROPT",
                            split="val", offset=0, sample_count=len(scenes), checkpoint=CHECKPOINT.as_posix(),
                            checkpoint_sha256=meta["checkpoint_sha256"], split_manifest=MANIFEST.as_posix(),
                            split_manifest_sha256=meta["split_manifest_sha256"], metric_profile=DEFAULT_PROFILE,
                            output_policy="clamp_round_uint8", result_dir=(out/method).as_posix(),
                            metadata_path=(out/"run_metadata.json").as_posix(), metrics=means,
                            metric_source=dict(type="json_paths", path=(out/"means.json").as_posix(),
                                               paths={k: [method, k] for k in means})))
    write_json(out / "means.json", summary)
    register_many(entries)


if __name__ == "__main__":
    main()
