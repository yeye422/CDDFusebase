"""Compare a two-stage E2 checkpoint with the saved reference E2 val1000."""

import csv
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from run_e2_aspr_pilot import METRICS, summarize, write_csv
from run_e2_matched_no_pac import register_many
from run_e2_rccr import EXPERIMENT as RCCR
from run_e2_scratch_twostage import BASE_ANCHOR, SOURCE_INTERVAL
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from test_OS import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


INTERVAL = replace(SOURCE_INTERVAL, name="e2_scratch_source_interval_v2",
                   experiment_id="E2-SCRATCH-SOURCE-INTERVAL-V2")
GROUP = "QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI"


def read_metric_rows(path, scene_ids):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["image", *METRICS]:
            raise ValueError(f"Unexpected metric columns: {path}")
        rows = [row for row in reader if row["image"] not in {"mean", "std"}]
    if [row["image"] for row in rows] != scene_ids:
        raise ValueError(f"Unexpected validation scene order: {path}")
    return [(tuple(float(row[name]) for name in METRICS), (), ()) for row in rows]


def write_nabf_strata(path, scene_ids, baseline_rows, candidate_rows):
    index = METRICS.index("Nabf")
    baseline, candidate = (np.asarray([row[0][index] for row in rows])
                           for rows in (baseline_rows, candidate_rows))
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("NAB/F must be finite to form the baseline strata")
    ranked = sorted(range(len(scene_ids)), key=lambda i: (baseline[i], scene_ids[i]))
    half = len(ranked) // 2
    rng = np.random.default_rng(42)
    records = []
    for group, indices in (("low_half", ranked[:half]), ("high_half", ranked[half:])):
        delta = candidate[indices] - baseline[indices]
        bootstrap = delta[rng.integers(delta.size, size=(2000, delta.size))].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap, (.025, .975))
        records.append((group, delta.size, baseline[indices].mean(), candidate[indices].mean(),
                        delta.mean(), ci_low, ci_high, (delta < 0).sum(), (delta > 0).sum()))
    write_csv(path, ("group", "common_count", "e2_nabf", "candidate_nabf", "delta_nabf",
                     "ci_low", "ci_high", "better_count", "worse_count"), records)


def validate_against_e2(experiment, group, candidate_key, candidate_label, *,
                        report_nabf_strata=False, sar_detail_ablation=False):
    output = experiment.result
    archived = RCCR.result / "e2" / "metrics.csv"
    archived_meta = RCCR.result / "run_metadata.json"
    archive = read_json(archived_meta)
    modes = {candidate_key: None}
    methods = {"e2": "E2 光学 Base（已有）", candidate_key: candidate_label}
    contrasts = [("e2", candidate_key)]
    if sar_detail_ablation:
        modes = {candidate_key: "spatial", "mean": "mean", "off": "off"}
        methods.update(mean="同权重，SAR 子带均匀收缩", off="同权重，关闭 SAR 子带收缩")
        contrasts.extend((key, candidate_key) for key in ("mean", "off"))
    checkpoints = {"e2": BASE_ANCHOR.final, **dict.fromkeys(modes, experiment.final)}
    checkpoint_hashes = {key: sha256_file(path) for key, path in checkpoints.items()}
    manifest_hash = sha256_file(experiment.manifest)
    scene_ids = read_json(experiment.manifest)["splits"]["val"]
    metadata = dict(split="val", offset=0, scene_ids=scene_ids, metric_profile=DEFAULT_PROFILE,
                    split_manifest=experiment.manifest.as_posix(), split_manifest_sha256=manifest_hash,
                    output_policy="clamp_then_uint8", checkpoint_hashes=checkpoint_hashes,
                    checkpoints={key: path.as_posix() for key, path in checkpoints.items()},
                    archived_metrics=archived.as_posix(), archived_metrics_sha256=sha256_file(archived),
                    archived_metadata=archived_meta.as_posix(), archived_metadata_sha256=sha256_file(archived_meta))
    if report_nabf_strata:
        metadata["nabf_baseline_strata"] = "Two equal halves sorted by E2 NAB/F, then scene ID"
    if sar_detail_ablation:
        metadata["sar_detail_support_modes"] = modes
    if experiment.uses_polarity:
        metadata["polarity_mode"] = "aligned"
    common_keys = ("split", "offset", "scene_ids", "metric_profile", "split_manifest_sha256", "output_policy")
    if (len(scene_ids) != experiment.val_count or len(set(scene_ids)) != experiment.val_count
        or any(archive.get(key) != metadata[key] for key in common_keys)
        or archive["checkpoint_hashes"]["e2"] != checkpoint_hashes["e2"]):
        raise ValueError("The saved E2 results differ from the fixed val1000 protocol")
    baseline_rows = read_metric_rows(archived, scene_ids)
    if output.exists():
        if read_json(output / "run_metadata.json") != metadata:
            raise FileExistsError(f"Preserve validation with a different protocol: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".e2-val-", dir=output.parent) as temporary:
            work = Path(temporary) / "results"
            (work / "e2").mkdir(parents=True)
            results = {"e2": baseline_rows}
            for key, mode in modes.items():
                candidate = work / key
                flags = ("--sar-detail-support", mode) if mode is not None else ()
                subprocess.run([sys.executable, "-u", experiment.evaluator, "--data-root", str(DATA),
                                "--split-manifest", str(experiment.manifest), "--split", "val",
                                "--limit", str(experiment.val_count), "--variant", experiment.variant,
                                "--checkpoint", str(experiment.final), "--output-dir", str(candidate),
                                "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp",
                                "--timing-warmup", "0", *flags], check=True)
                expected = dict(split="val", offset=0, selected_scene_ids=scene_ids, metric_profile=DEFAULT_PROFILE,
                                split_manifest_sha256=manifest_hash, output_range_policy="clamp",
                                checkpoint_sha256=checkpoint_hashes[key], variant=experiment.variant,
                                evaluated_pairs=experiment.val_count, metrics_only=False)
                if mode is not None:
                    expected["sar_detail_support"] = mode
                if experiment.uses_polarity:
                    expected["polarity_mode"] = metadata["polarity_mode"]
                candidate_meta = read_json(candidate / "run_metadata.json")
                if any(candidate_meta.get(field) != value for field, value in expected.items()):
                    raise ValueError(f"Unexpected validation metadata: {experiment.name} ({key})")
                for kind in ("gray", "rgb", "preview"):
                    verify_images(candidate / kind, scene_ids)
                raw_metrics = (candidate / "metrics.csv").rename(candidate / "evaluator_metrics.csv")
                results[key] = read_metric_rows(raw_metrics, scene_ids)
            summaries = summarize(work, scene_ids, results, contrasts,
                                  diagnostic_fields=(), source_metrics=())
            if report_nabf_strata:
                write_nabf_strata(work / "nabf_by_baseline_half.csv", scene_ids, baseline_rows, results[candidate_key])
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "registry_metrics.json", summaries)
            work.rename(output)

    entries = []
    for key, metrics in summaries.items():
        folder = output / key
        entries.append(dict(
            id=f"{group}-{key}", method=methods[key], dataset="QXS-SAROPT", task="fusion",
            status="development_validation", comparable_group=group, split="val", offset=0,
            sample_count=len(scene_ids), split_manifest=metadata["split_manifest"],
            split_manifest_sha256=manifest_hash, checkpoint=checkpoints[key].as_posix(),
            checkpoint_sha256=checkpoint_hashes[key], result_dir=folder.as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(), metric_profile=DEFAULT_PROFILE,
            output_policy=metadata["output_policy"], metrics=metrics,
            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in metrics}),
        ))
    register_many(entries)
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)


def validate():
    validate_against_e2(INTERVAL, GROUP, "interval", "E2 + 源特异梯度区间 v2")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    validate()
