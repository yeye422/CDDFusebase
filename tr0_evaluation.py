"""TR0 diagnostics, fixed proxy validation and paired full-image evaluation."""

import csv
import math
from pathlib import Path
import shutil
from statistics import fmean, pstdev
import sys
from time import perf_counter
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from compare_lwpd_prl import METRICS, read_metrics, screening
from e2_tr0 import E2ComplementaryLearning, E2ProgressiveJointDecoding
from run_e2_matched_no_pac import register_many
from run_e2_pac_mamba import write_json
from run_tr0_factorial import (
    MAIN_GROUPS, build_model, checkpoint_path, forward_losses, precision_context, read_json,
)
from train_direct_fusion import DirectFusionLosses, sha256_file
from tr0_complementary import ComplementaryReconstructionLoss, ComplementaryViewGenerator
from utils.dataset_os import OpticalSARDataset, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


POLICY = "clamp_then_uint8"


def provenance(checkpoint, manifest):
    return dict(checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                split_manifest=manifest.as_posix(), split_manifest_sha256=sha256_file(manifest))


def csv_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def register_trainings(config, run_id, completed, manifest):
    entries = []
    for arm, checkpoint in completed.items():
        history = checkpoint.parent / "history.csv"
        with history.open(newline="", encoding="utf-8") as handle:
            final = list(csv.DictReader(handle))[-1]
        fields = ("train_loss", "val_loss", "train_optical_reconstruction", "train_sar_reconstruction")
        entry_id = f"QXS-TR0-V1-{run_id}-{arm}-TRAIN"
        entries.append(dict(
            provenance(checkpoint, manifest), id=entry_id, comparable_group=entry_id,
            dataset="QXS-SAROPT", split="train", offset=0, sample_count=config["train_count"],
            task="training", status="training_complete", method=f"TR0 {arm}",
            result_dir=checkpoint.parent.as_posix(), metadata_path=(checkpoint.parent / "config.json").as_posix(),
            metric_profile="training_loss", output_policy="float_loss",
            metrics={field: float(final[field]) for field in fields},
            metric_source=dict(type="csv_row", path=history.as_posix(), match={"epoch": final["epoch"]},
                               columns={field: field for field in fields}),
            notes="从头训练。用于机制比较的更新步数、两源重建误差、梯度和耗时见 history.csv。",
        ))
    register_many(entries)


def verify_checkpoint(path, config, arm, *, proxy_end=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    end = config["reconstruction_epochs"] + config["proxy_epochs"]
    expected = end if proxy_end else end + config["fusion_epochs"]
    metadata = path.parent / "config.json"
    recorded = read_json(metadata)
    if (state["arm"] != arm or state["epoch"] != expected or state["stage"] != ("II" if proxy_end else "III")
            or state["config_sha256"] != sha256_file(metadata) or recorded["config"] != config
            or state["manifest_sha256"] != recorded["manifest_sha256"]
            or state["manifest_sha256"] != sha256_file(path.parents[1] / "screen_manifest.json")):
        raise ValueError(f"Unexpected checkpoint or training settings: {path}")
    return state


def reuse_baseline(config, details, ids, folder):
    archive = Path(config["baseline_screen_result"])
    archive_metrics = archive / "e2" / "metrics.csv"
    archive_metadata = archive / "run_metadata.json"
    if not archive_metrics.is_file() or not archive_metadata.is_file():
        return False
    meta = read_json(archive_metadata)
    if (meta["checkpoint_hashes"]["e2"] != details["checkpoint_sha256"]
            or meta["split_manifest_sha256"] != details["split_manifest_sha256"]
            or meta["metric_profile"] != DEFAULT_PROFILE or meta["output_policy"] != POLICY
            or meta["scene_ids"] != ids or list(read_metrics(archive_metrics)[0]) != ids):
        return False
    folder.mkdir(parents=True)
    shutil.copyfile(archive_metrics, folder / "metrics.csv")
    write_json(folder / "run_metadata.json", dict(
        details, selected_scene_ids=ids, split="val", offset=0, metric_profile=DEFAULT_PROFILE,
        output_range_policy="clamp", reuse_metrics=archive_metrics.as_posix(),
        reuse_metrics_sha256=sha256_file(archive_metrics), reuse_metadata=archive_metadata.as_posix(),
        reuse_metadata_sha256=sha256_file(archive_metadata),
    ))
    return True


def evaluate_arm(config, arm, path, manifest, split, device, folder):
    ids = read_json(manifest)["splits"]["val"]
    details = provenance(path, manifest)
    metadata = folder / "run_metadata.json"
    metrics = folder / "metrics.csv"
    if not folder.exists() and not (arm == "B0" and split == "screen" and reuse_baseline(config, details, ids, folder)):
        import test_OS

        types = {
            E2ComplementaryLearning.VARIANT: E2ComplementaryLearning,
            E2ProgressiveJointDecoding.VARIANT: E2ProgressiveJointDecoding,
            E2ProgressiveJointDecoding.EARLY_VARIANT: lambda: E2ProgressiveJointDecoding(early=True),
        }
        argv = ["test_OS.py", "--data-root", config["data_root"], "--split-manifest", str(manifest),
                "--split", "val", "--checkpoint", str(path), "--output-dir", str(folder),
                "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
        with patch.dict(test_OS.DIRECT_MODEL_TYPES, types), patch.object(sys, "argv", argv):
            test_OS.main(device=str(device))
    if not metadata.is_file() or not metrics.is_file():
        raise FileExistsError(f"Incomplete evaluation is preserved at {folder}. Use a new run ID.")
    meta = read_json(metadata)
    expected = {key: details[key] for key in ("checkpoint_sha256", "split_manifest_sha256")}
    expected.update(selected_scene_ids=ids, split="val", offset=0, metric_profile=DEFAULT_PROFILE,
                    output_range_policy="clamp")
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Evaluation provenance differs: {folder}")
    rows, _ = read_metrics(metrics)
    if list(rows) != ids:
        raise ValueError(f"Evaluation rows differ from the fixed split order: {folder}")
    return rows, details


def paired_summary(results):
    reference_ids = list(next(iter(results.values())))
    if any(list(rows) != reference_ids for rows in results.values()):
        raise ValueError("All methods must use the same ordered scenes")
    means = {arm: {} for arm in results}
    support, effects, comparisons = [], [], []
    contrasts = [("B0", method) for method in results if "B0" in results and method != "B0"]
    contrasts.extend((base, candidate) for base, candidate in (
        ("C00", "C01"), ("C10", "C11"), ("C00", "C10"), ("C01", "C11"),
        ("R8", "C00"), ("R8", "C01"), ("EARLY00", "C10"), ("EARLY01", "C11"),
    ) if base in results and candidate in results)
    for metric in METRICS:
        vectors = {arm: {scene: float(row[metric]) for scene, row in rows.items()} for arm, rows in results.items()}
        ids = [scene for scene in reference_ids if all(math.isfinite(vector[scene]) for vector in vectors.values())]
        common = set(ids)
        support.append(dict(metric=metric, n_common=len(ids), excluded_scene_ids=" ".join(
            scene for scene in reference_ids if scene not in common)))
        if not ids:
            continue
        sign = -1 if metric == "Nabf" else 1
        for arm, vector in vectors.items():
            means[arm][metric] = fmean(vector[scene] for scene in ids)
        for base, candidate in contrasts:
            before, after = means[base][metric], means[candidate][metric]
            delta = sign * (after - before)
            comparisons.append(dict(metric=metric, control=base, arm=candidate, n_common=len(ids),
                                    baseline=before, candidate=after, beneficial_change=delta,
                                    baseline_std=pstdev(vectors[base][scene] for scene in ids),
                                    candidate_std=pstdev(vectors[candidate][scene] for scene in ids),
                                    beneficial_relative_percent=100 * delta / abs(before) if before else None))
        if all(arm in vectors for arm in MAIN_GROUPS):
            a, b, c, d = (means[arm][metric] for arm in MAIN_GROUPS)
            effects.append(dict(metric=metric, n_common=len(ids), direction="up" if sign > 0 else "down",
                                delta_t=0.5 * ((b - a) + (d - c)), delta_a=0.5 * ((c - a) + (d - b)),
                                delta_ta=d - c - b + a))
            effects[-1].update({f"beneficial_{key}": sign * effects[-1][key]
                                for key in ("delta_t", "delta_a", "delta_ta")})
    return means, support, effects, comparisons


def evaluate_arms(config, root, output, arms, manifest, split, device, run_id):
    arms = tuple(sorted(arms))
    output = output / split
    results, sources, steps = {}, {}, {}
    for arm in arms:
        path = Path(config["baseline_checkpoint"]) if arm == "B0" else checkpoint_path(config, root, arm)
        if arm != "B0":
            state = verify_checkpoint(path, config, arm)
            steps[arm] = [row["train_optimizer_steps"] for row in state["history"]]
            del state
        results[arm], sources[arm] = evaluate_arm(config, arm, path, manifest, split, device, output / arm)
    if len({tuple(value) for value in steps.values()}) > 1:
        raise ValueError("Matched arms used different numbers of optimizer steps")
    arm_key = "-".join(sorted(arms))
    comparison = output / f"comparison-{arm_key}"
    means, support, effects, comparisons = paired_summary(results)
    checks = {}
    for arm in arms:
        records = [row for row in comparisons if row["control"] == "B0" and row["arm"] == arm]
        if all(any(row["metric"] == metric and row["beneficial_relative_percent"] is not None for row in records)
               for metric in ("MI", "FMI", "Qabf", "Nabf", "VIF", "SSIM", "MS_SSIM")):
            checks[arm] = screening(records)
    payload = dict(arms=sorted(arms), metrics=means, support=support, effects=effects,
                   comparisons=comparisons, checkpoints=sources, optimizer_steps=steps,
                   screening=checks, metric_profile=DEFAULT_PROFILE, output_policy=POLICY,
                   raw_metrics={arm: (output / arm / "metrics.csv").as_posix() for arm in arms})
    summary = comparison / "summary.json"
    if comparison.exists():
        if read_json(summary) != payload:
            raise FileExistsError(f"Existing comparison differs: {comparison}")
    else:
        comparison.mkdir(parents=True)
        for name, rows in (("common_scenes", support), ("factorial_effects", effects), ("paired_comparisons", comparisons)):
            if rows:
                csv_rows(comparison / f"{name}.csv", rows)
        write_json(summary, payload)
    group = f"QXS-TR0-V1-{run_id}-{split.upper()}-{arm_key}-REF-FMI"
    entries = [dict(
        sources[arm], id=f"{group}-{arm}", comparable_group=group,
        dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(results[arm]),
        task="fusion", status="development_validation", method=f"TR0 {arm}",
        result_dir=(output / arm).as_posix(), metadata_path=(output / arm / "run_metadata.json").as_posix(),
        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=means[arm],
        metric_source=dict(type="json_paths", path=summary.as_posix(),
                           paths={metric: ["metrics", arm, metric] for metric in means[arm]}),
        notes="使用同组各方法逐项共同有限的全图场景。原始逐图指标和共同样本数见比较目录。",
    ) for arm in arms]
    register_many(entries)
    print(f"Comparison: {summary}")


@torch.inference_mode()
def proxy_check(config, root, output, arms, manifest, device, run_id):
    splits, _ = manifest_pairs(Path(config["data_root"]), manifest)
    dataset = OpticalSARDataset(splits["val"], patch_size=config["patch_size"], augment=False)
    loader = DataLoader(dataset, batch_size=config["batch_size"])
    views = ComplementaryViewGenerator(**config["view_generator"])
    loss = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
    entries = []
    for arm in arms:
        plan = config["experiments"][arm]
        if not plan["proxy"]:
            raise ValueError(f"{arm} does not have a proxy training stage")
        path = checkpoint_path(config, root, arm, proxy_end=True)
        folder = output / "proxy" / arm
        metadata = folder / "run_metadata.json"
        metrics_path = folder / "metrics.csv"
        details = dict(provenance(path, manifest), precision="fp32",
                       proxy_seed=config["proxy_validation_seed"], scenes=len(dataset))
        if folder.exists():
            meta = read_json(metadata)
            if (any(meta.get(key) != value for key, value in details.items())
                    or meta["raw_metrics_sha256"] != sha256_file(metrics_path)):
                raise ValueError(f"Existing proxy diagnostics differ: {folder}")
            summary = meta["means"]
        else:
            saved = verify_checkpoint(path, config, arm, proxy_end=True)
            model = build_model(plan["architecture"]).to(device).eval()
            model.load_state_dict(saved["model"])
            del saved
            folder.mkdir(parents=True)
            summary = measure_proxy(config, device, model, loader, views, loss, metrics_path)
            write_json(metadata, dict(details, means=summary, raw_metrics_sha256=sha256_file(metrics_path)))
        group = f"QXS-TR0-V1-{run_id}-PROXY-{arm}"
        metrics = {f"{label}_{key}": value for label, numbers in summary.items() for key, value in numbers.items()}
        entries.append(dict(
            provenance(path, manifest), id=group, comparable_group=group, dataset="QXS-SAROPT",
            split="val", offset=0, sample_count=len(dataset), task="reconstruction", status="diagnostic_fixed_subset",
            method=f"TR0 {arm} 代理输入对照", result_dir=folder.as_posix(),
            metadata_path=(folder / "run_metadata.json").as_posix(), metric_profile="tr0_proxy_v1",
            output_policy="float_loss", metrics=metrics, metric_source=dict(type="json_paths",
                path=(folder / "run_metadata.json").as_posix(), paths={f"{label}_{key}": ["means", label, key]
                for label, numbers in summary.items() for key in numbers}),
            notes="固定 val64 同源光学代理诊断，使用第 8 轮权重。",
        ))
    register_many(entries)


def measure_proxy(config, device, model, loader, views, loss, metrics_path):
    generator = torch.Generator().manual_seed(config["proxy_validation_seed"])
    rows = []
    for optical, _, _, scenes in loader:
        optical = optical.to(device)
        first, second, _, _ = views(optical, generator)
        for label, pair in (("complementary", (first, second)), ("repeat_first", (first, first)),
                            ("repeat_second", (second, second))):
            prediction = model(*pair)
            for index, scene in enumerate(scenes):
                estimate, target = prediction[index:index + 1], optical[index:index + 1]
                total, pixels, gradient = loss(estimate, target)
                mse = float((estimate.float() - target).square().mean())
                rows.append(dict(image=scene, input=label, loss=float(total), mae=float(pixels),
                                 gradient_mae=float(gradient), psnr=-10 * math.log10(max(mse, 1e-12))))
    csv_rows(metrics_path, rows)
    return {label: {key: fmean(row[key] for row in rows if row["input"] == label)
                    for key in ("loss", "mae", "gradient_mae", "psnr")}
            for label in ("complementary", "repeat_first", "repeat_second")}


def diagnose(config, device):
    """Single synthetic patch, without any optimizer update."""
    from run_tr0_factorial import set_seed

    set_seed(config["seed"])
    size = config["patch_size"]
    optical = torch.rand(1, 1, size, size, device=device)
    sar = torch.rand_like(optical)
    losses = DirectFusionLosses().to(device)
    proxy_loss = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
    views = ComplementaryViewGenerator(**config["view_generator"])
    first, second, _, stats = views(optical, torch.Generator().manual_seed(config["proxy_seed"]))
    if not (float(stats["clear_first_only"]) >= 0.2 and float(stats["clear_second_only"]) >= 0.2
            and float(stats["clear_union"]) >= 0.85 and all(float(stats[k]) > 0 for k in (
                "degradation_first", "degradation_second"))):
        raise AssertionError("Generated proxy views do not meet TR0 constraints")
    print({name: float(value) for name, value in stats.items()}, flush=True)
    for architecture in ("e2", "progressive", "early"):
        set_seed(config["seed"])
        model = build_model(architecture).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with precision_context(config, device):
            total, _ = forward_losses(model, losses, proxy_loss, config, "II", optical, sar,
                                      torch.ones_like(sar), proxy_pair=(first, second),
                                      checkpointing=config["activation_checkpointing"])
        total.backward()
        if not torch.isfinite(total) or not any(p.grad is not None and torch.count_nonzero(p.grad)
                                             for p in model.base_fusion.parameters()):
            raise AssertionError("Missing finite Base Fusion gradient")
        if isinstance(model, E2ProgressiveJointDecoding):
            for unit in model.decoder.synthesis_stages:
                for branch in (unit.optical_message, unit.sar_message):
                    if not torch.count_nonzero(branch[0].weight.grad):
                        raise AssertionError("Missing gradient to PJD source branch")
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            odd = model(optical[..., :31, :33], sar[..., :31, :33])
            if odd.shape[-2:] != (31, 33) or not torch.isfinite(odd).all():
                raise AssertionError("Odd image prediction failed")
            if isinstance(model, E2ProgressiveJointDecoding):
                encoded = model.encode_pair(optical[..., :31, :33], sar[..., :31, :33])
                for active, base, detail in (((True, False), *encoded[:2]), ((False, True), *encoded[2:])):
                    inactive = 1 if active[0] else 0
                    def unexpected_source(*_):
                        raise AssertionError("Inactive synthesis branch received a forward call")

                    branches = (unit.optical_message if inactive == 0 else unit.sar_message
                                for unit in model.decoder.synthesis_stages)
                    hooks = [branch.register_forward_hook(unexpected_source) for branch in branches]
                    try:
                        reconstruction = model.decoder(base, detail if active[0] else None,
                                                       detail if active[1] else None, active)
                        if reconstruction.shape != odd.shape or not torch.isfinite(reconstruction).all():
                            raise AssertionError("Single-source decoding failed")
                    finally:
                        for hook in hooks:
                            hook.remove()
            model(optical, sar)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            model(optical, sar)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms = 1000 * (perf_counter() - started)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
                model(optical, sar)
        print(dict(architecture=architecture, parameters=sum(p.numel() for p in model.parameters()),
                   counted_inference_gflops=sum(event.flops for event in profile.key_averages()) / 1e9,
                   stage_ii_peak_mib=peak, single_inference_ms=inference_ms), flush=True)
        del model, total
