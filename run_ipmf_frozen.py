"""Train frozen-CF1 IPMF-V1 and compare it on the fixed QXS val1000."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import torch

import train_lwpd as trainer
from audit_ipmf_preflight import DTYPES, LR, OUTPUT as PREFLIGHT, PARENT, load_parent
from cfdr_evaluation import infer as infer_control
from e2_ipmf import IPMFE2, require_frozen_cf1
from hcfdr_comparison import OUTPUT as CF1_E04_ARCHIVE, reuse_metrics
from run_cfdr_frozen import PARENT as CF0
from run_cfdr_frozen_study import checkpoint as cfdr_checkpoint, comparison_dir
from run_e2_aspr_pilot import METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import config_values
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, verify_images, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file


ROOT = Path("models/e2_ipmf_v1_frozen/seed42")
OUTPUT = Path("test_result/e2_ipmf_v1_frozen/val1000")
GROUP = "QXS-IPMF-V1-FROZEN-SEED42"
CANDIDATE = "IPMF-V1"
METHODS = {"CF0": "光学 Base 锚定 E2", "CF1": "CF1 第 4 轮", CANDIDATE: "冻结 CF1 + IPMF-V1"}
STRUCTURE = ("SSIM", "MS_SSIM")
TARGET_DELTAS = {"SSIM": .003, "MS_SSIM": .003, "MI": -.030, "FMI": -.005,
                 "Qabf": -.002, "VIF": -.003, "Nabf": .001535}


def checkpoint(epoch):
    return ROOT / f"epoch_{epoch:03d}.pth"


def validation_dir(epoch):
    return OUTPUT / f"epoch_{epoch:03d}"


def training_inputs():
    parent, args, protocol = load_parent()
    report = read_json(PREFLIGHT)
    selected = report["selected_precision"]
    if report["protocol"] != protocol or selected not in DTYPES or not report["precision"][selected]["eligible"]:
        raise ValueError(f"IPMF preflight must pass for the current checkpoint, data, and sources: {PREFLIGHT}")
    args.no_amp = selected == "fp32"
    protocol["source_sha256"][Path(__file__).name] = sha256_file(Path(__file__))
    protocol.update(variant=IPMFE2.VARIANT, precision=selected, preflight=PREFLIGHT.as_posix(),
                    preflight_sha256=sha256_file(PREFLIGHT), args=config_values(args), lr=LR,
                    training_policy="ipmf_only_from_cf1_e04")
    return parent, args, protocol


def read_checkpoint(epoch, config_hash, cf1):
    path = checkpoint(epoch)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if ((saved["epoch"], saved["variant"], saved["config_sha256"]) != (epoch, IPMFE2.VARIANT, config_hash)
            or [row["epoch"] for row in saved["history"]] != list(range(1, epoch + 1))):
        raise ValueError(f"Unexpected IPMF checkpoint: {path}")
    require_frozen_cf1(saved["model"], cf1)
    return saved


def frozen_loss(model, losses, optical, sar, valid, args):
    fused, stats = model.fuse_and_reconstruct(optical, sar, model.encode_pair(optical, sar), return_stats=True)
    total, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    return total, dict(fusion=total, intensity=intensity, gradient=gradient, **stats)


def train_epochs(model, optimizer, scaler, loader, device, args, protocol, history, config_hash, target, cf1):
    parameters = {name: list(module.parameters()) for name, module in model.ipmf.named_children()}
    precision = protocol["precision"]
    losses = trainer.DirectFusionLosses().to(device)
    updates = 0
    norms = []

    def record_update(*_):
        nonlocal updates
        updates += 1

    def inspect_gradients(_):
        values = torch.stack([
            torch.stack([p.grad.detach().float().norm() for p in group if p.grad is not None]).norm()
            for group in parameters.values()
        ]).tolist()
        if all(math.isfinite(value) for value in values):
            norms.append(values)
        elif precision != "fp16":
            raise FloatingPointError(f"Nonfinite gradients in {precision} training")
        return {}

    def autocast(*, enabled):
        return torch.autocast("cuda", dtype=DTYPES[precision], enabled=enabled)

    hook = optimizer.register_step_post_hook(record_update)
    try:
        with (patch.object(trainer, "forward_losses", frozen_loss),
              patch.object(trainer, "group_gradient_norms", inspect_gradients),
              patch.object(trainer, "autocast", autocast)):
            for epoch in range(len(history) + 1, target + 1):
                trainer.set_seed(args.seed + epoch)
                loader.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                updates = 0
                norms.clear()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, loader, device, args, optimizer, scaler)
                planned = math.ceil(len(loader) / args.accumulation_steps)
                if not updates or not all(math.isfinite(value) for value in stats.values()):
                    raise FloatingPointError(f"Invalid IPMF training epoch {epoch}")
                state = model.state_dict()
                require_frozen_cf1(state, cf1)
                history.append(dict(
                    epoch=epoch, train_seconds=time.perf_counter() - started,
                    train_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    planned_updates=planned, effective_updates=updates, skipped_updates=planned - updates,
                    frozen_cf1_match=True, **{f"train_{key}": value for key, value in stats.items()},
                    **{f"train_grad_{name}": sum(column) / len(column)
                       for name, column in zip(parameters, zip(*norms))},
                ))
                path = checkpoint(epoch)
                temporary = path.with_suffix(".tmp")
                torch.save(dict(variant=model.variant, args={"variant": model.variant}, epoch=epoch,
                                config_sha256=config_hash, history=history, model=state,
                                optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), temporary)
                temporary.replace(path)
                write_history(ROOT / "history.csv", history)
                print(f"IPMF-V1 epoch {epoch}/{target}: loss={stats['loss']:.6f}; updates={updates}/{planned}", flush=True)
    finally:
        hook.remove()


def train(device, target):
    parent, args, protocol = training_inputs()
    if protocol["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The audited BF16 precision requires a BF16-capable CUDA device")
    existing = sorted(ROOT.glob("epoch_*.pth"))
    last = len(existing)
    if last > 8 or existing != [checkpoint(epoch) for epoch in range(1, last + 1)]:
        raise ValueError(f"Unexpected IPMF checkpoint sequence: {ROOT}")
    if target == 8:
        extension_gate(protocol)
    ROOT.mkdir(parents=True, exist_ok=True)
    config = ROOT / "config.json"
    if not config.exists():
        if existing:
            raise ValueError(f"Missing IPMF run configuration: {config}")
        temporary = config.with_suffix(".tmp")
        write_json(temporary, protocol)
        temporary.replace(config)
    if read_json(config) != protocol:
        raise ValueError(f"Preserve existing training with a different protocol: {config}")
    config_hash = sha256_file(config)
    saved = read_checkpoint(min(last, target), config_hash, parent["model"]) if last else None
    history = saved["history"] if saved else []
    log = ROOT / "history.csv"
    if last < target:
        trainer.set_seed(args.seed)
        model = IPMFE2().to(device)
        model.load_cf1(parent)
        optimizer = torch.optim.AdamW(model.ipmf.parameters(), lr=LR, weight_decay=args.weight_decay)
        scaler = trainer.GradScaler(enabled=protocol["precision"] == "fp16")
        if saved:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
        del saved
        require_frozen_cf1(model.state_dict(), parent["model"])
        loader, _, _, _ = trainer.build_loaders(args, device, ROOT)
        if len(loader.dataset) != 8000:
            raise ValueError("Expected the locked train8000")
        train_epochs(model, optimizer, scaler, loader, device, args, protocol, history, config_hash, target, parent["model"])
    if last <= target:
        write_history(log, history)
    final = checkpoint(target)
    entry_id = f"{GROUP}-TRAIN8000-EPOCH{target:02d}"
    register_many([dict(
        id=entry_id, comparable_group=entry_id, method=METHODS[CANDIDATE], task="training",
        status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
        checkpoint=final.as_posix(), checkpoint_sha256=sha256_file(final),
        parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"],
        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=protocol["split_manifest_sha256"],
        result_dir=ROOT.as_posix(), metadata_path=config.as_posix(), metric_profile="training_loss",
        output_policy="float_loss", metrics={"train_loss": history[-1]["train_loss"]},
        metric_source=dict(type="csv_row", path=log.as_posix(), match={"epoch": str(target)},
                           columns={"train_loss": "train_loss"}),
    )])


def published_rows(folder, arm, filename, ids, digest, epoch, path):
    metadata = reuse_metrics(folder, ids, digest, {arm: sha256_file(path)}, epoch)
    raw = folder / arm / filename
    return read_metric_rows(raw, ids), dict(
        metrics=raw.as_posix(), metrics_sha256=sha256_file(raw), metadata=metadata.as_posix(),
        metadata_sha256=sha256_file(metadata),
    )


def infer(path, folder, ids, digest, device, profile):
    import test_OS

    variant = IPMFE2.VARIANT
    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST), "--split", "val",
            "--limit", str(len(ids)), "--variant", variant, "--checkpoint", str(path), "--output-dir", str(folder),
            "--metric-profile", profile, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, {variant: IPMFE2}), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    metadata = read_json(folder / "run_metadata.json")
    expected = dict(split="val", offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids), variant=variant,
                    metric_profile=profile, split_manifest_sha256=digest, output_range_policy="clamp",
                    checkpoint_sha256=sha256_file(path), metrics_only=False)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected IPMF evaluation metadata: {folder}")
    for kind in ("gray", "rgb", "preview"):
        verify_images(folder / kind, ids)
    return read_metric_rows((folder / "metrics.csv").rename(folder / "evaluator_metrics.csv"), ids)


def screening(folder):
    with (folder / "paired_comparison.csv").open(newline="", encoding="utf-8") as file:
        rows = {(row["control"], row["metric"]): row for row in csv.DictReader(file)
                if row["candidate"] == CANDIDATE}
    checks = {}
    for metric, delta in TARGET_DELTAS.items():
        row = rows["CF1", metric]
        count = int(row["common_count"])
        candidate = float(row["candidate_mean"]) if count else None
        target = float(row["control_mean"]) + delta if count else None
        if metric == "Nabf" and count == 1000:
            target = .016700
        passed = bool(count and (candidate <= target if metric == "Nabf" else candidate >= target))
        checks[metric] = dict(common_count=count, candidate=candidate, target=target, passed=passed)
        if metric in STRUCTURE:
            floor = float(rows["CF0", metric]["control_mean"]) - .0005 if count else None
            checks[metric].update(cf0_floor=floor, passes_cf0_floor=bool(count and candidate >= floor),
                                  delta_vs_cf1=float(row["candidate_minus_control"]) if count else None)
    ssim, ms_ssim = (checks[name] for name in STRUCTURE)
    structure = (ssim["passed"] and ms_ssim["passes_cf0_floor"]) or (ms_ssim["passed"] and ssim["passes_cf0_floor"])
    fidelity = all(checks[name]["passed"] for name in checks.keys() - set(STRUCTURE))
    return dict(numeric_pass=structure and fidelity, fidelity_pass=fidelity, visual_review="pending", metrics=checks)


def extension_gate(training):
    ids = read_json(MANIFEST)["splits"]["val"]
    val = {}
    for epoch in (2, 4):
        folder = validation_dir(epoch)
        metadata = read_json(folder / "run_metadata.json")
        expected = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=epoch,
                        metric_profile=training["metric_profile"], output_policy=training["output_policy"],
                        split_manifest_sha256=training["split_manifest_sha256"],
                        training_config_sha256=sha256_file(ROOT / "config.json"))
        if (any(metadata.get(key) != value for key, value in expected.items())
                or metadata["checkpoint_hashes"][CANDIDATE] != sha256_file(checkpoint(epoch))
                or metadata["checkpoint_hashes"]["CF1"] != training["parent_checkpoint_sha256"]):
            raise ValueError(f"Unexpected validation provenance for extension: {folder}")
        val[epoch] = read_metric_rows(folder / CANDIDATE / "metrics.csv", ids)
    decision = screening(validation_dir(4))
    deltas = [decision["metrics"][name]["delta_vs_cf1"] for name in STRUCTURE]
    if not decision["fidelity_pass"] or any(value is None or value < 0 for value in deltas) or max(deltas) < .0015:
        raise ValueError("Epoch-4 val1000 does not meet the IPMF extension thresholds")
    for name, delta in zip(STRUCTURE, deltas):
        index = METRICS.index(name)
        if delta < .0015:
            continue
        cf1 = read_metric_rows(validation_dir(4) / "CF1/metrics.csv", ids)
        paired = [(b[0][index], a[0][index]) for a, b, baseline in zip(val[2], val[4], cf1)
                  if all(math.isfinite(row[0][index]) for row in (a, b, baseline))]
        if paired and sum(after - before for after, before in paired) > 0:
            return
    raise ValueError("Epoch-2 to epoch-4 validation lacks a positive paired structure trend")


def evaluate(device, epoch):
    parent, _, training = training_inputs()
    config = ROOT / "config.json"
    if read_json(config) != training:
        raise ValueError(f"Unexpected IPMF training protocol: {config}")
    config_hash = sha256_file(config)
    read_checkpoint(epoch, config_hash, parent["model"])
    ids = read_json(MANIFEST)["splits"]["val"]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected the locked val1000 split")
    digest = training["split_manifest_sha256"]
    paths = {"CF0": CF0, "CF1": PARENT, CANDIDATE: checkpoint(epoch)}
    methods = dict(METHODS)
    results, archives = {}, {}
    for arm, name in (("CF0", "metrics.csv"), ("CF1", "evaluator_metrics.csv")):
        try:
            results[arm], archives[arm] = published_rows(CF1_E04_ARCHIVE, arm, name, ids, digest, 4, paths[arm])
        except (OSError, ValueError, KeyError) as error:
            print(f"{arm} archive cannot be reused; evaluating the existing checkpoint: {error}", flush=True)
    cf1_e08 = cfdr_checkpoint("CF1", 8)
    if cf1_e08.is_file() and comparison_dir(8).is_dir():
        try:
            results["CF1-E08"], archives["CF1-E08"] = published_rows(
                comparison_dir(8), "CF1", "metrics.csv", ids, digest, 8, cf1_e08,
            )
            paths["CF1-E08"], methods["CF1-E08"] = cf1_e08, "CF1 第 8 轮"
        except (OSError, ValueError, KeyError) as error:
            print(f"Excluding the unmatched CF1 epoch-8 archive: {error}", flush=True)
    hashes = {arm: sha256_file(path) for arm, path in paths.items()}
    protocol = dict(split="val", offset=0, scene_ids=ids, device=str(device), frozen_epochs=epoch,
                    metric_profile=training["metric_profile"], output_policy=training["output_policy"],
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest, cached=archives,
                    checkpoints={arm: path.as_posix() for arm, path in paths.items()}, checkpoint_hashes=hashes,
                    training_config=config.as_posix(), training_config_sha256=config_hash,
                    source_sha256={**training["source_sha256"], **training["evaluator_sha256"]})
    output = validation_dir(epoch)
    if output.exists():
        if read_json(output / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous IPMF validation at {output}")
    else:
        work = output.parent / f".{output.name}-{uuid4().hex}"
        work.mkdir(parents=True)
        for arm, path in paths.items():
            folder = work / arm
            if arm in results:
                folder.mkdir()
            elif arm == CANDIDATE:
                results[arm] = infer(path, folder, ids, digest, device, training["metric_profile"])
            else:
                results[arm] = infer_control(arm, path, folder, "val", ids, digest, device)
        contrasts = (("CF0", "CF1"), *((arm, CANDIDATE) for arm in paths if arm != CANDIDATE))
        summaries = summarize(work, ids, results, contrasts, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        write_json(work / "screening.json", screening(work))
        work.rename(output)
    for arm in paths.keys() - archives.keys():
        for kind in ("gray", "rgb", "preview"):
            verify_images(output / arm / kind, ids)
    summary_path = output / "registry_metrics.json"
    summaries = read_json(summary_path)
    group = f"{GROUP}-VAL1000-EPOCH{epoch:02d}-REF-FMI"
    register_many([dict(
        id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split="val", offset=0,
        sample_count=len(ids), task="fusion", method=label, status="development_validation",
        result_dir=(output / arm).as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
        checkpoint=paths[arm].as_posix(), checkpoint_sha256=hashes[arm], split_manifest=MANIFEST.as_posix(),
        split_manifest_sha256=digest, metric_profile=training["metric_profile"],
        output_policy=training["output_policy"], metrics=summaries[arm],
        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                           paths={key: [arm, key] for key in summaries[arm]}),
        notes="每项指标基于同组共同有限场景。第 4 轮用于候选筛选。",
    ) for arm, label in methods.items()])
    print(f"Paired comparison and screening: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val"))
    parser.add_argument("--epoch", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.action == "train" and args.epoch == 2:
        parser.error("Use epoch 4 for the initial training budget")
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for IPMF")
    (train if args.action == "train" else evaluate)(torch.device(args.device), args.epoch)


if __name__ == "__main__":
    main()
