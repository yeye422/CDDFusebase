"""Train ASCD-V1 with a frozen CF0 frontend and evaluate epoch 4 on val1000."""

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
from cfdr_evaluation import POLICY, infer as infer_control
from e2_ascd import ASCDE2
from hcfdr_comparison import EVALUATOR_FILES, OUTPUT as CONTROL_ARCHIVE, reuse_metrics
from run_cfdr_frozen import PARENT, checkpoint_path as cfdr_checkpoint, load_backbone
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import STAGE_LEARNING_RATES, config_values
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, verify_images, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file
from utils.dataset_os import OpticalSARDataset, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


EPOCHS = 4
ROOT = Path("models/e2_ascd_v1_frozen/seed42")
OUTPUT = Path("test_result/e2_ascd_v1_frozen/val1000/epoch_004")
GROUP = "QXS-ASCD-V1-FROZEN-SEED42"
CANDIDATE = "ASCD-V1"
METHODS = {"CF0": "光学 Base 锚定 E2", "CF1": "CF1 原位加权", CANDIDATE: "冻结 E2 前端 + ASCD-V1"}
TRAIN_SOURCES = (
    "run_ascd_frozen.py", "e2_ascd.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py",
    "run_cfdr_frozen.py", "train_lwpd.py", "train_direct_fusion.py", "utils/dataset_os.py",
)


def checkpoint(epoch):
    return ROOT / f"epoch_{epoch:03d}.pth"


def training_inputs():
    backbone, args = load_backbone()
    if (args.seed, args.batch_size, args.accumulation_steps, args.patch_size,
            args.gradient_weight, args.no_amp) != (42, 2, 4, 256, 10., False):
        raise ValueError("Unexpected CF1 training recipe")
    manifest = read_json(MANIFEST)
    if manifest["seed"] != 20260924 or [len(manifest["splits"][key]) for key in ("train", "val", "test")] != [
        8000, 1000, 1000
    ]:
        raise ValueError("Expected the fixed QXSLAB-SAROPT 8000/1000/1000 split")
    protocol = dict(
        variant=ASCDE2.VARIANT, epochs=EPOCHS, training_policy="ascd_only_from_converged_cf0",
        parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=sha256_file(PARENT),
        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=sha256_file(MANIFEST),
        args=config_values(args), lr=STAGE_LEARNING_RATES["fusion_layers"],
        source_sha256={name: sha256_file(Path(name)) for name in TRAIN_SOURCES},
    )
    return ASCDE2.frontend_state(backbone), args, protocol


def require_frozen_weights(state, frontend):
    saved = ASCDE2.frontend_state(state)
    if saved.keys() != frontend.keys() or any(not torch.equal(saved[key].cpu(), value)
                                              for key, value in frontend.items()):
        raise ValueError("ASCD frontend differs from the CF0 checkpoint")


def read_checkpoint(epoch, config_hash, frontend):
    path = checkpoint(epoch)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["variant"], saved["epoch"], saved["config_sha256"]) != (ASCDE2.VARIANT, epoch, config_hash):
        raise ValueError(f"Unexpected ASCD checkpoint: {path}")
    if [row["epoch"] for row in saved["history"]] != list(range(1, epoch + 1)):
        raise ValueError(f"Unexpected ASCD history: {path}")
    require_frozen_weights(saved["model"], frontend)
    return saved


def frozen_losses(model, losses, optical, sar, valid, args):
    fused, stats = model(optical, sar, return_stats=True)
    loss, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    return loss, dict(fusion=loss, intensity=intensity, gradient=gradient, **stats)


def train(device):
    frontend, args, protocol = training_inputs()
    ROOT.mkdir(parents=True, exist_ok=True)
    config = ROOT / "config.json"
    if not config.exists():
        if any(ROOT.glob("epoch_*.pth")):
            raise ValueError(f"Missing ASCD training configuration: {config}")
        temporary = config.with_suffix(".tmp")
        write_json(temporary, protocol)
        temporary.replace(config)
    if read_json(config) != protocol:
        raise ValueError(f"Existing ASCD run uses a different training protocol: {config}")
    config_hash = sha256_file(config)
    saved_epochs = [epoch for epoch in range(1, EPOCHS + 1) if checkpoint(epoch).exists()]
    last = max(saved_epochs, default=0)
    if saved_epochs != list(range(1, last + 1)):
        raise ValueError(f"Missing intermediate ASCD checkpoint in {ROOT}")
    saved = read_checkpoint(last, config_hash, frontend) if last else None
    history = saved["history"] if saved else []
    log = ROOT / "history.csv"
    if last < EPOCHS:
        trainer.set_seed(args.seed)
        model = ASCDE2().to(device)
        model.load_frontend(frontend)
        optimizer = torch.optim.AdamW([dict(params=model.ascd.parameters(), group_name="ascd")],
                                      lr=protocol["lr"], weight_decay=args.weight_decay)
        scaler = trainer.GradScaler(enabled=True)
        if saved:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
        del saved
        training, _, _, _ = trainer.build_loaders(args, device, ROOT)
        if len(training.dataset) != 8000:
            raise ValueError("ASCD requires the original train8000")
        losses = trainer.DirectFusionLosses().to(device)
        with patch.object(trainer, "forward_losses", frozen_losses):
            for epoch in range(last + 1, EPOCHS + 1):
                trainer.set_seed(args.seed + epoch)
                training.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, training, device, args, optimizer, scaler)
                if not all(math.isfinite(value) for name, value in stats.items() if not name.startswith("grad_")):
                    raise FloatingPointError(f"Nonfinite ASCD statistics at epoch {epoch}")
                state = model.state_dict()
                require_frozen_weights(state, frontend)
                history.append(dict(epoch=epoch, train_seconds=time.perf_counter() - started,
                                    train_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                                    **{f"train_{k}": v if math.isfinite(v) else None for k, v in stats.items()}))
                path = checkpoint(epoch)
                temporary = path.with_suffix(".tmp")
                torch.save(dict(variant=model.variant, args={"variant": model.variant}, epoch=epoch, model=state,
                                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                config_sha256=config_hash, history=history), temporary)
                temporary.replace(path)
                write_history(log, history)
                print(f"ASCD-V1 epoch {epoch}/{EPOCHS}: train={stats['loss']:.6f}", flush=True)
    write_history(log, history)
    entry_id = f"{GROUP}-TRAIN8000-EPOCH04"
    register_many([dict(
        id=entry_id, comparable_group=entry_id, method=METHODS[CANDIDATE], task="training",
        status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
        checkpoint=checkpoint(EPOCHS).as_posix(), checkpoint_sha256=sha256_file(checkpoint(EPOCHS)),
        parent_checkpoint=protocol["parent_checkpoint"], parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"],
        split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
        result_dir=ROOT.as_posix(), metadata_path=config.as_posix(), metric_profile="training_loss",
        output_policy="float_loss", metrics={"train_loss": history[-1]["train_loss"]},
        metric_source=dict(type="csv_row", path=log.as_posix(), match={"epoch": str(EPOCHS)},
                           columns={"train_loss": "train_loss"}),
    )])


def smoke(device):
    frontend, args, _ = training_inputs()
    trainer.set_seed(args.seed)
    model = ASCDE2().to(device).train()
    model.load_frontend(frontend)
    splits, _ = manifest_pairs(DATA, MANIFEST)
    dataset = OpticalSARDataset(splits["train"][:1], patch_size=args.patch_size, training=True,
                               invalid_threshold=args.invalid_threshold, min_valid_fraction=args.min_valid_fraction)
    optical, sar, valid, scene = dataset[0]
    optical, sar, valid = (item[None].to(device) for item in (optical, sar, valid))
    losses = trainer.DirectFusionLosses().to(device)
    optimizer = torch.optim.AdamW(model.ascd.parameters(), lr=STAGE_LEARNING_RATES["fusion_layers"],
                                  weight_decay=args.weight_decay)
    scg = model.ascd.scg
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fused, stats = model(optical, sar, return_stats=True)
        loss, _, _ = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    with torch.no_grad():
        optical_mean, sar_mean = scg.smooth(torch.cat((optical, sar), dim=1)).split(1, dim=1)
        initial = optical_mean + .475 * (optical - optical_mean) + .475 * (sar - sar_mean)
        torch.testing.assert_close(fused, initial, rtol=0, atol=1e-4)
    before = scg.structure_head.weight.detach().clone()
    loss.backward()
    gradients = [p.grad for p in model.ascd.parameters() if p.grad is not None]
    if (fused.shape != optical.shape or not torch.isfinite(loss) or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
            or any(module.training for name, module in model.named_children() if name != "ascd")
            or any(p.requires_grad or p.grad is not None for name, p in model.named_parameters()
                   if name.startswith(model.FROZEN_PREFIXES))):
        raise AssertionError("ASCD smoke test failed")
    optimizer.step()
    if torch.equal(before, scg.structure_head.weight):
        raise AssertionError("ASCD structure head was not updated")
    require_frozen_weights(model.state_dict(), frontend)
    print(dict(scene=scene, loss=float(loss.detach()), output_range=(float(fused.detach().min()),
          float(fused.detach().max())), **{name: float(value) for name, value in stats.items()}), flush=True)


def infer_candidate(path, folder, ids, digest, device):
    import test_OS

    variant = ASCDE2.VARIANT
    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST), "--split", "val",
            "--limit", str(len(ids)), "--variant", variant, "--checkpoint", str(path), "--output-dir", str(folder),
            "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, {variant: ASCDE2}), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    meta = read_json(folder / "run_metadata.json")
    expected = dict(split="val", offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids), variant=variant,
                    metric_profile=DEFAULT_PROFILE, split_manifest_sha256=digest, output_range_policy="clamp",
                    checkpoint_sha256=sha256_file(path), metrics_only=False)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected ASCD evaluation metadata: {folder}")
    for kind in ("gray", "rgb", "preview"):
        verify_images(folder / kind, ids)
    return read_metric_rows((folder / "metrics.csv").rename(folder / "evaluator_metrics.csv"), ids)


def screening(folder):
    with (folder / "paired_comparison.csv").open(newline="", encoding="utf-8") as file:
        pairs = {(r["metric"], r["control"]): r for r in csv.DictReader(file) if r["candidate"] == CANDIDATE}
    checks = {}
    for metric, delta in {"SSIM": .003, "MS_SSIM": .003, "MI": -.030, "FMI": -.005,
                          "Qabf": -.002, "VIF": -.003, "Nabf": .001535}.items():
        row = pairs[metric, "CF1"]
        count = int(row["common_count"])
        candidate = float(row["candidate_mean"]) if count else None
        target = float(row["control_mean"]) + delta if count else None
        if metric == "Nabf" and count == 1000:
            target = .016700
        passed = bool(count and (candidate <= target if metric == "Nabf" else candidate >= target))
        checks[metric] = dict(common_count=count, candidate=candidate, target=target, passed=passed)
        if metric in ("SSIM", "MS_SSIM"):
            baseline = pairs[metric, "CF0"]
            floor = float(baseline["control_mean"]) - .0005 if count else None
            checks[metric].update(cf0_floor=floor, passes_cf0_floor=bool(count and candidate >= floor))
    ssim, ms_ssim = (checks[name] for name in ("SSIM", "MS_SSIM"))
    structure = (ssim["passed"] and ms_ssim["passes_cf0_floor"]) or (ms_ssim["passed"] and ssim["passes_cf0_floor"])
    others = all(checks[name]["passed"] for name in ("MI", "FMI", "Qabf", "VIF", "Nabf"))
    write_json(folder / "screening.json", dict(numeric_pass=structure and others, visual_review="pending", metrics=checks))


def evaluate(device):
    frontend, _, training = training_inputs()
    config = ROOT / "config.json"
    if read_json(config) != training:
        raise ValueError(f"Unexpected ASCD training protocol: {config}")
    read_checkpoint(EPOCHS, sha256_file(config), frontend)
    ids = read_json(MANIFEST)["splits"]["val"]
    digest = training["split_manifest_sha256"]
    paths = {"CF0": PARENT, "CF1": cfdr_checkpoint("CF1", EPOCHS), CANDIDATE: checkpoint(EPOCHS)}
    hashes = {arm: sha256_file(path) for arm, path in paths.items()}
    results, archives = {}, {}
    for arm in ("CF0", "CF1"):
        try:
            metadata = reuse_metrics(CONTROL_ARCHIVE, ids, digest, {arm: hashes[arm]}, EPOCHS)
            raw = CONTROL_ARCHIVE / arm / ("metrics.csv" if arm == "CF0" else "evaluator_metrics.csv")
            run_meta = (Path(read_json(metadata)["archive_cf0"]["archived_metadata"])
                        if arm == "CF0" else CONTROL_ARCHIVE / arm / "run_metadata.json")
            profile_meta = (Path(read_json(run_meta)["archive_cf0"]["archived_images"]) / "run_metadata.json"
                            if arm == "CF0" else run_meta)
            cached = read_metric_rows(raw, ids)
            archives[arm] = dict(metrics=raw.as_posix(), metrics_sha256=sha256_file(raw),
                                 metadata=metadata.as_posix(), metadata_sha256=sha256_file(metadata),
                                 profile_metadata=profile_meta.as_posix() if profile_meta.is_file() else None,
                                 profile_metadata_sha256=sha256_file(profile_meta) if profile_meta.is_file() else None)
            results[arm] = cached
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"{arm} archive unavailable or incompatible; evaluating existing checkpoint: {error}", flush=True)
    protocol = dict(split="val", offset=0, scene_ids=ids, device=str(device), frozen_epochs=EPOCHS,
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, cached=archives,
                    checkpoints={arm: path.as_posix() for arm, path in paths.items()}, checkpoint_hashes=hashes,
                    source_sha256={name: sha256_file(Path(name)) for name in (*TRAIN_SOURCES, *EVALUATOR_FILES)})
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous ASCD validation at {OUTPUT}")
    else:
        work = OUTPUT.parent / f".{OUTPUT.name}-{uuid4().hex}"
        work.mkdir(parents=True)
        profiles = {}
        for arm, path in paths.items():
            folder = work / arm
            if arm in results:
                folder.mkdir()
                profile_path = archives[arm]["profile_metadata"]
            else:
                results[arm] = (infer_candidate(path, folder, ids, digest, device) if arm == CANDIDATE else
                                infer_control(arm, path, folder, "val", ids, digest, device))
                profile_path = folder / "run_metadata.json"
            profile = read_json(Path(profile_path)) if profile_path else {}
            profiles[arm] = {key: profile.get(key) for key in (
                "device", "total_parameters", "timing_warmup", "inference_time_ms",
            )}
            profiles[arm]["archived"] = arm in archives
        contrasts = (("CF0", "CF1"), ("CF0", CANDIDATE), ("CF1", CANDIDATE))
        summaries = summarize(work, ids, results, contrasts, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        write_json(work / "model_profile.json", profiles)
        screening(work)
        work.rename(OUTPUT)
    summary_path = OUTPUT / "registry_metrics.json"
    summaries = read_json(summary_path)
    group = f"{GROUP}-VAL1000-EPOCH04-REF-FMI"
    register_many([dict(
        id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split="val", offset=0,
        sample_count=len(ids), task="fusion", method=label, status="development_validation",
        result_dir=(OUTPUT / arm).as_posix(), metadata_path=(OUTPUT / "run_metadata.json").as_posix(),
        checkpoint=paths[arm].as_posix(), checkpoint_sha256=hashes[arm], split_manifest=MANIFEST.as_posix(),
        split_manifest_sha256=digest, metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=summaries[arm],
        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                           paths={key: [arm, key] for key in summaries[arm]}),
        notes="每项指标在三组共同有限场景上汇总。",
    ) for arm, label in METHODS.items()])
    print(f"Paired comparison: {OUTPUT / 'paired_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ASCD protocol")
    {"smoke": smoke, "train": train, "val": evaluate}[args.action](torch.device(args.device))


if __name__ == "__main__":
    main()
