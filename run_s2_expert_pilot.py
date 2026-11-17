"""S2 pilot: matched CNN / dense heterogeneous experts / Top2 experts on frozen E2.

The study uses existing optical-Base E2 weights, original E2 losses, and VAL
only. It tests the utility of different fusion operations; it is not a final
from-scratch result and does not use any test-set images.

Windows CMD (repository root):
  .venv\\Scripts\\python.exe -m unittest -v test_s2_expert_pilot
  .venv\\Scripts\\python.exe run_s2_expert_pilot.py smoke --mode top2
  .venv\\Scripts\\python.exe run_s2_expert_pilot.py train-all
  .venv\\Scripts\\python.exe run_s2_expert_pilot.py eval-all
  .venv\\Scripts\\python.exe run_s2_expert_pilot.py compare
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import fmean, pstdev
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from s2_expert_models import MODES, HeterogeneousExpertFusion, S2ModelFactory, count_params
from maintain_experiment_index import sha256_file as digest
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator

CHECKPOINT = Path("models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth")
MANIFEST = Path("../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json")
DATA_ROOT = Path("../datasets/QXSLAB_SAROPT")
ROOT = Path("models/s2_expert_pilot_v2")
EVAL_ROOT = Path("test_result/s2_expert_pilot_v2")
PROTOCOL = "QXS-S2-EXPERT-PILOT-V2"
OUTPUT_POLICY = "clamp"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pilot_manifest(ns):
    from train_direct_fusion import deterministic_subset
    from utils.dataset_os import digest_splits, evenly_spaced, validate_manifest_splits

    source = read_json(ns.manifest)
    validate_manifest_splits(source)
    if not 0 < ns.train_limit <= len(source["splits"]["train"]):
        raise ValueError("Training subset size must be within the source train split")
    splits = dict(
        train=deterministic_subset(source["splits"]["train"], ns.train_limit, ns.seed),
        val=evenly_spaced(source["splits"]["val"], ns.val_limit),
        test=[],
    )
    return dict(
        dataset="QXS-SAROPT", optical_dir=source["optical_dir"], sar_dir=source["sar_dir"],
        splits=splits, splits_sha256=digest_splits(splits), seed=ns.seed,
        source_manifest=ns.manifest.as_posix(), source_manifest_sha256=digest(ns.manifest),
    )


def make_args(ns, manifest):
    return SimpleNamespace(
        data_root=ns.data_root, split_manifest=manifest,
        test_count=0, val_count=1000, optical_dir="GF2", sar_dir="GF3",
        train_limit=ns.train_limit, val_limit=ns.val_limit,
        patch_size=256, batch_size=ns.batch_size, num_workers=ns.num_workers,
        # Recreate workers after warmstart so their augmentation RNG also resets.
        persistent_workers=False,
        invalid_threshold=0, min_valid_fraction=.90, seed=ns.seed,
        gradient_weight=10., reconstruction_weight=.1,
        accumulation_steps=ns.accumulation_steps, no_amp=ns.no_amp,
        clip_grad=1., gradient_loss="baseline", structure_weight=0.,
    )


def validated_run(ns, mode):
    run = ROOT / f"{mode}_seed{ns.seed}"
    ck = run / "final.pth"
    manifest = run / "pilot_manifest.json"
    config = read_json(run / "config.json")
    split = read_json(manifest)
    if (config["protocol"] != PROTOCOL or config["mode"] != mode or config["seed"] != ns.seed
            or config["val_limit"] != ns.val_limit or len(split["splits"]["val"]) != ns.val_limit
            or config["manifest_sha256"] != digest(manifest)
            or split["source_manifest_sha256"] != digest(ns.manifest)):
        raise ValueError(f"Run does not match the requested pilot split or mode: {run}")
    if not ck.is_file():
        raise FileNotFoundError(ck)
    return ck, manifest, config, split["splits"]["val"]


def register_result(mode, ck, manifest, result, metadata, split, count, profile, output_policy, *, group, task, status, **extra):
    from run_e2_matched_no_pac import register_many

    register_many([dict(
        id=f"{group}-{mode}", comparable_group=group, method=f"S2 pilot {mode}",
        task=task, status=status, dataset="QXS-SAROPT", split=split, offset=0,
        sample_count=count, checkpoint=ck.as_posix(), checkpoint_sha256=digest(ck),
        split_manifest=manifest.as_posix(), split_manifest_sha256=digest(manifest),
        metric_profile=profile, output_policy=output_policy,
        result_dir=result.as_posix(), metadata_path=metadata.as_posix(), **extra,
    )])


def matched_group(config):
    from utils.dataset_os import digest_splits

    shared = {key: value for key, value in config.items() if key not in ("mode", "detail_parameters")}
    return f"{PROTOCOL}-SEED{config['seed']}-{digest_splits(shared)[:12]}"


def device_of(ns):
    device = torch.device(ns.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return device


def load_initial(ns, mode, device):
    from e2_base_anchor import OpticalBaseAnchoredE2
    state = torch.load(ns.checkpoint, map_location="cpu", weights_only=False)
    if state.get("variant") != OpticalBaseAnchoredE2.VARIANT:
        raise ValueError("Expected optical Base anchored E2 checkpoint")
    model = S2ModelFactory.build(mode)
    if mode == "e2_ft":
        model.load_state_dict(state["model"], strict=True)
    else:
        expected = {k for k in model.state_dict() if k.startswith("detail_fusion.")}
        source = {k: v for k, v in state["model"].items() if not k.startswith("detail_fusion.")}
        missing, unexpected = model.load_state_dict(source, strict=False)
        if set(missing) != expected or unexpected:
            raise RuntimeError(f"Unexpected source state: {missing}, {unexpected}")
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.detail_fusion.parameters():
        p.requires_grad_(True)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    return model, state


def fixed_features(model, optical, sar):
    with torch.no_grad():
        encoded = model.encode_pair(optical, sar)
        bo, do, bs, ds = encoded
        base = bo + model.base_fusion(bo, bs)
    return encoded, base


def fusion_loss(model, losses, optical, sar, valid, encoded, base):
    bo, do, bs, ds = encoded
    detail = model.detail_fusion(do, ds, base)
    fused = model.decoder(None, base, detail)[0]
    fusion, intensity, gradient = losses.fusion(optical, sar, fused, valid, 10.)
    recon_o = losses.reconstruction(optical, model.reconstruct(optical, bo, do))
    recon_s = losses.reconstruction(sar, model.reconstruct(sar, bs, ds), valid)
    reconstruction = .5 * (recon_o + recon_s)
    total = fusion + .1 * reconstruction
    return total, dict(fusion=float(fusion.detach()), intensity=float(intensity.detach()),
                       gradient=float(gradient.detach()), reconstruction=float(reconstruction.detach()))


def epoch(model, losses, loader, device, args, optimizer=None, scaler=None, *, max_batches=None):
    training = optimizer is not None
    model.train(training)
    for frozen in (model.encoder, model.lwpd, model.base_fusion):
        frozen.eval()
    total, samples = 0., 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for idx, (o, s, valid, _) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        o, s, valid = o.to(device), s.to(device), valid.to(device)
        encoded, base = fixed_features(model, o, s)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                loss, _ = fusion_loss(model, losses, o, s, valid, encoded, base)
            if training:
                scaler.scale(loss / args.accumulation_steps).backward()
                last = max_batches is not None and idx + 1 >= max_batches
                final = idx + 1 >= len(loader)
                if (idx + 1) % args.accumulation_steps == 0 or final or last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * len(o)
        samples += len(o)
    if not samples:
        raise RuntimeError("Empty epoch")
    return total / samples


def warmstart(model, source_state, loader, device, ns, optimizer):
    """Feature imitation before fusion fitting; identical for new-head variants."""
    from e2_base_anchor import OpticalBaseAnchoredE2
    from torch.nn import functional as F
    teacher = OpticalBaseAnchoredE2().to(device).eval()
    teacher.load_state_dict(source_state["model"], strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    model.train()
    for frozen in (model.encoder, model.lwpd, model.base_fusion):
        frozen.eval()
    previous = getattr(model.detail_fusion, "mode", None)
    if isinstance(model.detail_fusion, HeterogeneousExpertFusion):
        model.detail_fusion.mode = "dense"  # All three experts receive teacher gradients.
    steps = 0
    iterator = iter(loader)
    while steps < ns.warmstart_steps:
        try:
            o, s, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            o, s, _, _ = next(iterator)
        o, s = o.to(device), s.to(device)
        encoded, base = fixed_features(model, o, s)
        _, do, _, ds = encoded
        with torch.no_grad():
            target = teacher.detail_fusion(do, ds, base)
        predicted = model.detail_fusion(do, ds, base)
        loss = F.smooth_l1_loss(predicted.float(), target.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.detail_fusion.parameters(), 1.)
        optimizer.step()
        steps += 1
        if steps % 32 == 0 or steps == ns.warmstart_steps:
            print(f"warmstart={steps}/{ns.warmstart_steps} feature_loss={float(loss):.6f}", flush=True)
    if previous is not None:
        model.detail_fusion.mode = previous


def train(ns, mode):
    from train_direct_fusion import DirectFusionLosses
    import train_lwpd as trainer
    device = device_of(ns)
    if not ns.checkpoint.is_file() or not ns.manifest.is_file():
        raise FileNotFoundError("Missing source checkpoint or split manifest")
    selected = pilot_manifest(ns)
    trainer.set_seed(ns.seed)
    run = ROOT / f"{mode}_seed{ns.seed}"
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    manifest = run / "pilot_manifest.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    args = make_args(ns, manifest)
    loader, val_loader, split, subset = trainer.build_loaders(args, device, run)
    model, source_state = load_initial(ns, mode, device)
    source_sha = digest(ns.checkpoint)
    config = dict(
        protocol=PROTOCOL, mode=mode, seed=ns.seed, source_checkpoint=ns.checkpoint.as_posix(),
        source_checkpoint_sha256=source_sha, manifest_sha256=digest(manifest),
        saved_manifest_sha256=digest(split), subset_sha256=digest(subset),
        train_limit=ns.train_limit, val_limit=ns.val_limit,
        batch_size=ns.batch_size, accumulation_steps=ns.accumulation_steps,
        epochs=ns.epochs, warmstart_steps=ns.warmstart_steps,
        device=str(device), num_workers=ns.num_workers, no_amp=ns.no_amp,
        loss="E2 intensity + 10*gradient + 0.1*reconstruction",
        frozen=["encoder", "lwpd", "base_fusion"],
        detail_parameters=count_params(model.detail_fusion),
    )
    config_path = run / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if ns.warmstart_steps and mode != "e2_ft":
        optimizer = torch.optim.AdamW(model.detail_fusion.parameters(), lr=1e-4)
        warmstart(model, source_state, loader, device, ns, optimizer)
    # Reset the data sampler and augmentation RNG before the matched fusion stage.
    # Warmstart and model construction otherwise consume mode-dependent random draws.
    trainer.set_seed(ns.seed + 202611)
    loader.generator.manual_seed(ns.seed + 202611)
    optimizer = torch.optim.AdamW([
        {"params": model.detail_fusion.parameters(), "lr": 1e-4},
        {"params": model.decoder.parameters(), "lr": 1e-5},
    ], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not ns.no_amp)
    losses = DirectFusionLosses().to(device)
    history = []
    for i in range(ns.epochs):
        train_loss = epoch(model, losses, loader, device, args, optimizer, scaler)
        val_loss = epoch(model, losses, val_loader, device, args)
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise FloatingPointError(f"Non-finite {mode} loss at epoch {i + 1}")
        row = dict(epoch=i + 1, train_loss=train_loss, val_loss=val_loss)
        history.append(row)
        print(mode, row, flush=True)
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    saved = dict(
        mode=mode, variant=model.variant, model=model.state_dict(),
        args={"variant": model.variant}, config=config,
        source_checkpoint_sha256=source_sha,
    )
    ck = run / "final.pth"
    temporary = ck.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(ck)
    register_result(
        mode, ck, manifest, run, config_path, "train", ns.train_limit, "training_loss", "float_loss",
        group=f"{matched_group(config)}-TRAIN-{mode.upper()}", task="training", status="training_complete",
        parent_checkpoint=ns.checkpoint.as_posix(), parent_checkpoint_sha256=source_sha,
        metrics={key: history[-1][key] for key in ("train_loss", "val_loss")},
        metric_source=dict(type="csv_row", path=(run / "history.csv").as_posix(),
                           match={"epoch": ns.epochs}, columns={key: key for key in ("train_loss", "val_loss")}),
    )
    print("Saved:", ck)


def evaluate(ns, mode):
    """Run the standard evaluator on the recorded pilot validation split."""
    from e2_base_anchor import OpticalBaseAnchoredE2
    import test_OS
    ck, manifest, config, scene_ids = validated_run(ns, mode)
    result = EVAL_ROOT / f"{mode}_seed{ns.seed}_val{ns.val_limit}"
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {result}")
    registry = {OpticalBaseAnchoredE2.VARIANT: OpticalBaseAnchoredE2}
    for expert_mode in MODES:
        if expert_mode != "e2_ft":
            registry[f"s2_{expert_mode}_v1"] = (lambda m=expert_mode: S2ModelFactory.build(m))
    argv = [
        "test_OS.py", "--data-root", str(ns.data_root),
        "--split-manifest", str(manifest), "--split", "val",
        "--variant", "auto", "--checkpoint", str(ck),
        "--output-dir", str(result), "--offset", "0",
        "--limit", str(ns.val_limit), "--metric-profile", DEFAULT_PROFILE,
        "--output-range-policy", OUTPUT_POLICY, "--metrics-only",
        "--timing-warmup", "2",
    ]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, registry), patch.object(sys, "argv", argv):
        test_OS.main()
    read_evaluation(result, ck, manifest, scene_ids, mode)
    register_result(
        mode, ck, manifest, result, result / "run_metadata.json", "val", len(scene_ids),
        DEFAULT_PROFILE, OUTPUT_POLICY, group=f"{matched_group(config)}-VAL{len(scene_ids)}-{mode.upper()}",
        task="fusion", status="development_validation",
        parent_checkpoint=config["source_checkpoint"], parent_checkpoint_sha256=config["source_checkpoint_sha256"],
        metric_source=dict(type="csv_mean", path=(result / "metrics.csv").as_posix()),
        notes="原始逐图结果。四组共同有限样本汇总由 compare 生成。",
    )
    print("Saved validation metrics:", result / "metrics.csv")


def read_evaluation(result, ck, manifest, scene_ids, mode):
    from e2_base_anchor import OpticalBaseAnchoredE2

    metadata = read_json(result / "run_metadata.json")
    expected = dict(
        checkpoint_sha256=digest(ck), split_manifest_sha256=digest(manifest),
        selected_scene_ids=scene_ids, split="val", offset=0, evaluated_pairs=len(scene_ids),
        metric_profile=DEFAULT_PROFILE, output_range_policy=OUTPUT_POLICY,
        variant=OpticalBaseAnchoredE2.VARIANT if mode == "e2_ft" else f"s2_{mode}_v1",
    )
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Mismatched evaluation protocol: {result}")
    with (result / "metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
        samples = [row for row in csv.DictReader(handle) if row["image"] not in ("mean", "std")]
    if [row["image"] for row in samples] != scene_ids:
        raise ValueError(f"Mismatched evaluation scenes: {result}")
    return samples, metadata["device"]


def paired_summary(rows, scene_ids):
    data = []
    for metric in FusionEvaluator.METRIC_NAMES:
        columns = [[float(row[metric]) for row in rows[mode]] for mode in MODES]
        common = [i for i, values in enumerate(zip(*columns)) if all(map(math.isfinite, values))]
        row = dict(metric=metric, common_count=len(common),
                   excluded_scene_ids=" ".join(scene for i, scene in enumerate(scene_ids) if i not in common))
        for mode, column in zip(MODES, columns):
            values = [column[i] for i in common]
            row[mode] = fmean(values) if values else None
            row[f"{mode}_std"] = pstdev(values) if values else None
            row[f"{mode}_finite_count"] = sum(map(math.isfinite, column))
        direction = -1 if metric == "Nabf" else 1
        for mode in MODES:
            row[f"{mode}_vs_e2_ft"] = direction * (row[mode] - row["e2_ft"]) if common else None
        row["dense_vs_conv"] = direction * (row["dense"] - row["conv"]) if common else None
        row["top2_vs_dense"] = direction * (row["top2"] - row["dense"]) if common else None
        data.append(row)
    return data


def compare(ns):
    rows, devices, groups = {}, set(), set()
    expected_scenes = None
    for mode in MODES:
        ck, manifest, config, scene_ids = validated_run(ns, mode)
        result = EVAL_ROOT / f"{mode}_seed{ns.seed}_val{ns.val_limit}"
        rows[mode], device = read_evaluation(result, ck, manifest, scene_ids, mode)
        if expected_scenes is not None and scene_ids != expected_scenes:
            raise ValueError("S2 comparison requires the same validation scene order")
        expected_scenes = scene_ids
        devices.add(device)
        groups.add(matched_group(config))
    if len(groups) != 1 or len(devices) != 1:
        raise ValueError("S2 comparison requires matched training and evaluation protocols")
    data = paired_summary(rows, expected_scenes)
    report = EVAL_ROOT / f"comparison_seed{ns.seed}_val{ns.val_limit}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise FileExistsError(f"Preserving existing comparison: {report}")
    with report.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print("Saved comparison:", report)


def smoke(ns, mode):
    device = device_of(ns)
    torch.manual_seed(ns.seed)
    model = S2ModelFactory.build(mode).to(device)
    model.train()
    o, s, b = (torch.randn(1, 64, 32, 32, device=device) for _ in range(3))
    out = model.detail_fusion(o, s, b)
    if out.shape != o.shape or not torch.isfinite(out).all():
        raise RuntimeError("Invalid forward")
    out.square().mean().backward()
    gradients = [p.grad for p in model.detail_fusion.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Invalid gradients")
    print("SMOKE OK:", mode, "detail_parameters=", count_params(model.detail_fusion))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "train-all", "eval", "eval-all", "compare"))
    parser.add_argument("--mode", choices=MODES, default="top2")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmstart-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    ns = parser.parse_args()
    if ns.epochs <= 0 or ns.batch_size <= 0 or ns.accumulation_steps <= 0 or ns.val_limit <= 0:
        parser.error("Epochs, batch size, accumulation and val-limit must be positive")
    if ns.action == "smoke":
        smoke(ns, ns.mode)
    elif ns.action == "train":
        train(ns, ns.mode)
    elif ns.action == "train-all":
        for mode in MODES:
            train(ns, mode)
    elif ns.action == "eval":
        evaluate(ns, ns.mode)
    elif ns.action == "eval-all":
        for mode in MODES:
            evaluate(ns, mode)
    else:
        compare(ns)


if __name__ == "__main__":
    main()
