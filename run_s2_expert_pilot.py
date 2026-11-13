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
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from s2_expert_models import MODES, HeterogeneousExpertFusion, S2ModelFactory, count_params

CHECKPOINT = Path("models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth")
MANIFEST = Path("../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json")
DATA_ROOT = Path("../datasets/QXSLAB_SAROPT")
ROOT = Path("models/s2_expert_pilot_v1")
EVAL_ROOT = Path("test_result/s2_expert_pilot_v1")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_args(ns):
    return SimpleNamespace(
        data_root=ns.data_root, split_manifest=ns.manifest,
        test_count=0, val_count=1000, optical_dir="GF2", sar_dir="GF3",
        train_limit=ns.train_limit, val_limit=ns.val_limit,
        patch_size=256, batch_size=ns.batch_size, num_workers=ns.num_workers,
        invalid_threshold=0, min_valid_fraction=.90, seed=ns.seed,
        gradient_weight=10., reconstruction_weight=.1,
        accumulation_steps=ns.accumulation_steps, no_amp=ns.no_amp,
        clip_grad=1., gradient_loss="baseline", structure_weight=0.,
    )


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
    args = make_args(ns)
    trainer.set_seed(ns.seed)
    run = ROOT / f"{mode}_seed{ns.seed}"
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    loader, val_loader, split, subset = trainer.build_loaders(args, device, run)
    model, source_state = load_initial(ns, mode, device)
    source_sha = digest(ns.checkpoint)
    config = dict(
        mode=mode, seed=ns.seed, source_checkpoint=str(ns.checkpoint),
        source_checkpoint_sha256=source_sha, manifest_sha256=digest(ns.manifest),
        saved_manifest_sha256=digest(split), subset_sha256=digest(subset),
        train_limit=ns.train_limit, val_limit=ns.val_limit,
        batch_size=ns.batch_size, accumulation_steps=ns.accumulation_steps,
        epochs=ns.epochs, warmstart_steps=ns.warmstart_steps,
        loss="E2 intensity + 10*gradient + 0.1*reconstruction",
        frozen=["encoder", "lwpd", "base_fusion"],
        detail_parameters=count_params(model.detail_fusion),
    )
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not ns.no_amp)
    losses = DirectFusionLosses().to(device)
    history = []
    for i in range(ns.epochs):
        train_loss = epoch(model, losses, loader, device, args, optimizer, scaler)
        val_loss = epoch(model, losses, val_loader, device, args)
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
    torch.save(saved, run / "final.pth")
    print("Saved:", run / "final.pth")


def evaluate(ns, mode):
    """Run the repository's standard paper-profile evaluator on val, never test."""
    from e2_base_anchor import OpticalBaseAnchoredE2
    import test_OS
    ck = ROOT / f"{mode}_seed{ns.seed}" / "final.pth"
    if not ck.exists():
        raise FileNotFoundError(ck)
    result = EVAL_ROOT / f"{mode}_seed{ns.seed}_val{ns.val_limit}"
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {result}")
    registry = {OpticalBaseAnchoredE2.VARIANT: OpticalBaseAnchoredE2}
    for expert_mode in MODES:
        if expert_mode != "e2_ft":
            registry[f"s2_{expert_mode}_v1"] = (lambda m=expert_mode: S2ModelFactory.build(m))
    argv = [
        "test_OS.py", "--data-root", str(ns.data_root),
        "--split-manifest", str(ns.manifest), "--split", "val",
        "--variant", "auto", "--checkpoint", str(ck),
        "--output-dir", str(result), "--offset", "0",
        "--limit", str(ns.val_limit), "--metric-profile", "paper",
        "--output-range-policy", "clamp", "--metrics-only",
        "--timing-warmup", "2",
    ]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, registry), patch.object(sys, "argv", argv):
        test_OS.main()
    print("Saved paper metrics:", result / "metrics.csv")


def compare(ns):
    import run_e2_pac_mamba as protocol
    rows, summaries = {}, {}
    for mode in MODES:
        source = EVAL_ROOT / f"{mode}_seed{ns.seed}_val{ns.val_limit}" / "metrics.csv"
        metadata = source.parent / "run_metadata.json"
        if not source.exists() or not metadata.exists():
            raise FileNotFoundError(f"Missing evaluation for {mode}: {source}")
        values, avg = protocol.read_metrics(source)
        rows[mode] = values
        summaries[mode] = (avg, json.loads(metadata.read_text(encoding="utf-8")))
    keys = (
        "split_manifest_sha256", "selected_scene_ids", "split", "offset",
        "evaluated_pairs", "metric_profile", "output_range_policy",
    )
    baseline = summaries["e2_ft"][1]
    for mode in MODES:
        current = summaries[mode][1]
        if any(current.get(k) != baseline.get(k) for k in keys):
            raise ValueError(f"Mismatched evaluation protocol: {mode}")
        if set(rows[mode]) != set(rows["e2_ft"]):
            raise ValueError(f"Mismatched evaluated samples: {mode}")
    report = EVAL_ROOT / f"comparison_seed{ns.seed}_val{ns.val_limit}.csv"
    data = []
    for metric in ("FMI", "Qabf", "Nabf", "SSIM", "MS_SSIM", "VIF", "MI"):
        row = {"metric": metric}
        for mode in MODES:
            row[mode] = float(summaries[mode][0][metric])
            row[mode + "_vs_e2_ft"] = (row[mode] - float(summaries["e2_ft"][0][metric])) * (-1 if metric == "Nabf" else 1)
        data.append(row)
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
