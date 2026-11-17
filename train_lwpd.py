"""Progressive-unfreezing trainer for the LWPD E1-E4 ablations.

This entry point deliberately does not reuse the fully frozen direct-fusion
loop.  Warm-up is short; the encoder tail is enabled during local adaptation,
and the shared encoder is optimized at a small learning rate during the joint
stage.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from direct_fusion import DirectFusionCDDFuse
from lwpd_ll_context import LL_CONTEXT_STATS
from utils.fusion_structure_loss import FusionStructureLoss
from utils.pa_directional_loss import PADirectionalGradientLoss
from train_direct_fusion import (
    DirectFusionLosses,
    deterministic_subset,
    sha256_file,
    worker_seed,
)
from utils.dataset_os import (
    OpticalSARDataset,
    digest_splits,
    manifest_pairs,
    split_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=tuple(DirectFusionCDDFuse.LWPD_VARIANTS),
        default="lwpd_e4",
    )
    parser.add_argument("--data-root", type=Path, default=Path("../datasets/Dongying"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--optical-dir", default="GF2")
    parser.add_argument("--sar-dir", default="GF3")
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--warmup-lwpd-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-wavelet-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-fusion-lr", type=float, default=5e-5)
    parser.add_argument("--local-lwpd-lr", type=float, default=1e-4)
    parser.add_argument("--local-wavelet-lr", type=float, default=1e-5)
    parser.add_argument("--local-fusion-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lwpd-lr", type=float, default=5e-5)
    parser.add_argument("--joint-wavelet-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ll-context-lr", type=float, default=1e-4)
    parser.add_argument("--local-ll-context-lr", type=float, default=1e-4)
    parser.add_argument("--joint-ll-context-lr", type=float, default=5e-5)
    parser.add_argument("--joint-fusion-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=2e-5)
    parser.add_argument("--encoder-tail-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-early-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--gradient-loss", choices=("baseline", "pa_directional"), default="baseline")
    parser.add_argument("--gradient-weight", type=float, default=10.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--structure-weight", type=float, default=0.0,
                        help="Additional valid-window 1-SSIM fusion loss; 0 preserves the original loss path.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/lwpd"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_groups(model):
    blocks = list(model.encoder.encoder_level1.children())
    split = max(1, len(blocks) // 2)
    early_modules = [model.encoder.patch_embed, *blocks[:split]]
    tail_modules = blocks[split:]
    early = [parameter for module in early_modules for parameter in module.parameters()]
    tail = [parameter for module in tail_modules for parameter in module.parameters()]
    return early, tail


def make_optimizer(model, args):
    encoder_early, encoder_tail = encoder_groups(model)
    groups = [
        {"params": encoder_early, "lr": 0.0, "group_name": "encoder_early"},
        {"params": encoder_tail, "lr": 0.0, "group_name": "encoder_tail"},
        {
            "params": list(model.decoder.parameters()),
            "lr": args.decoder_lr,
            "group_name": "decoder",
        },
        {
            "params": list(model.base_fusion.parameters()),
            "lr": args.warmup_fusion_lr,
            "group_name": "base_fusion",
        },
        {
            "params": list(model.detail_fusion.parameters()),
            "lr": args.warmup_fusion_lr,
            "group_name": "detail_fusion",
        },
    ]
    wavelet_parameters = list(model.lwpd.haar.parameters())
    wavelet_parameter_ids = {id(parameter) for parameter in wavelet_parameters}
    ll_context_parameters = list(model.lwpd.ll_context.parameters()) if hasattr(model.lwpd, 'll_context') else []
    ll_context_ids = {id(parameter) for parameter in ll_context_parameters}
    lwpd_parameters = [
        parameter
        for parameter in model.lwpd.parameters()
        if id(parameter) not in wavelet_parameter_ids | ll_context_ids
    ]
    if lwpd_parameters:
        groups.append(
            {
                "params": lwpd_parameters,
                "lr": args.warmup_lwpd_lr,
                "group_name": "lwpd",
            }
        )
    if wavelet_parameters:
        groups.append(
            {
                "params": wavelet_parameters,
                "lr": args.warmup_wavelet_lr,
                "group_name": "wavelet",
            }
        )
    if ll_context_parameters:
        groups.append({'params':ll_context_parameters,'lr':args.warmup_ll_context_lr,'group_name':'ll_context'})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def stage_for_epoch(epoch, args):
    if epoch < args.warmup_epochs:
        return "warmup"
    if epoch < args.warmup_epochs + args.local_epochs:
        return "local"
    return "joint"


def configure_stage(model, optimizer, stage, args):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_early, encoder_tail = encoder_groups(model)
    if stage in ("local", "joint"):
        for parameter in encoder_tail:
            parameter.requires_grad_(True)
    if stage == "joint":
        for parameter in encoder_early:
            parameter.requires_grad_(True)
    for module in (model.lwpd, model.decoder, model.base_fusion, model.detail_fusion):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    rates = {
        "encoder_early": args.encoder_early_lr if stage == "joint" else 0.0,
        "encoder_tail": args.encoder_tail_lr if stage in ("local", "joint") else 0.0,
        "decoder": args.decoder_lr,
        "base_fusion": (
            args.warmup_fusion_lr
            if stage == "warmup"
            else args.local_fusion_lr
            if stage == "local"
            else args.joint_fusion_lr
        ),
        "detail_fusion": (
            args.warmup_fusion_lr
            if stage == "warmup"
            else args.local_fusion_lr
            if stage == "local"
            else args.joint_fusion_lr
        ),
        "lwpd": (
            args.warmup_lwpd_lr
            if stage == "warmup"
            else args.local_lwpd_lr
            if stage == "local"
            else args.joint_lwpd_lr
        ),
        "wavelet": (
            args.warmup_wavelet_lr
            if stage == "warmup"
            else args.local_wavelet_lr
            if stage == "local"
            else args.joint_wavelet_lr
        ),
    }
    for group in optimizer.param_groups:
        if group['group_name']=='ll_context':
            group['lr']=getattr(args,f'{stage}_ll_context_lr')
        elif model.variant=='lwpd_e2_llctx_v1' and group['group_name']=='decoder':
            group['lr']=rates['base_fusion']
        else:
            group["lr"] = rates[group["group_name"]]


def forward_losses(model, losses, optical, sar, valid, args, fusion_transform=None):
    encoded, diagnostics = model.encode_pair(
        optical, sar, return_diagnostics=True
    )
    optical_base, optical_detail, sar_base, sar_detail = encoded
    optical_hat = model.reconstruct(optical, optical_base, optical_detail)
    sar_hat = model.reconstruct(sar, sar_base, sar_detail)
    optical_reconstruction = losses.reconstruction(optical, optical_hat)
    sar_reconstruction = losses.reconstruction(sar, sar_hat, valid)
    reconstruction = 0.5 * (optical_reconstruction + sar_reconstruction)
    fusion_inputs, fusion_diagnostics = model.prepare_fusion_inputs(encoded, sar)
    fused_base, fused_detail = model.fuse_features(*fusion_inputs)
    fused = model.decoder(None, fused_base, fused_detail)[0]
    if fusion_transform is not None:
        fused = fusion_transform(optical, sar, fused, valid)
    fusion, intensity, gradient = losses.fusion(
        optical, sar, fused, valid, args.gradient_weight
    )
    if getattr(args, "gradient_loss", "baseline") == "pa_directional":
        gradient = losses.pa_directional(optical, sar, fused, valid)
        fusion = intensity + float(args.gradient_weight) * gradient
    total = fusion + float(args.reconstruction_weight) * reconstruction
    components = {
        "fusion": fusion,
        "intensity": intensity,
        "gradient": gradient,
        "reconstruction": reconstruction,
        "gate_mean": 0.5
        * (
            diagnostics["optical_gates"].mean()
            + diagnostics["sar_gates"].mean()
        ),
        "sar_coefficient_mean": diagnostics["sar_coefficients"].mean(),
        "wavelet_deviation": diagnostics["wavelet_deviation"],
        **fusion_diagnostics,
    }
    if hasattr(model.lwpd,'ll_context'):
        components.update({k:diagnostics[k] for k in LL_CONTEXT_STATS})
    structure_weight = float(getattr(args, "structure_weight", 0.0))
    if structure_weight > 0:
        structure, structure_stats = losses.fusion_structure(optical, sar, fused, valid)
        total = total + structure_weight * structure
        components.update(structure_stats)
        components["structure"] = structure
        components["weighted_structure"] = structure_weight * structure
        components["weighted_gradient"] = float(args.gradient_weight) * gradient
        components["weighted_reconstruction"] = float(args.reconstruction_weight) * reconstruction
    return total, components


def group_gradient_norms(optimizer):
    norms = {}
    for group in optimizer.param_groups:
        squared = 0.0
        found = False
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
                found = True
        norms[group["group_name"]] = math.sqrt(squared) if found else 0.0
    return norms


def run_epoch(model, losses, loader, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    gradient_totals = {}
    gradient_steps = 0
    sample_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda" and not args.no_amp
    for batch_index, (optical, sar, valid, _) in enumerate(loader):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with autocast(enabled=amp_enabled):
                total, components = forward_losses(
                    model, losses, optical, sar, valid, args
                )
                scaled_total = total / args.accumulation_steps
            if training:
                scaler.scale(scaled_total).backward()
                should_step = (
                    (batch_index + 1) % args.accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    current_norms = group_gradient_norms(optimizer)
                    for name, value in current_norms.items():
                        gradient_totals[name] = gradient_totals.get(name, 0.0) + value
                    gradient_steps += 1
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        args.clip_grad,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        batch_size = optical.shape[0]
        sample_count += batch_size
        for name, value in {"loss": total, **components}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    averages = {name: value / sample_count for name, value in totals.items()}
    if training and gradient_steps:
        averages.update(
            {
                f"grad_{name}": value / gradient_steps
                for name, value in gradient_totals.items()
            }
        )
    return averages


def save_checkpoint(path, epoch, stage, model, optimizer, scaler, args, best_val, *, continuation=None):
    payload = {
        "epoch": epoch,
        "stage": stage,
        "variant": args.variant,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "lowest_val_loss": best_val,
        "args": vars(args),
        "CDDF_Encoder": model.encoder.state_dict(),
        "CDDF_Decoder": model.decoder.state_dict(),
        "DirectBaseFusion": model.base_fusion.state_dict(),
        "DirectDetailFusion": model.detail_fusion.state_dict(),
        "LWPD": model.lwpd.state_dict(),
    }
    if continuation is not None:
        payload["continuation"] = continuation
    torch.save(payload, path)


def maybe_load_direct_fusion(model, checkpoint):
    loaded = {}
    for key, module in (
        ("DirectBaseFusion", model.base_fusion),
        ("DirectDetailFusion", model.detail_fusion),
    ):
        state = checkpoint.get(key)
        if state is None:
            loaded[key] = False
            continue
        try:
            module.load_state_dict(state)
            loaded[key] = True
        except RuntimeError:
            loaded[key] = False
    return loaded


def build_loaders(args, device, run_dir):
    if args.split_manifest:
        splits, split_metadata = manifest_pairs(args.data_root, args.split_manifest)
    else:
        splits = split_pairs(
            args.data_root,
            args.test_count,
            args.val_count,
            args.optical_dir,
            args.sar_dir,
        )
        split_metadata = None
    split_stems = {
        name: [pair[0].stem for pair in pairs] for name, pairs in splits.items()
    }
    saved_manifest = {
        "optical_dir": (
            split_metadata.get("optical_dir", args.optical_dir)
            if split_metadata
            else args.optical_dir
        ),
        "sar_dir": (
            split_metadata.get("sar_dir", args.sar_dir)
            if split_metadata
            else args.sar_dir
        ),
        "splits": split_stems,
        "splits_sha256": digest_splits(split_stems),
    }
    saved_path = run_dir / "split_manifest.json"
    saved_path.write_text(
        json.dumps(saved_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_pairs = deterministic_subset(splits["train"], args.train_limit, args.seed)
    val_pairs = list(splits["val"][: args.val_limit or None])
    subset = {
        "seed": args.seed,
        "source_split_sha256": saved_manifest["splits_sha256"],
        "train_stems": [pair[0].stem for pair in train_pairs],
        "val_stems": [pair[0].stem for pair in val_pairs],
    }
    subset_path = run_dir / "pilot_subset.json"
    subset_path.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_dataset = OpticalSARDataset(
        train_pairs,
        patch_size=args.patch_size,
        training=True,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
    )
    val_dataset = OpticalSARDataset(
        val_pairs,
        patch_size=args.patch_size,
        training=False,
        invalid_threshold=args.invalid_threshold,
        min_valid_fraction=args.min_valid_fraction,
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
        "persistent_workers": args.num_workers > 0 and getattr(args, "persistent_workers", True),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, saved_path, subset_path


def main():
    args = parse_args()
    if not math.isfinite(args.structure_weight) or args.structure_weight < 0:
        raise ValueError("structure-weight must be finite and non-negative")
    if args.structure_weight > 0 and args.variant != "lwpd_e2":
        raise ValueError("The general structure-loss protocol is restricted to lwpd_e2")
    if args.gradient_loss == "pa_directional" and (args.variant != "lwpd_e2" or args.structure_weight != 0):
        raise ValueError("E2-G1 requires lwpd_e2 and structure-weight=0")
    if args.warmup_epochs < 0 or args.local_epochs < 1 or args.joint_epochs < 1:
        raise ValueError("Use warmup>=0, local>=1, and joint>=1 epochs")
    if args.accumulation_steps < 1:
        raise ValueError("accumulation-steps must be positive")
    set_seed(args.seed)
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = requested_device
    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, saved_split, subset_path = build_loaders(
        args, device, run_dir
    )
    model = DirectFusionCDDFuse(args.variant).to(device)
    initial = torch.load(args.init_checkpoint, map_location="cpu")
    if args.variant=='lwpd_e2_llctx_v1':
        model.load_e2_llctx_checkpoint(initial)
        loaded_fusion={'DirectBaseFusion':True,'DirectDetailFusion':True,'LWPD_E2':True}
    else:
        model.load_backbone_checkpoint(initial)
        loaded_fusion = maybe_load_direct_fusion(model, initial)
    losses = DirectFusionLosses().to(device)
    if args.structure_weight > 0:
        losses.fusion_structure = FusionStructureLoss().to(device)
    if args.gradient_loss == "pa_directional":
        losses.pa_directional = PADirectionalGradientLoss().to(device)
    optimizer = make_optimizer(model, args)
    scaler = GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        if resume.get("variant") != args.variant:
            raise ValueError("Resume checkpoint variant mismatch")
        previous_args = resume.get("args", {})
        if previous_args.get("gradient_loss", "baseline") != args.gradient_loss:
            raise ValueError("Resume gradient loss mismatch")
        for key, default in (("structure_weight", 0.0), ("gradient_weight", 10.0),
                             ("reconstruction_weight", 0.1)):
            if float(previous_args.get(key, default)) != float(getattr(args, key)):
                raise ValueError(f"Resume loss configuration mismatch: {key}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume.get("lowest_val_loss", best_val))

    total_epochs = args.warmup_epochs + args.local_epochs + args.joint_epochs
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "init_checkpoint": str(args.init_checkpoint),
        "resume": str(args.resume) if args.resume else None,
        "init_checkpoint_sha256": sha256_file(args.init_checkpoint),
        "saved_split_manifest": str(saved_split),
        "saved_split_manifest_sha256": sha256_file(saved_split),
        "pilot_subset_sha256": sha256_file(subset_path),
        "loaded_direct_fusion": loaded_fusion,
        "total_epochs": total_epochs,
        "parameter_counts": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (
                ("encoder", model.encoder),
                ("lwpd", model.lwpd),
                ("decoder", model.decoder),
                ("base_fusion", model.base_fusion),
                ("detail_fusion", model.detail_fusion),
            )
        },
    }
    if args.gradient_loss == "pa_directional":
        config["pa_directional_spec"] = {**losses.pa_directional.spec,
            "source_sha256": sha256_file(Path(__file__).parent / "utils/pa_directional_loss.py"),
            "phase_source_sha256": sha256_file(Path(__file__).parent / "phase_antisymmetry.py")}
    if args.structure_weight > 0:
        config["fusion_structure_spec"] = {
            "version": "e2_general_ssim_v1", "window_size": 11, "sigma": 1.5,
            "range": 1.0, "reduction": "per-image source mean, then batch mean",
            "sar_validity": "all 11x11 support pixels valid; optical-only fallback",
            "loss": "1-SSIM (not DSSIM)",
            "source_sha256": sha256_file(Path(__file__).parent / "utils/fusion_structure_loss.py"),
        }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"variant={args.variant}, train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, total_epochs={total_epochs}"
    )
    if args.dry_run:
        stage = stage_for_epoch(0, args)
        configure_stage(model, optimizer, stage, args)
        optical, sar, valid, _ = next(iter(val_loader))
        total, components = forward_losses(
            model,
            losses,
            optical.to(device),
            sar.to(device),
            valid.to(device),
            args,
        )
        total.backward()
        gradients_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        (run_dir / "dry_run_diagnostics.json").write_text(
            json.dumps({"loss": float(total.detach()),
                        "components": {k: float(v.detach()) for k, v in components.items()},
                        "optimizer_updated": False,
                        "gradients_finite": gradients_finite}, indent=2) + "\n", encoding="utf-8"
        )
        if args.variant=='lwpd_e2_llctx_v1' and not gradients_finite:
            raise RuntimeError('Non-finite LL-context dry-run gradients')
        print(
            f"dry run stage={stage}, loss={float(total):.6f}, "
            f"gate_mean={float(components['gate_mean']):.6f}"
        )
        if args.structure_weight > 0:
            print(f"structure={float(components['structure']):.6f}, "
                  f"weighted_structure={float(components['weighted_structure']):.6f}")
        return

    history_path = run_dir / "history.csv"
    scalar_fields = (
        "loss",
        "fusion",
        "intensity",
        "gradient",
        "reconstruction",
        "gate_mean",
        "sar_coefficient_mean",
        "wavelet_deviation",
    )
    if args.variant=='lwpd_e2_llctx_v1':scalar_fields+=LL_CONTEXT_STATS
    if args.structure_weight > 0:
        scalar_fields += ("structure", "structure_optical", "structure_sar",
                          "structure_sar_available", "structure_valid_fraction",
                          "weighted_structure", "weighted_gradient", "weighted_reconstruction")
    gradient_fields = tuple(
        f"grad_{name}"
        for name in (
            "encoder_early",
            "encoder_tail",
            "decoder",
            "base_fusion",
            "detail_fusion",
            "lwpd",
            "wavelet",
        )
    )
    if args.variant=='lwpd_e2_llctx_v1':gradient_fields+=('grad_ll_context',)
    fieldnames = (
        "epoch",
        "stage",
        "seconds",
        *(f"train_{name}" for name in scalar_fields),
        *(f"val_{name}" for name in scalar_fields),
        *gradient_fields,
    )
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, total_epochs):
        stage = stage_for_epoch(epoch, args)
        configure_stage(model, optimizer, stage, args)
        started = time.time()
        train_stats = run_epoch(
            model, losses, train_loader, device, args, optimizer, scaler
        )
        with torch.no_grad():
            val_stats = run_epoch(model, losses, val_loader, device, args)
        seconds = time.time() - started
        row = {"epoch": epoch + 1, "stage": stage, "seconds": seconds}
        for prefix, stats in (("train", train_stats), ("val", val_stats)):
            for name in scalar_fields:
                row[f"{prefix}_{name}"] = stats.get(name, "")
        for name in gradient_fields:
            row[name] = train_stats.get(name, "")
        with history_path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        improved = val_stats["loss"] < best_val
        if improved:
            best_val = val_stats["loss"]
        save_checkpoint(
            run_dir / f"epoch_{epoch + 1:03d}.pth",
            epoch,
            stage,
            model,
            optimizer,
            scaler,
            args,
            best_val,
        )
        if improved:
            save_checkpoint(
                run_dir / "lowest_val_loss_diagnostic.pth",
                epoch,
                stage,
                model,
                optimizer,
                scaler,
                args,
                best_val,
            )
        print(
            f"epoch={epoch + 1}/{total_epochs} stage={stage} "
            f"train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"gate={val_stats['gate_mean']:.4f} seconds={seconds:.1f}"
        )


if __name__ == "__main__":
    main()
