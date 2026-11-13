"""Progressive trainer for PAC-Mamba initialized from the verified LWPD E4."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from pac_mamba import PACMambaCDDFuse, PACMambaE2CDDFuse
from train_direct_fusion import DirectFusionLosses, sha256_file
from train_lwpd import (
    build_loaders,
    encoder_groups,
    group_gradient_norms,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-variant", choices=("lwpd_e4", "lwpd_e2"), default="lwpd_e4")
    parser.add_argument("--data-root", type=Path, default=Path("../datasets/QXSLAB_SAROPT"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--optical-dir", default="GF2")
    parser.add_argument("--sar-dir", default="GF3")
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--invalid-threshold", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--joint-epochs", type=int, default=7)
    parser.add_argument("--warmup-pac-lr", type=float, default=1e-4)
    parser.add_argument("--local-pac-lr", type=float, default=1e-4)
    parser.add_argument("--joint-pac-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-existing-lr", type=float, default=1e-5)
    parser.add_argument("--local-existing-lr", type=float, default=2e-5)
    parser.add_argument("--joint-existing-lr", type=float, default=1e-5)
    parser.add_argument("--encoder-tail-lr", type=float, default=5e-6)
    parser.add_argument("--encoder-early-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--gradient-weight", type=float, default=10.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/pac_mamba"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def pac_parameters(model):
    modules = (
        model.detail_fusion.reliability,
        model.detail_fusion.joint_projection,
        model.detail_fusion.ssm,
        model.detail_fusion.update_projection,
    )
    parameters = [parameter for module in modules for parameter in module.parameters()]
    parameters.append(model.detail_fusion.update_logit)
    return parameters


def make_optimizer(model, args):
    encoder_early, encoder_tail = encoder_groups(model)
    groups = [
        {"params": encoder_early, "lr": 0.0, "group_name": "encoder_early"},
        {"params": encoder_tail, "lr": 0.0, "group_name": "encoder_tail"},
        {
            "params": list(model.lwpd.parameters()),
            "lr": args.warmup_existing_lr,
            "group_name": "lwpd",
        },
        {
            "params": list(model.base_fusion.parameters()),
            "lr": args.warmup_existing_lr,
            "group_name": "base_fusion",
        },
        {
            "params": list(model.detail_fusion.keep_fusion.parameters()),
            "lr": args.warmup_existing_lr,
            "group_name": "keep_fusion",
        },
        {
            "params": list(model.decoder.parameters()),
            "lr": args.warmup_existing_lr,
            "group_name": "decoder",
        },
        {
            "params": pac_parameters(model),
            "lr": args.warmup_pac_lr,
            "group_name": "pac",
        },
    ]
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
    for module in (
        model.lwpd,
        model.base_fusion,
        model.detail_fusion,
        model.decoder,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    existing_lr = (
        args.warmup_existing_lr
        if stage == "warmup"
        else args.local_existing_lr
        if stage == "local"
        else args.joint_existing_lr
    )
    pac_lr = (
        args.warmup_pac_lr
        if stage == "warmup"
        else args.local_pac_lr
        if stage == "local"
        else args.joint_pac_lr
    )
    rates = {
        "encoder_early": args.encoder_early_lr if stage == "joint" else 0.0,
        "encoder_tail": args.encoder_tail_lr if stage in ("local", "joint") else 0.0,
        "lwpd": existing_lr,
        "base_fusion": existing_lr,
        "keep_fusion": existing_lr,
        "decoder": existing_lr,
        "pac": pac_lr,
    }
    for group in optimizer.param_groups:
        group["lr"] = rates[group["group_name"]]


def forward_losses(model, losses, optical, sar, valid, args):
    evidence = model.compute_phase_evidence(optical, sar)
    encoded, evidence, lwpd_diagnostics = model.encode_pair(
        optical,
        sar,
        evidence=evidence,
        return_diagnostics=True,
    )
    optical_base, optical_detail, sar_base, sar_detail = encoded
    optical_hat = model.reconstruct(optical, optical_base, optical_detail)
    sar_hat = model.reconstruct(sar, sar_base, sar_detail)
    optical_reconstruction = losses.reconstruction(optical, optical_hat)
    sar_reconstruction = losses.reconstruction(sar, sar_hat, valid)
    reconstruction = 0.5 * (optical_reconstruction + sar_reconstruction)

    fused_base = model.base_fusion(optical_base, sar_base)
    fused_detail, pac_diagnostics = model.detail_fusion(
        optical_detail,
        sar_detail,
        fused_base,
        evidence["optical_pa"],
        evidence["sar_pa"],
        return_diagnostics=True,
    )
    fused = model.decoder(None, fused_base, fused_detail)[0]
    fusion, intensity, gradient = losses.fusion(
        optical, sar, fused, valid, args.gradient_weight
    )
    total = fusion + float(args.reconstruction_weight) * reconstruction
    update_rms = torch.sqrt(pac_diagnostics["update"].square().mean() + 1e-12)
    keep_rms = torch.sqrt(pac_diagnostics["keep"].square().mean() + 1e-12)
    components = {
        "fusion": fusion,
        "intensity": intensity,
        "gradient": gradient,
        "reconstruction": reconstruction,
        "lwpd_gate_mean": 0.5
        * (
            lwpd_diagnostics["optical_gates"].mean()
            + lwpd_diagnostics["sar_gates"].mean()
        ),
        "phase_match_mean": pac_diagnostics["phase_match"].mean(),
        "reliability_mean": pac_diagnostics["reliability"].mean(),
        "interaction_mean": pac_diagnostics["interaction"].mean(),
        "update_rms": update_rms,
        "relative_update_rms": update_rms / keep_rms.clamp_min(1e-6),
        "update_scale": pac_diagnostics["update_scale"],
    }
    return total, components


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
                    norms = group_gradient_norms(optimizer)
                    for name, value in norms.items():
                        if torch.isfinite(torch.tensor(value)):
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


def save_checkpoint(path, epoch, stage, model, optimizer, scaler, args, best_val):
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "variant": model.VARIANT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lowest_val_loss": best_val,
            "args": {**vars(args), "variant": model.VARIANT},
        },
        path,
    )


def main():
    args = parse_args()
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
    model_type = PACMambaE2CDDFuse if args.backbone_variant == "lwpd_e2" else PACMambaCDDFuse
    model = model_type().to(device)
    initial = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_lwpd_checkpoint(initial)
    losses = DirectFusionLosses().to(device)
    optimizer = make_optimizer(model, args)
    scaler = GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        if resume.get("variant") != model.VARIANT:
            raise ValueError("Resume checkpoint variant mismatch")
        for key in ("gradient_weight", "reconstruction_weight", "seed", "warmup_epochs", "local_epochs", "joint_epochs"):
            if resume.get("args", {}).get(key) != getattr(args, key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume.get("lowest_val_loss", best_val))

    total_epochs = args.warmup_epochs + args.local_epochs + args.joint_epochs
    config = {
        **vars(args),
        "variant": model.VARIANT,
        "data_root": str(args.data_root),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "init_checkpoint": str(args.init_checkpoint),
        "resume": str(args.resume) if args.resume else None,
        "init_checkpoint_sha256": sha256_file(args.init_checkpoint),
        "saved_split_manifest": str(saved_split),
        "saved_split_manifest_sha256": sha256_file(saved_split),
        "pilot_subset_sha256": sha256_file(subset_path),
        "total_epochs": total_epochs,
        "parameter_counts": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (
                ("encoder", model.encoder),
                ("lwpd", model.lwpd),
                ("base_fusion", model.base_fusion),
                ("pac_detail", model.detail_fusion),
                ("decoder", model.decoder),
            )
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"variant={model.VARIANT}, train={len(train_loader.dataset)}, "
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
        diagnostics = {"loss": float(total.detach()),
                       "components": {key: float(value.detach()) for key, value in components.items()},
                       "gradients_finite": all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                       "updated_parameters": False}
        (run_dir / "dry_run_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
        if not diagnostics["gradients_finite"]:
            raise RuntimeError("Non-finite dry-run gradient")
        print(
            f"dry run stage={stage}, loss={float(total):.6f}, "
            f"interaction={float(components['interaction_mean']):.6f}, "
            f"relative_update={float(components['relative_update_rms']):.6f}"
        )
        return


    scalar_fields = (
        "loss",
        "fusion",
        "intensity",
        "gradient",
        "reconstruction",
        "lwpd_gate_mean",
        "phase_match_mean",
        "reliability_mean",
        "interaction_mean",
        "update_rms",
        "relative_update_rms",
        "update_scale",
    )
    gradient_names = (
        "encoder_early",
        "encoder_tail",
        "lwpd",
        "base_fusion",
        "keep_fusion",
        "decoder",
        "pac",
    )
    fieldnames = (
        "epoch",
        "stage",
        "seconds",
        *(f"train_{name}" for name in scalar_fields),
        *(f"val_{name}" for name in scalar_fields),
        *(f"grad_{name}" for name in gradient_names),
    )
    history_path = run_dir / "history.csv"
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
        for name in gradient_names:
            row[f"grad_{name}"] = train_stats.get(f"grad_{name}", "")
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
            f"interaction={val_stats['interaction_mean']:.4f} "
            f"relative_update={val_stats['relative_update_rms']:.4f} "
            f"seconds={seconds:.1f}"
        )


if __name__ == "__main__":
    main()

