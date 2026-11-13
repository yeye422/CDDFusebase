"""Train LWPD E2 with Sobel overshoot loss from random initialization."""
import csv
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import run_e2_pac_mamba as protocol
from run_e2_matched_no_pac import register
from train_e2_pac_overshoot import OvershootLosses, WEIGHT
import train_lwpd as trainer


ORIGINAL_CONFIG = Path("models/lwpd_qxs_ablation_v1/e2_seed42/config.json")
MODEL = Path("models/e2_scratch_overshoot_v1")
RUN = MODEL / "seed42"
FINAL = RUN / "epoch_016.pth"


def scratch_args():
    original = json.loads(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    if original["variant"] != "lwpd_e2":
        raise ValueError("Expected the original E2 training configuration")
    # The shared parser requires an initialization argument. This entry point
    # only borrows its defaults and never enters its checkpoint loading path.
    argv = ["train_lwpd.py", "--variant", "lwpd_e2", "--init-checkpoint", "unused",
            "--run-name", "seed42", "--split-manifest", str(protocol.MANIFEST)]
    with patch.object(sys, "argv", argv):
        args = trainer.parse_args()
    skip = {"init_checkpoint", "resume", "output_dir", "run_name", "dry_run"}
    for key in (vars(args).keys() & original.keys()) - skip:
        value = original[key]
        setattr(args, key, Path(value) if isinstance(getattr(args, key), Path) else value)
    args.init_checkpoint = None
    args.initialization = "random"
    # Train all E2 modules immediately. The former warmup froze a pretrained
    # encoder, which would leave a randomly initialized encoder fixed.
    args.warmup_epochs = args.local_epochs = 0
    args.joint_epochs = 16
    for key in ("encoder_early_lr", "encoder_tail_lr", "decoder_lr",
                "joint_fusion_lr", "joint_lwpd_lr"):
        setattr(args, key, 1e-4)
    if args.gradient_loss != "baseline" or args.structure_weight != 0:
        raise ValueError("Expected the original E2 loss")
    return args


def training_args():
    args = scratch_args()
    args.output_dir = MODEL
    args.overshoot_weight = WEIGHT
    args.overshoot_loss_source_sha256 = protocol.sha256_file(
        Path(__file__).with_name("train_e2_pac_overshoot.py")
    )
    return args


def register_training(args):
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["initialization"] != "random" or config["init_checkpoint"] is not None
        or config["overshoot_weight"] != WEIGHT
        or config["overshoot_loss_source_sha256"] != args.overshoot_loss_source_sha256):
        raise ValueError("Completed run differs from the scratch protocol")
    history = RUN / "history.csv"
    with history.open(encoding="utf-8", newline="") as f:
        row = next(row for row in csv.DictReader(f) if row["epoch"] == "16")
    metrics = {k: float(row[k]) for k in ("train_loss", "val_loss", "train_overshoot", "val_overshoot")}
    register(dict(
        id="E2-SCRATCH-OVERSHOOT-V1-SEED42-TRAIN8000", status="training_complete", task="training",
        method="LWPD E2 from random initialization with L2 Sobel overshoot penalty (2.5)",
        comparable_group="QXS-E2-SCRATCH-OVERSHOOT-V1-TRAIN8000", dataset="QXS-SAROPT",
        split="train", offset=0, sample_count=8000, split_manifest=protocol.MANIFEST.as_posix(),
        split_manifest_sha256=protocol.sha256_file(protocol.MANIFEST),
        checkpoint=FINAL.as_posix(), checkpoint_sha256=protocol.sha256_file(FINAL),
        result_dir=RUN.as_posix(), metadata_path=config_path.as_posix(),
        metric_profile="training_loss", output_policy="float_loss", metrics=metrics,
        metric_source=dict(type="csv_row", path=history.as_posix(), match={"epoch": "16"},
                           columns={k: k for k in metrics}),
        notes="完整E2从随机初始化训练16轮，全部可学习模块从第1轮参与。固定Haar仍按设计初始化。",
    ))


def train(args):
    run = args.output_dir / args.run_name
    if run.exists():
        raise FileExistsError(f"Preserve incomplete run: {run}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    trainer.set_seed(args.seed)
    run.mkdir(parents=True)
    train_loader, val_loader, saved_split, subset = trainer.build_loaders(args, device, run)
    model = trainer.DirectFusionCDDFuse(args.variant).to(device)
    losses = OvershootLosses().to(device)
    optimizer = trainer.make_optimizer(model, args)
    trainer.configure_stage(model, optimizer, "joint", args)
    scaler = trainer.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.write_json(run / "config.json", dict(
        config, total_epochs=args.joint_epochs, saved_split_manifest_sha256=protocol.sha256_file(saved_split),
        pilot_subset_sha256=protocol.sha256_file(subset),
    ))
    best = math.inf
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = None
        for epoch in range(args.joint_epochs):
            train_stats = trainer.run_epoch(model, losses, train_loader, device, args, optimizer, scaler)
            val_stats = trainer.run_epoch(model, losses, val_loader, device, args)
            row = {"epoch": epoch + 1, "stage": "joint"}
            for label, stats in (("train", train_stats), ("val", val_stats)):
                # The shared forward adds the new term to the fusion aggregate.
                stats["overshoot"] = (stats["fusion"] - stats["intensity"]
                                      - args.gradient_weight * stats["gradient"]) / WEIGHT
                row.update({f"{label}_{key}": value for key, value in stats.items()})
            if not all(math.isfinite(row[k]) for k in ("train_loss", "val_loss")):
                raise RuntimeError("Non-finite training or validation loss")
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            best = min(best, row["val_loss"])
            path = run / f"epoch_{epoch + 1:03d}.pth"
            temp = path.with_suffix(".tmp")
            trainer.save_checkpoint(temp, epoch, "joint", model, optimizer, scaler, args, best)
            temp.replace(path)
            print(f"epoch={epoch + 1}/{args.joint_epochs} train={row['train_loss']:.6f} "
                  f"val={row['val_loss']:.6f} val_overshoot={row['val_overshoot']:.6f}", flush=True)


def main():
    os.chdir(protocol.ROOT)
    args = training_args()
    if not FINAL.is_file():
        train(args)
    register_training(args)


if __name__ == "__main__":
    main()
