"""Two-epoch E2 + PAC warmup with an evaluator-aligned overshoot penalty."""
import csv
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

import run_e2_pac_mamba as protocol
from run_e2_matched_no_pac import register
import train_pac_mamba as trainer


WEIGHT = 2.5
MODEL = Path("models/e2_pac_overshoot_v1")
RUN = MODEL / "seed42"


class OvershootLosses(trainer.DirectFusionLosses):
    def magnitude(self, image):
        gx = F.conv2d(image, self.sobel.weightx, padding=1)
        gy = F.conv2d(image, self.sobel.weighty, padding=1)
        return torch.linalg.vector_norm(torch.stack((gx, gy)), dim=0)

    def overshoot(self, optical, sar, fused, valid):
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                optical_mag = self.magnitude(optical.float())
                sar_mag = self.magnitude(sar.float())
                upper = torch.maximum(optical_mag, sar_mag)
                # All nine input pixels must be valid; zero padding excludes the border.
                mask = F.avg_pool2d((valid > 0.5).float(), 3, stride=1, padding=1) == 1
            excess = (self.magnitude(fused.float()) - upper).relu()
            return excess.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)

    def fusion(self, optical, sar, fused, valid, gradient_weight):
        total, intensity, gradient = super().fusion(optical, sar, fused, valid, gradient_weight)
        return total + WEIGHT * self.overshoot(optical, sar, fused, valid), intensity, gradient


def register_training():
    history = RUN / "history.csv"
    with history.open(encoding="utf-8", newline="") as f:
        row = next(row for row in csv.DictReader(f) if row["epoch"] == "2")
    metrics = {k: float(row[k]) for k in ("train_loss", "val_loss", "train_overshoot", "val_overshoot")}
    ck = RUN / "epoch_002.pth"
    register(dict(
        id="E2-PAC-OVERSHOOT-V1-SEED42-TRAIN", status="training_complete", task="training",
        method="E2 + PAC two-epoch warmup with L2 Sobel overshoot penalty (2.5)",
        comparable_group="QXS-E2-PAC-OVERSHOOT-V1-TRAIN", dataset="QXS-SAROPT", split="train",
        offset=0, sample_count=8000, split_manifest=protocol.MANIFEST.as_posix(),
        split_manifest_sha256=protocol.sha256_file(protocol.MANIFEST),
        checkpoint=ck.as_posix(), checkpoint_sha256=protocol.sha256_file(ck),
        parent_checkpoint=protocol.INIT.as_posix(), parent_checkpoint_sha256=protocol.sha256_file(protocol.INIT),
        result_dir=RUN.as_posix(), metadata_path=(RUN / "config.json").as_posix(),
        metric_profile="training_loss", output_policy="float_loss", metrics=metrics,
        metric_source=dict(type="csv_row", path=history.as_posix(), match={"epoch": "2"},
                           columns={k: k for k in metrics}),
        notes="两轮热身，编码器冻结。原梯度损失保留，新增有效SAR邻域内的L2 Sobel单侧过冲惩罚。",
    ))


def train():
    if (RUN / "epoch_002.pth").is_file():
        register_training()
        return
    if RUN.exists():
        raise FileExistsError(f"Preserve incomplete run: {RUN}")
    argv = protocol.train_args()
    argv[argv.index("--output-dir") + 1] = str(MODEL)
    with patch.object(sys, "argv", argv):
        args = trainer.parse_args()
    args.local_epochs = args.joint_epochs = 0
    args.overshoot_weight = WEIGHT
    trainer.set_seed(args.seed)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched warmup")
    RUN.mkdir(parents=True)
    train_loader, val_loader, saved_split, subset_path = trainer.build_loaders(args, device, RUN)
    model = trainer.PACMambaE2CDDFuse().to(device)
    model.load_lwpd_checkpoint(torch.load(args.init_checkpoint, map_location="cpu", weights_only=False))
    losses = OvershootLosses().to(device)
    optimizer = trainer.make_optimizer(model, args)
    trainer.configure_stage(model, optimizer, "warmup", args)
    scaler = trainer.GradScaler(enabled=not args.no_amp)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.write_json(RUN / "config.json", dict(
        config, variant=model.VARIANT, total_epochs=args.warmup_epochs,
        init_checkpoint_sha256=protocol.sha256_file(args.init_checkpoint),
        saved_split_manifest_sha256=protocol.sha256_file(saved_split),
        pilot_subset_sha256=protocol.sha256_file(subset_path),
    ))
    best = float("inf")
    with (RUN / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = None
        for epoch in range(args.warmup_epochs):
            row = {"epoch": epoch + 1, "stage": "warmup"}
            for label, stats in (
                ("train", trainer.run_epoch(model, losses, train_loader, device, args, optimizer, scaler)),
                ("val", trainer.run_epoch(model, losses, val_loader, device, args)),
            ):
                # The original fusion total now includes the weighted extra term.
                stats["overshoot"] = (stats["fusion"] - stats["intensity"]
                                      - args.gradient_weight * stats["gradient"]) / WEIGHT
                row.update({f"{label}_{key}": value for key, value in stats.items()})
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            best = min(best, row["val_loss"])
            path = RUN / f"epoch_{epoch + 1:03d}.pth"
            temp = path.with_suffix(".tmp")
            trainer.save_checkpoint(temp, epoch, "warmup", model, optimizer, scaler, args, best)
            temp.replace(path)
            print(f"epoch={epoch+1}/2 train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
                  f"val_overshoot={row['val_overshoot']:.6f}", flush=True)
    register_training()


if __name__ == "__main__":
    os.chdir(protocol.ROOT)
    train()
