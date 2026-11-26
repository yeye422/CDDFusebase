"""Train CFDR from the archived optical-Base E2 reconstruction checkpoint."""

import argparse
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cfdr import ARMS, CFDRE2
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import STAGE_LEARNING_RATES, configure, config_values
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file
import train_lwpd as trainer
from utils.dataset_os import OpticalSARDataset, manifest_pairs


BASELINE = Path("models/e2_scratch_base_anchor_v1/seed42")
ROOT = Path("models/e2_cfdr_v21/seed42")
PARENT = BASELINE / "epoch_004.pth"
VALIDATION_EPOCHS = (4, 8, 12)


def checkpoint_path(arm, fusion_epoch):
    folder = BASELINE if arm == "CF0" else ROOT / arm
    return folder / f"epoch_{4 + fusion_epoch:03d}.pth"


def load_parent():
    # The historical local checkpoint contains pathlib values in its argument dict.
    state = torch.load(PARENT, map_location="cpu", weights_only=False)
    config = read_json(BASELINE / "config.json")
    args = argparse.Namespace(**state["args"])
    if ((state["epoch"], state["stage"], state["variant"]) != (3, "reconstruction", OpticalBaseAnchoredE2.VARIANT)
            or any(config.get(key) != value for key, value in config_values(args).items())
            or Path(args.split_manifest).resolve() != MANIFEST.resolve()
            or (args.seed, args.reconstruction_epochs, args.joint_epochs, args.gradient_loss,
                args.gradient_weight, args.reconstruction_weight, args.structure_weight, args.no_amp)
            != (42, 4, 12, "baseline", 10.0, 0.1, 0.0, False)
            or config["stage_learning_rates"] != STAGE_LEARNING_RATES):
        raise ValueError("Unexpected CF0 reconstruction checkpoint or training protocol")
    args.data_root, args.split_manifest = DATA, MANIFEST
    return state, args


def initialize(arm, device, parent, args):
    trainer.set_seed(args.seed)
    model = CFDRE2(arm).to(device)
    missing, unexpected = model.load_state_dict(parent["model"], strict=False)
    if set(missing) != {name for name in model.state_dict() if name.startswith("cfdr.")} or unexpected:
        raise ValueError("CF0 weights do not match the CFDR backbone")
    optimizer = trainer.make_optimizer(model, args)
    optimizer.load_state_dict(parent["optimizer"])
    # Retain the six archived optimizer groups and their state. The new heads
    # take the fusion-layer learning rate and start with empty AdamW state.
    detail_group = next(group for group in optimizer.param_groups if group["group_name"] == "detail_fusion")
    detail_group["params"].extend(model.cfdr.parameters())
    scaler = trainer.GradScaler(enabled=True)
    scaler.load_state_dict(parent["scaler"])
    configure(model, optimizer, args, reconstruction=False)
    return model, optimizer, scaler


def forward_losses(model, losses, optical, sar, valid, args):
    encoded, diagnostics = model.encode_pair(optical, sar, return_diagnostics=True)
    ob, od, sb, sd = encoded
    optical_rec = losses.reconstruction(optical, model.reconstruct(optical, ob, od))
    sar_rec = losses.reconstruction(sar, model.reconstruct(sar, sb, sd), valid)
    reconstruction = 0.5 * (optical_rec + sar_rec)
    fused, stats = model.fuse_and_reconstruct(optical, sar, encoded, return_stats=True)
    fusion, intensity, gradient = losses.fusion(optical, sar, fused, valid, args.gradient_weight)
    return fusion + args.reconstruction_weight * reconstruction, dict(
        fusion=fusion, intensity=intensity, gradient=gradient, reconstruction=reconstruction,
        optical_reconstruction=optical_rec, sar_reconstruction=sar_rec,
        gate_mean=0.5 * (diagnostics["optical_gates"].mean() + diagnostics["sar_gates"].mean()), **stats,
    )


def train_arm(arm, device):
    parent, args = load_parent()
    folder = ROOT / arm
    final = checkpoint_path(arm, 12)
    metadata = folder / "config.json"
    protocol = dict(arm=arm, args=config_values(args), split_manifest_sha256=sha256_file(MANIFEST),
                    parent_checkpoint=PARENT.as_posix(), parent_checkpoint_sha256=sha256_file(PARENT),
                    stage_learning_rates=STAGE_LEARNING_RATES, stage_ii_epochs=12,
                    data_order="seed + global epoch; same for CF1 and CF2",
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "e2_cfdr.py", "run_cfdr.py", "train_lwpd.py", "train_direct_fusion.py",
                        "run_e2_scratch_twostage.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py",
                        "utils/dataset_os.py",
                    )})
    folder.mkdir(parents=True, exist_ok=True)
    if metadata.is_file() and read_json(metadata) != protocol:
        raise ValueError(f"Existing run protocol differs: {folder}")
    if not metadata.exists():
        write_json(metadata, protocol)
    config_hash = sha256_file(metadata)
    history = []
    checkpoints = sorted(folder.glob("epoch_???.pth"))
    state = None
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        history = state["history"]
        if (state["arm"] != arm or state["config_sha256"] != config_hash
                or state["epoch"] != len(history) + 3
                or [path.name for path in checkpoints] != [f"epoch_{i:03d}.pth" for i in range(5, len(history) + 5)]):
            raise ValueError(f"Unexpected CFDR checkpoint or history: {checkpoints[-1]}")
    if not final.is_file():
        model, optimizer, scaler = initialize(arm, device, parent, args)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
        del parent, state
        training, validation, _, _ = trainer.build_loaders(args, device, folder)
        if (len(training.dataset), len(validation.dataset)) != (8000, 1000):
            raise ValueError("CFDR requires the original train8000 and val1000")
        losses = trainer.DirectFusionLosses().to(device)
        with patch.object(trainer, "forward_losses", forward_losses):
            for epoch in range(5 + len(history), 17):
                configure(model, optimizer, args, reconstruction=False)
                trainer.set_seed(args.seed + epoch)
                training.generator.manual_seed(args.seed + epoch)
                row = dict(epoch=epoch, fusion_epoch=epoch - 4)
                for label, loader, opt in (("train", training, optimizer), ("val", validation, None)):
                    started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)
                    stats = trainer.run_epoch(model, losses, loader, device, args, opt, scaler)
                    if not all(math.isfinite(value) for key, value in stats.items() if not key.startswith("grad_")):
                        raise FloatingPointError(f"Nonfinite {arm} {label} statistics at epoch {epoch}")
                    row.update({f"{label}_{key}": value if math.isfinite(value) else None for key, value in stats.items()},
                               **{f"{label}_seconds": time.perf_counter() - started,
                                  f"{label}_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20})
                history.append(row)
                path = checkpoint_path(arm, epoch - 4)
                temp = path.with_suffix(".tmp")
                torch.save(dict(arm=arm, epoch=epoch - 1, stage="fusion", variant=model.variant,
                                args={"variant": model.variant}, config_sha256=config_hash, history=history,
                                model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), temp)
                temp.replace(path)
                write_history(folder / "history.csv", history)
                print(f"{arm} fusion epoch {epoch - 4}/12: train={row['train_loss']:.6f} val={row['val_loss']:.6f}", flush=True)
    write_history(folder / "history.csv", history)
    metrics = {name: history[-1][name] for name in ("train_loss", "val_loss")}
    return dict(id=f"QXS-CFDR-V21-{arm}-SEED42-TRAIN8000", comparable_group=f"QXS-CFDR-V21-{arm}-TRAIN8000",
                method=f"CFDR-V2.1 {arm}", task="training", status="training_complete", dataset="QXS-SAROPT",
                split="train", offset=0, sample_count=8000, checkpoint=final.as_posix(),
                checkpoint_sha256=sha256_file(final), parent_checkpoint=PARENT.as_posix(),
                parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"], split_manifest=MANIFEST.as_posix(),
                split_manifest_sha256=protocol["split_manifest_sha256"], result_dir=folder.as_posix(),
                metadata_path=metadata.as_posix(), metric_profile="training_loss", output_policy="float_loss",
                metrics=metrics, metric_source=dict(type="csv_row", path=(folder / "history.csv").as_posix(),
                                                    match={"epoch": "16"}, columns={key: key for key in metrics}))


def smoke(device):
    parent, args = load_parent()
    splits, _ = manifest_pairs(DATA, MANIFEST)
    optical, sar, valid, scene = OpticalSARDataset(splits["val"][:1], patch_size=args.patch_size)[0]
    optical, sar, valid = (value[None].to(device) for value in (optical, sar, valid))
    losses = trainer.DirectFusionLosses().to(device)
    for arm in ARMS:
        model, optimizer, scaler = initialize(arm, device, parent, args)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, stats = forward_losses(model, losses, optical, sar, valid, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        heads = (model.cfdr.gain, model.cfdr.neighbors, model.cfdr.strength) if arm == "CF2" else (model.cfdr.gain,)
        if not torch.isfinite(loss) or not all(torch.isfinite(p.weight.grad).all() and torch.count_nonzero(p.weight.grad)
                                               for p in heads):
            raise AssertionError(f"Invalid CFDR gradients: {arm}")
        with torch.inference_mode():
            output = model(optical[..., :31, :33], sar[..., :31, :33])
            if output.shape != (1, 1, 31, 33) or not torch.isfinite(output).all():
                raise AssertionError(f"Unexpected odd-size output: {arm}")
        print(dict(arm=arm, scene=scene, loss=float(loss), parameters=sum(p.numel() for p in model.parameters()),
                   **{key: float(value) for key, value in stats.items() if key.startswith("cfdr_")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val", "test"))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--fusion-epochs", choices=VALIDATION_EPOCHS, nargs="+", type=int, default=(12,))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched CFDR protocol")
    device = torch.device(args.device)
    if args.action == "smoke":
        smoke(device)
    elif args.action == "train":
        register_many([train_arm(arm, device) for arm in dict.fromkeys(args.arms)])
    else:
        from cfdr_evaluation import evaluate

        if args.action == "test" and tuple(args.fusion_epochs) != (12,):
            parser.error("test uses the fixed final checkpoint")
        for epoch in dict.fromkeys(args.fusion_epochs):
            evaluate(args.action, epoch, device)


if __name__ == "__main__":
    main()
