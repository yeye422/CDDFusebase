"""Resume frozen ASCD-V1 from epoch 4 and test the epoch-8 checkpoint."""

import argparse
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

import run_ascd_frozen as initial
import train_lwpd as trainer
from e2_ascd import ASCDE2
from run_ascd_frozen_test import OUTPUT as INITIAL_TEST, evaluate_checkpoint
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import read_json, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file


START = initial.EPOCHS
FINAL = 8
ROOT = initial.ROOT / "to_epoch_008"
OUTPUT = INITIAL_TEST.with_name("epoch_008")


def checkpoint(epoch):
    return ROOT / f"epoch_{epoch:03d}.pth"


def continuation_inputs():
    frontend, args, protocol = initial.training_inputs()
    parent_config = initial.ROOT / "config.json"
    if read_json(parent_config) != protocol:
        raise ValueError(f"Unexpected epoch-{START} training protocol: {parent_config}")
    parent_hash = sha256_file(parent_config)
    saved = initial.read_checkpoint(START, parent_hash, frontend)
    parent = initial.checkpoint(START)
    protocol["source_sha256"][Path(__file__).name] = sha256_file(Path(__file__))
    protocol.update(epochs=FINAL, continuation_checkpoint=parent.as_posix(),
                    continuation_checkpoint_sha256=sha256_file(parent),
                    parent_config_sha256=parent_hash)
    return frontend, args, protocol, saved


def read_continuation(epoch, config_hash, frontend, parent_history):
    path = checkpoint(epoch)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if ((saved["variant"], saved["epoch"], saved["config_sha256"]) != (ASCDE2.VARIANT, epoch, config_hash)
            or saved["history"][:START] != parent_history
            or [row["epoch"] for row in saved["history"]] != list(range(1, epoch + 1))):
        raise ValueError(f"Unexpected ASCD continuation checkpoint: {path}")
    initial.require_frozen_weights(saved["model"], frontend)
    return saved


def train(device):
    frontend, args, protocol, saved = continuation_inputs()
    ROOT.mkdir(parents=True, exist_ok=True)
    config = ROOT / "config.json"
    paths = sorted(ROOT.glob("epoch_*.pth"))
    last = START + len(paths)
    if last > FINAL or paths != [checkpoint(epoch) for epoch in range(START + 1, last + 1)]:
        raise ValueError(f"Unexpected ASCD continuation checkpoint sequence: {ROOT}")
    if not config.exists():
        if paths:
            raise ValueError(f"Missing ASCD continuation configuration: {config}")
        temporary = config.with_suffix(".tmp")
        write_json(temporary, protocol)
        temporary.replace(config)
    if read_json(config) != protocol:
        raise ValueError(f"Unexpected ASCD continuation protocol: {config}")
    config_hash = sha256_file(config)
    if paths:
        saved = read_continuation(last, config_hash, frontend, saved["history"])
    history = saved["history"]
    log = ROOT / "history.csv"
    if last < FINAL:
        model = ASCDE2().to(device)
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.AdamW([dict(params=model.ascd.parameters(), group_name="ascd")],
                                      lr=protocol["lr"], weight_decay=args.weight_decay)
        optimizer.load_state_dict(saved["optimizer"])
        scaler = trainer.GradScaler(enabled=True)
        scaler.load_state_dict(saved["scaler"])
        del saved
        training, _, _, _ = trainer.build_loaders(args, device, ROOT)
        if len(training.dataset) != 8000:
            raise ValueError("ASCD requires the original train8000")
        losses = trainer.DirectFusionLosses().to(device)
        with patch.object(trainer, "forward_losses", initial.frozen_losses):
            for epoch in range(last + 1, FINAL + 1):
                trainer.set_seed(args.seed + epoch)
                training.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, training, device, args, optimizer, scaler)
                if not all(math.isfinite(value) for name, value in stats.items() if not name.startswith("grad_")):
                    raise FloatingPointError(f"Nonfinite ASCD statistics at epoch {epoch}")
                state = model.state_dict()
                initial.require_frozen_weights(state, frontend)
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
                print(f"ASCD-V1 epoch {epoch}/{FINAL}: train={stats['loss']:.6f}", flush=True)
    write_history(log, history)
    entry_id = f"{initial.GROUP}-TRAIN8000-EPOCH{FINAL:02d}"
    register_many([dict(
        id=entry_id, comparable_group=entry_id, method="冻结 E2 前端 + ASCD-V1 续训", task="training",
        status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
        checkpoint=checkpoint(FINAL).as_posix(), checkpoint_sha256=sha256_file(checkpoint(FINAL)),
        parent_checkpoint=protocol["continuation_checkpoint"],
        parent_checkpoint_sha256=protocol["continuation_checkpoint_sha256"],
        split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
        result_dir=ROOT.as_posix(), metadata_path=config.as_posix(), metric_profile="training_loss",
        output_policy="float_loss", metrics={"train_loss": history[-1]["train_loss"]},
        metric_source=dict(type="csv_row", path=log.as_posix(), match={"epoch": str(FINAL)},
                           columns={"train_loss": "train_loss"}),
    )])


def test(device):
    frontend, _, protocol, parent = continuation_inputs()
    config = ROOT / "config.json"
    if read_json(config) != protocol:
        raise ValueError(f"Unexpected ASCD continuation protocol: {config}")
    read_continuation(FINAL, sha256_file(config), frontend, parent["history"])
    evaluate_checkpoint(device, FINAL, checkpoint(FINAL), config, protocol["split_manifest_sha256"], OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ASCD continuation")
    {"train": train, "test": test}[args.action](torch.device(args.device))


if __name__ == "__main__":
    main()
