"""Continue TR0 C00 Stage I into the C02 source-reconstruction control."""

import argparse
from pathlib import Path
import os

import torch

from run_tr0_factorial import (
    build_loaders, build_model, checkpoint_path, configure_stage, lock_manifests, make_optimizer,
    read_json, run_epoch, set_seed, sha256_file, write_history, write_json,
)
from train_direct_fusion import DirectFusionLosses
from tr0_evaluation import T1_ARMS, evaluate_arms, register_trainings


ARM = "C02"
ROOT = Path("models/tr0_v1/seed42")
OUTPUT = Path("test_result/tr0_v1/seed42")


def load_stage_one(config, root, manifest):
    epoch = config["reconstruction_epochs"]
    path = root / "C00" / f"epoch_{epoch:03d}.pth"
    metadata = path.parent / "config.json"
    recorded = read_json(metadata)
    state = torch.load(path, map_location="cpu", weights_only=True)
    manifest_hash = sha256_file(manifest)
    if (recorded["config"] != config or recorded["arm"] != "C00" or recorded["device"] != "cuda"
            or recorded["manifest_sha256"] != manifest_hash or state["manifest_sha256"] != manifest_hash
            or state["config_sha256"] != sha256_file(metadata)
            or (state["arm"], state["epoch"], state["stage"]) != ("C00", epoch, "I")):
        raise ValueError(f"Unexpected Stage I checkpoint or training protocol: {path}")
    for name, digest in recorded["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Training source differs from C00: {name}")
    parent = dict(checkpoint=path.as_posix(), checkpoint_sha256=sha256_file(path), epoch=epoch,
                  arm="C00", optimizer_restored=True)
    return state, parent, recorded["source_sha256"]


def training_cost(history, inherited_epochs):
    def summarize(rows):
        return dict(epochs=len(rows), optimizer_steps=sum(row["train_optimizer_steps"] for row in rows),
                    train_seconds=sum(row["train_seconds"] for row in rows),
                    val_seconds=sum(row["val_seconds"] for row in rows),
                    peak_train_mib=max(row["train_peak_allocated_mib"] for row in rows))

    return dict(equivalent_total=summarize(history), inherited=summarize(history[:inherited_epochs]),
                executed_for_c02=summarize(history[inherited_epochs:]))


def validate_history(history, stages, config):
    batch_budget = (config["train_count"] + config["batch_size"] - 1) // config["batch_size"]
    steps = (batch_budget + config["accumulation_steps"] - 1) // config["accumulation_steps"]
    if not 0 < len(history) <= len(stages) or any(
            (row["epoch"], row["stage"], row["train_optimizer_steps"]) != (index, stage, steps)
            for index, (stage, row) in enumerate(zip(stages, history), start=1)):
        raise ValueError("C02 history differs from the fixed stage or optimizer-step budget")


def train_c02(config, root, manifest, device):
    state, parent, sources = load_stage_one(config, root, manifest)
    stages = ["I"] * (config["reconstruction_epochs"] + config["proxy_epochs"]) + ["III"] * config["fusion_epochs"]
    folder = root / ARM
    metadata = folder / "config.json"
    run = dict(config=config, arm=ARM, device=str(device), parent=parent, stages=stages,
               manifest_sha256=sha256_file(manifest),
               source_sha256={**sources, "run_tr0_c02.py": sha256_file(Path(__file__))})
    folder.mkdir(parents=True, exist_ok=True)
    if metadata.is_file():
        if read_json(metadata) != run:
            raise ValueError(f"Existing C02 training settings differ: {folder}")
    else:
        write_json(metadata, run)
    config_hash = sha256_file(metadata)
    checkpoints = sorted(folder.glob("epoch_*.pth"))
    if checkpoints:
        inherited = state["history"]
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if (state["arm"] != ARM or state["config_sha256"] != config_hash
                or state["manifest_sha256"] != run["manifest_sha256"] or state["parent"] != parent
                or state["history"][:parent["epoch"]] != inherited
                or state["epoch"] != len(state["history"]) or state["stage"] != state["history"][-1]["stage"]
                or checkpoints[-1].name != f"epoch_{state['epoch']:03d}.pth"):
            raise ValueError(f"Existing C02 checkpoint provenance differs: {checkpoints[-1]}")
    history = state["history"]
    validate_history(history, stages, config)
    if len(history) < len(stages):
        set_seed(config["seed"])
        data_args = argparse.Namespace(
            **{key: config[key] for key in ("batch_size", "patch_size", "min_valid_fraction", "invalid_threshold", "seed")},
            data_root=Path(config["data_root"]), split_manifest=manifest, optical_dir="GF2", sar_dir="GF3",
            train_limit=0, val_limit=0, num_workers=0,
        )
        training, validation, _, _ = build_loaders(data_args, device, folder)
        if (len(training.dataset), len(validation.dataset)) != (config["train_count"], config["screen_count"]):
            raise ValueError("Resolved dataset sizes differ from the TR0 protocol")
        model = build_model("e2").to(device)
        losses = DirectFusionLosses().to(device)
        optimizer = make_optimizer(model, config)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        del state
        torch.backends.cudnn.deterministic = True
        for epoch in range(len(history) + 1, len(stages) + 1):
            stage = stages[epoch - 1]
            configure_stage(model, optimizer, stage, config)
            row = dict(epoch=epoch, stage=stage)
            for label, loader, update in (("train", training, optimizer), ("val", validation, None)):
                stats = run_epoch(model, losses, None, None, loader, config, stage, None, epoch, device, update)
                row.update({f"{label}_{key}": value for key, value in stats.items()})
            history.append(row)
            validate_history(history, stages, config)
            checkpoint = folder / f"epoch_{epoch:03d}.pth"
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(dict(epoch=epoch, stage=stage, arm=ARM, variant=model.variant,
                            args={"variant": model.variant}, model=model.state_dict(),
                            optimizer=optimizer.state_dict(), history=history, parent=parent,
                            manifest_sha256=run["manifest_sha256"], config_sha256=config_hash), temporary)
            temporary.replace(checkpoint)
            write_history(folder / "history.csv", history)
            print(f"C02 epoch {epoch}/{len(stages)} stage {stage}: "
                  f"train={row['train_loss']:.6f} val={row['val_loss']:.6f}", flush=True)
    write_history(folder / "history.csv", history)
    write_json(folder / "training_cost.json", training_cost(history, parent["epoch"]))
    return checkpoint_path(config, root, ARM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "screen", "confirm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    config = read_json("configs/tr0_v1.json")
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("C02 requires the CUDA BF16 setup used for C00")
    manifests = lock_manifests(config, ROOT)
    if args.action == "train":
        final = train_c02(config, ROOT, manifests["screen"], device)
        register_trainings(config, "seed42", {ARM: final}, manifests["screen"])
    else:
        evaluate_arms(config, ROOT, OUTPUT, (*T1_ARMS, ARM), manifests[args.action], args.action, device, "seed42")


if __name__ == "__main__":
    main()
