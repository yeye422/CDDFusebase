"""Train and evaluate the static C05-F fidelity-companion experiment."""

import argparse
import os
from pathlib import Path

import torch

from run_tr0_hcsl import OUTPUT as TR0_OUTPUT, ROOT as TR0_ROOT, load_parent
from run_tr0_factorial import (
    build_loaders, build_model, checkpoint_path, configure_stage, forward_losses,
    lock_manifests, make_optimizer, precision_context, read_json, run_epoch, set_seed,
    sha256_file, write_history, write_json,
)
from train_direct_fusion import DirectFusionLosses
from tr0_complementary import ComplementaryReconstructionLoss
from tr0_evaluation import FPCL_ARM, FPCL_ARMS, evaluate_arms, register_trainings
from tr0_fidelity import FidelityViewGenerator, task_assignments
from utils.dataset_os import OpticalSARDataset, manifest_pairs


ROOT = Path("models/tr0_fpcl_v1/seed42")
OUTPUT = Path("test_result/tr0_fpcl_v1/seed42")


def train_static(config, fpcl, root, manifest, device):
    state, parent, sources = load_parent(config, TR0_ROOT, manifest)
    stages = ["I"] * config["reconstruction_epochs"] + ["II"] * config["proxy_epochs"] + ["III"] * config["fusion_epochs"]
    folder = root / FPCL_ARM
    folder.mkdir(parents=True, exist_ok=True)
    assignment = task_assignments(config, fpcl)
    tasks = folder / "proxy_task_manifest.json"
    if tasks.exists() and read_json(tasks) != assignment:
        raise ValueError(f"Existing proxy task manifest differs: {tasks}")
    if not tasks.exists():
        write_json(tasks, assignment)
    metadata = folder / "config.json"
    run = dict(config=config, arm=FPCL_ARM, device=str(device), parent=parent, stages=stages, fpcl=fpcl,
               task_manifest=tasks.as_posix(), task_manifest_sha256=sha256_file(tasks),
               manifest_sha256=sha256_file(manifest), source_sha256={**sources, **{
                   name: sha256_file(Path(name)) for name in ("run_tr0_fpcl.py", "tr0_fidelity.py", "run_tr0_hcsl.py")
               }})
    if metadata.exists() and read_json(metadata) != run:
        raise ValueError(f"Existing C05-F training settings differ: {folder}")
    if not metadata.exists():
        write_json(metadata, run)
    config_hash = sha256_file(metadata)
    checkpoints = sorted(folder.glob("epoch_*.pth"))
    if checkpoints:
        inherited = state["history"]
        path = checkpoints[-1]
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["arm"] != FPCL_ARM or state["parent"] != parent or state["config_sha256"] != config_hash
                or state["manifest_sha256"] != run["manifest_sha256"] or state["history"][:parent["epoch"]] != inherited
                or state["epoch"] != len(state["history"]) or state["stage"] != state["history"][-1]["stage"]
                or path.name != f"epoch_{state['epoch']:03d}.pth"):
            raise ValueError(f"Existing C05-F checkpoint provenance differs: {path}")
    history = state["history"]
    steps = history[0]["train_optimizer_steps"]
    if len(history) > len(stages) or any(
        (row["epoch"], row["stage"], row["train_optimizer_steps"]) != (epoch, stage, steps)
        for epoch, (row, stage) in enumerate(zip(history, stages), start=1)
    ):
        raise ValueError("C05-F history differs from the fixed stage or optimizer-step budget")
    if len(history) < len(stages):
        set_seed(config["seed"])
        data_args = argparse.Namespace(
            **{key: config[key] for key in ("batch_size", "patch_size", "min_valid_fraction", "invalid_threshold", "seed")},
            data_root=Path(config["data_root"]), split_manifest=manifest, optical_dir="GF2", sar_dir="GF3",
            train_limit=0, val_limit=0, num_workers=0,
        )
        train_loader, val_loader, _, _ = build_loaders(data_args, device, folder)
        if (len(train_loader.dataset), len(val_loader.dataset)) != (config["train_count"], config["screen_count"]):
            raise ValueError("Resolved dataset sizes differ from the TR0 protocol")
        model = build_model("e2").to(device)
        optimizer = make_optimizer(model, config)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        del state
        losses = DirectFusionLosses().to(device)
        proxy = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
        views = FidelityViewGenerator(config["view_generator"], assignment)
        torch.backends.cudnn.deterministic = True
        for epoch in range(len(history) + 1, len(stages) + 1):
            stage = stages[epoch - 1]
            configure_stage(model, optimizer, stage, config)
            row = dict(epoch=epoch, stage=stage)
            for label, loader, update in (("train", train_loader, optimizer), ("val", val_loader, None)):
                stats = run_epoch(model, losses, proxy, views, loader, config, stage, "complementary", epoch, device, update)
                row.update({f"{label}_{name}": value for name, value in stats.items()})
            if row["train_optimizer_steps"] != steps:
                raise ValueError("C05-F update count differs from the inherited schedule")
            if stage == "II" and any(row[f"{label}_identity_fraction"] != fpcl["identity_fraction"]
                                     for label in ("train", "val")):
                raise ValueError("C05-F proxy task counts differ from the saved list")
            history.append(row)
            path = folder / f"epoch_{epoch:03d}.pth"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(epoch=epoch, stage=stage, arm=FPCL_ARM, variant=model.variant,
                            args={"variant": model.variant}, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            history=history, parent=parent, manifest_sha256=run["manifest_sha256"],
                            config_sha256=config_hash), temporary)
            temporary.replace(path)
            write_history(folder / "history.csv", history)
            print(f"{FPCL_ARM} epoch {epoch}/{len(stages)} stage {stage}: "
                  f"train={row['train_loss']:.6f} val={row['val_loss']:.6f}", flush=True)
    write_history(folder / "history.csv", history)
    return checkpoint_path(config, root, FPCL_ARM)


def smoke(config, fpcl, manifest, device):
    state, _, _ = load_parent(config, TR0_ROOT, manifest)
    assignment = task_assignments(config, fpcl)
    if assignment != task_assignments(config, fpcl) or any(
        len(plan["identity_positions"]) != plan["samples"] * fpcl["identity_fraction"] for plan in assignment.values()
    ):
        raise AssertionError("Task assignments are not reproducible at the declared budget")
    splits, _ = manifest_pairs(Path(config["data_root"]), manifest)
    optical, sar, valid, scene = OpticalSARDataset(splits["val"][:1], patch_size=config["patch_size"], augment=False)[0]
    optical, sar, valid = (tensor.unsqueeze(0).to(device) for tensor in (optical, sar, valid))
    generator_seed = config["proxy_validation_seed"]
    # Route the same scene through both tasks to check exact identity and V1 reuse.
    test_schedule = {str(generator_seed): dict(samples=2, identity_positions=[0])}
    views = FidelityViewGenerator(config["view_generator"], test_schedule)
    source = optical.expand(2, -1, -1, -1)
    first, second, _, stats = views(source, torch.Generator().manual_seed(generator_seed))
    v1_first, v1_second, _, _ = views.complementary(source, torch.Generator().manual_seed(generator_seed))
    for actual, expected in ((first[0], optical[0]), (second[0], optical[0]),
                             (first[1], v1_first[1]), (second[1], v1_second[1])):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if float(stats["identity_fraction"]) != 0.5:
        raise AssertionError("The two-task smoke schedule was not consumed")
    model = build_model("e2").to(device)
    optimizer = make_optimizer(model, config)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if not optimizer.state:
        raise AssertionError("C01 optimizer state is empty")
    configure_stage(model, optimizer, "II", config)
    losses = DirectFusionLosses().to(device)
    proxy = ComplementaryReconstructionLoss(config["proxy_gradient_weight"])
    for index, task in enumerate(("identity", "complementary")):
        optimizer.zero_grad(set_to_none=True)
        pair = (first[index:index + 1], second[index:index + 1])
        with precision_context(config, device):
            loss, _ = forward_losses(model, losses, proxy, config, "II", optical, sar, valid,
                                     proxy_pair=pair, checkpointing=config["activation_checkpointing"])
        loss.backward()
        if not torch.isfinite(loss) or not any(
            p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
            for p in model.detail_fusion.parameters()
        ):
            raise AssertionError(f"Invalid Stage II gradient for {task}")
        print(dict(scene=scene, task=task, loss=float(loss)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "screen", "confirm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    config = read_json("configs/tr0_v1.json")
    fpcl = read_json("configs/tr0_fpcl_v1.json")
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("C05-F requires the CUDA BF16 setup used for C01")
    if args.action == "smoke":
        smoke(config, fpcl, lock_manifests(config, TR0_ROOT)["screen"], device)
        return
    manifests = lock_manifests(config, ROOT)
    if args.action == "train":
        final = train_static(config, fpcl, ROOT, manifests["screen"], device)
        register_trainings(config, "seed42", {FPCL_ARM: final}, manifests["screen"])
    else:
        evaluate_arms(config, ROOT, OUTPUT, FPCL_ARMS, manifests[args.action], args.action, device, "seed42",
                      fpcl=fpcl, archive_root=TR0_ROOT, archive_output=TR0_OUTPUT)


if __name__ == "__main__":
    main()
