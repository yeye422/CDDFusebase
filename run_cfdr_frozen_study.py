"""Continue CF1, CF2 and H2 together and evaluate each matched frozen milestone."""

import argparse
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

import cfdr_evaluation as evaluation
from e2_cfdr import VARIANTS
from e2_hcfdr import VARIANT as H2_VARIANT
from hcfdr_comparison import EVALUATOR_FILES, METHODS, OUTPUT as EPOCH_FOUR_OUTPUT, evaluate_epoch_four, reuse_metrics
import run_cfdr_frozen as cfdr
import run_hcfdr_frozen as hcfd
from run_e2_aspr_pilot import write_csv
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import MANIFEST, read_json, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file
import train_lwpd as trainer


ARMS = ("CF1", "CF2", "H2")
MILESTONES = (4, 6, 8)
OUTPUT = Path("test_result/e2_hcfdr_v1_frozen/val1000")


def checkpoint(arm, epoch):
    return hcfd.h2_checkpoint(epoch) if arm == "H2" else cfdr.checkpoint_path(arm, epoch)


def variant(arm):
    return H2_VARIANT if arm == "H2" else VARIANTS[arm]


def comparison_dir(epoch):
    return OUTPUT / f"epoch_{epoch:03d}_joint_v1"


def read_state(arm, epoch, backbone, args):
    path = checkpoint(arm, epoch)
    metadata = path.parent / "config.json"
    config = read_json(metadata)
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = dict(arm=arm, epochs=epoch, args=cfdr.config_values(args),
                    parent_checkpoint_sha256=sha256_file(cfdr.PARENT),
                    split_manifest_sha256=sha256_file(MANIFEST), lr=cfdr.STAGE_LEARNING_RATES["fusion_layers"])
    if epoch > 2:
        previous = checkpoint(arm, epoch - 2)
        expected.update(continuation_checkpoint=previous.as_posix(), continuation_checkpoint_sha256=sha256_file(previous))
    if (any(config.get(key) != value for key, value in expected.items())
            or (state["arm"], state["epoch"], state["variant"], state["config_sha256"])
            != (arm, epoch, variant(arm), sha256_file(metadata))
            or [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1))
            or any(not torch.equal(value, state["model"][key]) for key, value in backbone.items())):
        raise ValueError(f"Checkpoint differs from its matched frozen recipe: {path}")
    return state, config


def train_to(arm, epochs, device):
    backbone, args = cfdr.load_backbone()
    parent_epoch = epochs - 2
    state, parent_config = read_state(arm, parent_epoch, backbone, args)
    inherited = state["history"]
    final = checkpoint(arm, epochs)
    folder, parent = final.parent, checkpoint(arm, parent_epoch)
    metadata = folder / "config.json"
    sources = {name: sha256_file(Path(name)) for name in (
        "run_cfdr_frozen_study.py", "run_cfdr_frozen.py", "run_hcfdr_frozen.py", "e2_cfdr.py",
        "e2_hcfdr.py", "train_lwpd.py", "train_direct_fusion.py", "utils/dataset_os.py",
        "e2_base_anchor.py", "direct_fusion.py", "net.py", "lwpd.py",
    )}
    protocol = dict(parent_config, epochs=epochs, continuation_checkpoint=parent.as_posix(),
                    continuation_checkpoint_sha256=sha256_file(parent), source_sha256=sources)
    folder.mkdir(parents=True, exist_ok=True)
    if metadata.exists() and read_json(metadata) != protocol:
        raise ValueError(f"Existing continuation protocol differs: {folder}")
    if not metadata.exists():
        write_json(metadata, protocol)
    config_hash = sha256_file(metadata)
    history = list(inherited)
    checkpoints = sorted(folder.glob("epoch_???.pth"))
    if checkpoints:
        latest = parent_epoch + len(checkpoints)
        if latest > epochs or [p.name for p in checkpoints] != [
            f"epoch_{i:03d}.pth" for i in range(parent_epoch + 1, latest + 1)
        ]:
            raise ValueError(f"Unexpected checkpoint sequence: {folder}")
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        history = state["history"]
        if ((state["arm"], state["epoch"], state["variant"], state["config_sha256"])
                != (arm, latest, variant(arm), config_hash) or history[:parent_epoch] != inherited
                or [row["epoch"] for row in history] != list(range(1, latest + 1))
                or any(not torch.equal(value, state["model"][key]) for key, value in backbone.items())):
            raise ValueError(f"Unexpected continuation checkpoint: {checkpoints[-1]}")
    if not final.is_file():
        trainer.set_seed(args.seed)
        model = (hcfd.FrozenHCFDRE2(backbone) if arm == "H2" else cfdr.FrozenCFDRE2(arm, backbone)).to(device)
        model.load_state_dict(state["model"])
        optimizer = torch.optim.AdamW([dict(params=list(model.cfdr.parameters()), group_name="cfdr")],
                                      lr=protocol["lr"], weight_decay=args.weight_decay)
        optimizer.load_state_dict(state["optimizer"])
        scaler = trainer.GradScaler(enabled=True)
        scaler.load_state_dict(state["scaler"])
        del state, backbone
        loader, _, _, _ = trainer.build_loaders(args, device, folder)
        if len(loader.dataset) != 8000:
            raise ValueError("Expected the locked train8000")
        losses = trainer.DirectFusionLosses().to(device)
        original_norms = trainer.group_gradient_norms

        def head_norms(optimizer):
            norms = original_norms(optimizer)
            for name, module in model.cfdr.named_children():
                squares = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
                if squares:
                    norms[f"head_{name}"] = float(torch.stack(squares).sum().sqrt())
            return norms

        with patch.object(trainer, "forward_losses", cfdr.frozen_losses), patch.object(trainer, "group_gradient_norms", head_norms):
            for epoch in range(len(history) + 1, epochs + 1):
                trainer.set_seed(args.seed + epoch)
                loader.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, loader, device, args, optimizer, scaler)
                if not all(math.isfinite(value) for key, value in stats.items() if not key.startswith("grad_")):
                    raise FloatingPointError(f"Nonfinite {arm} statistics at epoch {epoch}")
                row = {f"train_{key}": value if math.isfinite(value) else None for key, value in stats.items()}
                row.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_seconds=time.perf_counter() - started,
                           train_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
                history.append(row)
                path = folder / f"epoch_{epoch:03d}.pth"
                temporary = path.with_suffix(".tmp")
                torch.save(dict(arm=arm, epoch=epoch, variant=model.variant, args={"variant": model.variant},
                                history=history, config_sha256=config_hash, model=model.state_dict(),
                                optimizer=optimizer.state_dict(), scaler=scaler.state_dict()), temporary)
                temporary.replace(path)
                write_history(folder / "history.csv", history)
                print(f"Frozen {arm} epoch {epoch}/{epochs}: loss={row['train_loss']:.6f}", flush=True)
    log = folder / "history.csv"
    write_history(log, history)
    group = f"QXS-CFDR-STUDY-FROZEN-{arm}-SEED42-TRAIN8000-EPOCH{epochs:02d}"
    return dict(id=group, comparable_group=group, method=f"{METHODS[arm]} 冻结续训", task="training",
                status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
                checkpoint=final.as_posix(), checkpoint_sha256=sha256_file(final), parent_checkpoint=parent.as_posix(),
                parent_checkpoint_sha256=protocol["continuation_checkpoint_sha256"], split_manifest=MANIFEST.as_posix(),
                split_manifest_sha256=protocol["split_manifest_sha256"], result_dir=folder.as_posix(),
                metadata_path=metadata.as_posix(), metric_profile="training_loss", output_policy="float_loss",
                metrics={"train_loss": history[-1]["train_loss"]}, metric_source=dict(type="csv_row", path=log.as_posix(),
                    match={"epoch": str(epochs)}, columns={"train_loss": "train_loss"}))


def evaluate_milestone(epochs, device):
    backbone, args = cfdr.load_backbone()
    histories = {arm: read_state(arm, epochs, backbone, args)[0]["history"] for arm in ARMS}
    del backbone
    ids = read_json(MANIFEST)["splits"]["val"]
    if len(ids) != 1000:
        raise ValueError("Expected the locked val1000")
    digest = sha256_file(MANIFEST)
    base_hash = sha256_file(cfdr.PARENT)
    checkpoints = {"CF0": cfdr.PARENT, **{arm: checkpoint(arm, epochs) for arm in ARMS}}
    methods, cached, archive = dict(METHODS), {}, {}
    prior = comparison_dir(epochs - 2)
    prior_hashes = {"CF0": base_hash, **{arm: sha256_file(checkpoint(arm, epochs - 2)) for arm in ARMS}}
    metadata = reuse_metrics(prior, ids, digest, prior_hashes, epochs - 2)
    previous = read_json(metadata)
    for name, raw_checkpoint in previous["checkpoints"].items():
        if name == "CF0":
            continue
        archive_key = f"{name}_E{epochs - 2:02d}" if name in ARMS else name
        path = Path(raw_checkpoint)
        if sha256_file(path) != previous["checkpoint_hashes"][name]:
            raise ValueError(f"Archived series checkpoint differs: {path}")
        checkpoints[archive_key] = path
        arm, round_name = archive_key.split("_E")
        methods[archive_key] = f"{METHODS[arm]} 第 {int(round_name)} 轮"
        metrics = prior / name / "metrics.csv"
        cached[archive_key] = read_metric_rows(metrics, ids)
        archive[archive_key] = dict(metrics=metrics.as_posix(), sha256=sha256_file(metrics))
    baseline = read_metric_rows(prior / "CF0/metrics.csv", ids)
    protocol = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=epochs, device=str(device),
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                    metric_profile=evaluation.DEFAULT_PROFILE, output_policy=evaluation.POLICY,
                    checkpoint_hashes={arm: sha256_file(path) for arm, path in checkpoints.items()},
                    checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()},
                    archive_cf0=dict(metrics=(prior / "CF0/metrics.csv").as_posix(),
                                     sha256=sha256_file(prior / "CF0/metrics.csv")),
                    archive_metadata=dict(path=metadata.as_posix(), sha256=sha256_file(metadata)), cached=archive,
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "run_cfdr_frozen_study.py", "e2_cfdr.py", "e2_hcfdr.py", *EVALUATOR_FILES,
                    )})
    contrasts = (("CF0", "CF1"), ("CF0", "CF2"), ("CF0", "H2"), ("CF1", "CF2"), ("CF1", "H2"), ("CF2", "H2"),
                 *((f"{arm}_E{round_:02d}", arm) for round_ in (2, epochs - 2) for arm in ARMS))

    def infer(arm, path, folder, split, ids, digest, device):
        if arm == "H2":
            return hcfd.infer(path, folder, ids, digest, device)
        return evaluation.infer(arm, path, folder, split, ids, digest, device)

    output = comparison_dir(epochs)
    evaluation.evaluate_group(output, f"QXS-CFDR-STUDY-FROZEN-SEED42-VAL1000-EPOCH{epochs:02d}-REF-FMI", "val",
                              device, checkpoints, protocol, baseline, methods=methods, contrasts=contrasts,
                              cached=cached, infer_method=infer)
    write_curves(output, histories, epochs)


def write_curves(output, histories, epochs):
    lr = cfdr.STAGE_LEARNING_RATES["fusion_layers"]
    rows = [{"arm": arm, "lr": lr, **row} for arm, history in histories.items() for row in history]
    fields = ["arm", "epoch", "lr", *dict.fromkeys(key for row in rows for key in row if key not in ("arm", "epoch", "lr"))]
    write_csv(output / "training_curves.csv", fields, ([row.get(key) for key in fields] for row in rows))
    stats = read_json(output / "registry_metrics.json")
    excluded = read_json(output / "excluded_scene_ids.json")
    metrics = tuple(excluded)
    visual_rows = [("CF0", 0, *[stats["CF0"].get(metric) for metric in metrics])]
    for epoch in range(2, epochs + 1, 2):
        for arm in ARMS:
            values = stats[arm if epoch == epochs else f"{arm}_E{epoch:02d}"]
            visual_rows.append((arm, epoch, *(values.get(metric) for metric in metrics)))
    visual_rows.append(("共同有效场景数", "", *(1000 - len(excluded[metric]) for metric in metrics)))
    write_csv(output / "validation_curves.csv", ("arm", "epoch", *metrics), visual_rows)


def run(device, target):
    for epoch in MILESTONES:
        if epoch > target:
            break
        print(f"=== Frozen CFDR milestone {epoch}: CF1, CF2, H2 and val1000 ===", flush=True)
        if epoch == 4:
            register_many([cfdr.train_arm(arm, device, epoch) for arm in cfdr.ARMS])
            hcfd.train(device, epoch)
            evaluate_epoch_four(device)
            histories = {arm: torch.load(checkpoint(arm, epoch), map_location="cpu", weights_only=True)["history"] for arm in ARMS}
            write_curves(EPOCH_FOUR_OUTPUT, histories, epoch)
        else:
            register_many([train_to(arm, epoch, device) for arm in ARMS])
            evaluate_milestone(epoch, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to-epoch", type=int, choices=MILESTONES, default=8)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CFDR study")
    run(torch.device(args.device), args.to_epoch)


if __name__ == "__main__":
    main()
