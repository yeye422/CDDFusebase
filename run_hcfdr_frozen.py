"""Frozen-head H-CFDR screen against the completed CF0 and frozen CF2."""

import argparse
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4

import torch

from e2_hcfdr import HCFDRE2, HeterogeneousReconstruction, VARIANT
from run_cfdr_frozen import (EPOCHS, EXTENDED_EPOCHS, PARENT, FrozenCFDRE2, checkpoint_path,
                             frozen_losses, load_backbone)
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import STAGE_LEARNING_RATES, config_values
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, verify_images, write_json
from run_tr0_factorial import write_history
from train_direct_fusion import sha256_file
import train_lwpd as trainer
from utils.dataset_os import OpticalSARDataset, manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path("models/e2_hcfdr_v1_frozen/seed42/H2")
FINAL = ROOT / f"epoch_{EPOCHS:03d}.pth"
INITIAL_RUNNER_SHA256 = "c6e1bbc6a3cbad3ce0f0c4655db8c1d2f1c5affa99988024531d87d452e4ff92"
ARCHIVE = Path("test_result/e2_cfdr_v21_frozen/val1000/epoch_002")
OUTPUT = Path("test_result/e2_hcfdr_v1_frozen/val1000/epoch_002")
POLICY = "clamp_then_uint8"
METHODS = {"H0": "光学 Base 锚定 E2", "H1": "冻结 CFDR V2.1（CF2）", "H2": "冻结 H-CFDR"}


class FrozenHCFDRE2(FrozenCFDRE2):
    def __init__(self, backbone):
        super().__init__("CF2", backbone)
        self.cfdr = HeterogeneousReconstruction(self.cfdr)
        self.variant = VARIANT
        self.train(False)


def training_protocol(args):
    baseline = checkpoint_path("CF2")
    comparable = dict(args=config_values(args), parent_checkpoint=PARENT.as_posix(),
                      parent_checkpoint_sha256=sha256_file(PARENT), split_manifest_sha256=sha256_file(MANIFEST),
                      epochs=EPOCHS, lr=STAGE_LEARNING_RATES["fusion_layers"])
    metadata = baseline.parent / "config.json"
    control = read_json(metadata)
    if any(control.get(key) != value for key, value in comparable.items()) or control.get("training_policy") != "cfdr_only_from_converged_cf0":
        raise ValueError("Archived H1 does not match the frozen H2 training protocol")
    for name, digest in control["source_sha256"].items():
        if name != "run_cfdr_frozen.py" and sha256_file(Path(name)) != digest:
            raise ValueError(f"H1 training source differs: {name}")
    return dict(comparable, arm="H2", training_policy="hcfdr_only_from_converged_cf0",
                h1_checkpoint=baseline.as_posix(), h1_checkpoint_sha256=sha256_file(baseline),
                h1_config_sha256=sha256_file(metadata), source_sha256={name: sha256_file(Path(name)) for name in (
                    "run_hcfdr_frozen.py", "e2_hcfdr.py", "run_cfdr_frozen.py", "e2_cfdr.py", "train_lwpd.py",
                )})


def h2_checkpoint(epochs):
    if epochs == EPOCHS:
        return FINAL
    return ROOT.parent / f"to_epoch_{epochs:03d}" / ROOT.name / f"epoch_{epochs:03d}.pth"


def read_h2_checkpoint(path, metadata, epoch):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if ((state["arm"], state["epoch"], state["variant"]) != ("H2", epoch, VARIANT)
            or state["config_sha256"] != sha256_file(metadata)
            or [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1))):
        raise ValueError(f"Unexpected H2 checkpoint: {path}")
    return state


def two_epoch_protocol(protocol):
    saved = read_json(ROOT / "config.json")
    original = dict(protocol, source_sha256={**protocol["source_sha256"], "run_hcfdr_frozen.py": INITIAL_RUNNER_SHA256})
    if saved not in (protocol, original):
        raise ValueError("H2 epoch-2 protocol differs from the matched recipe")
    return saved


def train(device, epochs=EPOCHS):
    backbone, args = load_backbone()
    protocol = training_protocol(args)
    final = h2_checkpoint(epochs)
    folder = final.parent
    state, inherited = None, []
    first_saved_epoch = 1
    if epochs == EXTENDED_EPOCHS:
        two_epoch_protocol(protocol)
        state = read_h2_checkpoint(FINAL, ROOT / "config.json", EPOCHS)
        if any(not torch.equal(value, state["model"][key]) for key, value in backbone.items()):
            raise ValueError("H2 epoch 2 changed the frozen backbone")
        inherited = state["history"]
        first_saved_epoch = EPOCHS + 1
        protocol.update(epochs=epochs, continuation_checkpoint=FINAL.as_posix(),
                        continuation_checkpoint_sha256=sha256_file(FINAL))
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "config.json"
    if epochs == EPOCHS and metadata.exists():
        protocol = two_epoch_protocol(protocol)
    if metadata.exists() and read_json(metadata) != protocol:
        raise ValueError(f"Existing H2 training protocol differs: {folder}")
    if not metadata.exists():
        write_json(metadata, protocol)
    config_hash = sha256_file(metadata)
    checkpoints = sorted(folder.glob("epoch_???.pth"))
    history = list(inherited)
    if checkpoints:
        current_epoch = first_saved_epoch + len(checkpoints) - 1
        state = read_h2_checkpoint(checkpoints[-1], metadata, current_epoch)
        history = state["history"]
        if (current_epoch > epochs or history[:len(inherited)] != inherited
                or [p.name for p in checkpoints] != [f"epoch_{i:03d}.pth" for i in range(first_saved_epoch, current_epoch + 1)]):
            raise ValueError("Unexpected H2 checkpoint sequence")
    if not final.is_file():
        trainer.set_seed(args.seed)
        model = FrozenHCFDRE2(backbone).to(device)
        optimizer = torch.optim.AdamW([dict(params=list(model.cfdr.parameters()), group_name="cfdr")],
                                      lr=protocol["lr"], weight_decay=args.weight_decay)
        scaler = trainer.GradScaler(enabled=True)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
        del state, backbone
        training, _, _, _ = trainer.build_loaders(args, device, folder)
        if len(training.dataset) != 8000:
            raise ValueError("Expected the original train8000")
        losses = trainer.DirectFusionLosses().to(device)
        with patch.object(trainer, "forward_losses", frozen_losses):
            for epoch in range(len(history) + 1, epochs + 1):
                trainer.set_seed(args.seed + epoch)
                training.generator.manual_seed(args.seed + epoch)
                started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                stats = trainer.run_epoch(model, losses, training, device, args, optimizer, scaler)
                if not all(math.isfinite(value) for key, value in stats.items() if not key.startswith("grad_")):
                    raise FloatingPointError(f"Nonfinite H2 statistics at epoch {epoch}")
                row = {f"train_{key}": value if math.isfinite(value) else None for key, value in stats.items()}
                row.update(epoch=epoch, train_seconds=time.perf_counter() - started,
                           train_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
                history.append(row)
                path = folder / f"epoch_{epoch:03d}.pth"
                temporary = path.with_suffix(".tmp")
                torch.save(dict(arm="H2", epoch=epoch, variant=VARIANT, args={"variant": VARIANT}, history=history,
                                config_sha256=config_hash, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scaler=scaler.state_dict()), temporary)
                temporary.replace(path)
                write_history(folder / "history.csv", history)
                print(f"H2 epoch {epoch}/{epochs}: loss={row['train_loss']:.6f}", flush=True)
    log = folder / "history.csv"
    write_history(log, history)
    entry_id = "QXS-HCFDR-V1-FROZEN-H2-SEED42-TRAIN8000"
    if epochs != EPOCHS:
        entry_id += f"-EPOCH{epochs:02d}"
    register_many([dict(id=entry_id, comparable_group=entry_id, method=METHODS["H2"], task="training",
                        status="training_complete", dataset="QXS-SAROPT", split="train", offset=0, sample_count=8000,
                        checkpoint=final.as_posix(), checkpoint_sha256=sha256_file(final),
                        parent_checkpoint=protocol.get("continuation_checkpoint", PARENT.as_posix()),
                        parent_checkpoint_sha256=protocol.get("continuation_checkpoint_sha256", protocol["parent_checkpoint_sha256"]),
                        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=protocol["split_manifest_sha256"],
                        result_dir=folder.as_posix(), metadata_path=metadata.as_posix(), metric_profile="training_loss",
                        output_policy="float_loss", metrics={"train_loss": history[-1]["train_loss"]},
                        metric_source=dict(type="csv_row", path=log.as_posix(), match={"epoch": str(epochs)},
                                           columns={"train_loss": "train_loss"}))])


def archived_controls(ids, digest, training):
    meta_path = ARCHIVE / "run_metadata.json"
    meta = read_json(meta_path)
    expected = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=EPOCHS, split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY)
    if (any(meta.get(key) != value for key, value in expected.items())
            or meta["checkpoint_hashes"]["CF0"] != training["parent_checkpoint_sha256"]
            or meta["checkpoint_hashes"]["CF2"] != training["h1_checkpoint_sha256"]
            or any(meta["source_sha256"].get(name) != sha256_file(Path(name)) for name in (
                "test_OS.py", "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py",
            ))):
        raise ValueError("The archived H0 and H1 do not match the proposed H2 evaluation")
    raw = {"H0": ARCHIVE / "CF0/metrics.csv", "H1": ARCHIVE / "CF2/evaluator_metrics.csv"}
    return {arm: read_metric_rows(path, ids) for arm, path in raw.items()}, dict(
        archived_metadata=meta_path.as_posix(), archived_metadata_sha256=sha256_file(meta_path),
        archived_metrics={arm: dict(path=path.as_posix(), sha256=sha256_file(path)) for arm, path in raw.items()},
    )


def infer(checkpoint, folder, ids, digest, device):
    import test_OS

    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST), "--split", "val",
            "--limit", str(len(ids)), "--variant", VARIANT, "--checkpoint", str(checkpoint), "--output-dir", str(folder),
            "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, {VARIANT: HCFDRE2}), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    meta = read_json(folder / "run_metadata.json")
    expected = dict(split="val", offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids), variant=VARIANT,
                    metric_profile=DEFAULT_PROFILE, split_manifest_sha256=digest, output_range_policy="clamp",
                    checkpoint_sha256=sha256_file(checkpoint), metrics_only=False)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected H2 evaluation metadata: {folder}")
    raw = (folder / "metrics.csv").rename(folder / "evaluator_metrics.csv")
    return read_metric_rows(raw, ids)


def evaluate(device):
    backbone, args = load_backbone()
    training = two_epoch_protocol(training_protocol(args))
    config_path = ROOT / "config.json"
    state = torch.load(FINAL, map_location="cpu", weights_only=True)
    if (read_json(config_path) != training or state["config_sha256"] != sha256_file(config_path)
            or (state["arm"], state["epoch"], state["variant"]) != ("H2", EPOCHS, VARIANT)
            or any(not torch.equal(value, state["model"][key]) for key, value in backbone.items())):
        raise ValueError("H2 checkpoint or frozen backbone differs from the protocol")
    del backbone, state
    ids = read_json(MANIFEST)["splits"]["val"]
    if len(ids) != 1000:
        raise ValueError("Expected the fixed val1000")
    digest = sha256_file(MANIFEST)
    results, archive = archived_controls(ids, digest, training)
    checkpoints = {"H0": PARENT.as_posix(), "H1": training["h1_checkpoint"], "H2": FINAL.as_posix()}
    hashes = {arm: sha256_file(Path(path)) for arm, path in checkpoints.items()}
    protocol = dict(split="val", scene_ids=ids, checkpoints=checkpoints, checkpoint_hashes=hashes,
                    frozen_epochs=EPOCHS, split_manifest_sha256=digest, metric_profile=DEFAULT_PROFILE,
                    output_policy=POLICY, device=str(device), archive=archive,
                    evaluator_sha256=sha256_file(Path("run_hcfdr_frozen.py")))
    metadata = OUTPUT / "run_metadata.json"
    summary = OUTPUT / "registry_metrics.json"
    if OUTPUT.exists():
        saved = read_json(metadata)
        if saved not in (protocol, dict(protocol, evaluator_sha256=INITIAL_RUNNER_SHA256)):
            raise FileExistsError(f"Existing results have different provenance: {OUTPUT}")
    else:
        work = OUTPUT.parent / f".{OUTPUT.name}-{uuid4().hex}"
        (work / "H0").mkdir(parents=True)
        (work / "H1").mkdir()
        results["H2"] = infer(FINAL, work / "H2", ids, digest, device)
        summaries = summarize(work, ids, results, (("H0", "H1"), ("H0", "H2"), ("H1", "H2")),
                              diagnostic_fields=(), source_metrics=())
        write_json(work / summary.name, summaries)
        write_json(work / metadata.name, protocol)
        work.rename(OUTPUT)
    for kind in ("gray", "rgb", "preview"):
        verify_images(OUTPUT / "H2" / kind, ids)
    summaries = read_json(summary)
    group = "QXS-HCFDR-V1-FROZEN-SEED42-VAL1000-EPOCH02-REF-FMI"
    print(f"Paired comparisons: {OUTPUT / 'paired_comparison.csv'}", flush=True)
    register_many([dict(id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split="val", offset=0,
                        sample_count=len(ids), task="fusion", status="development_validation", method=label,
                        checkpoint=checkpoints[arm], checkpoint_sha256=hashes[arm], result_dir=(OUTPUT / arm).as_posix(),
                        metadata_path=metadata.as_posix(), split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=summaries[arm],
                        metric_source=dict(type="json_paths", path=summary.as_posix(),
                                           paths={key: [arm, key] for key in summaries[arm]}))
                   for arm, label in METHODS.items()])


def smoke(device):
    backbone, args = load_backbone()
    training_protocol(args)
    splits, _ = manifest_pairs(DATA, MANIFEST)
    optical, sar, valid, scene = OpticalSARDataset(splits["val"][:1], patch_size=args.patch_size)[0]
    optical, sar, valid = (item[None].to(device) for item in (optical, sar, valid))
    trainer.set_seed(args.seed)
    model = FrozenHCFDRE2(backbone).to(device).train()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, stats = frozen_losses(model, trainer.DirectFusionLosses().to(device), optical, sar, valid, args)
    loss.backward()
    grad = model.cfdr.adaptation.weight.grad
    if (not torch.isfinite(loss) or not torch.isfinite(grad).all() or not torch.count_nonzero(grad)
            or any(p.grad is not None for name, p in model.named_parameters() if not name.startswith("cfdr."))):
        raise AssertionError("H2 adaptation or backbone freezing failed")
    print(dict(scene=scene, loss=float(loss), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               **{key: float(value) for key, value in stats.items() if key.startswith("hcfdr_")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--to-epoch", type=int, choices=(EPOCHS, EXTENDED_EPOCHS), default=EPOCHS)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched H-CFDR protocol")
    device = torch.device(args.device)
    if args.action == "smoke":
        smoke(device)
    elif args.action == "train":
        train(device, args.to_epoch)
    elif args.to_epoch == EXTENDED_EPOCHS:
        from hcfdr_comparison import evaluate_epoch_four

        evaluate_epoch_four(device)
    else:
        evaluate(device)


if __name__ == "__main__":
    main()
