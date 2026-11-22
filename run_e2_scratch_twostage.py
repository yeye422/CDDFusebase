"""Train and validate a scratch LWPD E2 baseline using two stages."""
import argparse
import csv
from dataclasses import dataclass
from functools import partial
import io
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, quantiles
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import e2_aspr
import e2_polarity_fusion
import e2_rccr
import e2_sar_directional
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_base_anchor_pac import OpticalBaseAnchoredPACE2
from e2_source_interval_loss import INTERVAL, INTERVAL_STATS, SourceIntervalFusionLosses
from e2_structure_joint import GEOMETRY, ROUTING_STATS, StructureGuidedE2
from e2_structure_joint import forward_losses as structure_joint_forward_losses
import run_e2_pac_mamba as protocol
from run_e2_matched_no_pac import register
from train_e2_full_overshoot import scratch_args
import train_e2_pac_overshoot as strict_overshoot
import train_lwpd as trainer
from train_pac_mamba import forward_losses as pac_forward_losses
from utils.dataset_os import evenly_spaced


RECON_EPOCHS = 4
FUSION_EPOCHS = 12
TOTAL_EPOCHS = RECON_EPOCHS + FUSION_EPOCHS
STAGE_LEARNING_RATES = dict(reconstruction=1e-4, fusion_backbone=5e-5, fusion_layers=1e-4)
PILOT_VAL_GROUP = "QXS-E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-VAL64"
FIELDS = ("loss", "optical_reconstruction", "sar_reconstruction", "reconstruction",
          "fusion", "intensity", "gradient", "gate_mean")


@dataclass(frozen=True)
class Experiment:
    name: str
    experiment_id: str
    variant: str
    evaluator: str
    method: str
    notes: str
    uses_source_interval: bool = False
    train_count: int = 8000
    val_count: int = 1000
    matched_pilot_control: bool = False
    fixed_aspr: bool = False
    uses_strict_overshoot: bool = False

    @property
    def run_dir(self):
        return Path("models") / self.name / "seed42"

    @property
    def final(self):
        return self.run_dir / f"epoch_{TOTAL_EPOCHS:03d}.pth"

    @property
    def result(self):
        return Path("test_result") / self.name / f"val{self.val_count}" / "seed42"

    @property
    def is_pilot(self):
        return self.val_count < 1000

    @property
    def manifest(self):
        if self.matched_pilot_control:
            return SOURCE_INTERVAL_PILOT.manifest
        return self.run_dir / "pilot_manifest.json" if self.is_pilot else protocol.MANIFEST

    @property
    def validation_group(self):
        return PILOT_VAL_GROUP if self.is_pilot else protocol.GROUP

    def build_model(self):
        if self.uses_rccr:
            return e2_rccr.RCCRL2E2()
        if self.uses_structure_joint:
            return StructureGuidedE2()
        if self.uses_pac:
            return OpticalBaseAnchoredPACE2()
        if self.uses_sar_directional:
            return e2_sar_directional.SARDirectionalE2()
        if self.uses_polarity:
            return e2_polarity_fusion.PolarityFusionE2()
        if self.variant == OpticalBaseAnchoredE2.VARIANT:
            return OpticalBaseAnchoredE2()
        return trainer.DirectFusionCDDFuse(self.variant)

    @property
    def uses_pac(self):
        return self.variant == OpticalBaseAnchoredPACE2.VARIANT

    @property
    def uses_structure_joint(self):
        return self.variant == StructureGuidedE2.VARIANT

    @property
    def uses_rccr(self):
        return self.variant == e2_rccr.RCCRL2E2.VARIANT

    @property
    def uses_sar_directional(self):
        return self.variant == e2_sar_directional.SARDirectionalE2.VARIANT

    @property
    def uses_polarity(self):
        return self.variant == e2_polarity_fusion.PolarityFusionE2.VARIANT


BASELINE = Experiment(
    "e2_scratch_twostage_v1", "E2-SCRATCH-TWOSTAGE-V1", "lwpd_e2", "test_OS.py",
    "Scratch LWPD E2, 4 reconstruction + 12 fusion epochs, original losses",
    "全模型随机初始化，同一次运行内从重建切换到融合，保留固定Haar与原融合损失。",
)
BASE_ANCHOR = Experiment(
    "e2_scratch_base_anchor_v1", "E2-SCRATCH-BASE-ANCHOR-V1", OpticalBaseAnchoredE2.VARIANT,
    "e2_base_anchor.py", "Scratch LWPD E2 with optical Base anchor and zero-initialized correction",
    "全模型从头训练4轮重建和12轮融合。融合Base由光学Base与零初始化跨模态修正相加。",
)
BASE_ANCHOR_PAC = Experiment(
    "e2_scratch_base_anchor_pac_v1", "E2-SCRATCH-BASE-ANCHOR-PAC-V1", OpticalBaseAnchoredPACE2.VARIANT,
    "e2_base_anchor_pac.py", "Scratch optical Base anchored LWPD E2 with PAC-Mamba Detail",
    "从头训练4轮重建和12轮融合，保留光学Base锚点，融合阶段加入零初始化PAC-Mamba细节残差。",
)
STRUCTURE_JOINT = Experiment(
    "e2_scratch_struct_joint_v1", "E2-SCRATCH-STRUCT-JOINT-V1", StructureGuidedE2.VARIANT,
    "e2_structure_joint.py", "Scratch LWPD E2 with structure-guided Base–Detail joint fusion",
    "随机初始化，4轮重建加12轮融合。低频修正受粗尺度结构引导，高频按共同、光学独有和SAR独有分流。",
)
SOURCE_INTERVAL = Experiment(
    "e2_scratch_source_interval_v1", "E2-SCRATCH-SOURCE-INTERVAL-V1",
    OpticalBaseAnchoredE2.VARIANT, "e2_base_anchor.py",
    "Scratch optical Base anchored LWPD E2 with a source-specific gradient interval loss",
    "光学Base结构从头训练4轮重建和12轮融合，融合阶段加入可信源方向梯度下限和带容差的上限。",
    uses_source_interval=True,
)
SOURCE_INTERVAL_PILOT = Experiment(
    "e2_scratch_source_interval_pilot_v1", "E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1",
    OpticalBaseAnchoredE2.VARIANT, "e2_base_anchor.py",
    "Scratch optical Base anchored LWPD E2 with a source-specific gradient interval loss, pilot",
    "固定训练子集1024对、均匀抽取验证64对。用于小样本可行性筛选，训练预算不同于完整基线。",
    uses_source_interval=True, train_count=1024, val_count=64,
)
ORIGINAL_LOSS_PILOT = Experiment(
    "e2_scratch_original_loss_pilot_v1", "E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1",
    OpticalBaseAnchoredE2.VARIANT, "e2_base_anchor.py",
    "Scratch optical Base anchored LWPD E2, matched original-loss pilot",
    "与分区梯度损失试跑共用固定训练1024对和验证64对。相同两阶段安排，融合阶段使用原E2损失。",
    train_count=1024, val_count=64, matched_pilot_control=True,
)
FIXED_ASPR_PILOT = Experiment(
    "e2_scratch_aspr_fixed_pilot_v1", "E2-SCRATCH-ASPR-FIXED-PILOT-V1",
    OpticalBaseAnchoredE2.VARIANT, "run_e2_aspr_pilot.py", "LWPD E2 + fixed-weight ASPR",
    "与原损失E2试跑匹配，从头训练4轮重建和12轮融合。融合阶段使用固定结构权重与6步PCG。",
    train_count=1024, val_count=64, matched_pilot_control=True, fixed_aspr=True,
)


def pilot_manifest(experiment, seed):
    source = json.loads(protocol.MANIFEST.read_text(encoding="utf-8"))
    splits = {
        "train": trainer.deterministic_subset(source["splits"]["train"], experiment.train_count, seed),
        "val": evenly_spaced(source["splits"]["val"], experiment.val_count),
        "test": [],
    }
    if len(splits["train"]) != experiment.train_count:
        raise ValueError("Training subset size exceeds the source split")
    return dict(
        dataset="QXS-SAROPT", optical_dir=source["optical_dir"], sar_dir=source["sar_dir"],
        splits=splits, splits_sha256=trainer.digest_splits(splits), seed=seed,
        source_manifest=protocol.MANIFEST.as_posix(),
        source_manifest_sha256=protocol.sha256_file(protocol.MANIFEST),
    )


def training_args(experiment=BASELINE):
    args = scratch_args()
    args.output_dir = experiment.run_dir.parent
    args.variant = experiment.variant
    args.split_manifest = experiment.manifest
    args.reconstruction_epochs = RECON_EPOCHS
    args.joint_epochs = FUSION_EPOCHS
    args.overshoot_weight = 0.0
    if experiment.uses_structure_joint:
        args.structure_geometry = GEOMETRY
    if experiment.uses_source_interval:
        args.source_interval = INTERVAL
        args.structure_geometry = GEOMETRY
        args.overshoot_weight = INTERVAL["upper_weight"]
    elif experiment.uses_strict_overshoot:
        args.overshoot_weight = strict_overshoot.WEIGHT
        args.overshoot_loss_source_sha256 = protocol.sha256_file(Path(strict_overshoot.__file__))
    return args


def reconstruction_forward(model, losses, optical, sar, valid, args):
    optical_base, optical_detail, sar_base, sar_detail = model.encode_pair(optical, sar)
    optical_loss = losses.reconstruction(optical, model.reconstruct(optical, optical_base, optical_detail))
    sar_loss = losses.reconstruction(sar, model.reconstruct(sar, sar_base, sar_detail), valid)
    total = 0.5 * (optical_loss + sar_loss)
    return total, dict(optical_reconstruction=optical_loss, sar_reconstruction=sar_loss, reconstruction=total)


def configure(model, optimizer, args, reconstruction):
    trainer.configure_stage(model, optimizer, "joint", args)
    if isinstance(model, e2_rccr.RCCRL2E2):
        model.rccr.requires_grad_(not reconstruction)
    if isinstance(model, e2_sar_directional.SARDirectionalE2):
        model.sar_directional.requires_grad_(not reconstruction)
    for group in optimizer.param_groups:
        fusion = group["group_name"] in ("base_fusion", "detail_fusion")
        if fusion and reconstruction:
            group["lr"] = 0.0
            for parameter in group["params"]:
                parameter.requires_grad_(False)
        else:
            key = "reconstruction" if reconstruction else "fusion_layers" if fusion else "fusion_backbone"
            group["lr"] = STAGE_LEARNING_RATES[key]


def interval_forward(original_forward):
    def forward(model, losses, optical, sar, valid, args):
        total, components = original_forward(model, losses, optical, sar, valid, args)
        return total, {**components, **losses.interval_stats}
    return forward


def strict_overshoot_forward(original_forward):
    def forward(model, losses, optical, sar, valid, args):
        total, components = original_forward(model, losses, optical, sar, valid, args)
        base_fusion = components["intensity"] + args.gradient_weight * components["gradient"]
        components["overshoot"] = (components["fusion"] - base_fusion) / args.overshoot_weight
        return total, components
    return forward


def config_values(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def verify_pilot_control(args):
    reference_run = SOURCE_INTERVAL_PILOT.run_dir
    reference = json.loads((reference_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest != pilot_manifest(SOURCE_INTERVAL_PILOT, args.seed):
        raise ValueError("The source-interval pilot manifest differs from the fixed protocol")
    if reference["total_epochs"] != TOTAL_EPOCHS or reference["stage_learning_rates"] != STAGE_LEARNING_RATES:
        raise ValueError("The source-interval pilot used a different training schedule")
    if reference["source_interval"] != INTERVAL:
        raise ValueError("The existing source-interval pilot used different loss settings")
    comparable = config_values(args)
    differences = [key for key, value in comparable.items()
                   if key not in ("output_dir", "overshoot_weight") and reference.get(key) != value]
    if differences:
        raise ValueError(f"Unmatched training settings: {', '.join(sorted(differences))}")
    for name, key in (("split_manifest.json", "saved_split_manifest_sha256"),
                      ("pilot_subset.json", "pilot_subset_sha256")):
        if protocol.sha256_file(reference_run / name) != reference[key]:
            raise ValueError(f"The source-interval pilot has an unexpected {name}")
    return reference


def load_continuation(run, args):
    """Validate the saved run and load its latest committed epoch on CPU."""
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    expected = dict(config_values(args), total_epochs=TOTAL_EPOCHS,
                    stage_learning_rates=STAGE_LEARNING_RATES)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved training settings differ from this run: {run}")
    for name, key in (("split_manifest.json", "saved_split_manifest_sha256"),
                      ("pilot_subset.json", "pilot_subset_sha256")):
        if protocol.sha256_file(run / name) != config[key]:
            raise ValueError(f"Saved training pairs differ from the run configuration: {name}")
    checkpoints = sorted(run.glob("epoch_???.pth"))
    expected_names = [f"epoch_{epoch:03d}.pth" for epoch in range(1, len(checkpoints) + 1)]
    if not checkpoints or [path.name for path in checkpoints] != expected_names:
        raise ValueError(f"Expected a contiguous sequence of committed checkpoints in {run}")
    path = checkpoints[-1]
    state = torch.load(path, map_location="cpu", weights_only=False)
    epoch = len(checkpoints)
    stage = "reconstruction" if epoch <= RECON_EPOCHS else "fusion"
    if (state["epoch"] != epoch - 1 or state["stage"] != stage or state["variant"] != args.variant
            or {key: str(value) if isinstance(value, Path) else value
                for key, value in state["args"].items()} != config_values(args)):
        raise ValueError(f"Checkpoint does not match the saved run: {path}")
    return path, state, config


def resume_history(path, fields, start_epoch):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if reader.fieldnames != fields or [row["epoch"] for row in rows[:start_epoch]] != [
        str(epoch) for epoch in range(1, start_epoch + 1)
    ]:
        raise ValueError(f"History does not match the checkpoint sequence: {path}")
    # The history row is flushed before its checkpoint is committed. A crash may leave one extra row.
    tail = rows[start_epoch:]
    if tail:
        if len(tail) != 1 or tail[0]["epoch"] != str(start_epoch + 1):
            raise ValueError(f"Unexpected uncommitted history in {path}")
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:start_epoch])
        temporary.replace(path)


def continuation_state(loader, device, provenance):
    return dict(provenance=provenance, sampler=loader.generator.get_state(),
                python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_random_state(saved, loader, device, seed, start_epoch):
    if saved is None:
        # The original checkpoint format did not persist the sampler or augmentation RNG.
        trainer.set_seed(seed + start_epoch)
        loader.generator.manual_seed(seed + start_epoch)
        return
    loader.generator.set_state(saved["sampler"])
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"], device)


def train(experiment=BASELINE, *, resume=False):
    save_continuation = (experiment.uses_strict_overshoot or experiment.uses_polarity
                         or getattr(experiment, "save_continuation", False))
    if resume and not save_continuation:
        raise ValueError("This experiment does not support continuation")
    args = training_args(experiment)
    run = experiment.run_dir
    if experiment.uses_rccr:
        args.rccr = e2_rccr.SPEC
        args.rccr_source_sha256 = protocol.sha256_file(Path(e2_rccr.__file__))
    if experiment.uses_sar_directional:
        args.sar_directional = e2_sar_directional.SPEC
        args.sar_directional_source_sha256 = protocol.sha256_file(Path(e2_sar_directional.__file__))
    if experiment.uses_polarity:
        args.polarity_fusion = e2_polarity_fusion.SPEC
        args.polarity_fusion_source_sha256 = protocol.sha256_file(Path(e2_polarity_fusion.__file__))
        args.polarity_difference_source_sha256 = protocol.sha256_file(Path(e2_aspr.__file__))
    if (experiment.uses_rccr or save_continuation or experiment.uses_sar_directional) and experiment.final.is_file():
        existing = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in config_values(args).items()):
            raise ValueError(f"Preserve completed run with a different protocol: {run}")
    if not experiment.final.is_file():
        if run.exists() and not resume:
            raise FileExistsError(f"Preserve incomplete run: {run}")
        resumed = load_continuation(run, args) if resume else None
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        reference = verify_pilot_control(args) if experiment.matched_pilot_control else None
        if experiment.fixed_aspr:
            args.aspr_fixed = e2_aspr.SPEC
            args.aspr_source_sha256 = protocol.sha256_file(Path(e2_aspr.__file__))
        trainer.set_seed(args.seed)
        if resumed:
            _, _, saved_config = resumed
            # Resolve the current data without replacing the original run's manifests.
            with tempfile.TemporaryDirectory(prefix=".resume-", dir=run) as temporary:
                training, validation, split, subset = trainer.build_loaders(args, device, Path(temporary))
                if (protocol.sha256_file(split) != saved_config["saved_split_manifest_sha256"]
                        or protocol.sha256_file(subset) != saved_config["pilot_subset_sha256"]):
                    raise ValueError("Current train/validation pairs differ from the saved run")
        else:
            run.mkdir(parents=True)
            if experiment.is_pilot and not experiment.matched_pilot_control:
                protocol.write_json(experiment.manifest, pilot_manifest(experiment, args.seed))
            training, validation, split, subset = trainer.build_loaders(args, device, run)
        if (len(training.dataset), len(validation.dataset)) != (experiment.train_count, experiment.val_count):
            raise ValueError("Dataset sizes differ from the experiment protocol")
        if reference is not None and (
            protocol.sha256_file(split) != reference["saved_split_manifest_sha256"]
            or protocol.sha256_file(subset) != reference["pilot_subset_sha256"]
        ):
            raise ValueError("Resolved train/validation pairs do not match the source-interval pilot")
        model = experiment.build_model().to(device)
        loss_type = trainer.DirectFusionLosses
        if experiment.uses_source_interval:
            loss_type = SourceIntervalFusionLosses
        elif experiment.uses_strict_overshoot:
            loss_type = strict_overshoot.OvershootLosses
        losses = loss_type().to(device)
        optimizer = trainer.make_optimizer(model, args)
        if experiment.uses_rccr:
            # RCCR follows the existing Base fusion schedule, including its Stage I freeze.
            group = next(group for group in optimizer.param_groups if group["group_name"] == "base_fusion")
            group["params"].extend(model.rccr.parameters())
        if experiment.uses_sar_directional:
            group = next(group for group in optimizer.param_groups if group["group_name"] == "detail_fusion")
            group["params"].extend(model.sar_directional.parameters())
        scaler = trainer.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
        if not resumed:
            protocol.write_json(run / "config.json", dict(
                config_values(args), total_epochs=TOTAL_EPOCHS,
                saved_split_manifest_sha256=protocol.sha256_file(split),
                pilot_subset_sha256=protocol.sha256_file(subset), stage_learning_rates=STAGE_LEARNING_RATES,
            ))
        stats_fields = FIELDS
        if experiment.uses_pac:
            stats_fields += ("interaction_mean", "relative_update_rms", "update_scale")
        elif experiment.uses_structure_joint:
            stats_fields += ROUTING_STATS
        elif experiment.uses_source_interval:
            stats_fields += INTERVAL_STATS
        elif experiment.uses_rccr:
            stats_fields += e2_rccr.STATS
        elif experiment.uses_strict_overshoot:
            stats_fields += ("overshoot",)
        elif experiment.uses_sar_directional:
            stats_fields += e2_sar_directional.STATS
        elif experiment.uses_polarity:
            stats_fields += e2_polarity_fusion.STATS
        fields = ["epoch", "stage", *(f"{label}_{key}" for label in ("train", "val") for key in stats_fields)]
        history = run / "history.csv"
        start_epoch = 0
        best = math.inf
        provenance = None
        if resumed:
            checkpoint, state, _ = resumed
            start_epoch = state["epoch"] + 1
            best = state["lowest_val_loss"]
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            rng = state.get("continuation")
            restore_random_state(rng, training, device, args.seed, start_epoch)
            provenance = dict(checkpoint=checkpoint.as_posix(), checkpoint_sha256=protocol.sha256_file(checkpoint),
                              restored_rng=rng is not None, previous=rng["provenance"] if rng else None)
            resume_history(history, fields, start_epoch)
            print(f"Resuming at epoch {start_epoch + 1} from {checkpoint} (restored RNG: {rng is not None})", flush=True)
        with history.open("a" if resumed else "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not resumed:
                writer.writeheader()
            fusion_forward = pac_forward_losses if experiment.uses_pac else trainer.forward_losses
            if experiment.uses_structure_joint:
                fusion_forward = structure_joint_forward_losses
            elif experiment.uses_source_interval:
                fusion_forward = interval_forward(fusion_forward)
            elif experiment.fixed_aspr:
                fusion_forward = partial(fusion_forward, fusion_transform=e2_aspr.refine)
            elif experiment.uses_strict_overshoot:
                fusion_forward = strict_overshoot_forward(fusion_forward)
            elif experiment.uses_polarity:
                fusion_forward = e2_polarity_fusion.forward_losses
            for epoch in range(start_epoch, TOTAL_EPOCHS):
                reconstruction = epoch < RECON_EPOCHS
                stage = "reconstruction" if reconstruction else "fusion"
                configure(model, optimizer, args, reconstruction)
                forward = reconstruction_forward if reconstruction else fusion_forward
                if experiment.uses_rccr and not reconstruction:
                    limit = "first_fusion_max_gate" if epoch == RECON_EPOCHS else "max_gate"
                    forward = partial(e2_rccr.forward_losses, max_gate=args.rccr[limit])
                with patch.object(trainer, "forward_losses", forward):
                    train_stats = trainer.run_epoch(model, losses, training, device, args, optimizer, scaler)
                    val_stats = trainer.run_epoch(model, losses, validation, device, args)
                if not all(math.isfinite(stats["loss"]) for stats in (train_stats, val_stats)):
                    raise RuntimeError(f"Non-finite {stage} loss at epoch {epoch + 1}")
                row = dict(epoch=epoch + 1, stage=stage)
                for label, stats in (("train", train_stats), ("val", val_stats)):
                    row.update({f"{label}_{key}": stats[key] for key in stats_fields if key in stats})
                writer.writerow(row)
                f.flush()
                if epoch == RECON_EPOCHS:
                    best = math.inf
                best = min(best, val_stats["loss"])
                ck = run / f"epoch_{epoch + 1:03d}.pth"
                temp = ck.with_suffix(".tmp")
                state = continuation_state(training, device, provenance) if save_continuation else None
                trainer.save_checkpoint(temp, epoch, stage, model, optimizer, scaler, args, best, continuation=state)
                temp.replace(ck)
                print(f"epoch={epoch + 1}/{TOTAL_EPOCHS} stage={stage} "
                      f"train={train_stats['loss']:.6f} val={val_stats['loss']:.6f}", flush=True)
    register_training(experiment)


def entry_base(experiment, checkpoint=None):
    checkpoint = checkpoint if checkpoint is not None else experiment.final
    return dict(
        dataset="QXS-SAROPT", offset=0, split_manifest=experiment.manifest.as_posix(),
        split_manifest_sha256=protocol.sha256_file(experiment.manifest),
        method=experiment.method,
        checkpoint=checkpoint.as_posix(), checkpoint_sha256=protocol.sha256_file(checkpoint),
        notes=experiment.notes,
    )


def register_training(experiment):
    run = experiment.run_dir
    history = run / "history.csv"
    with history.open(encoding="utf-8", newline="") as f:
        row = next(row for row in csv.DictReader(f) if row["epoch"] == str(TOTAL_EPOCHS))
    names = ("train_loss", "val_loss")
    if experiment.uses_strict_overshoot:
        names += ("train_overshoot", "val_overshoot")
    if experiment.uses_sar_directional:
        names += tuple(f"val_{name}" for name in e2_sar_directional.STATS)
    if experiment.uses_polarity:
        names += tuple(f"val_{name}" for name in e2_polarity_fusion.STATS)
    metrics = {k: float(row[k]) for k in names}
    register(dict(
        entry_base(experiment), id=f"{experiment.experiment_id}-SEED42-TRAIN{experiment.train_count}",
        status="training_complete", task="training",
        comparable_group=f"QXS-{experiment.experiment_id}-TRAIN{experiment.train_count}",
        split="train", sample_count=experiment.train_count, result_dir=run.as_posix(),
        metadata_path=(run / "config.json").as_posix(), metric_profile="training_loss", output_policy="float_loss",
        metrics=metrics, metric_source=dict(type="csv_row", path=history.as_posix(),
                                            match={"epoch": str(TOTAL_EPOCHS)}, columns={k: k for k in metrics}),
    ))


def validate(experiment=BASELINE, epoch=TOTAL_EPOCHS):
    checkpoint = experiment.run_dir / f"epoch_{epoch:03d}.pth"
    entry = entry_base(experiment, checkpoint)
    result = experiment.result
    experiment_id = f"{experiment.experiment_id}-SEED42-VAL{experiment.val_count}"
    if epoch != TOTAL_EPOCHS:
        result = result.parent / f"seed42_epoch_{epoch:03d}"
        experiment_id += f"-EPOCH{epoch:03d}"
        entry["notes"] += f"固定验证第{epoch}轮权重。"
    metrics = result / "metrics.csv"
    metadata = result / "run_metadata.json"
    ref = json.loads((protocol.BASELINE / "run_metadata.json").read_text(encoding="utf-8"))
    expected = {key: ref[key] for key in ("split_manifest_sha256", "selected_scene_ids", "split", "offset",
                                        "evaluated_pairs", "metric_profile", "output_range_policy", "metric_names")}
    if experiment.is_pilot:
        pilot = json.loads(experiment.manifest.read_text(encoding="utf-8"))
        if pilot != pilot_manifest(experiment, seed=42) or pilot["source_manifest_sha256"] != ref["split_manifest_sha256"]:
            raise ValueError("Pilot manifest differs from the fixed subset protocol")
        ids = evenly_spaced(ref["selected_scene_ids"], experiment.val_count)
        if pilot["splits"]["val"] != ids:
            raise ValueError("Pilot scene order differs from the fixed validation reference")
        expected.update(split_manifest_sha256=entry["split_manifest_sha256"],
                        selected_scene_ids=ids, evaluated_pairs=experiment.val_count)
    if not metrics.is_file():
        if result.exists():
            raise FileExistsError(f"Preserve incomplete validation: {result}")
        protocol.run([sys.executable, "-u", experiment.evaluator, "--data-root", "../datasets/QXSLAB_SAROPT",
                      "--split-manifest", experiment.manifest, "--split", "val", "--variant", experiment.variant,
                      "--checkpoint", checkpoint, "--output-dir", result, "--offset", "0", "--limit", str(experiment.val_count),
                      "--metric-profile", "paper", "--output-range-policy", "clamp", "--timing-warmup", "10"])
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"Validation differs from the original E2 protocol: {key}")
    if (meta["checkpoint_sha256"] != entry["checkpoint_sha256"] or meta["variant"] != experiment.variant
        or meta["metrics_only"]):
        raise ValueError("Unexpected validation run")
    rows, mean = protocol.read_metrics(metrics)
    if len(rows) != experiment.val_count or set(rows) != set(meta["selected_scene_ids"]):
        raise ValueError("Validation scene IDs differ from the metadata")
    if any(not math.isfinite(float(row[k])) for row in rows.values() for k in protocol.METRICS):
        raise ValueError("Non-finite validation metric")
    if any(len(list((result / d).glob("*.png"))) != experiment.val_count for d in ("gray", "rgb", "preview")):
        raise ValueError("Incomplete saved validation images")
    register(dict(
        entry, id=experiment_id,
        status="diagnostic_fixed_subset" if experiment.is_pilot else "development_validation",
        task="fusion", comparable_group=experiment.validation_group, split="val", sample_count=experiment.val_count,
        result_dir=result.as_posix(), metadata_path=metadata.as_posix(), metric_profile="paper",
        output_policy="clamp_then_uint8", metrics={k: float(mean[k]) for k in protocol.METRICS},
        metric_source=dict(type="csv_row", path=metrics.as_posix(), match={"image": "mean"},
                           columns={k: k for k in protocol.METRICS}),
    ))
    if experiment.matched_pilot_control and epoch == TOTAL_EPOCHS:
        compare_matched_pilots(experiment, SOURCE_INTERVAL_PILOT)


def compare_matched_pilots(control, candidate):
    control_config = json.loads((control.run_dir / "config.json").read_text(encoding="utf-8"))
    candidate_config = verify_pilot_control(training_args(control))
    ignored = {"output_dir", "overshoot_weight", "source_interval", "structure_geometry"}
    if ({k: v for k, v in control_config.items() if k not in ignored}
        != {k: v for k, v in candidate_config.items() if k not in ignored}):
        raise ValueError("Saved pilot runs have different settings outside the loss")
    if control_config["overshoot_weight"] != 0 or "source_interval" in control_config:
        raise ValueError("The matched control must use the original E2 loss")

    control_meta, candidate_meta = (
        json.loads((experiment.result / "run_metadata.json").read_text(encoding="utf-8"))
        for experiment in (control, candidate)
    )
    for key in ("split_manifest_sha256", "selected_scene_ids", "split", "offset", "evaluated_pairs",
                "metric_profile", "output_range_policy", "metric_names"):
        if control_meta[key] != candidate_meta[key]:
            raise ValueError(f"Cannot pair pilot outputs with different {key}")
    for experiment, meta in ((control, control_meta), (candidate, candidate_meta)):
        if meta["checkpoint_sha256"] != protocol.sha256_file(experiment.final):
            raise ValueError(f"Pilot checkpoint does not match saved results: {experiment.name}")
    original_rows, original_mean = protocol.read_metrics(control.result / "metrics.csv")
    candidate_rows, candidate_mean = protocol.read_metrics(candidate.result / "metrics.csv")
    ids = set(control_meta["selected_scene_ids"])
    if len(ids) != control.val_count or set(original_rows) != ids or set(candidate_rows) != ids:
        raise ValueError("Pilot metric files do not contain the fixed validation scenes")

    records = [protocol.paired_record(key, original_rows, candidate_rows, original_mean, candidate_mean)
               for key in protocol.METRICS]
    worst8 = sorted(ids, key=lambda i: (-float(original_rows[i]["Nabf"]), i))[:8]
    worst_rows = [{i: rows[i] for i in worst8} for rows in (original_rows, candidate_rows)]
    worst_means = [{"Nabf": mean(float(row["Nabf"]) for row in rows.values())} for rows in worst_rows]
    tail = protocol.paired_record("Nabf", *worst_rows, *worst_means)
    tail["metric"] = "Nabf_control_worst8"
    records.append(tail)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    report = control.result / "source_interval_vs_original_loss.csv"
    content = buffer.getvalue()
    if report.exists():
        if report.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Existing paired comparison differs: {report}")
    else:
        report.write_bytes(content.encode("utf-8"))
    for record in records:
        if record["metric"] in ("Nabf", "Qabf", "MS_SSIM", "FMI", "VIF", "Nabf_control_worst8"):
            print(f"{record['metric']}: {record['baseline']:.6f} -> {record['candidate']:.6f} "
                  f"(new minus original {record['raw_change']:+.6f})")
    p95 = [quantiles([float(row["Nabf"]) for row in rows.values()], n=100, method="inclusive")[94]
           for rows in (original_rows, candidate_rows)]
    print(f"Nabf P95: {p95[0]:.6f} -> {p95[1]:.6f}")
    print(f"Paired comparison (original loss -> source interval): {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "train-val"))
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--optical-base-anchor", action="store_true",
                          help="Use an optical Base skip with a zero-initialized cross-modal correction")
    variants.add_argument("--pac-mamba", action="store_true",
                          help="Add PAC-Mamba Detail fusion to the optical Base anchored model")
    variants.add_argument("--structure-joint", action="store_true",
                          help="Use structure-guided Base and directional Detail fusion")
    variants.add_argument("--source-interval", action="store_true",
                          help="Train optical Base E2 with the source-specific gradient interval loss")
    parser.add_argument("--pilot", action="store_true",
                        help="Use the fixed train1024/val64 split with --source-interval or --optical-base-anchor")
    parser.add_argument("--epoch", type=int, choices=range(RECON_EPOCHS + 1, TOTAL_EPOCHS + 1),
                        help="Evaluate a saved fusion checkpoint; valid only with val")
    args = parser.parse_args()
    if args.epoch is not None and args.action != "val":
        parser.error("--epoch is only available with val")
    if args.pilot and not (args.source_interval or args.optical_base_anchor):
        parser.error("--pilot requires --source-interval or --optical-base-anchor")
    os.chdir(protocol.ROOT)
    if args.source_interval:
        experiment = SOURCE_INTERVAL_PILOT if args.pilot else SOURCE_INTERVAL
    elif args.structure_joint:
        experiment = STRUCTURE_JOINT
    elif args.pac_mamba:
        experiment = BASE_ANCHOR_PAC
    elif args.optical_base_anchor:
        experiment = ORIGINAL_LOSS_PILOT if args.pilot else BASE_ANCHOR
    else:
        experiment = BASELINE
    if args.action in ("train", "train-val"):
        train(experiment)
    if args.action in ("val", "train-val"):
        validate(experiment, args.epoch if args.epoch is not None else TOTAL_EPOCHS)


if __name__ == "__main__":
    main()
