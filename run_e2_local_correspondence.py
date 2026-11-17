"""Train a scratch E2 candidate and pair its validation with the archived E2 pilot."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
import e2_local_correspondence as correspondence
from e2_local_correspondence import LocalCorrespondenceE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    Experiment, ORIGINAL_LOSS_PILOT, STAGE_LEARNING_RATES, TOTAL_EPOCHS, config_values,
    pilot_manifest, reconstruction_forward, train, training_args,
)
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from train_lwpd import DirectFusionLosses, forward_losses
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
ARCHIVE = Path("test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val")
GROUP = "QXS-E2-LOCAL-CORRESPONDENCE-V1-SEED42-VAL64-REFERENCE-FMI"
OUTPUT_POLICY = "clamp_then_uint8"
DIAGNOSTICS = ("match_mass", "offcenter_mass", "optical_zero_exchange_fraction", "sar_zero_exchange_fraction")


class LocalCorrespondencePilot(Experiment):
    def build_model(self):
        return LocalCorrespondenceE2()


PILOT = LocalCorrespondencePilot(
    "e2_scratch_local_correspondence_pilot_v1", "E2-SCRATCH-LOCAL-CORRESPONDENCE-PILOT-V1",
    LocalCorrespondenceE2.VARIANT, "run_e2_local_correspondence.py",
    "光学 Base E2 + 可拒配的局部 Detail 交换",
    "从头初始化，4轮重建+12轮原损失融合。保留两路原位输入，替换第三条膨胀卷积分支。",
    train_count=1024, val_count=64,
)


def source_manifest():
    baseline = ORIGINAL_LOSS_PILOT
    manifest = read_json(baseline.manifest)
    if manifest != pilot_manifest(PILOT, 42):
        raise ValueError("The source pilot no longer matches the archived train1024/val64 split")
    return manifest


def read_scene_rows(path, features, scene_ids):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["image"] not in ("mean", "std")]
    if [row["image"] for row in rows] != scene_ids:
        raise ValueError(f"Unexpected scene order or duplicate scene IDs: {path}")
    return [[float(row[name]) for name in features] for row in rows]


def archived_baseline(scene_ids):
    baseline = ORIGINAL_LOSS_PILOT
    meta_path = ARCHIVE / "run_metadata.json"
    meta = read_json(meta_path)
    expected = dict(scene_ids=scene_ids, metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY,
                    split_manifest_sha256=sha256_file(baseline.manifest))
    if (any(meta.get(key) != value for key, value in expected.items())
            or Path(meta["checkpoints"]["e2"]) != baseline.final
            or meta["checkpoint_hashes"]["e2"] != sha256_file(baseline.final)):
        raise ValueError("Archived E2 results do not match the evaluation protocol")
    folder = ARCHIVE / "e2"
    verify_images(folder / "gray", scene_ids)
    metrics = read_scene_rows(folder / "metrics.csv", METRICS, scene_ids)
    sources = read_scene_rows(folder / "diagnostics.csv", SOURCE_METRICS, scene_ids)
    return [(row, source, [None] * len(DIAGNOSTICS)) for row, source in zip(metrics, sources)]


def protocol_lock():
    paths = (Path(correspondence.__file__), ROOT / "run_e2_scratch_twostage.py",
             ROOT / "direct_fusion.py", ROOT / "lwpd.py", ROOT / "e2_base_anchor.py",
             ROOT / "train_lwpd.py", ROOT / "train_direct_fusion.py", ROOT / "utils/dataset_os.py")
    return dict(config=config_values(training_args(PILOT)),
                original_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                sources={path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
                total_epochs=TOTAL_EPOCHS, stage_learning_rates=STAGE_LEARNING_RATES)


def lock_training(*, create):
    path = PILOT.run_dir.parent / "workflow_protocol.json"
    expected = protocol_lock()
    if path.is_file():
        if read_json(path) != expected:
            raise ValueError(f"Preserve existing training with a different protocol: {path}")
    elif create and not PILOT.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        raise FileNotFoundError(f"Missing training protocol: {path}")
    return path


def verify_training(config, expected, *, resolved):
    differing = [key for key, value in expected.items()
                 if key not in ("variant", "split_manifest", "output_dir") and config.get(key) != value]
    if differing:
        raise ValueError(f"Training settings differ from archived E2: {', '.join(sorted(differing))}")
    if resolved:
        baseline = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
        for filename, key in (("split_manifest.json", "saved_split_manifest_sha256"),
                              ("pilot_subset.json", "pilot_subset_sha256")):
            if config.get(key) != baseline[key] or sha256_file(PILOT.run_dir / filename) != baseline[key]:
                raise ValueError(f"Pilot training/validation pairs differ: {filename}")


@torch.inference_mode()
def evaluate_scene(model, pair, device, gray_dir):
    optical, sar = read_pair(pair)
    o, s = (to_tensor(image, device) for image in (optical, sar))
    features, _ = model.prepare_fusion_inputs(model.encode_pair(o, s), s)
    base, (detail, stats) = model.fuse_features(*features, return_diagnostics=True)
    fused = model.reconstruct(o, base, detail)
    if not torch.isfinite(fused).all():
        raise FloatingPointError(f"Non-finite fused image: {pair[0]}")
    gray = fused[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    if not cv2.imwrite(str(gray_dir / f"{pair[0].stem}.png"), gray):
        raise OSError(f"Cannot save fused image: {pair[0]}")
    scores = FusionEvaluator.metric_vector(gray, optical, sar)
    sources = [metric(gray, image) for metric in (FusionEvaluator._ssim_pair, FusionEvaluator._ms_ssim_pair)
               for image in (optical, sar)]
    return scores, sources, [float(stats[key]) for key in DIAGNOSTICS]


def validate(manifest):
    lock = lock_training(create=False)
    candidate_config = read_json(PILOT.run_dir / "config.json")
    expected = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    verify_training(candidate_config, expected, resolved=True)
    recorded_settings = read_json(lock)["config"]
    if (read_json(PILOT.manifest) != manifest or sha256_file(PILOT.manifest) != sha256_file(ORIGINAL_LOSS_PILOT.manifest)
            or any(candidate_config.get(key) != value for key, value in recorded_settings.items())):
        raise ValueError("Candidate training settings or split do not match the pilot record")
    scene_ids = manifest["splits"]["val"]
    baseline_rows = archived_baseline(scene_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash = sha256_file(PILOT.final)
    metadata = dict(
        dataset="QXS-SAROPT", split="val", offset=0, scene_ids=scene_ids, evaluated_pairs=len(scene_ids),
        metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY, device=str(device),
        checkpoint=PILOT.final.as_posix(), checkpoint_sha256=checkpoint_hash,
        split_manifest=PILOT.manifest.as_posix(), split_manifest_sha256=sha256_file(PILOT.manifest),
        training_protocol_sha256=sha256_file(lock),
        baseline_checkpoint=ORIGINAL_LOSS_PILOT.final.as_posix(), baseline_checkpoint_sha256=sha256_file(ORIGINAL_LOSS_PILOT.final),
        baseline_metadata=(ARCHIVE / "run_metadata.json").as_posix(),
        baseline_metrics_sha256=sha256_file(ARCHIVE / "e2/metrics.csv"),
        baseline_diagnostics_sha256=sha256_file(ARCHIVE / "e2/diagnostics.csv"),
        evaluator_sha256=sha256_file(Path(__file__)),
    )
    output = PILOT.result
    if output.is_dir():
        if read_json(output / "run_metadata.json") != metadata:
            raise ValueError(f"Preserve existing results with different provenance: {output}")
        summaries = read_json(output / "registry_metrics.json")
        verify_images(output / "candidate/gray", scene_ids)
    else:
        state = torch.load(PILOT.final, map_location="cpu", weights_only=False)
        saved_settings = config_values(argparse.Namespace(**state["args"]))
        if ((state["variant"], state["epoch"], state["stage"]) != (PILOT.variant, TOTAL_EPOCHS - 1, "fusion")
                or saved_settings != recorded_settings):
            raise ValueError("Expected the final fusion checkpoint")
        model = PILOT.build_model().to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        splits, _ = manifest_pairs(DATA, PILOT.manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".local-correspondence-", dir=output.parent) as temp:
            pending = Path(temp) / "result"
            (pending / "e2").mkdir(parents=True)
            gray_dir = pending / "candidate/gray"
            gray_dir.mkdir(parents=True)
            candidate_rows = []
            for i, pair in enumerate(splits["val"], 1):
                candidate_rows.append(evaluate_scene(model, pair, device, gray_dir))
                print(f"val {i}/{len(scene_ids)} {pair[0].stem}", flush=True)
            summaries = summarize(pending, scene_ids, {"e2": baseline_rows, "candidate": candidate_rows},
                                  (("e2", "candidate"),), diagnostic_fields=DIAGNOSTICS)
            write_json(pending / "run_metadata.json", metadata)
            write_json(pending / "registry_metrics.json", summaries)
            pending.rename(output)
    metrics = summaries["candidate"]
    metric_path = output / "candidate/metrics.csv"
    register_many([dict(
        id=f"{GROUP}-candidate", comparable_group=GROUP, task="fusion", status="development_validation",
        method=PILOT.method, dataset=metadata["dataset"], split="val", offset=0, sample_count=len(scene_ids),
        checkpoint=metadata["checkpoint"], checkpoint_sha256=checkpoint_hash,
        baseline_checkpoint=metadata["baseline_checkpoint"], baseline_checkpoint_sha256=metadata["baseline_checkpoint_sha256"],
        split_manifest=metadata["split_manifest"], split_manifest_sha256=metadata["split_manifest_sha256"],
        metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY, result_dir=output.as_posix(),
        metadata_path=(output / "run_metadata.json").as_posix(), metrics=metrics,
        metric_source=dict(type="csv_row", path=metric_path.as_posix(), match={"image": "mean"},
                           columns={name: name for name in metrics}),
        notes="原损失E2为已训练对照，候选从头训练。两法按每项指标共同有限场景配对。",
    )])
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)


def smoke(manifest, scene_id):
    if scene_id not in manifest["splits"]["val"]:
        raise ValueError("Smoke test scene must be in the matched validation split")
    optical, sar = read_pair(locate_scene(DATA, manifest, scene_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = training_args(PILOT)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model = PILOT.build_model().to(device)
    o, s = (to_tensor(image, device) for image in (optical, sar))
    # Preserve the real source images while keeping the smoke test small.
    o, s = (image[..., :64, :64] for image in (o, s))
    valid = (s > 0).float()
    losses = DirectFusionLosses().to(device)
    reconstruction, _ = reconstruction_forward(model, losses, o, s, valid, args)
    assert torch.isfinite(reconstruction)
    with torch.autocast("cuda", enabled=device.type == "cuda" and not args.no_amp):
        loss, _ = forward_losses(model, losses, o, s, valid, args)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite single-scene loss")
    loss.backward()
    for layer in (model.detail_fusion.optical_value, model.detail_fusion.sar_value):
        gradient = layer.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().any():
            raise FloatingPointError("Missing, zero or non-finite exchange gradient")
    with torch.inference_mode():
        features, _ = model.prepare_fusion_inputs(model.encode_pair(o, s), s)
        base, (detail, stats) = model.fuse_features(*features, return_diagnostics=True)
        torch.testing.assert_close(model.reconstruct(o, base, detail), model(o, s))
    print(f"Smoke passed: scene {scene_id} 64x64, loss={float(loss):.6f}, "
          f"match mass={float(stats['match_mass']):.6f}; no optimizer update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train-val", "train", "val", "smoke"), nargs="?", default="train-val")
    parser.add_argument("--scene-id", default="211", help="Smoke-only fixed validation scene")
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = source_manifest()
    if args.action == "smoke":
        smoke(manifest, args.scene_id)
        return
    if args.action in ("train-val", "train"):
        baseline = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
        lock = protocol_lock()
        expected = dict(lock["config"], total_epochs=lock["total_epochs"], stage_learning_rates=lock["stage_learning_rates"])
        verify_training(expected, {key: baseline[key] for key in expected}, resolved=False)
        archived_baseline(manifest["splits"]["val"])
        lock_training(create=True)
        train(PILOT)
    if args.action in ("train-val", "val"):
        validate(manifest)


if __name__ == "__main__":
    main()
