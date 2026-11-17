"""Train matched phase guidance variants from scratch, or compare saved results."""

import argparse
import json
import os
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import torch
from torch.nn import functional as F

from diagnose_e2_information_path import DEFAULT_MANIFEST, image_u8, locate_scene, read_pair, to_tensor
from e2_base_anchor_pac import OpticalBaseAnchoredPACE2
from e2_phase_guide import MODES, VARIANT_PREFIX, SignedPhaseGuideE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize, write_csv
from run_e2_local_correspondence import archived_baseline, read_scene_rows, source_manifest, verify_training
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    Experiment, ORIGINAL_LOSS_PILOT, STAGE_LEARNING_RATES, TOTAL_EPOCHS,
    config_values, pilot_manifest, reconstruction_forward, train, training_args,
)
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from train_lwpd import DirectFusionLosses
from train_pac_mamba import forward_losses
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
RESULTS = Path("test_result/e2_signed_pac_v1")
OUTPUT_POLICY = "clamp_then_uint8"
METRICS = FusionEvaluator.METRIC_NAMES


class PhaseGuidePilot(Experiment):
    @property
    def uses_pac(self):
        return True

    def build_model(self):
        return SignedPhaseGuideE2(self.variant[len(VARIANT_PREFIX):])


LABELS = {
    "feature": "有符号 Haar Detail + 跨模态 SSM，纯特征交互",
    "pa": "有符号 Haar Detail + 跨模态 SSM，PA 引导",
    "pc": "有符号 Haar Detail + 跨模态 SSM，PC 引导",
    "pa_shifted": "有符号 Haar Detail + 跨模态 SSM，空间平移 PA",
}
PILOTS = {
    mode: PhaseGuidePilot(f"e2_signed_pac_v1_{mode}_pilot", f"E2-SIGNED-PAC-V1-{mode.upper()}",
                          VARIANT_PREFIX + mode, "run_e2_phase_guide.py", method,
                          "从头初始化。固定 train1024/val64，4 轮重建、12 轮原损失融合。",
                          train_count=1024, val_count=64)
    for mode, method in LABELS.items()
}


def locked_training(pilot, create=False):
    expected = dict(config_values(training_args(pilot)), total_epochs=TOTAL_EPOCHS,
                    stage_learning_rates=STAGE_LEARNING_RATES)
    baseline_config = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    verify_training(expected, {key: baseline_config[key] for key in expected}, resolved=False)
    files = ("run_e2_phase_guide.py", "e2_phase_guide.py", "pac_mamba.py", "e2_base_anchor_pac.py", "e2_base_anchor.py", "lwpd.py",
             "direct_fusion.py", "run_e2_scratch_twostage.py", "train_pac_mamba.py", "train_lwpd.py", "utils/dataset_os.py")
    lock = dict(settings=expected, source_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                sources={name: sha256_file(Path(name)) for name in files})
    path = pilot.run_dir.parent / "workflow_protocol.json"
    if path.exists():
        if read_json(path) != lock:
            raise ValueError(f"Existing experiment uses another protocol: {path}")
    elif create and not pilot.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)
    else:
        raise FileNotFoundError(path)
    return lock, path


def validate_checkpoint(pilot, manifest):
    lock, lock_path = locked_training(pilot)
    if read_json(pilot.manifest) != manifest or sha256_file(pilot.manifest) != lock["source_manifest_sha256"]:
        raise ValueError("The trained variant did not use the shared manifest.")
    config = read_json(pilot.run_dir / "config.json")
    if any(config.get(key) != value for key, value in lock["settings"].items()):
        raise ValueError("The trained variant did not use the shared settings.")
    verify_training(config, read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json"), resolved=False)
    baseline = ORIGINAL_LOSS_PILOT.run_dir
    for name in ("split_manifest.json", "pilot_subset.json"):
        if sha256_file(pilot.run_dir / name) != sha256_file(baseline / name):
            raise ValueError(f"Resolved data differ from the original pilot: {name}")
    return lock_path


@torch.inference_mode()
def evaluate(pilot, mode, manifest, device):
    lock_path = validate_checkpoint(pilot, manifest)
    selected = manifest_pairs(DATA, pilot.manifest)[0]["val"]
    scene_ids = [optical.stem for optical, _ in selected]
    output = RESULTS / "raw_val64" / mode
    meta = dict(scene_ids=scene_ids, split_manifest=pilot.manifest.as_posix(),
                split_manifest_sha256=sha256_file(pilot.manifest), device=str(device),
                checkpoint=pilot.final.as_posix(), checkpoint_sha256=sha256_file(pilot.final),
                training_protocol_sha256=sha256_file(lock_path), metric_profile=DEFAULT_PROFILE,
                output_policy=OUTPUT_POLICY, evaluator_sha256=sha256_file(Path(__file__)))
    if output.exists():
        if read_json(output / "run_metadata.json") != meta:
            raise ValueError(f"Preserve existing results with different provenance: {output}")
        verify_images(output / "gray", scene_ids)
    else:
        state = torch.load(pilot.final, map_location="cpu", weights_only=False)
        if (state["variant"], state["epoch"], state["stage"]) != (pilot.variant, TOTAL_EPOCHS - 1, "fusion"):
            raise ValueError("Expected the final fusion checkpoint for the selected variant.")
        model = pilot.build_model().to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".phase-guide-", dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            (pending / "gray").mkdir(parents=True)
            values, stats = [], []
            for scene, paths in zip(scene_ids, selected):
                optical, sar = read_pair(paths)
                image, features = model(*(to_tensor(x, device) for x in (optical, sar)), return_features=True)
                gray = image_u8(image)
                if not cv2.imwrite(str(pending / "gray" / f"{scene}.png"), gray):
                    raise OSError(f"Cannot save {scene}.")
                pac = features["pac"]
                applied = pac["update"] * pac["update_scale"]
                ratio = applied.float().norm() / pac["keep"].float().norm().clamp_min(1e-6)
                stats.append((scene, float(pac["interaction"].mean()), float(ratio)))
                values.append((scene, *FusionEvaluator.metric_vector(gray, optical, sar)))
                print(f"{mode} val {len(values)}/{len(scene_ids)} {scene}", flush=True)
            write_csv(pending / "metrics.csv", ("image", *METRICS), values)
            write_csv(pending / "diagnostics.csv", ("image", "interaction_mean", "relative_applied_update_rms"), stats)
            write_json(pending / "run_metadata.json", meta)
            pending.rename(output)
    return output, meta


def compare(manifest, modes, label, device):
    missing = [mode for mode in modes if not PILOTS[mode].final.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed checkpoints: {', '.join(missing)}")
    scene_ids = manifest["splits"]["val"]
    rows = {"e2": [(metrics, [], []) for metrics, _, _ in archived_baseline(scene_ids)]}
    raw = {}
    for mode in modes:
        source, provenance = evaluate(PILOTS[mode], mode, manifest, device)
        rows[mode] = [(values, [], []) for values in read_scene_rows(source / "metrics.csv", METRICS, scene_ids)]
        raw[mode] = dict(provenance, raw_result=source.as_posix(), raw_metrics_sha256=sha256_file(source / "metrics.csv"))
    group = f"QXS-E2-SIGNED-PAC-V1-VAL64-REF-FMI-{label.upper()}"
    output = RESULTS / f"val64_{label}"
    archive_path = Path("test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val")
    E2 = ORIGINAL_LOSS_PILOT
    meta = dict(scene_ids=scene_ids, candidates=raw, metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY,
                baseline_result=archive_path.as_posix(), baseline_checkpoint_sha256=sha256_file(E2.final),
                baseline_metrics_sha256=sha256_file(archive_path / "e2/metrics.csv"),
                split_manifest_sha256=sha256_file(E2.manifest))
    if output.exists():
        if read_json(output / "run_metadata.json") != meta:
            raise ValueError(f"Preserve existing comparisons: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        with TemporaryDirectory(prefix=".paired-", dir=RESULTS) as temporary:
            pending = Path(temporary) / "comparison"
            for name in rows:
                (pending / name).mkdir(parents=True)
            pairs = [("e2", "feature")]
            pairs.extend((mode, "pa") for mode in modes if mode != "pa")
            summaries = summarize(pending, scene_ids, rows, pairs, diagnostic_fields=(), source_metrics=())
            counts = []
            for control, candidate in pairs:
                for metric, sign in (("Qabf", 1), ("FMI", 1), ("Nabf", -1), ("SCD", 1)):
                    column = METRICS.index(metric)
                    changes = [(rows[candidate][i][0][column] - rows[control][i][0][column]) * sign
                               for i in range(len(scene_ids))
                               if all(isfinite(group_rows[i][0][column]) for group_rows in rows.values())]
                    counts.append((control, candidate, metric, len(changes),
                                   sum(x > 0 for x in changes), sum(x == 0 for x in changes),
                                   sum(x < 0 for x in changes)))
            write_csv(pending / "scene_direction_counts.csv",
                      ("control", "candidate", "metric", "common_count", "better", "ties", "worse"), counts)
            write_json(pending / "run_metadata.json", meta)
            write_json(pending / "registry_metrics.json", summaries)
            pending.rename(output)
    entries = []
    for mode, metrics in summaries.items():
        checkpoint = E2.final if mode == "e2" else PILOTS[mode].final
        entries.append(dict(
            id=f"{group}-{mode}", comparable_group=group, task="fusion", status="development_validation",
            method="归档光学 Base E2" if mode == "e2" else LABELS[mode],
            dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(scene_ids),
            checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
            split_manifest=E2.manifest.as_posix(), split_manifest_sha256=meta["split_manifest_sha256"],
            metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY, result_dir=output.as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(), metrics=metrics,
            metric_source=dict(type="csv_row", path=(output / mode / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in metrics}),
        ))
    register_many(entries)
    print(f"Paired results: {output / 'paired_comparison.csv'}", flush=True)


def smoke(manifest, device):
    scene = manifest["splits"]["val"][0]
    o, s = (to_tensor(x, device)[..., :64, :64] for x in read_pair(locate_scene(DATA, manifest, scene)))
    valid = (s > 0).float()
    snapshot = None
    aligned = None
    for mode, pilot in PILOTS.items():
        torch.manual_seed(42)
        model = pilot.build_model().to(device)
        args = training_args(pilot)
        losses = DirectFusionLosses().to(device)
        if snapshot is None:
            snapshot = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), snapshot[key], rtol=0, atol=0)
        reconstruction, _ = reconstruction_forward(model, losses, o, s, valid, args)
        assert torch.isfinite(reconstruction)
        with torch.inference_mode():
            output, features = model(o, s, return_features=True)
            keep = model.reconstruct(o, features["fused_base"], features["pac"]["keep"])
            torch.testing.assert_close(output, keep, rtol=0, atol=0)
            layer = model.detail_fusion
            for key in ("optical_detail", "sar_detail"):
                detail = features[key]
                assert F.avg_pool2d(detail, 2).norm() < 1e-6 * detail.norm()
                assert layer.signed_bands(detail).norm() > 0.9 * detail.norm()
            assert torch.isfinite(features["pac"]["interaction"]).all()
        if mode in ("pa", "pa_shifted"):
            evidence = model.compute_phase_evidence(o, s)
            _, weights = layer.phase_factor(evidence["optical_pa"], evidence["sar_pa"], (32, 32))
            torch.testing.assert_close(weights.mean(), weights.new_tensor(1.0))
            if mode == "pa":
                aligned = weights
            else:
                torch.testing.assert_close(weights.flatten().sort().values, aligned.flatten().sort().values)
        with torch.autocast("cuda", enabled=device.type == "cuda" and not args.no_amp):
            loss, _ = forward_losses(model, losses, o, s, valid, args)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = layer.update_projection[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().any()
        if mode == "pa":
            with torch.inference_mode():
                # Open the zero-initialized projection only for the dependency check.
                layer.update_projection[-1].weight.normal_(0, 0.001)
                optical, sar = (features[key] for key in ("optical_detail", "sar_detail"))
                base = features["fused_base"]
                phase = (evidence["optical_pa"], evidence["sar_pa"])
                actual = layer(optical, sar, base, *phase, return_diagnostics=True)[1]["update"]
                for left, right in ((torch.zeros_like(optical), sar), (optical, torch.zeros_like(sar))):
                    control = layer(left, right, base, *phase, return_diagnostics=True)[1]["update"]
                    assert (actual - control).norm() > 1e-5 * actual.norm()
        print(f"{mode}: scene {scene}, loss={float(loss):.6f}, nonzero update gradient; no optimizer step", flush=True)


@torch.inference_mode()
def audit_legacy(device):
    checkpoint = Path("models/e2_scratch_base_anchor_pac_v1/seed42/epoch_016.pth")
    manifest = read_json(DEFAULT_MANIFEST)
    scene = manifest["splits"]["val"][0]
    files = locate_scene(DATA, manifest, scene)
    model = OpticalBaseAnchoredPACE2().to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["variant"] != model.VARIANT:
        raise ValueError("Expected the legacy anchored PAC checkpoint.")
    model.load_state_dict(state["model"], strict=True)
    _, optical, _, sar = model.encode_pair(*(to_tensor(image, device) for image in read_pair(files)))
    pooled = [F.avg_pool2d(x, 2) for x in (optical, sar)]
    layer = model.detail_fusion

    def update(a, b):
        joint = layer.joint_projection(torch.cat((a, b, (a-b).abs()), dim=1))
        return layer.update_projection(layer.ssm(joint))

    actual = update(*pooled)
    zero = update(*(torch.zeros_like(x) for x in pooled))
    rms = lambda x: float(x.double().square().mean().sqrt())
    print(json.dumps(dict(scene=scene, optical_rms=rms(optical), sar_rms=rms(sar),
                          optical_pooled_rms=rms(pooled[0]), sar_pooled_rms=rms(pooled[1]),
                          delta_relative_difference_with_zero=rms(actual-zero)/rms(actual)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "audit-legacy", "train", "val-primary", "val-full"))
    parser.add_argument("--mode", choices=MODES, help="Required for train.")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action == "audit-legacy":
        audit_legacy(device)
        return
    manifest = source_manifest()
    if args.action == "smoke":
        smoke(manifest, device)
    elif args.action == "train":
        if args.mode is None:
            parser.error("--mode is required for train")
        pilot = PILOTS[args.mode]
        if pilot_manifest(pilot, 42) != manifest:
            raise ValueError("Expected the archived train1024/val64 split.")
        locked_training(pilot, create=True)
        train(pilot)
    else:
        label = args.action[len("val-"):]
        compare(manifest, MODES if label == "full" else MODES[:2], label, device)


if __name__ == "__main__":
    main()
