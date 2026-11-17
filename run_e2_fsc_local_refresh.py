"""Run matched, archived-safe local LL refresh and feature-input controls."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import torch

import run_e2_phase_guide as phase
from diagnose_e2_information_path import image_u8, locate_scene, read_pair, to_tensor
from e2_fsc_local_refresh import LocalRefreshFSCMambaE2 as FSCMambaE2, MODES, VARIANT_PREFIX, CHUNK_SIZE, CARRY
from e2_fsc_mamba import CONTEXT_SCALE, scan_with_state
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize, write_csv
from run_e2_local_correspondence import archived_baseline, read_scene_rows, source_manifest, verify_training
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    Experiment, ORIGINAL_LOSS_PILOT, STAGE_LEARNING_RATES, TOTAL_EPOCHS,
    config_values, pilot_manifest, train, training_args,
)
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from train_lwpd import DirectFusionLosses, forward_losses
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
RESULTS = Path("test_result/e2_fsc_local_refresh_v1")
ARCHIVE = Path("test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val")
LABELS = dict(e2="归档光学 Base E2", feature="纯特征 SSM",
              local_feature="每16步持续输入 LL 分段摘要",
              local_refresh="每16步 LL 分段状态刷新（carry=0.5）",
              local_shuffled="每16步错配 LL 分段状态刷新")


class FSCPilot(Experiment):
    def build_model(self):
        return FSCMambaE2(self.variant[len(VARIANT_PREFIX):])


PILOTS = {
    mode: FSCPilot(f"e2_fsc_local_refresh_v1_{mode}_pilot", f"E2-FSC-LOCAL-REFRESH-V1-{mode.upper()}",
                  VARIANT_PREFIX + mode, "run_e2_fsc_mamba.py", LABELS[mode],
                  "从头初始化，固定 train1024/val64，4 轮重建及 12 轮原损失融合。",
                  train_count=1024, val_count=64)
    for mode in MODES
}


def fsc_lock(pilot, create=False):
    settings = dict(config_values(training_args(pilot)), total_epochs=TOTAL_EPOCHS,
                    stage_learning_rates=STAGE_LEARNING_RATES)
    baseline = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    verify_training(settings, {key: baseline[key] for key in settings}, resolved=False)
    files = ("run_e2_fsc_local_refresh.py", "e2_fsc_local_refresh.py", "run_e2_fsc_mamba.py", "e2_fsc_mamba.py", "e2_phase_guide.py", "e2_base_anchor_pac.py",
             "pac_mamba.py", "lwpd.py", "e2_base_anchor.py", "net.py", "direct_fusion.py",
             "run_e2_scratch_twostage.py", "train_lwpd.py", "train_direct_fusion.py", "utils/dataset_os.py")
    expected = dict(settings=settings, source_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                    sources={path: sha256_file(Path(path)) for path in files})
    path = pilot.run_dir.parent / "workflow_protocol.json"
    if path.is_file():
        if read_json(path) != expected:
            raise ValueError(f"Existing FSC run uses another protocol: {path}")
    elif create and not pilot.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
    else:
        raise FileNotFoundError(path)
    return expected, path


@torch.inference_mode()
def evaluate_fsc(mode, manifest, device):
    pilot = PILOTS[mode]
    with patch.object(phase, "locked_training", fsc_lock):
        lock_path = phase.validate_checkpoint(pilot, manifest)
    selected = manifest_pairs(DATA, pilot.manifest)[0]["val"]
    ids = [optical.stem for optical, _ in selected]
    if ids != manifest["splits"]["val"]:
        raise ValueError("Validation order differs from the shared manifest.")
    output = RESULTS / "raw_val64" / mode
    meta = dict(scene_ids=ids, checkpoint=pilot.final.as_posix(), checkpoint_sha256=sha256_file(pilot.final),
                split_manifest=pilot.manifest.as_posix(), split_manifest_sha256=sha256_file(pilot.manifest),
                training_protocol_sha256=sha256_file(lock_path), evaluator_sha256=sha256_file(Path(__file__)),
                metric_profile=DEFAULT_PROFILE, output_policy=phase.OUTPUT_POLICY, device=str(device))
    if output.exists():
        if read_json(output / "run_metadata.json") != meta:
            raise ValueError(f"Existing validation uses another protocol: {output}")
        verify_images(output / "gray", ids)
        return output, meta
    state = torch.load(pilot.final, map_location="cpu", weights_only=False)
    if (state["variant"], state["epoch"], state["stage"]) != (pilot.variant, TOTAL_EPOCHS - 1, "fusion"):
        raise ValueError("Expected the final matched FSC fusion checkpoint.")
    model = pilot.build_model().to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fsc-val-", dir=output.parent) as temporary:
        pending = Path(temporary) / "result"
        (pending / "gray").mkdir(parents=True)
        rows = []
        for scene, pair in zip(ids, selected):
            optical, sar = read_pair(pair)
            o, s = (to_tensor(image, device) for image in (optical, sar))
            gray = image_u8(model(o, s))
            if not cv2.imwrite(str(pending / "gray" / f"{scene}.png"), gray):
                raise OSError(f"Cannot save {mode} image {scene}")
            rows.append((scene, *FusionEvaluator.metric_vector(gray, optical, sar)))
            print(f"{mode}: {len(rows)}/{len(ids)} {scene}", flush=True)
        write_csv(pending / "metrics.csv", ("image", *phase.METRICS), rows)
        write_json(pending / "run_metadata.json", meta)
        pending.rename(output)
    return output, meta


def validate(manifest, device):
    reference = phase.PILOTS["feature"]
    checkpoints = {mode: pilot.final for mode, pilot in PILOTS.items()}
    checkpoints.update(feature=reference.final, e2=ORIGINAL_LOSS_PILOT.final)
    missing = [name for name, path in checkpoints.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed checkpoints: " + ", ".join(missing))
    ids = manifest["splits"]["val"]
    rows = {"e2": [(values, [], []) for values, _, _ in archived_baseline(ids)]}
    archive_metrics = ARCHIVE / "e2/metrics.csv"
    provenance = {"e2": dict(checkpoint_sha256=sha256_file(checkpoints["e2"]),
                              native_metrics_sha256=sha256_file(archive_metrics), native_metrics=archive_metrics.as_posix())}
    for mode in ("feature", *MODES):
        if mode == "feature":
            raw, info = phase.evaluate(reference, mode, manifest, device)
        else:
            raw, info = evaluate_fsc(mode, manifest, device)
        metrics = raw / "metrics.csv"
        rows[mode] = [(value, [], []) for value in read_scene_rows(metrics, phase.METRICS, ids)]
        provenance[mode] = dict(info, native_metrics=metrics.as_posix(), native_metrics_sha256=sha256_file(metrics))
    output = RESULTS / "val64_matched"
    meta = dict(scene_ids=ids, methods=provenance, metric_profile=DEFAULT_PROFILE, output_policy=phase.OUTPUT_POLICY,
                split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
                split_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest))
    if output.exists():
        if read_json(output / "run_metadata.json") != meta:
            raise ValueError(f"Preserve existing matched results: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        with TemporaryDirectory(prefix=".fsc-paired-", dir=RESULTS) as temporary:
            pending = Path(temporary) / "comparison"
            for name in rows:
                (pending / name).mkdir(parents=True)
            pairs = (("e2", "feature"), ("feature", "local_refresh"),
                     ("local_feature", "local_refresh"),
                     ("local_shuffled", "local_refresh"))
            summaries = summarize(pending, ids, rows, pairs, diagnostic_fields=(), source_metrics=())
            write_json(pending / "run_metadata.json", meta)
            write_json(pending / "registry_metrics.json", summaries)
            pending.rename(output)
    group = "QXS-E2-FSC-LOCAL-REFRESH-V1-VAL64-REF-FMI"
    entries = []
    for mode, metrics in summaries.items():
        path = output / mode / "metrics.csv"
        entries.append(dict(id=f"{group}-{mode}", task="fusion", status="development_validation",
                            comparable_group=group, method=LABELS[mode], dataset="QXS-SAROPT",
                            split="val", offset=0, sample_count=len(ids), checkpoint=checkpoints[mode].as_posix(),
                            checkpoint_sha256=provenance[mode]["checkpoint_sha256"],
                            split_manifest=meta["split_manifest"], split_manifest_sha256=meta["split_manifest_sha256"],
                            metric_profile=DEFAULT_PROFILE, output_policy=phase.OUTPUT_POLICY,
                            result_dir=output.as_posix(), metadata_path=(output/"run_metadata.json").as_posix(),
                            metrics=metrics, metric_source=dict(type="csv_row", path=path.as_posix(),
                                                               match={"image": "mean"}, columns={k:k for k in metrics})))
    register_many(entries)
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)


def smoke(manifest, device, full):
    scene = manifest["splits"]["val"][0]
    patch_size = 256 if full else 64
    optical, sar = (to_tensor(image, device)[..., :patch_size, :patch_size]
                    for image in read_pair(locate_scene(DATA, manifest, scene)))
    valid = (sar > 0).float()
    matched_weights, updates = {}, {}
    for mode, pilot in PILOTS.items():
        torch.manual_seed(42)
        model = pilot.build_model().to(device)
        current = model.state_dict()
        if not matched_weights:
            matched_weights = {key: value.cpu().clone() for key, value in current.items()}
        assert current.keys() == matched_weights.keys()
        for key, value in current.items():
            torch.testing.assert_close(value.cpu(), matched_weights[key], rtol=0, atol=0)
        branch = model.detail_fusion
        with torch.no_grad():
            bo, od, bs, sd = model.encode_pair(optical, sar)
            fused_base = bo + model.base_fusion(bo, bs)
            expected = model.reconstruct(optical, fused_base, branch.keep_fusion(od, sd, fused_base))
            torch.testing.assert_close(model(optical, sar), expected, rtol=0, atol=0)
            for x in (od, sd):
                bands = branch.bands(x)
                assert bands[0].norm() < 1e-6 * x.norm() and sum(b.norm() for b in bands[1:]) > 0
            scan = branch.ssm.core.row_scan
            joint = branch.signed_joint(od, sd)
            row_index = joint.shape[-2] // 2
            x = joint[0, :, row_index, :].T[None]
            initial = x.new_zeros(1, scan.inner, scan.state_channels)
            torch.testing.assert_close(scan_with_state(scan, x, initial), scan(x), rtol=1e-5, atol=1e-6)
            low = branch.ssm.ll_projection(torch.cat((branch.bands(bo)[0], branch.bands(bs)[0]), dim=1)).tanh()
            summary = branch.ssm.terminal(low[0, :, row_index, :].T[None])
            injected = branch.ssm.initial_state(CONTEXT_SCALE * summary, scan)
            head, middle, tail = (section.square().mean().sqrt().item() for section in
                                  (scan_with_state(scan, x, injected) - scan_with_state(scan, x, initial)).chunk(3, dim=1))
            if mode == "local_refresh":
                print(f"{scene}: single-initial-state reference head={head:.3e}, middle={middle:.3e}, tail={tail:.3e}", flush=True)
            # Open the common zero projection to test gradients through LL transfer.
            branch.update_projection[-1].weight.fill_(0.001)
            updates[mode] = branch.update(od, sd, bo, bs).cpu()
        losses = DirectFusionLosses().to(device)
        args = training_args(pilot)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        mixed = device.type == "cuda" and not args.no_amp
        with torch.autocast("cuda", enabled=mixed):
            loss, _ = forward_losses(model, losses, optical, sar, valid, args)
        assert torch.isfinite(loss)
        # The normal training loop uses a GradScaler. Reproduce loss scaling
        # here without stepping an optimizer or changing a checkpoint.
        (loss * (1024 if mixed else 1)).backward()
        for parameter in (branch.ssm.ll_projection.weight, branch.ssm.decay_logit, branch.update_projection[-1].weight):
            gradient = parameter.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().any()
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        print(f"{mode} {patch_size}x{patch_size}: loss={float(loss):.6f}, peak={peak:.2f} GiB, no optimizer step", flush=True)
        del loss, branch, model
    assert (updates["local_feature"] - updates["local_refresh"]).norm() > 0
    assert (updates["local_shuffled"] - updates["local_refresh"]).norm() > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val"))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--full", action="store_true", help="Use the full single image for smoke.")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = source_manifest()
    if args.action == "smoke":
        smoke(manifest, device, args.full)
    elif args.action == "train":
        if args.mode is None:
            parser.error("--mode is required for train.")
        pilot = PILOTS[args.mode]
        if pilot_manifest(pilot, 42) != manifest:
            raise ValueError("Expected the archived train1024/val64 split.")
        fsc_lock(pilot, create=True)
        train(pilot)
    else:
        validate(manifest, device)


if __name__ == "__main__":
    main()
