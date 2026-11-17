"""Matched pure-feature, positional and SAR-position-shuffled SSM pilot."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import run_e2_phase_guide as phase
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from e2_base_anchor import OpticalBaseAnchoredE2
from e2_pg_mamba import MODES, POSITION_SCALE, VARIANT_PREFIX, PGMambaE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize, write_csv
from run_e2_local_correspondence import archived_baseline, read_scene_rows, source_manifest, verify_training
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    ORIGINAL_LOSS_PILOT, STAGE_LEARNING_RATES, TOTAL_EPOCHS, config_values,
    pilot_manifest, reconstruction_forward, train, training_args,
)
from run_qxs_reference_test1000 import DATA, read_json, write_json
from train_lwpd import DirectFusionLosses
from train_pac_mamba import forward_losses
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parent
RESULTS = Path("test_result/e2_pg_mamba_v1")
E2_ARCHIVE = Path("test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val")
CROP_SIZE, CROP_OFFSET, MARGIN = 224, 2, 16
LABELS = dict(e2="归档光学 Base E2", feature="纯特征 SSM，零位置码",
              pg="共享固定二维位置码 SSM", sar_shuffled="仅 SAR 位置码打乱的 SSM")


class PGPilot(phase.PhaseGuidePilot):
    def build_model(self):
        return PGMambaE2(self.variant[len(VARIANT_PREFIX):])


PILOTS = {"feature": phase.PILOTS["feature"]}
PILOTS.update({
    mode: PGPilot(f"e2_pg_mamba_v1_{mode}_pilot", f"E2-PG-MAMBA-V1-{mode.upper()}",
                  VARIANT_PREFIX + mode, "run_e2_pg_mamba.py", LABELS[mode],
                  "从头初始化，沿用 train1024/val64、4 轮重建和 12 轮原损失融合。",
                  train_count=1024, val_count=64)
    for mode in MODES
})


def pg_lock(pilot, create=False):
    settings = dict(config_values(training_args(pilot)), total_epochs=TOTAL_EPOCHS,
                    stage_learning_rates=STAGE_LEARNING_RATES)
    reference = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    verify_training(settings, {key: reference[key] for key in settings}, resolved=False)
    paths = ("run_e2_pg_mamba.py", "e2_pg_mamba.py", "e2_phase_guide.py", "e2_base_anchor_pac.py", "pac_mamba.py",
             "lwpd.py", "e2_base_anchor.py", "direct_fusion.py", "net.py", "run_e2_scratch_twostage.py",
             "train_lwpd.py", "train_pac_mamba.py", "train_direct_fusion.py", "utils/dataset_os.py")
    expected = dict(settings=settings, source_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                    sources={name: sha256_file(Path(name)) for name in paths})
    path = pilot.run_dir.parent / "workflow_protocol.json"
    if path.is_file():
        if read_json(path) != expected:
            raise ValueError(f"Existing experiment uses another protocol: {path}")
    elif create and not pilot.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
    else:
        raise FileNotFoundError(path)
    return expected, path


@torch.inference_mode()
def crop_rms(model, o, s):
    size, step, margin = CROP_SIZE, CROP_OFFSET, MARGIN
    left = model(o[..., :size, :size], s[..., :size, :size])
    right = model(o[..., step:size+step, step:size+step], s[..., step:size+step, step:size+step])
    overlap_left = left[..., step+margin:-margin, step+margin:-margin]
    overlap_right = right[..., margin:-(step+margin), margin:-(step+margin)]
    if not torch.isfinite(overlap_left).all() or not torch.isfinite(overlap_right).all():
        raise FloatingPointError("Non-finite crop comparison.")
    return float((overlap_left - overlap_right).float().square().mean().sqrt())


@torch.inference_mode()
def evaluate_crop(mode, selected, checkpoint, device):
    directory = RESULTS / "crop_val64" / mode
    scene_ids = [pair[0].stem for pair in selected]
    metadata = dict(scene_ids=scene_ids, checkpoint_sha256=sha256_file(checkpoint), device=str(device),
                    crop_size=CROP_SIZE, offset_xy=[CROP_OFFSET, CROP_OFFSET], margin=MARGIN,
                    evaluator_sha256=sha256_file(Path(__file__)))
    if directory.exists():
        if read_json(directory / "run_metadata.json") != metadata:
            raise ValueError(f"Existing crop results use another protocol: {directory}")
    else:
        model = (OpticalBaseAnchoredE2() if mode == "e2" else PILOTS[mode].build_model()).to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["variant"] != model.variant:
            raise ValueError(f"Checkpoint variant differs from the requested model: {checkpoint}")
        model.load_state_dict(state["model"], strict=True)
        directory.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".crop-", dir=directory.parent) as temporary:
            output = Path(temporary) / "crop"
            output.mkdir()
            values = []
            for scene, pair in zip(scene_ids, selected):
                o, s = (to_tensor(image, device) for image in read_pair(pair))
                values.append((scene, crop_rms(model, o, s)))
            write_csv(output / "crop.csv", ("image", "crop_rms"), values)
            write_json(output / "run_metadata.json", metadata)
            output.rename(directory)
    return directory / "crop.csv", metadata


def native_results(mode, manifest, device):
    pilot = PILOTS[mode]
    if mode == "feature":
        return phase.evaluate(pilot, mode, manifest, device)
    # Reuse the signed-Haar pilot evaluator with a distinct output and lock.
    with patch.object(phase, "RESULTS", RESULTS), patch.object(phase, "locked_training", pg_lock):
        return phase.evaluate(pilot, mode, manifest, device)


def compare(manifest, modes, label, device):
    missing = [s for s in modes if not PILOTS[s].final.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed checkpoints: " + ", ".join(missing))
    scene_ids = manifest["splits"]["val"]
    reference = ORIGINAL_LOSS_PILOT
    selected = manifest_pairs(DATA, reference.manifest)[0]["val"]
    if [o.stem for o, _ in selected] != scene_ids:
        raise ValueError("Validation order differs from the archive.")
    native = {"e2": [row[0] for row in archived_baseline(scene_ids)]}
    native_artifacts = {"e2": (E2_ARCHIVE / "e2/metrics.csv").as_posix()}
    for mode in modes:
        raw, _ = native_results(mode, manifest, device)
        native[mode] = read_scene_rows(raw / "metrics.csv", phase.METRICS, scene_ids)
        native_artifacts[mode] = (raw / "metrics.csv").as_posix()
    records, provenance = {}, {}
    for mode, values in native.items():
        checkpoint = reference.final if mode == "e2" else PILOTS[mode].final
        raw_csv = Path(native_artifacts[mode])
        provenance[mode] = dict(checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                                native_metrics=raw_csv.as_posix(), native_metrics_sha256=sha256_file(raw_csv))
        if label == "full":
            crop_csv, crop_meta = evaluate_crop(mode, selected, checkpoint, device)
            crop = read_scene_rows(crop_csv, ("crop_rms",), scene_ids)
            records[mode] = [(value, error, []) for value, error in zip(values, crop)]
            provenance[mode].update(crop=crop_meta, crop_metrics=crop_csv.as_posix(),
                                    crop_metrics_sha256=sha256_file(crop_csv))
        else:
            records[mode] = [(value, [], []) for value in values]
    group = f"QXS-E2-PG-MAMBA-V1-VAL64-REF-FMI-{label.upper()}"
    output = RESULTS / f"val64_{label}"
    meta = dict(scenes=scene_ids, methods=provenance, metric_profile=DEFAULT_PROFILE,
                output_policy=phase.OUTPUT_POLICY, split_manifest=reference.manifest.as_posix(),
                split_manifest_sha256=sha256_file(reference.manifest))
    if output.exists():
        if read_json(output / "run_metadata.json") != meta:
            raise ValueError(f"Existing comparison uses another protocol: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        with TemporaryDirectory(prefix=".paired-", dir=RESULTS) as temporary:
            pending = Path(temporary) / "result"
            for mode in records:
                (pending / mode).mkdir(parents=True)
            contrasts = [("e2", "feature")]
            if label == "full":
                contrasts += [("e2", "pg"), ("feature", "pg"), ("sar_shuffled", "pg")]
            crop_metric = ("crop_rms",) if label == "full" else ()
            summaries = summarize(pending, scene_ids, records, contrasts, diagnostic_fields=(), source_metrics=crop_metric)
            write_json(pending / "run_metadata.json", meta)
            write_json(pending / "registry_metrics.json", summaries)
            pending.rename(output)
    entries = []
    for mode, metrics in summaries.items():
        path = output / mode / "metrics.csv"
        entries.append(dict(id=f"{group}-{mode}", method=LABELS[mode], comparable_group=group, task="fusion",
                            status="development_validation", dataset="QXS-SAROPT", split="val", offset=0,
                            sample_count=len(scene_ids), checkpoint=provenance[mode]["checkpoint"],
                            checkpoint_sha256=provenance[mode]["checkpoint_sha256"],
                            split_manifest=meta["split_manifest"], split_manifest_sha256=meta["split_manifest_sha256"],
                            metric_profile=DEFAULT_PROFILE, output_policy=phase.OUTPUT_POLICY,
                            result_dir=output.as_posix(), metadata_path=(output/"run_metadata.json").as_posix(),
                            metrics=metrics, metric_source=dict(type="csv_row", path=path.as_posix(),
                                                               match={"image": "mean"}, columns={k:k for k in metrics})))
    register_many(entries)
    print(f"Paired results: {output / 'paired_comparison.csv'}")


def smoke(manifest, device):
    scene = manifest["splits"]["val"][0]
    o, s = (to_tensor(image, device)[..., :64, :64] for image in read_pair(locate_scene(DATA, manifest, scene)))
    valid = (s > 0).float()
    weights, maps, updates, gate = {}, {}, {}, {}
    for mode, pilot in PILOTS.items():
        torch.manual_seed(42)
        model = pilot.build_model().to(device)
        if mode != "feature":
            current = {name.replace(".ssm.core.", ".ssm."): tensor for name, tensor in model.state_dict().items()}
            assert weights.keys() == current.keys()
            for name, parameter in current.items():
                torch.testing.assert_close(parameter.cpu(), weights[name], rtol=0, atol=0)
        else:
            weights = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
        args = training_args(pilot)
        losses = DirectFusionLosses().to(device)
        reconstruction, _ = reconstruction_forward(model, losses, o, s, valid, args)
        assert torch.isfinite(reconstruction)
        with torch.inference_mode():
            image, features = model(o, s, return_features=True)
            detail = features["pac"]
            torch.testing.assert_close(image, model.reconstruct(o, features["fused_base"], detail["keep"]), rtol=0, atol=0)
            gate[mode] = detail["reliability"]
            if mode != "feature":
                maps[mode] = model.detail_fusion.ssm.position_code(torch.zeros(1, 64, 32, 32, device=device))
        with torch.autocast("cuda", enabled=device.type == "cuda" and not args.no_amp):
            loss, _ = forward_losses(model, losses, o, s, valid, args)
        assert torch.isfinite(loss)
        loss.backward()
        last = model.detail_fusion.update_projection[-1]
        assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all() and last.weight.grad.abs().any()
        with torch.inference_mode():
            # Identical temporary open projection for the position intervention.
            last.weight.fill_(0.001)
            _, f = model(o, s, return_features=True)
            updates[mode] = f["pac"]["update"]
        print(f"{mode} on {scene}: loss={float(loss):.6f}, finite nonzero projection gradient; no optimizer update")
    for mode in MODES:
        torch.testing.assert_close(gate["feature"], gate[mode], rtol=0, atol=0)
        assert (updates["feature"] - updates[mode]).norm() > 0
    shared_o, shared_s = maps["pg"].chunk(2, 1)
    shuffle_o, shuffle_s = maps["sar_shuffled"].chunk(2, 1)
    torch.testing.assert_close(shared_o, shared_s, rtol=0, atol=0)
    torch.testing.assert_close(shuffle_o, shared_o, rtol=0, atol=0)
    torch.testing.assert_close(shared_s.flatten(2).sort().values, shuffle_s.flatten(2).sort().values, rtol=0, atol=0)
    assert not torch.equal(shared_s, shuffle_s)
    assert (updates["pg"] - updates["sar_shuffled"]).norm() > 0
    print(f"Identical trainable state and gate, fixed code scale {POSITION_SCALE}, distinct SSM updates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "val-baseline", "val"))
    parser.add_argument("--mode", choices=tuple(PILOTS))
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = source_manifest()
    if args.action == "smoke":
        smoke(manifest, device)
    elif args.action == "train":
        if args.mode is None:
            parser.error("--mode is required for train")
        pilot = PILOTS[args.mode]
        if pilot_manifest(pilot, 42) != manifest:
            raise ValueError("Expected the archived train1024/val64 split.")
        locker = phase.locked_training if args.mode == "feature" else pg_lock
        locker(pilot, create=True)
        train(pilot)
    elif args.action == "val-baseline":
        compare(manifest, ("feature",), "baseline", device)
    else:
        compare(manifest, tuple(PILOTS), "full", device)


if __name__ == "__main__":
    main()
