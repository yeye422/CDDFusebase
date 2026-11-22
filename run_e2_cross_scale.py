"""Matched cross-scale E2 pilot. Training/batch validation are user-run CMD actions."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
import torch

from e2_cross_scale import CrossScaleE2, DIAGNOSTICS, MODES, SPEC, VARIANT_PREFIX
from diagnose_e2_information_path import read_pair, to_tensor
from diagnose_fcsr_decoder import gradients, signed_transfer, source_specific_mask
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    Experiment, ORIGINAL_LOSS_PILOT, STAGE_LEARNING_RATES, TOTAL_EPOCHS,
    config_values, pilot_manifest, train, training_args,
)
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
ARCHIVE = Path("test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val")
POLICY = "clamp_then_uint8"
EDGE_METRICS = ("optical_candidate_transfer", "sar_candidate_transfer")
ALL_SOURCE_METRICS = (*SOURCE_METRICS, *EDGE_METRICS)
ALL_DIAGNOSTICS = (*DIAGNOSTICS, "optical_candidate_pixels", "sar_candidate_pixels")
CONDITIONS = ("aligned", "shifted", "disabled")


class CrossScalePilot(Experiment):
    @property
    def save_continuation(self):
        return True

    def build_model(self):
        return CrossScaleE2(self.variant[len(VARIANT_PREFIX):])


def experiment(mode="relation", tag="v1"):
    if mode not in MODES or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", tag):
        raise ValueError("Unknown mode or unsafe run tag")
    return CrossScalePilot(
        f"e2_cross_scale_pilot_{tag}_{mode}", f"E2-CROSS-SCALE-{tag.upper()}-{mode.upper()}",
        VARIANT_PREFIX + mode, "run_e2_cross_scale.py",
        f"光学Base E2，源内跨尺度条件（{mode}）",
        "从头4轮重建+12轮原损失融合；保留原多尺度头、Base及Decoder。",
        train_count=1024, val_count=64,
    )


def canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def read_rows(path, fields, scene_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["image"] not in ("mean", "std")]
    if [r["image"] for r in rows] != scene_ids:
        raise ValueError(f"Missing, duplicated, or reordered scenes: {path}")
    return [[float(r[name]) if r[name] not in ("", None) else math.nan for name in fields]
            for r in rows]


def baseline_cache(scene_ids):
    baseline = ORIGINAL_LOSS_PILOT
    metadata = read_json(ARCHIVE / "run_metadata.json")
    expected = dict(scene_ids=scene_ids, metric_profile=DEFAULT_PROFILE, output_policy=POLICY,
                    split_manifest_sha256=sha256_file(baseline.manifest))
    if (any(metadata.get(k) != v for k, v in expected.items())
            or Path(metadata["checkpoints"]["e2"]) != baseline.final
            or metadata["checkpoint_hashes"]["e2"] != sha256_file(baseline.final)):
        raise ValueError("Archived E2 does not match checkpoint, split, or evaluation profile")
    folder = ARCHIVE / "e2"
    verify_images(folder / "gray", scene_ids)
    metrics = read_rows(folder / "metrics.csv", METRICS, scene_ids)
    sources = read_rows(folder / "diagnostics.csv", SOURCE_METRICS, scene_ids)
    fingerprint = dict(
        metadata=sha256_file(ARCHIVE / "run_metadata.json"),
        metrics=sha256_file(folder / "metrics.csv"),
        diagnostics=sha256_file(folder / "diagnostics.csv"),
        gray={scene: sha256_file(folder / "gray" / f"{scene}.png") for scene in scene_ids},
    )
    return list(zip(metrics, sources)), fingerprint


def check_training_settings(config, baseline, *, resolved=False, run=None):
    ignored = {"variant", "output_dir", "split_manifest"}
    differences = [key for key, value in config.items()
                   if key not in ignored and baseline.get(key) != value]
    if differences:
        raise ValueError(f"Training budget differs from archived E2: {', '.join(sorted(differences))}")
    if config.get("init_checkpoint") is not None or config.get("initialization") != "random":
        raise ValueError("New candidates must start from scratch")
    if resolved:
        for filename, key in (("split_manifest.json", "saved_split_manifest_sha256"),
                              ("pilot_subset.json", "pilot_subset_sha256")):
            if sha256_file(run / filename) != baseline[key]:
                raise ValueError(f"Resolved training pairs differ: {filename}")


def require_cuda_runtime():
    """Fail before creating a run when the selected interpreter cannot train."""
    if torch.version.cuda is None:
        raise RuntimeError(
            f"CPU-only PyTorch detected: {torch.__version__}. "
            "Install the matching CUDA build in this .venv; an NVIDIA GPU/driver alone is insufficient."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} has CUDA {torch.version.cuda}, but no usable CUDA device. "
            "Check nvidia-smi, the driver, and CUDA_VISIBLE_DEVICES before starting training."
        )
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0))


def preflight(pilot, *, require_gpu=True):
    runtime = require_cuda_runtime() if require_gpu else None
    if not require_gpu:
        saved_lock = pilot.run_dir.parent / "workflow_protocol.json"
        if saved_lock.is_file():
            runtime = read_json(saved_lock).get("training_runtime")
    baseline = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    config = config_values(training_args(pilot))
    check_training_settings(config, baseline)
    manifest = read_json(ORIGINAL_LOSS_PILOT.manifest)
    if manifest != pilot_manifest(pilot, 42):
        raise ValueError("Expected the archived train1024/val64 manifest")
    _, archive = baseline_cache(manifest["splits"]["val"])
    files = (
        "e2_cross_scale.py", "run_e2_cross_scale.py", "run_e2_scratch_twostage.py",
        "direct_fusion.py", "lwpd.py", "e2_base_anchor.py", "net.py",
        "train_lwpd.py", "train_direct_fusion.py", "utils/dataset_os.py",
        "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py",
        "diagnose_fcsr_decoder.py", "run_e2_aspr_pilot.py", "run_e2_matched_no_pac.py",
    )
    lock = canonical(dict(config=config, spec=SPEC, total_epochs=TOTAL_EPOCHS,
                          training_runtime=runtime,
                          stage_learning_rates=STAGE_LEARNING_RATES,
                          manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                          baseline_checkpoint_sha256=sha256_file(ORIGINAL_LOSS_PILOT.final),
                          archive=archive, sources={p: sha256_file(ROOT / p) for p in files}))
    return manifest, lock


def training_lock(pilot, expected, *, create=False):
    path = pilot.run_dir.parent / "workflow_protocol.json"
    if path.exists():
        if read_json(path) != expected:
            if not create or pilot.run_dir.exists():
                raise ValueError(f"Preserve a different source/configuration snapshot: {path}")
            # An earlier preflight could create only this lock before CUDA
            # initialization failed. Preserve that snapshot, then allow a new
            # first attempt; never refresh a run with a seed42 directory.
            backup = path.with_name(f"workflow_protocol.prestart_{sha256_file(path)[:12]}.json")
            payload = path.read_bytes()
            if backup.exists():
                if backup.read_bytes() != payload:
                    raise ValueError(f"Conflicting pre-start provenance: {backup}")
            else:
                with backup.open("xb") as handle:
                    handle.write(payload)
            write_json(path, expected)
    elif create and not pilot.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        raise FileNotFoundError(f"Missing training provenance lock: {path}")
    return path


def proxy_readout(gray, optical, sar):
    """Same quantized-image source-gradient proxy for cached E2 and candidates."""
    o, s, f = [to_tensor(x, torch.device("cpu")) for x in (optical, sar, gray)]
    go, gs, gf = [gradients(x) for x in (o, s, f)]
    valid = s > 0
    masks = (source_specific_mask(go, gs, valid), source_specific_mask(gs, go, valid))
    transfers = [signed_transfer(g, gf, mask) for g, mask in zip((go, gs), masks)]
    return [math.nan if x is None else x for x in transfers], [int(m.sum()) for m in masks]


def save_visuals(folder, scene, gray, optical_color, optical, sar):
    for name in ("gray", "rgb", "preview"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    ycc = cv2.cvtColor(optical_color, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = gray
    images = dict(gray=gray, rgb=cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR),
                  preview=np.concatenate((optical, sar, gray), axis=1))
    for name, image in images.items():
        if not cv2.imwrite(str(folder / name / f"{scene}.png"), image):
            raise OSError(f"Could not save {scene}/{name}")


def seal_result(folder):
    files = {p.relative_to(folder).as_posix(): sha256_file(p)
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "result_checksums.json"}
    write_json(folder / "result_checksums.json", files)


def verify_result(folder, metadata=None, *, image_keys=()):
    if metadata is not None and read_json(folder / "run_metadata.json") != metadata:
        raise ValueError(f"Preserve output with different provenance: {folder}")
    files = read_json(folder / "result_checksums.json")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
              if p.is_file() and p.name != "result_checksums.json"}
    if actual != set(files) or any(sha256_file(folder / name) != value for name, value in files.items()):
        raise ValueError(f"Missing, extra, or modified result files: {folder}")
    ids = read_json(folder / "run_metadata.json")["scene_ids"]
    for key in image_keys:
        for name in ("gray", "rgb", "preview"):
            verify_images(folder / key / name, ids)


def register_validation(output, metadata, summaries, methods, group):
    entries = []
    for key, origin in methods.items():
        scores = summaries[key]
        folder = output / key
        entries.append(dict(
            id=f"{group}-{key}", comparable_group=group, task="fusion",
            status="development_validation", method=origin["method"], dataset="QXS-SAROPT",
            split="val", offset=0, sample_count=len(metadata["scene_ids"]),
            checkpoint=origin["checkpoint"], checkpoint_sha256=origin["checkpoint_sha256"],
            split_manifest=metadata["split_manifest"],
            split_manifest_sha256=metadata["split_manifest_sha256"],
            result_dir=folder.as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
            metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=scores,
            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in scores}),
            notes="逐项取全部方法共同有限场景；源候选传递为无标注弱代理。同权重关闭/错位不是独立训练消融。",
        ))
    register_many(entries)


@torch.inference_mode()
def validate(pilot, manifest, lock, device):
    lock_path = training_lock(pilot, lock)
    config = read_json(pilot.run_dir / "config.json")
    baseline_config = read_json(ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    expected = dict(lock["config"], total_epochs=TOTAL_EPOCHS,
                    stage_learning_rates=STAGE_LEARNING_RATES)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Saved candidate configuration differs from training lock")
    check_training_settings(expected, baseline_config)
    check_training_settings({k: config[k] for k in expected}, baseline_config,
                            resolved=True, run=pilot.run_dir)
    if read_json(pilot.manifest) != manifest:
        raise ValueError("Candidate manifest changed")
    ids = manifest["splits"]["val"]
    baseline_rows, _ = baseline_cache(ids)
    methods = {"e2": dict(method="已归档光学Base E2", checkpoint=ORIGINAL_LOSS_PILOT.final.as_posix(),
                           checkpoint_sha256=sha256_file(ORIGINAL_LOSS_PILOT.final))}
    for condition in CONDITIONS:
        methods[condition] = dict(method=f"{pilot.method}，condition={condition}",
                                  checkpoint=pilot.final.as_posix(), checkpoint_sha256=sha256_file(pilot.final))
    splits, _ = manifest_pairs(DATA, pilot.manifest)
    if [pair[0].stem for pair in splits["val"]] != ids:
        raise ValueError("Resolved validation scene order differs")
    input_hashes = {pair[0].stem: [sha256_file(p) for p in pair] for pair in splits["val"]}
    metadata = canonical(dict(
        scene_ids=ids, dataset="QXS-SAROPT", split="val", offset=0, evaluated_pairs=len(ids),
        split_manifest=pilot.manifest.as_posix(), split_manifest_sha256=sha256_file(pilot.manifest),
        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, device=str(device),
        training_protocol_sha256=sha256_file(lock_path), spec=SPEC, methods=methods,
        primary_condition="shifted" if pilot.variant.endswith("_shuffled") else "aligned",
        baseline_source=(ARCHIVE / "run_metadata.json").as_posix(), archive=lock["archive"],
        source_sha256=sha256_file(Path(__file__)),
        input_hashes=input_hashes, implementation_sources=lock["sources"],
        runtime=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
        training_settings={k: v for k, v in expected.items()
                           if k not in ("variant", "output_dir", "split_manifest")},
    ))
    output = pilot.result
    if output.exists():
        verify_result(output, metadata, image_keys=methods)
        summaries = read_json(output / "registry_metrics.json")
    else:
        saved = torch.load(pilot.final, map_location="cpu", weights_only=False)
        if ((saved["variant"], saved["epoch"], saved["stage"]) != (pilot.variant, TOTAL_EPOCHS - 1, "fusion")
                or config_values(argparse.Namespace(**saved["args"])) != lock["config"]):
            raise ValueError("Expected final fusion checkpoint with locked settings")
        model = pilot.build_model().to(device).eval()
        model.load_state_dict(saved["model"], strict=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cross-scale-", dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            for key in methods:
                (pending / key).mkdir(parents=True)
            results = {key: [] for key in methods}
            for index, pair in enumerate(splits["val"]):
                scene = ids[index]
                optical, sar = read_pair(pair)
                color = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
                cached = cv2.imread(str(ARCHIVE / "e2/gray" / f"{scene}.png"), cv2.IMREAD_GRAYSCALE)
                if cached is None or cached.shape != optical.shape or color is None:
                    raise ValueError(f"Invalid archived/source image: {scene}")
                transfer, counts = proxy_readout(cached, optical, sar)
                scores, source_scores = baseline_rows[index]
                results["e2"].append((scores, [*source_scores, *transfer], [None] * len(DIAGNOSTICS) + counts))
                save_visuals(pending / "e2", scene, cached, color, optical, sar)
                o, s = [to_tensor(image, device) for image in (optical, sar)]
                encoded = model.encode_pair(o, s)
                for condition in CONDITIONS:
                    base, (detail, stats) = model.fuse_features(*encoded, condition=condition, return_diagnostics=True)
                    fused = model.reconstruct(o, base, detail)
                    if not torch.isfinite(fused).all():
                        raise FloatingPointError(f"Non-finite output: {scene}/{condition}")
                    gray = fused[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
                    scores = FusionEvaluator.metric_vector(gray, optical, sar)
                    sources = [metric(gray, im) for metric in (FusionEvaluator._ssim_pair, FusionEvaluator._ms_ssim_pair)
                               for im in (optical, sar)]
                    transfer, counts = proxy_readout(gray, optical, sar)
                    diagnostics = [float(stats[name]) for name in DIAGNOSTICS] + counts
                    results[condition].append((scores, [*sources, *transfer], diagnostics))
                    save_visuals(pending / condition, scene, gray, color, optical, sar)
                print(f"val {index + 1}/{len(ids)} {scene}", flush=True)
            contrasts = [("e2", c) for c in CONDITIONS] + [("shifted", "aligned"), ("disabled", "aligned")]
            summaries = summarize(pending, ids, results, contrasts,
                                  diagnostic_fields=ALL_DIAGNOSTICS, source_metrics=ALL_SOURCE_METRICS)
            write_json(pending / "run_metadata.json", metadata)
            write_json(pending / "registry_metrics.json", summaries)
            seal_result(pending)
            pending.rename(output)
    group = f"QXS-E2-CROSS-SCALE-{pilot.experiment_id}-VAL64-REFERENCE-FMI"
    register_validation(output, metadata, summaries, methods, group)
    print(f"Paired validation: {output / 'paired_comparison.csv'}")


def compare(tag, *, include_shuffled=False):
    modes = ("relation", "wide", "shuffled") if include_shuffled else ("relation", "wide")
    results, methods, inputs = {}, {}, {}
    reference = None
    for mode in modes:
        pilot = experiment(mode, tag)
        verify_result(pilot.result, image_keys=("e2", *CONDITIONS))
        meta = read_json(pilot.result / "run_metadata.json")
        common = {k: meta[k] for k in (
            "scene_ids", "split_manifest_sha256", "metric_profile", "output_policy", "archive",
            "spec", "implementation_sources", "training_settings", "device", "input_hashes", "runtime",
        )}
        if reference is not None and common != reference:
            raise ValueError("Cannot compare different scenes, baselines, or evaluation profiles")
        reference = common
        primary = meta["primary_condition"]
        inputs[mode] = dict(metadata=(pilot.result / "run_metadata.json").as_posix(),
                            metadata_sha256=sha256_file(pilot.result / "run_metadata.json"), condition=primary)
        selections = [(mode, primary)]
        if "e2" not in results:
            selections.insert(0, ("e2", "e2"))
        for key, source_key in selections:
            folder = pilot.result / source_key
            scores = read_rows(folder / "metrics.csv", METRICS, meta["scene_ids"])
            sources = read_rows(folder / "diagnostics.csv", ALL_SOURCE_METRICS, meta["scene_ids"])
            diagnostics = read_rows(folder / "diagnostics.csv", ALL_DIAGNOSTICS, meta["scene_ids"])
            results[key] = [(m, s, d) for m, s, d in zip(scores, sources, diagnostics)]
            methods[key] = dict(meta["methods"][source_key], image_source=folder.as_posix())
    metadata = canonical(dict(reference, methods=methods, inputs=inputs,
                              split_manifest=experiment("relation", tag).manifest.as_posix(), spec=SPEC))
    output = Path(f"test_result/e2_cross_scale_comparison_{tag}/{'_'.join(modes)}_val64_seed42")
    if output.exists():
        verify_result(output, metadata)
        summaries = read_json(output / "registry_metrics.json")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cross-scale-compare-", dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            for key in results:
                (pending / key).mkdir(parents=True)
            contrasts = [("e2", mode) for mode in modes] + [("wide", "relation")]
            if include_shuffled:
                contrasts.append(("shuffled", "relation"))
            summaries = summarize(pending, reference["scene_ids"], results, contrasts,
                                  diagnostic_fields=ALL_DIAGNOSTICS, source_metrics=ALL_SOURCE_METRICS)
            write_json(pending / "run_metadata.json", metadata)
            write_json(pending / "registry_metrics.json", summaries)
            seal_result(pending)
            pending.rename(output)
    group = f"QXS-E2-CROSS-SCALE-{tag.upper()}-COMPARE-{'-'.join(modes).upper()}-VAL64-REFERENCE-FMI"
    register_validation(output, metadata, summaries, methods, group)
    print(f"Common finite-scene comparison: {output / 'paired_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "val", "train-val", "compare"),
                        nargs="?", default="train-val")
    parser.add_argument("--mode", choices=MODES, default="relation")
    parser.add_argument("--tag", default="v1", help="New tag for a deliberately separate run; never overwrite old runs")
    parser.add_argument("--resume", action="store_true", help="Resume the same run including optimizer and RNG")
    parser.add_argument("--include-shuffled", action="store_true", help="Include separately trained shuffled control in compare")
    args = parser.parse_args()
    os.chdir(ROOT)
    pilot = experiment(args.mode, args.tag)
    if args.action == "compare":
        compare(args.tag, include_shuffled=args.include_shuffled)
        return
    if args.resume and args.action not in ("train", "train-val"):
        parser.error("--resume is only valid for training the same run")
    manifest, lock = preflight(pilot, require_gpu=args.action != "val")
    if args.action == "preflight":
        print(f"Preflight passed: {pilot.variant}; {lock['training_runtime']}; "
              "archived E2 reused; no training/model inference")
        return
    if args.action in ("train", "train-val"):
        training_lock(pilot, lock, create=True)
        train(pilot, resume=args.resume)
    if args.action in ("val", "train-val"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validate(pilot, manifest, lock, device)


if __name__ == "__main__":
    main()
