"""User-run E2 source-objective pilot; defaults to a one-scene frozen probe."""

import argparse
from contextlib import ExitStack
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import torch

import e2_source_objective as candidate
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, SOURCE_METRICS, summarize
from run_e2_local_correspondence import ARCHIVE, archived_baseline, read_scene_rows
from run_e2_matched_no_pac import register_many
import run_e2_scratch_twostage as protocol
from run_qxs_reference_test1000 import DATA, read_json, verify_images, write_json
from train_direct_fusion import DirectFusionLosses
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


# Python 3.8 may set __file__ to a relative CLI path. Resolve it before main()
# changes cwd, including invocation from a parent directory.
SCRIPT_PATH = Path(__file__).resolve()
MODULE_PATH = Path(candidate.__file__).resolve()
ROOT = SCRIPT_PATH.parent
ARMS = ("objective", "readout", "joint")
OUTPUT_POLICY = "clamp_then_uint8"
STATS = (*candidate.LOSS_STATS, *candidate.READOUT_STATS)
BASE_TRAINING_ARGS = protocol.training_args


def runtime_info():
    packages = {}
    for name in ("numpy", "scipy", "scikit-image", "kornia", "opencv-python", "einops"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version.split()[0], executable=str(Path(sys.executable).resolve()),
                torch=str(torch.__version__), cuda_build=torch.version.cuda, packages=packages)


def ensure_training_cuda():
    if torch.cuda.is_available():
        return
    reason = ("This PyTorch build is CPU-only." if torch.version.cuda is None else
              "This CUDA-enabled PyTorch cannot access a GPU; check the driver and device visibility.")
    raise RuntimeError(
        f"CUDA training preflight failed.\nPython: {sys.executable}\n"
        f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}\n{reason}\n"
        'Select a verified CUDA environment with set "E2_PYTHON=<absolute python.exe path>" '
        'and run run_e2_source_objective.cmd check before training.'
    )


class SourcePilot(protocol.Experiment):
    def build_model(self):
        return (candidate.SourceObjectiveE2() if self.variant == candidate.SourceObjectiveE2.VARIANT
                else candidate.SourceReadoutE2())


def experiment(arm):
    if arm not in ARMS:
        raise ValueError("Unknown source-objective arm")
    return SourcePilot(
        name=f"e2_source_objective_v1_{arm}", experiment_id=f"E2-SOURCE-OBJECTIVE-V1-{arm.upper()}",
        variant=candidate.SourceObjectiveE2.VARIANT if arm == "objective" else candidate.SourceReadoutE2.VARIANT,
        evaluator="test_OS.py", method=f"LWPD E2 source-objective ablation: {arm}",
        notes="固定train1024/val64，整套从头4轮重建+12轮融合；复用已有E2，不默认重训对照。",
        train_count=1024, val_count=64,
    )


def training_args(exp):
    args = BASE_TRAINING_ARGS(exp)
    arm = exp.name.rsplit("_", 1)[-1]
    args.source_objective_arm = arm
    args.source_objective = json.loads(json.dumps(candidate.SPEC)) if arm != "readout" else None
    args.source_readout = candidate.READOUT_SPEC if arm != "objective" else None
    args.source_objective_sha256 = sha256_file(MODULE_PATH)
    args.source_runtime = runtime_info()
    return args


def source_manifest(exp):
    original = read_json(protocol.ORIGINAL_LOSS_PILOT.manifest)
    if original != protocol.pilot_manifest(exp, 42):
        raise ValueError("Archived baseline split no longer matches train1024/val64")
    baseline = read_json(protocol.ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    ignored = {"variant", "output_dir", "split_manifest", "source_objective_arm", "source_objective",
               "source_readout", "source_objective_sha256", "source_runtime"}
    settings = protocol.config_values(training_args(exp))
    different = [key for key, value in settings.items() if key not in ignored and baseline.get(key) != value]
    if different or baseline.get("total_epochs") != protocol.TOTAL_EPOCHS:
        raise ValueError(f"Baseline training differs outside the declared ablation: {different}")
    if baseline.get("stage_learning_rates") != protocol.STAGE_LEARNING_RATES:
        raise ValueError("Baseline learning-rate schedule differs")
    return original


def signature(exp):
    paths = (SCRIPT_PATH, MODULE_PATH, ROOT / "run_e2_scratch_twostage.py",
             ROOT / "train_e2_full_overshoot.py", ROOT / "train_lwpd.py", ROOT / "train_direct_fusion.py",
             ROOT / "direct_fusion.py", ROOT / "net.py", ROOT / "lwpd.py", ROOT / "e2_base_anchor.py",
             ROOT / "utils/dataset_os.py", ROOT / "utils/fusion_metrics.py",
             ROOT / "utils/reference_metrics.py", ROOT / "utils/Evaluator.py", ROOT / "utils/legacy_dssim.py")
    return json.loads(json.dumps(dict(
        settings=protocol.config_values(training_args(exp)),
        baseline_manifest_sha256=sha256_file(protocol.ORIGINAL_LOSS_PILOT.manifest),
        sources={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths},
        epochs=protocol.TOTAL_EPOCHS, learning_rates=protocol.STAGE_LEARNING_RATES,
    )))


def lock_training(exp, create=False):
    path = exp.run_dir.parent / "workflow_protocol.json"
    expected = signature(exp)
    if path.exists():
        if read_json(path) != expected:
            if not create or exp.run_dir.exists():
                raise ValueError(f"Preserve run with different sources/settings: {path}")
            # A failed preflight may have left only a lock, with no training
            # directory or weights. Preserve that record before refreshing it.
            previous = path.read_bytes()
            archive = path.with_name(f"workflow_protocol.unstarted-{sha256_file(path)[:12]}.json")
            if archive.exists():
                if archive.read_bytes() != previous:
                    raise ValueError(f"Conflicting unstarted protocol archive: {archive}")
            else:
                with archive.open("xb") as handle:
                    handle.write(previous)
            temporary = path.with_suffix(".pending.json")
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.replace(path)
            print(f"Refreshed unstarted protocol; previous lock preserved: {archive}")
    elif create and not exp.run_dir.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        raise FileNotFoundError(f"Missing training protocol: {path}")
    return path


def train(arm):
    ensure_training_cuda()
    # Validate the loss API before a lock or training directory is created.
    (candidate.SourceContrastLosses if arm != "readout" else DirectFusionLosses)()
    exp = experiment(arm)
    manifest = source_manifest(exp)
    # Validate reusable baseline evidence before creating or starting a run.
    archived_baseline(manifest["splits"]["val"])
    lock = lock_training(exp, create=True)
    if exp.final.exists():
        config = read_json(exp.run_dir / "config.json")
        if any(config.get(k) != v for k, v in read_json(lock)["settings"].items()):
            raise ValueError("Completed checkpoint has different training settings")
    loss_type = candidate.SourceContrastLosses if arm != "readout" else DirectFusionLosses
    with ExitStack() as stack:
        stack.enter_context(patch.object(protocol, "training_args", training_args))
        stack.enter_context(patch.object(protocol, "reconstruction_forward", candidate.reconstruction_forward))
        stack.enter_context(patch.object(protocol, "FIELDS", (*protocol.FIELDS, *STATS)))
        stack.enter_context(patch.object(protocol.trainer, "DirectFusionLosses", loss_type))
        stack.enter_context(patch.object(protocol.trainer, "forward_losses", candidate.forward_losses))
        protocol.train(exp)


def blank_diagnostics(rows):
    return [(row[0], row[1], [None] * len(STATS)) for row in rows]


def write_artifacts(folder):
    payload = {path.relative_to(folder).as_posix(): sha256_file(path)
               for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "artifacts_sha256.json"}
    write_json(folder / "artifacts_sha256.json", payload)


def verify_artifacts(folder):
    recorded = read_json(folder / "artifacts_sha256.json")
    actual = {path.relative_to(folder).as_posix(): sha256_file(path)
              for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "artifacts_sha256.json"}
    if recorded != actual:
        raise ValueError(f"Saved result files changed: {folder}")


def image_scores(image, optical, sar):
    gray = image[0, 0].detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    scores = FusionEvaluator.metric_vector(gray, optical, sar)
    sources = [metric(gray, source) for metric in (FusionEvaluator._ssim_pair, FusionEvaluator._ms_ssim_pair)
               for source in (optical, sar)]
    return gray, scores, sources


def save_images(folder, pair, gray, optical, sar):
    color = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
    if color is None:
        raise OSError(f"Cannot read optical image: {pair[0]}")
    ycc = cv2.cvtColor(color, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = gray
    rgb = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    preview = np.concatenate((color, cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR), rgb), axis=1)
    for name, value in (("gray", gray), ("rgb", rgb), ("preview", preview)):
        target = folder / name / (pair[0].stem + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), value):
            raise OSError(f"Cannot save {target}")


def register_results(output, metadata, summaries, checkpoints, group, *, task="fusion"):
    entries = []
    for key, checkpoint in checkpoints.items():
        metrics = summaries[key]
        entries.append(dict(
            id=f"{group}-{key}", comparable_group=group, dataset="QXS-SAROPT", split="val", offset=0,
            task=task, status="development_validation" if task == "fusion" else "diagnostic_fixed_subset",
            method=f"E2 source-objective v1: {key}", sample_count=len(metadata["scene_ids"]),
            checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
            split_manifest=metadata["split_manifest"], split_manifest_sha256=metadata["split_manifest_sha256"],
            metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY, result_dir=(output / key).as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(), metrics=metrics,
            metric_source=dict(type="csv_row", path=(output / key / "metrics.csv").as_posix(),
                               match={"image": "mean"}, columns={name: name for name in metrics}),
            notes="逐项共同有限场景；保留逐图值。e2复用既有权重/输出，诊断干预不修改权重。",
        ))
    register_many(entries)


def validate(arm):
    exp = experiment(arm)
    manifest = source_manifest(exp)
    lock = lock_training(exp)
    settings = read_json(lock)["settings"]
    config = read_json(exp.run_dir / "config.json")
    if any(config.get(k) != v for k, v in settings.items()):
        raise ValueError("Saved training settings do not match the locked objective")
    if read_json(exp.manifest) != manifest:
        raise ValueError("Candidate manifest differs from the reusable baseline")
    baseline_config = read_json(protocol.ORIGINAL_LOSS_PILOT.run_dir / "config.json")
    for name, key in (("split_manifest.json", "saved_split_manifest_sha256"),
                      ("pilot_subset.json", "pilot_subset_sha256")):
        if config.get(key) != baseline_config[key] or sha256_file(exp.run_dir / name) != baseline_config[key]:
            raise ValueError("Resolved pairs or crops differ from the archived baseline")
    ids = manifest["splits"]["val"]
    baseline = blank_diagnostics(archived_baseline(ids))
    metadata = dict(arm=arm, scene_ids=ids, dataset="QXS-SAROPT", split="val", offset=0,
                    metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY,
                    split_manifest=exp.manifest.as_posix(), split_manifest_sha256=sha256_file(exp.manifest),
                    checkpoint=exp.final.as_posix(), checkpoint_sha256=sha256_file(exp.final),
                    baseline_checkpoint=protocol.ORIGINAL_LOSS_PILOT.final.as_posix(),
                    baseline_checkpoint_sha256=sha256_file(protocol.ORIGINAL_LOSS_PILOT.final),
                    baseline_metadata_sha256=sha256_file(ARCHIVE / "run_metadata.json"),
                    baseline_metrics_sha256=sha256_file(ARCHIVE / "e2/metrics.csv"),
                    baseline_diagnostics_sha256=sha256_file(ARCHIVE / "e2/diagnostics.csv"),
                    training_protocol_sha256=sha256_file(lock), evaluator_sha256=sha256_file(SCRIPT_PATH))
    output = exp.result
    if output.exists():
        verify_artifacts(output)
        if read_json(output / "run_metadata.json") != metadata:
            raise ValueError(f"Preserve results with different provenance: {output}")
        summaries = read_json(output / "registry_metrics.json")
        for name in ("gray", "rgb", "preview"):
            verify_images(output / "candidate" / name, ids)
    else:
        state = torch.load(exp.final, map_location="cpu", weights_only=False)
        if (state["variant"], state["epoch"], state["stage"]) != (exp.variant, protocol.TOTAL_EPOCHS - 1, "fusion"):
            raise ValueError("Expected the final committed fusion checkpoint")
        if protocol.config_values(argparse.Namespace(**state["args"])) != settings:
            raise ValueError("Checkpoint configuration differs from the training lock")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = exp.build_model().to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        losses = candidate.SourceContrastLosses().to(device)
        splits, _ = manifest_pairs(DATA, exp.manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".source-objective-", dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            (pending / "e2").mkdir(parents=True)
            rows = []
            with torch.inference_mode():
                for index, pair in enumerate(splits["val"], 1):
                    optical, sar = read_pair(pair)
                    o, s = (to_tensor(im, device) for im in (optical, sar))
                    fused = model(o, s)
                    losses.fusion(o, s, fused, (s > 0).float(), 10.0)
                    stats = dict(losses.stats, **getattr(model.decoder, "stats", {}))
                    gray, scores, sources = image_scores(fused, optical, sar)
                    save_images(pending / "candidate", pair, gray, optical, sar)
                    rows.append((scores, sources, [float(stats[key]) if key in stats else None for key in STATS]))
                    print(f"val {index}/{len(ids)} {pair[0].stem}", flush=True)
            summaries = summarize(pending, ids, {"e2": baseline, "candidate": rows},
                                  (("e2", "candidate"),), diagnostic_fields=STATS)
            write_json(pending / "run_metadata.json", metadata)
            write_json(pending / "registry_metrics.json", summaries)
            write_artifacts(pending)
            pending.rename(output)
    register_results(output, metadata, summaries,
                     {"e2": protocol.ORIGINAL_LOSS_PILOT.final, "candidate": exp.final},
                     f"QXS-E2-SOURCE-OBJECTIVE-V1-{arm.upper()}-VAL64-REFERENCE-FMI")
    print(f"Paired results: {output / 'paired_comparison.csv'}")


def smoke(arm):
    """Synthetic forward/backward only: no optimizer, dataset, or checkpoint."""
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = experiment(arm).build_model()
    loss = candidate.SourceContrastLosses() if arm != "readout" else DirectFusionLosses()
    o, s = torch.rand(1, 1, 17, 19), torch.rand(1, 1, 17, 19)
    args = argparse.Namespace(gradient_weight=10.0, reconstruction_weight=0.1)
    for forward in (candidate.reconstruction_forward, candidate.forward_losses):
        model.zero_grad(set_to_none=True)
        value, _ = forward(model, loss, o, s, torch.ones_like(s), args)
        value.backward()
        if not torch.isfinite(value) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                           for p in model.parameters()):
            raise FloatingPointError("Non-finite synthetic smoke result")
    print(f"Synthetic smoke passed: {arm}, 17x19, no optimizer update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "probe", "smoke", "train", "val", "train-val", "compare"),
                        nargs="?", default="probe")
    parser.add_argument("--arm", choices=ARMS, default="objective")
    parser.add_argument("--scene-id", default="211", help="Frozen probe only; one fixed val scene")
    parser.add_argument("--output-dir", type=Path, help="Frozen probe only; refuses an existing directory")
    parser.add_argument("--project-detail", action="store_true", help="Probe-only bounded latent-target fit")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action != "probe" and (args.output_dir is not None or args.project_detail or args.scene_id != "211"):
        parser.error("--scene-id, --output-dir and --project-detail are probe-only")
    if args.action == "check":
        print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
        ensure_training_cuda()
        criterion = candidate.SourceContrastLosses()
        print(f"Reconstruction loss ready: {type(criterion.ssim).__name__}")
        print(f"CUDA ready: {torch.cuda.get_device_name(0)}; no training started")
    elif args.action == "smoke":
        smoke(args.arm)
    elif args.action == "probe":
        from diagnose_e2_source_objective import probe
        probe(args.scene_id, args.output_dir, project=args.project_detail)
    elif args.action == "compare":
        compare()
    else:
        if args.action in ("train", "train-val"):
            train(args.arm)
        if args.action in ("val", "train-val"):
            validate(args.arm)


def compare():
    """Combine completed cells only; never starts missing training/evaluation."""
    first = experiment("objective")
    manifest = source_manifest(first)
    ids = manifest["splits"]["val"]
    rows = {"e2": blank_diagnostics(archived_baseline(ids))}
    checkpoints = {"e2": protocol.ORIGINAL_LOSS_PILOT.final}
    origins = {}
    for arm in ARMS:
        exp = experiment(arm)
        verify_artifacts(exp.result)
        meta = read_json(exp.result / "run_metadata.json")
        if (meta["scene_ids"] != ids or meta["metric_profile"] != DEFAULT_PROFILE
                or meta["output_policy"] != OUTPUT_POLICY or meta["checkpoint_sha256"] != sha256_file(exp.final)
                or meta["baseline_checkpoint_sha256"] != sha256_file(checkpoints["e2"])
                or read_json(exp.manifest) != manifest):
            raise ValueError(f"Unmatched saved ablation cell: {arm}")
        lock_training(exp)
        metrics = read_scene_rows(exp.result / "candidate/metrics.csv", METRICS, ids)
        sources = read_scene_rows(exp.result / "candidate/diagnostics.csv", SOURCE_METRICS, ids)
        rows[arm] = [(m, s, [None] * len(STATS)) for m, s in zip(metrics, sources)]
        checkpoints[arm] = exp.final
        origins[arm] = {"metadata": sha256_file(exp.result / "run_metadata.json"),
                        "metrics": sha256_file(exp.result / "candidate/metrics.csv"),
                        "diagnostics": sha256_file(exp.result / "candidate/diagnostics.csv")}
    output = Path("test_result/e2_source_objective_v1/four_cell_val64/seed42")
    metadata = dict(scene_ids=ids, metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY,
                    split_manifest=protocol.ORIGINAL_LOSS_PILOT.manifest.as_posix(),
                    split_manifest_sha256=sha256_file(protocol.ORIGINAL_LOSS_PILOT.manifest), origins=origins,
                    baseline_metadata_sha256=sha256_file(ARCHIVE / "run_metadata.json"),
                    baseline_metrics_sha256=sha256_file(ARCHIVE / "e2/metrics.csv"),
                    baseline_diagnostics_sha256=sha256_file(ARCHIVE / "e2/diagnostics.csv"),
                    checkpoint_hashes={key: sha256_file(path) for key, path in checkpoints.items()},
                    evaluator_sha256=sha256_file(SCRIPT_PATH))
    if output.exists():
        verify_artifacts(output)
        if read_json(output / "run_metadata.json") != metadata:
            raise ValueError(f"Preserve an existing four-cell comparison: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".four-cell-", dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            for key in rows:
                (pending / key).mkdir(parents=True)
            summaries = summarize(pending, ids, rows,
                                  (("e2", "objective"), ("e2", "readout"), ("objective", "joint"),
                                   ("readout", "joint")), diagnostic_fields=STATS)
            write_json(pending / "run_metadata.json", metadata)
            write_json(pending / "registry_metrics.json", summaries)
            write_artifacts(pending)
            pending.rename(output)
    register_results(output, metadata, summaries, checkpoints,
                     "QXS-E2-SOURCE-OBJECTIVE-V1-FOUR-CELL-VAL64-REFERENCE-FMI")
    print(f"Four-cell comparison: {output / 'paired_comparison.csv'}")


if __name__ == "__main__":
    main()
