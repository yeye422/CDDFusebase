"""Train/evaluate OGDR on QXS without changing the original LWPD-E2 trainer.

Examples:
  python run_e2_ogdr.py smoke --mode guided
  python run_e2_ogdr.py check --mode guided
  python run_e2_ogdr.py train-val --mode guided --scale pilot
  python run_e2_ogdr.py resume --mode guided --scale pilot
  python run_e2_ogdr.py train-val --mode guided --scale full

Each mode receives an independent scratch seed42 run (4 reconstruction + 12
fusion epochs); original losses and input split are preserved. Validation uses
the reference_v2_candidate_with_paper_fmi profile and paired scene-level
comparisons with the archived, matching optical-Base E2.
"""

import argparse
import csv
import os
from pathlib import Path
import tempfile

import cv2
import torch
from filelock import FileLock

from diagnose_e2_information_path import read_pair, to_tensor
from e2_ogdr import MODES, OGDRE2, VARIANT_PREFIX
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, summarize
from run_e2_local_correspondence import ARCHIVE, archived_baseline
from run_e2_matched_no_pac import register_many
from run_e2_scratch_twostage import (
    BASE_ANCHOR, ORIGINAL_LOSS_PILOT, Experiment, STAGE_LEARNING_RATES,
    TOTAL_EPOCHS, config_values, pilot_manifest, train, training_args,
)
from run_e2_source_interval_val import RCCR, read_metric_rows
from run_e2_cross_scale import check_training_settings
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


ROOT = Path(__file__).resolve().parent
POLICY = "clamp_then_uint8"
DIAGNOSTICS = ("update_rms", "relative_update_rms", "guide_mean", "kernel_entropy")


class OGDRExperiment(Experiment):
    save_continuation = True

    def build_model(self):
        return OGDRE2(self.variant[len(VARIANT_PREFIX):])


def experiment(mode="guided", scale="pilot"):
    if mode not in MODES or scale not in ("pilot", "full"):
        raise ValueError("Unsupported OGDR mode or scale")
    count, val = (1024, 64) if scale == "pilot" else (8000, 1000)
    name = f"e2_ogdr_{mode}_{scale}_v1"
    return OGDRExperiment(
        name, f"E2-OGDR-{mode.upper()}-{scale.upper()}-V1",
        VARIANT_PREFIX + mode, "test_OS.py",
        f"光学Base E2 + OGDR Detail重建（{mode}，{scale}）",
        "随机初始化、4轮单源重建+12轮原损失融合；零初始化的局部差分残差。",
        train_count=count, val_count=val,
    )


def baseline_experiment(scale):
    return ORIGINAL_LOSS_PILOT if scale == "pilot" else BASE_ANCHOR


def reference_rows(exp, scale, ids):
    if scale == "pilot":
        return archived_baseline(ids)
    path = RCCR.result / "e2" / "metrics.csv"
    meta = read_json(RCCR.result / "run_metadata.json")
    control = BASE_ANCHOR
    if (meta.get("split") != "val" or meta.get("scene_ids") != ids
            or meta.get("metric_profile") != DEFAULT_PROFILE
            or meta.get("output_policy") != POLICY
            or meta.get("split_manifest_sha256") != sha256_file(exp.manifest)
            or meta.get("checkpoint_hashes", {}).get("e2") != sha256_file(control.final)):
        raise ValueError("Archived full E2 evaluation does not match current protocol")
    return read_metric_rows(path, ids)


def check(exp, scale):
    control = baseline_experiment(scale)
    if not control.final.is_file():
        raise FileNotFoundError(f"Missing matched optical-Base E2 checkpoint: {control.final}")
    settings = config_values(training_args(exp))
    baseline_config = read_json(control.run_dir / "config.json")
    check_training_settings(settings, baseline_config)
    if (baseline_config["total_epochs"] != TOTAL_EPOCHS
            or baseline_config["stage_learning_rates"] != STAGE_LEARNING_RATES):
        raise ValueError("The baseline training schedule differs")
    if scale == "pilot":
        if pilot_manifest(exp, 42) != read_json(control.manifest):
            raise ValueError("Pilot subset differs from the archived original-loss E2 pilot")
        ids = read_json(control.manifest)["splits"]["val"]
        archived_baseline(ids)
    else:
        meta = read_json(exp.manifest)
        if (len(meta["splits"]["train"]), len(meta["splits"]["val"])) != (8000, 1000):
            raise ValueError("Expected the fixed train8000/val1000 split")
        reference_rows(exp, scale, meta["splits"]["val"])
    print(f"Protocol matched: {exp.name}, train{exp.train_count}/val{exp.val_count}, "
          "seed42, original E2 losses, 4+12 epochs")


def smoke(mode):
    # Synthetic test: never writes weights, never touches the QXS dataset.
    from run_e2_scratch_twostage import configure
    import train_lwpd as trainer

    exp = experiment(mode, "pilot")
    args = training_args(exp)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = exp.build_model()
    losses = trainer.DirectFusionLosses()
    optimizer = trainer.make_optimizer(model, args)
    configure(model, optimizer, args, reconstruction=False)
    optical = torch.rand(1, 1, 21, 23)
    sar = torch.rand(1, 1, 21, 23)
    loss, _ = trainer.forward_losses(model, losses, optical, sar,
                                     torch.ones_like(sar), args)
    if not torch.isfinite(loss):
        raise FloatingPointError("OGDR smoke loss is non-finite")
    loss.backward()
    grad = model.detail_fusion.output_projection.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not grad.abs().any():
        raise FloatingPointError("OGDR output projection has no valid fusion gradient")
    print(f"OGDR smoke OK: {mode}, loss={float(loss):.6f}, no optimizer step")


def conditions(mode):
    options = {"candidate": None, "off": "off"}
    if mode == "guided":
        options["shifted"] = "shifted"
    elif mode == "shifted":
        options["aligned"] = "aligned"
    return options


@torch.inference_mode()
def validate(exp, scale):
    if not exp.final.is_file():
        raise FileNotFoundError(f"Missing candidate checkpoint: {exp.final}")
    config = read_json(exp.run_dir / "config.json")
    expected_settings = config_values(training_args(exp))
    if any(config.get(k) != v for k, v in expected_settings.items()):
        raise ValueError("Training config differs from the declared OGDR experiment")
    manifest = read_json(exp.manifest)
    if scale == "pilot" and manifest != pilot_manifest(exp, 42):
        raise ValueError("The candidate pilot manifest no longer matches the fixed QXS subset")
    ids = manifest["splits"]["val"]
    control_rows = reference_rows(exp, scale, ids)
    splits, _ = manifest_pairs(DATA, exp.manifest)
    if [pair[0].stem for pair in splits["val"]] != ids:
        raise ValueError("Validation scene order differs from the split manifest")
    saved = torch.load(exp.final, map_location="cpu", weights_only=False)
    if (saved["variant"] != exp.variant or saved["epoch"] != TOTAL_EPOCHS - 1
            or saved["stage"] != "fusion"
            or config_values(argparse.Namespace(**saved["args"])) != expected_settings):
        raise ValueError("Expected the final OGDR checkpoint with its matching training settings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.build_model().to(device).eval()
    model.load_state_dict(saved["model"], strict=True)

    mode = exp.variant[len(VARIANT_PREFIX):]
    variants = conditions(mode)
    control = baseline_experiment(scale)
    metadata = dict(
        dataset="QXS-SAROPT", split="val", offset=0, scene_ids=ids,
        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, mode=mode, scale=scale,
        conditions=variants, checkpoint=exp.final.as_posix(),
        checkpoint_sha256=sha256_file(exp.final),
        baseline_checkpoint=control.final.as_posix(),
        baseline_checkpoint_sha256=sha256_file(control.final),
        split_manifest=exp.manifest.as_posix(),
        split_manifest_sha256=sha256_file(exp.manifest),
        config_sha256=sha256_file(exp.run_dir / "config.json"),
        guide="Sobel of smoothed optical Detail absolute-channel activity",
        device=str(device),
    )
    output = exp.result
    if output.exists():
        if read_json(output / "run_metadata.json") != metadata:
            raise FileExistsError(f"Existing validation provenance differs: {output}")
        summaries = read_json(output / "registry_metrics.json")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ogdr-", dir=output.parent) as temporary:
            work = Path(temporary) / "result"
            for key in ("e2", *variants):
                (work / key / "gray").mkdir(parents=True)
            results = {"e2": control_rows, **{key: [] for key in variants}}
            diagnostics = {key: [] for key in variants}
            for index, pair in enumerate(splits["val"], 1):
                optical, sar = read_pair(pair)
                o, s = (to_tensor(image, device) for image in (optical, sar))
                encoded = model.encode_pair(o, s)
                for key, condition in variants.items():
                    base, (detail, stats) = model.fuse_features(
                        *encoded, condition=condition, return_diagnostics=True)
                    fused = model.reconstruct(o, base, detail)
                    if not torch.isfinite(fused).all():
                        raise FloatingPointError(f"Non-finite fusion: {pair[0].stem}/{key}")
                    gray = fused[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
                    path = work / key / "gray" / f"{pair[0].stem}.png"
                    if not cv2.imwrite(str(path), gray):
                        raise OSError(f"Could not write {path}")
                    results[key].append((FusionEvaluator.metric_vector(gray, optical, sar), (), ()))
                    diagnostics[key].append([pair[0].stem, *[float(stats[k]) for k in DIAGNOSTICS]])
                print(f"OGDR {scale} val {index}/{len(ids)}: {pair[0].stem}", flush=True)
            contrasts = [("e2", "candidate"), ("off", "candidate")]
            contrasts.extend([(key, "candidate") for key in variants if key not in ("candidate", "off")])
            summaries = summarize(work, ids, results, contrasts,
                                  diagnostic_fields=(), source_metrics=())
            for key, rows in diagnostics.items():
                with (work / key / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("image", *DIAGNOSTICS))
                    writer.writerows(rows)
            write_json(work / "run_metadata.json", metadata)
            write_json(work / "registry_metrics.json", summaries)
            work.rename(output)

    group = (f"QXS-E2-OGDR-V1-{scale.upper()}-SEED42-"
             f"VAL{exp.val_count}-REFERENCE-FMI")
    entries = []
    for key, metrics in summaries.items():
        source = control if key == "e2" else exp
        folder = output / key
        entries.append(dict(
            id=f"{group}-{mode}-{key}", comparable_group=group, task="fusion",
            status="development_validation", method=f"{source.method} [{key}]",
            dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(ids),
            checkpoint=source.final.as_posix(),
            checkpoint_sha256=sha256_file(source.final),
            split_manifest=exp.manifest.as_posix(),
            split_manifest_sha256=metadata["split_manifest_sha256"],
            metric_profile=DEFAULT_PROFILE, output_policy=POLICY,
            result_dir=folder.as_posix(),
            metadata_path=(output / "run_metadata.json").as_posix(),
            metrics=metrics, metric_source=dict(
                type="csv_row", path=(folder / "metrics.csv").as_posix(),
                match={"image": "mean"}, columns={name: name for name in metrics}),
            notes="Same checkpoint controls are frozen interventions, not independently trained models.",
        ))
    register_many(entries)
    print("Paired metrics:", output / "paired_comparison.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="check",
                        choices=("smoke", "check", "train", "resume", "val", "train-val"))
    parser.add_argument("--mode", choices=MODES, default="guided")
    parser.add_argument("--scale", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.action == "smoke":
        smoke(args.mode)
        return
    exp = experiment(args.mode, args.scale)
    check(exp, args.scale)
    if args.action == "check":
        return
    exp.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(exp.run_dir.parent / ".seed42.lock"), timeout=0):
        if args.action in ("train", "train-val", "resume"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use a CUDA-enabled PyTorch build for training")
            train(exp, resume=args.action == "resume")
        if args.action in ("val", "train-val", "resume"):
            validate(exp, args.scale)


if __name__ == "__main__":
    main()
