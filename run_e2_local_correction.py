"""Matched E2 versus E2 + local correction pilot. Batch actions are user-run."""

import argparse
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

import e2_local_correction as correction
import run_e2_scratch_twostage as protocol
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_e2_cross_scale import baseline_cache, check_training_settings, seal_result, training_lock, verify_result
from run_e2_local_correspondence import ARCHIVE
from run_e2_matched_no_pac import register_many
from run_e2_source_objective import image_scores, runtime_info, save_images
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


ROOT = Path(__file__).resolve().parent
BASE = protocol.ORIGINAL_LOSS_PILOT
EXTRA_FIELDS = ("correction_reconstruction", "ranking", "coarse_fusion", "correction_mae")


class CorrectionPilot(protocol.Experiment):
    save_continuation = True

    def build_model(self):
        return correction.LocalCorrectionE2()


PILOT = CorrectionPilot("e2_local_correction_pilot_v1", "E2-LOCAL-CORRECTION-PILOT-V1",
                        correction.LocalCorrectionE2.VARIANT, "test_OS.py",
                        "光学 Base E2＋初融图引导的源别局部修正",
                        "整网从头训练 4+12 轮，比较完整模块与归档 E2。", train_count=1024, val_count=64)


def preflight():
    manifest = read_json(BASE.manifest)
    if protocol.pilot_manifest(PILOT, 42) != manifest:
        raise ValueError("Pilot split differs from the archived E2 control")
    config = protocol.config_values(protocol.training_args(PILOT))
    ref = read_json(BASE.run_dir / "config.json")
    check_training_settings(config, ref)
    if (ref["stage_learning_rates"] != protocol.STAGE_LEARNING_RATES
            or ref["total_epochs"] != protocol.TOTAL_EPOCHS):
        raise ValueError("Archived E2 training schedule differs")
    rows, archive = baseline_cache(manifest["splits"]["val"])
    sources = ("e2_local_correction.py", "run_e2_local_correction.py", "run_e2_scratch_twostage.py",
               "train_lwpd.py", "train_direct_fusion.py", "direct_fusion.py", "e2_base_anchor.py",
               "net.py", "lwpd.py", "utils/dataset_os.py", "utils/fusion_metrics.py",
               "utils/reference_metrics.py", "utils/Evaluator.py", "run_e2_source_objective.py",
               "run_e2_cross_scale.py", "run_e2_aspr_pilot.py")
    lock = dict(config=config, spec=correction.SPEC, epochs=protocol.TOTAL_EPOCHS,
                learning_rates=protocol.STAGE_LEARNING_RATES, archive=archive,
                manifest_sha256=sha256_file(BASE.manifest), baseline_sha256=sha256_file(BASE.final),
                sources={s: sha256_file(ROOT / s) for s in sources})
    return manifest, rows, lock


def train(lock, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected interpreter; set E2_PYTHON to a CUDA Python.")
    training_lock(PILOT, lock, create=True)
    with patch.object(protocol, "reconstruction_forward", correction.reconstruction_forward), \
            patch.object(protocol.trainer, "forward_losses", correction.fusion_forward), \
            patch.object(protocol, "FIELDS", (*protocol.FIELDS, *EXTRA_FIELDS)):
        protocol.train(PILOT, resume=resume)


@torch.inference_mode()
def validate(manifest, baseline, lock):
    lock_path = training_lock(PILOT, lock)
    config = read_json(PILOT.run_dir / "config.json")
    check_training_settings(lock["config"], read_json(BASE.run_dir / "config.json"),
                            resolved=True, run=PILOT.run_dir)
    if (sha256_file(PILOT.manifest) != lock["manifest_sha256"]
            or any(config.get(k) != v for k, v in lock["config"].items())):
        raise ValueError("Candidate training differs from the locked protocol")
    ids = manifest["splits"]["val"]
    splits, _ = manifest_pairs(DATA, PILOT.manifest)
    if [p[0].stem for p in splits["val"]] != ids:
        raise ValueError("Validation images differ from the paired baseline")
    output = PILOT.result
    methods = {"e2": BASE, "candidate": PILOT}
    metadata = dict(scene_ids=ids, dataset="QXS-SAROPT", split="val", offset=0,
                    split_manifest=PILOT.manifest.as_posix(), split_manifest_sha256=sha256_file(PILOT.manifest),
                    training_lock_sha256=sha256_file(lock_path), metric_profile=DEFAULT_PROFILE,
                    output_policy="clamp_then_uint8", baseline_archive=ARCHIVE.as_posix(),
                    checkpoints={key: exp.final.as_posix() for key, exp in methods.items()},
                    checkpoint_hashes={key: sha256_file(exp.final) for key, exp in methods.items()},
                    inputs={a.stem: [sha256_file(a), sha256_file(b)] for a, b in splits["val"]})
    if output.exists():
        verify_result(output, metadata, image_keys=("candidate",))
        summaries = read_json(output / "registry_metrics.json")
    else:
        state = torch.load(PILOT.final, map_location="cpu", weights_only=False)
        if ((state["variant"], state["epoch"], state["stage"]) != (PILOT.variant, protocol.TOTAL_EPOCHS - 1, "fusion")
                or protocol.config_values(argparse.Namespace(**state["args"])) != lock["config"]):
            raise ValueError("Unexpected final checkpoint")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PILOT.build_model().to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            pending = Path(temporary) / "result"
            (pending / "e2").mkdir(parents=True)
            results = {"e2": [(metrics, sources, ()) for metrics, sources in baseline], "candidate": []}
            for i, pair in enumerate(splits["val"], 1):
                optical, sar = read_pair(pair)
                image = model(to_tensor(optical, device), to_tensor(sar, device))
                if not torch.isfinite(image).all():
                    raise FloatingPointError(f"Non-finite output for {pair[0]}")
                gray, metrics, sources = image_scores(image, optical, sar)
                save_images(pending / "candidate", pair, gray, optical, sar)
                results["candidate"].append((metrics, sources, ()))
                print(f"val {i}/{len(ids)} {pair[0].stem}", flush=True)
            summaries = summarize(pending, ids, results, [("e2", "candidate")], diagnostic_fields=())
            write_json(pending / "registry_metrics.json", summaries)
            write_json(pending / "run_metadata.json", metadata)
            seal_result(pending)
            pending.rename(output)
    group = "QXS-E2-LOCAL-CORRECTION-V1-VAL64-REFERENCE-FMI"
    entries = []
    for key, exp in methods.items():
        scores = summaries[key]
        folder = output / key
        entries.append(dict(id=f"{group}-{key}", comparable_group=group, task="fusion",
                            status="development_validation", method=exp.method, dataset="QXS-SAROPT",
                            split="val", offset=0, sample_count=len(ids), checkpoint=exp.final.as_posix(),
                            checkpoint_sha256=metadata["checkpoint_hashes"][key], split_manifest=PILOT.manifest.as_posix(),
                            split_manifest_sha256=metadata["split_manifest_sha256"], result_dir=folder.as_posix(),
                            metadata_path=(output / "run_metadata.json").as_posix(), metric_profile=DEFAULT_PROFILE,
                            output_policy=metadata["output_policy"], metrics=scores,
                            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                                               match={"image": "mean"}, columns={k: k for k in scores}),
                            notes="完整模块与归档 E2；逐项采用两法共同有限场景。"))
    register_many(entries)
    print(f"Paired comparison: {output / 'paired_comparison.csv'}")


def smoke(manifest):
    from e2_base_anchor import OpticalBaseAnchoredE2
    from test_OS import resolve_variant

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair = locate_scene(DATA, manifest, manifest["splits"]["val"][0])
    optical, sar = [to_tensor(x, device)[..., :49, :51] for x in read_pair(pair)]
    valid = (sar > 0).float()
    torch.manual_seed(42)
    model = PILOT.build_model().to(device).eval()
    torch.manual_seed(42)
    base = OpticalBaseAnchoredE2().to(device).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(optical, sar), base(optical, sar), rtol=0, atol=0)
    assert resolve_variant({"args": {"variant": PILOT.variant}}, "auto") == PILOT.variant
    args = protocol.training_args(PILOT)
    losses = protocol.trainer.DirectFusionLosses().to(device)
    optimizer = protocol.trainer.make_optimizer(model, args)
    head = model.decoder.correction
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in head.parameters())
    # Make the candidates different for the scorer gradient test, without an optimizer step.
    with torch.no_grad():
        head.optical_head[-1].bias.fill_(0.2)
        head.sar_head[-1].bias.fill_(-0.2)
    for reconstruction, forward in ((True, correction.reconstruction_forward), (False, correction.fusion_forward)):
        protocol.configure(model, optimizer, args, reconstruction)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and not args.no_amp):
            loss, components = forward(model, losses, optical, sar, valid, args)
        loss.backward()
        for layer in (head.optical_head[-1], head.sar_head[-1], head.score[-1]):
            grad = layer.weight.grad
            if grad is None or not torch.isfinite(grad).all() or not grad.abs().any():
                raise FloatingPointError("Invalid candidate/scorer gradient")
        if not all(torch.isfinite(x) for x in (loss, *components.values())):
            raise FloatingPointError("Nonfinite loss")
        print(f"{forward.__name__}: {float(loss):.6f}")
    print(f"Smoke passed on {pair[0].stem}, 49x51, {device}, no optimizer step.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "train-val", "resume-val", "val"))
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest, baseline, lock = preflight()
    if args.action == "check":
        print(runtime_info(), "cuda_available=", torch.cuda.is_available(), "matched validation pairs=", len(baseline))
    elif args.action == "smoke":
        smoke(manifest)
    else:
        if args.action in ("train-val", "resume-val"):
            train(lock, resume=args.action == "resume-val")
        validate(manifest, baseline, lock)


if __name__ == "__main__":
    main()
