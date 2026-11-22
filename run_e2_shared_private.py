"""Matched train1024/val64 pilot for patch-level shared/private Detail (direction B)."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
import os
from unittest.mock import patch

import cv2
from filelock import FileLock
import numpy as np
import torch

import e2_shared_private as candidate
import run_e2_local_correction as workflow
import run_e2_scratch_twostage as protocol
import run_e2_source_interval_val as full_evaluation
from run_e2_local_correspondence_full import check as check_full_budget
from run_e2_cross_scale import check_training_settings
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_dose import read_probe
from diagnose_e2_primitive_render import CENTER, LABELS, sampling_grid, source_contrast
from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import METRICS, summarize, write_csv
from run_e2_local_correction_v2 import NEGATIVE, fixed_probes
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json


class SharedPrivatePilot(protocol.Experiment):
    save_continuation = True

    def build_model(self):
        return candidate.SharedPrivateE2()


PILOT = SharedPrivatePilot("e2_shared_private_patch_pilot_v1", "E2-SHARED-PRIVATE-PATCH-V1",
                          candidate.SharedPrivateE2.VARIANT, "test_OS.py",
                          "E2＋Patch 共享–特有 Detail（B）", "seed42，从头训练 4+12 轮，复用 E2 对照。",
                          train_count=1024, val_count=64)
GROUP = "QXS-E2-SHARED-PRIVATE-PATCH-V1-VAL64-REFERENCE-FMI"
FULL = replace(PILOT, name="e2_shared_private_patch_full_v1", experiment_id="E2-SHARED-PRIVATE-PATCH-FULL-V1",
               notes="完整 train8000/val1000、seed42，随机初始化，B 的 4+12 轮训练与完整 E2 配对。",
               train_count=8000, val_count=1000)
FULL_GROUP = "QXS-E2-SHARED-PRIVATE-PATCH-FULL-V1-VAL1000-REFERENCE-FMI"
STANDARD_CONFIGURE = protocol.configure


def configure(model, optimizer, args, reconstruction):
    STANDARD_CONFIGURE(model, optimizer, args, reconstruction)
    if reconstruction:
        # Only the source projections and readouts participate in stage I.
        head = model.detail_fusion
        for layer in (head.optical_projection, head.sar_projection, head.optical_readout, head.sar_readout):
            layer.requires_grad_(True)
        group = next(g for g in optimizer.param_groups if g["group_name"] == "detail_fusion")
        group["lr"] = protocol.STAGE_LEARNING_RATES["reconstruction"]


def preflight(experiment=PILOT):
    if experiment.is_pilot:
        with patch.object(workflow, "PILOT", experiment):
            manifest, rows, lock = workflow.preflight()
    else:
        check_full_budget(experiment)
        manifest, rows = read_json(experiment.manifest), None
        lock = dict(config=protocol.config_values(protocol.training_args(experiment)),
                    total_epochs=protocol.TOTAL_EPOCHS, learning_rates=protocol.STAGE_LEARNING_RATES,
                    manifest_sha256=sha256_file(experiment.manifest), baseline_sha256=sha256_file(protocol.BASE_ANCHOR.final),
                    sources={name:sha256_file(workflow.ROOT/name) for name in (
                        "run_e2_scratch_twostage.py", "train_lwpd.py", "train_direct_fusion.py", "direct_fusion.py",
                        "lwpd.py", "e2_base_anchor.py", "net.py", "utils/dataset_os.py", "run_e2_source_interval_val.py",
                        "run_e2_aspr_pilot.py", "test_OS.py", "utils/fusion_metrics.py", "utils/reference_metrics.py")})
    lock["spec"] = candidate.SPEC
    for name in ("run_e2_shared_private.py", "e2_shared_private.py", "run_e2_local_correction_v2.py",
                 "diagnose_e2_contour_evidence.py", "diagnose_e2_primitive_render.py", "diagnose_e2_primitive_dose.py"):
        lock["sources"][name] = sha256_file(workflow.ROOT / name)
    lock["probes"] = {p.name: sha256_file(p) for p in (LABELS / "source_labels.csv", NEGATIVE)}
    return manifest, rows, lock


def train(lock, resume, experiment=PILOT):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a CUDA Python with E2_PYTHON")
    workflow.training_lock(experiment, lock, create=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(protocol, "configure", configure))
        stack.enter_context(patch.object(protocol, "reconstruction_forward", partial(candidate.stage_loss, fusion=False)))
        stack.enter_context(patch.object(protocol.trainer, "forward_losses", partial(candidate.stage_loss, fusion=True)))
        stack.enter_context(patch.object(protocol, "FIELDS", (*protocol.FIELDS, "patch_alignment")))
        protocol.train(experiment, resume=resume)


@torch.inference_mode()
def private_readouts(model, optical, sar):
    """Suppress each private group separately, holding shared context and weights fixed."""
    bo, do, bs, ds = model.encode_pair(optical, sar)
    head = model.detail_fusion
    zo, zs = head.project(do, ds)
    base = bo + model.base_fusion(bo, bs)
    def suppress_private(z):
        common, native = z.split((candidate.SPEC["shared_channels"], candidate.SPEC["private_channels"]), dim=1)
        return torch.cat((common, torch.zeros_like(native)), dim=1)
    return {mode: model.decoder(None, base, head.aggregate(a, b, base))[0]
            for mode, a, b in (("candidate", zo, zs), ("optical_private_off", suppress_private(zo), zs),
                               ("sar_private_off", zo, suppress_private(zs)))}


def paired_report(folder, ids, results, contrasts, *, experiment=PILOT, baseline_gray=None, **kwargs):
    summaries = summarize(folder, ids, results, contrasts, **kwargs)
    base, proposed = [np.array([row[0] for row in results[k]], dtype=float) for k in ("e2", "candidate")]
    delta = proposed - base
    counts = []
    for j, name in enumerate(METRICS):
        d = delta[:, j][np.isfinite(delta[:, j])]
        good = -d if name == "Nabf" else d
        counts.append((name, len(d), int((good > 0).sum()), int((good == 0).sum()), int((good < 0).sum())))
    write_csv(folder / "paired_counts.csv", ("metric", "n", "better", "equal", "worse"), counts)
    core = delta[:, [METRICS.index(k) for k in ("Qabf", "Nabf", "VIF")]]
    valid = np.isfinite(core).all(1)
    win = valid & (core[:, 0] > 0) & (core[:, 1] < 0) & (core[:, 2] > 0)
    workflow.write_json(folder / "joint.json", dict(common=int(valid.sum()), jointly_better=int(win.sum()),
                        scene_ids=[s for s, w in zip(ids, win) if w]))
    probes = fixed_probes()
    probes["4880"].append(("sar_response", np.asarray(read_json(NEGATIVE)["decisions"]["4880-B"]["sar_segment"])))
    readings = []
    manifest = read_json(experiment.manifest)
    if baseline_gray is None:
        baseline_gray = workflow.ARCHIVE / "e2/gray"
    for scene, lines in probes.items():
        optical, radar = read_pair(locate_scene(DATA, manifest, scene))
        images = {key: cv2.imread(str(parent/f"{scene}.png"), cv2.IMREAD_UNCHANGED)
                  for key, parent in (("e2", baseline_gray), ("candidate", folder/"candidate/gray"))}
        if any(image is None or image.shape != optical.shape for image in images.values()):
            raise ValueError(f"Invalid saved probe image for {scene}")
        if scene == "4880":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = experiment.build_model().to(device).eval()
            model.load_state_dict(torch.load(experiment.final, map_location="cpu", weights_only=False)["model"])
            outputs = private_readouts(model, to_tensor(optical, device), to_tensor(radar, device))
            for key, out in outputs.items():
                im = out[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
                if key == "candidate" and not np.array_equal(im, images[key]):
                    raise ValueError("Private readout diagnostic does not reproduce candidate PNG")
                images[key] = im
        for label, segment in lines:
            _, coords = sampling_grid(segment)
            src = radar if label == "sar_response" else optical
            sign = 0 if label == "parallel_negative" else int(np.sign(source_contrast(src, coords, label == "sar_response")))
            for key, image in images.items():
                response = profile(image, coords)[:, CENTER]
                hit = (sign * response if sign else np.abs(response)) >= MIN_SUPPORT
                bounds = np.flatnonzero(np.diff(np.r_[False, hit, False]))
                reading = read_probe(image, coords, sign)
                readings.append((scene, label, key, sign, len(hit), int(hit.sum()),
                                 int((bounds[1::2]-bounds[::2]).max(initial=0)), reading["peak_offset_px"],
                                 reading["center_jump_gray"] if sign else None))
    write_csv(folder / "fixed_probes.csv", ("scene","probe","method","sign","samples","hits","longest",
                                            "peak_offset_px","center_jump_gray"), readings)
    return summaries


def register_pair(entries):
    if len(entries) != 2:
        raise ValueError("Expected two matched methods")
    pair = []
    for entry, (name, exp) in zip(entries, (("e2", workflow.BASE), ("candidate", PILOT))):
        if entry["checkpoint"] != exp.final.as_posix() or entry["result_dir"] != (PILOT.result / name).as_posix():
            raise ValueError("Unexpected registration paths")
        pair.append(dict(entry, id=f"{GROUP}-{name}", comparable_group=GROUP))
    register_many(pair)


def smoke(manifest, experiment=PILOT):
    from test_OS import resolve_variant
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = experiment.build_model().to(device)
    assert resolve_variant({"args":{"variant":model.variant}}, "auto") == PILOT.variant
    head = model.detail_fusion
    optical, sar = (to_tensor(x, device)[..., :63, :61] for x in read_pair(locate_scene(DATA, manifest, "4880")))
    valid = (sar > 0).float()
    args = protocol.training_args(experiment)
    losses = protocol.trainer.DirectFusionLosses().to(device)
    opt = protocol.trainer.make_optimizer(model, args)
    for fusion in (False, True):
        configure(model, opt, args, not fusion)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda" and not args.no_amp):
            loss, stats = candidate.stage_loss(model, losses, optical, sar, valid, args, fusion=fusion)
        loss.backward()
        if not all(torch.isfinite(x) for x in (loss, *stats.values())):
            raise FloatingPointError("Nonfinite stage loss")
        layers = [head.optical_projection, head.sar_projection, head.optical_readout, head.sar_readout]
        if fusion:
            layers += [head.common[0], head.private_gain, head.output[-1]]
        for layer in layers:
            grad = layer.weight.grad
            if grad is None or not torch.isfinite(grad).all() or not grad.any():
                raise FloatingPointError(f"Missing/invalid gradient on {layer}")
        print(f"fusion={fusion}, loss={loss.item():.6f}, alignment={stats['patch_alignment'].item():.6f}")
    with torch.inference_mode():
        fused = model(optical, sar)
        outputs = private_readouts(model, optical, sar)
        torch.testing.assert_close(fused, outputs.pop("candidate"))
        for name, im in outputs.items():
            delta = (fused - im).abs().mean().item()
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError("Inactive private path")
            print(f"{name}, random-weight output MAE={delta:.7f}")
    print("4880 single crop: forward/backward passed, no optimizer step.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check","smoke","train-val","resume-val","val"), default="check", nargs="?")
    parser.add_argument("--full", action="store_true", help="Use train8000/val1000 and the archived full E2 control")
    args = parser.parse_args()
    os.chdir(workflow.ROOT)
    experiment = FULL if args.full else PILOT
    manifest, rows, lock = preflight(experiment)
    if args.action == "check":
        print(f"{experiment.variant}: train{experiment.train_count}, {experiment.val_count} cached E2 validation scenes, CUDA={torch.cuda.is_available()}")
    elif args.action == "smoke":
        smoke(manifest, experiment)
    else:
        experiment.run_dir.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(experiment.run_dir.parent / ".workflow.lock"), timeout=0):
            if args.action != "val":
                train(lock, args.action == "resume-val", experiment)
            if args.full:
                workflow.training_lock(experiment, lock)
                reference = read_json(protocol.BASE_ANCHOR.run_dir / "config.json")
                check_training_settings(lock["config"], reference, resolved=True, run=experiment.run_dir)
                full_report = partial(paired_report, experiment=experiment, baseline_gray=protocol.BASE_ANCHOR.result/"gray")
                with patch.object(full_evaluation, "summarize", full_report):
                    full_evaluation.validate_against_e2(experiment, FULL_GROUP, "candidate", experiment.method)
                return
            with patch.object(workflow, "PILOT", PILOT), patch.object(workflow, "register_many", register_pair), \
                    patch.object(workflow, "summarize", paired_report):
                workflow.validate(manifest, rows, lock)


if __name__ == "__main__":
    main()
