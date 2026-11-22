"""Source-supervised E2 correction pilot, paired only with the archived E2 control."""
import argparse
import csv
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import e2_local_correction_v2 as v2
import run_e2_local_correction as v1
import run_e2_scratch_twostage as protocol
from diagnose_e2_contour_evidence import MIN_SUPPORT, profile
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_primitive_dose import read_probe
from diagnose_e2_primitive_render import CENTER, LABELS, endpoints, sampling_grid, source_contrast, write_csv
from diagnose_stg_no_train import DEFAULT_LABELS
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_e2_cross_scale import check_training_settings, seal_result, training_lock, verify_result
from run_e2_matched_no_pac import register_many
from run_e2_source_objective import image_scores, save_images
from run_qxs_reference_test1000 import DATA, read_json, write_json
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE


class V2Pilot(v1.CorrectionPilot):
    def build_model(self):
        return v2.LocalCorrectionV2()


PILOT = V2Pilot("e2_local_correction_pilot_v2", "E2-LOCAL-CORRECTION-PILOT-V2",
                v2.LocalCorrectionV2.VARIANT, "test_OS.py", "E2＋源别候选直接监督 V2",
                "从头训练 4+12 轮，比较完整 V2 与归档 E2。", train_count=1024, val_count=64)
BASE = v1.BASE
NEGATIVE = Path("test_result/e2_local_supervision_v1/smoke_4880_B/summary.json")


def preflight():
    manifest, baseline, lock = v1.preflight()
    config = protocol.config_values(protocol.training_args(PILOT))
    check_training_settings(config, read_json(BASE.run_dir / "config.json"))
    if manifest != protocol.pilot_manifest(PILOT, 42):
        raise ValueError("V2 pilot split differs from E2")
    lock.update(config=config, spec=v2.SPEC)
    for name in ("run_e2_local_correction_v2.py", "e2_local_correction_v2.py",
                 "diagnose_e2_contour_evidence.py", "diagnose_e2_primitive_dose.py",
                 "diagnose_e2_primitive_render.py"):
        lock["sources"][name] = sha256_file(v1.ROOT / name)
    lock["probe_inputs"] = {str(path.relative_to(v1.ROOT).as_posix()): sha256_file(path)
                            for path in (LABELS / "source_labels.csv", LABELS / "sample.json",
                                         DEFAULT_LABELS / "sample.json", v1.ROOT / NEGATIVE)}
    return manifest, baseline, lock


def fixed_probes():
    with (LABELS / "source_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        targets = {r["scene"]: endpoints(r) for r in csv.DictReader(handle)
                   if r["scene"] in ("4880", "10095") and r["id"] == "B"}
    offset = read_json(NEGATIVE)["decisions"]["4880-B"]["false_translation"]
    return {"4880": [("B", targets["4880"]), ("parallel_negative", targets["4880"] + offset)],
            "10095": [("B", targets["10095"])]}


@torch.inference_mode()
def scene_probes(model, optical, sar, probes):
    device = next(model.parameters()).device
    o, s = (to_tensor(x, device) for x in (optical, sar))
    encoded = model.encode_pair(o, s)
    base, detail = model.fuse_features(*encoded)
    coarse, context = model.decoder(None, base, detail)
    final, candidates, _ = model.decoder.correction(o, s, coarse, context, encoded[1], encoded[3])
    images = torch.cat((candidates[:, :2], torch.where(s > 0, candidates[:, 2:3], coarse), final), 1)
    images = images[0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    rows = []
    for name, segment in probes:
        _, coords = sampling_grid(segment)
        # Negative readout is unsigned, so either polarity of a false edge counts.
        sign = 0 if name == "parallel_negative" else int(np.sign(source_contrast(optical, coords, False)))
        for key, im in zip(("initial", "optical_candidate", "sar_candidate", "final"), images):
            response = profile(im, coords)[:, CENTER]
            active = (sign * response if sign else abs(response)) >= MIN_SUPPORT
            bounds = np.flatnonzero(np.diff(np.r_[False, active, False]))
            reading = read_probe(im, coords, sign)
            if not sign:
                reading["center_jump_gray"] = None
            rows.append(dict(probe=name, method=key, sign=sign, samples=len(active), hits=int(active.sum()),
                             longest=int((bounds[1::2] - bounds[::2]).max(initial=0)), **reading))
    return rows


def train(manifest, lock, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a CUDA interpreter with E2_PYTHON.")
    training_lock(PILOT, lock, create=True)
    original_save = protocol.trainer.save_checkpoint
    probes = fixed_probes()

    def save(path, epoch, stage, model, *args, **kwargs):
        if epoch + 1 in (4, 8, 12, 16):
            rows = []
            for scene, lines in probes.items():
                optical, sar = read_pair(locate_scene(DATA, manifest, scene))
                rows.extend(dict(epoch=epoch + 1, scene=scene, **r)
                            for r in scene_probes(model, optical, sar, lines))
            target = PILOT.run_dir / f"probes_epoch_{epoch + 1:03d}.csv"
            pending = target.with_suffix(".pending.csv")
            write_csv(pending, rows)
            pending.replace(target)
        original_save(path, epoch, stage, model, *args, **kwargs)

    with patch.object(protocol, "reconstruction_forward", v2.reconstruction_forward), \
            patch.object(protocol.trainer, "forward_losses", v2.fusion_forward), \
            patch.object(protocol.trainer, "save_checkpoint", save), \
            patch.object(protocol, "FIELDS", (*protocol.FIELDS, *v2.FIELDS)):
        protocol.train(PILOT, resume=resume)


@torch.inference_mode()
def validate(manifest, baseline, lock):
    lock_path = training_lock(PILOT, lock)
    config = read_json(PILOT.run_dir / "config.json")
    check_training_settings(lock["config"], read_json(BASE.run_dir / "config.json"), resolved=True, run=PILOT.run_dir)
    if (sha256_file(PILOT.manifest) != lock["manifest_sha256"] or
            any(config.get(k) != v for k, v in lock["config"].items())):
        raise ValueError("V2 training differs from the locked protocol")
    ids = manifest["splits"]["val"]
    pairs = manifest_pairs(DATA, PILOT.manifest)[0]["val"]
    if [a.stem for a, b in pairs] != ids:
        raise ValueError("Validation scene order differs")
    pilot_methods = {"e2": BASE, "candidate": PILOT}
    # A descriptive slice of the same val64, excluding the two archived manual panels and smoke scene.
    reviewed = set(read_json(LABELS / "sample.json")["scene_ids"]) | set(read_json(DEFAULT_LABELS / "sample.json")["scene_ids"]) | {"211"}
    metadata = dict(scene_ids=ids, other_val_ids=[i for i in ids if i not in reviewed], dataset="QXS-SAROPT",
                    split="val", offset=0, split_manifest=PILOT.manifest.as_posix(),
                    split_manifest_sha256=lock["manifest_sha256"], metric_profile=DEFAULT_PROFILE,
                    output_policy="clamp_then_uint8", training_lock_sha256=sha256_file(lock_path),
                    checkpoints={k: exp.final.as_posix() for k, exp in pilot_methods.items()},
                    checkpoint_hashes={k: sha256_file(exp.final) for k, exp in pilot_methods.items()},
                    baseline_archive=v1.ARCHIVE.as_posix(), inputs={a.stem: [sha256_file(a), sha256_file(b)] for a,b in pairs})
    output = PILOT.result
    if output.exists():
        verify_result(output, metadata, image_keys=("candidate",))
        metrics = read_json(output / "registry_metrics.json")
    else:
        state = torch.load(PILOT.final, map_location="cpu", weights_only=False)
        if ((state["variant"], state["epoch"], state["stage"]) != (PILOT.variant, protocol.TOTAL_EPOCHS - 1, "fusion")
                or protocol.config_values(argparse.Namespace(**state["args"])) != lock["config"]):
            raise ValueError("Expected the final V2 checkpoint")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PILOT.build_model().to(device).eval()
        model.load_state_dict(state["model"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            pending = Path(tmp) / "result"
            (pending / "e2").mkdir(parents=True)
            rows = {"e2": [(m, s, ()) for m, s in baseline], "candidate": []}
            for i, pair in enumerate(pairs):
                optical, sar = read_pair(pair)
                image = model(to_tensor(optical, device), to_tensor(sar, device))
                if not torch.isfinite(image).all():
                    raise FloatingPointError(f"Non-finite output for {ids[i]}")
                gray, scores, sources = image_scores(image, optical, sar)
                save_images(pending / "candidate", pair, gray, optical, sar)
                rows["candidate"].append((scores, sources, ()))
                print(f"val {i+1}/{len(ids)} {ids[i]}", flush=True)
            metrics = summarize(pending, ids, rows, [("e2", "candidate")], diagnostic_fields=())
            subset = pending / "other_val"
            for key in rows:
                (subset / key).mkdir(parents=True)
            subset_rows = {k: [v for scene, v in zip(ids, values) if scene not in reviewed] for k, values in rows.items()}
            summarize(subset, metadata["other_val_ids"], subset_rows, [("e2", "candidate")], diagnostic_fields=())
            write_json(pending / "registry_metrics.json", metrics)
            write_json(pending / "run_metadata.json", metadata)
            seal_result(pending)
            pending.rename(output)
    group = "QXS-E2-LOCAL-CORRECTION-V2-VAL64-REFERENCE-FMI"
    entries = []
    for name, exp in pilot_methods.items():
        folder = output / name
        scores = metrics[name]
        entries.append(dict(**protocol.entry_base(exp), id=f"{group}-{name}", comparable_group=group, task="fusion",
                            status="development_validation", split="val", sample_count=len(ids),
                            result_dir=folder.as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
                            metric_profile=DEFAULT_PROFILE, output_policy=metadata["output_policy"], metrics=scores,
                            metric_source=dict(type="csv_row", path=(folder / "metrics.csv").as_posix(),
                                               match={"image": "mean"}, columns={k: k for k in scores})))
    register_many(entries)
    print(f"Paired results: {output / 'paired_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "train-val", "resume-val", "val"), default="check", nargs="?")
    args = parser.parse_args()
    os.chdir(v1.ROOT)
    manifest, baseline, lock = preflight()
    if args.action == "check":
        print(f"V2: {len(manifest['splits']['train'])} training pairs, {len(baseline)} cached E2 validation rows, CUDA={torch.cuda.is_available()}")
    elif args.action == "smoke":
        # Run the identical V1 identity, optimizer membership and AMP gradient checks with V2.
        with patch.object(v1, "PILOT", PILOT), patch.object(v1, "correction", v2):
            v1.smoke(manifest)
    else:
        if args.action in ("train-val", "resume-val"):
            train(manifest, lock, args.action == "resume-val")
        validate(manifest, baseline, lock)


if __name__ == "__main__":
    main()
