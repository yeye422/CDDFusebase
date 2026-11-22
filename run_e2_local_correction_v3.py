"""Matched E2/V3 pilot, using the same archived baseline and evaluation as V2."""
import argparse
import os
from unittest.mock import patch

import torch

import e2_local_correction_v3 as v3
import run_e2_local_correction_v2 as previous
import run_e2_scratch_twostage as protocol
from diagnose_e2_information_path import locate_scene, read_pair
from diagnose_stg_no_train import write_csv
from maintain_experiment_index import sha256_file
from run_e2_cross_scale import check_training_settings, training_lock
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, read_json


class V3Pilot(previous.V2Pilot):
    def build_model(self):
        return v3.LocalCorrectionV3()


PILOT = V3Pilot("e2_local_correction_pilot_v3", "E2-LOCAL-CORRECTION-PILOT-V3",
                v3.LocalCorrectionV3.VARIANT, "test_OS.py", "E2＋两读出共享的局部冲突 Sobel 监督",
                "V2 网络与候选辅助任务，融合阶段按固定源图冲突掩码加权；从头 4+12 轮。",
                train_count=1024, val_count=64)
GROUP = "QXS-E2-LOCAL-CORRECTION-V3-VAL64-REFERENCE-FMI"


def preflight():
    manifest, baseline, lock = previous.preflight()
    config = protocol.config_values(protocol.training_args(PILOT))
    check_training_settings(config, read_json(previous.BASE.run_dir / "config.json"))
    if manifest != protocol.pilot_manifest(PILOT, 42):
        raise ValueError("V3 split differs from the archived E2 control")
    lock.update(config=config, spec=v3.SPEC)
    for name in ("run_e2_local_correction_v3.py", "e2_local_correction_v3.py", "utils/loss.py"):
        lock["sources"][name] = sha256_file(previous.v1.ROOT / name)
    return manifest, baseline, lock


def train(manifest, lock, resume):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set E2_PYTHON to a CUDA interpreter")
    training_lock(PILOT, lock, create=True)
    save_checkpoint = protocol.trainer.save_checkpoint
    probes = previous.fixed_probes()

    def save(path, epoch, stage, model, *args, **kwargs):
        if epoch + 1 in (4, 5, 6, 7, 8, 12, 16):
            rows = []
            for scene, lines in probes.items():
                o, s = read_pair(locate_scene(DATA, manifest, scene))
                rows.extend(dict(epoch=epoch + 1, scene=scene, **r)
                            for r in previous.scene_probes(model, o, s, lines))
            dest = PILOT.run_dir / f"probes_epoch_{epoch + 1:03d}.csv"
            pending = dest.with_suffix(".pending.csv")
            write_csv(pending, rows)
            pending.replace(dest)
        save_checkpoint(path, epoch, stage, model, *args, **kwargs)

    with patch.object(protocol, "reconstruction_forward", v3.reconstruction_forward), \
            patch.object(protocol.trainer, "forward_losses", v3.fusion_forward), \
            patch.object(protocol.trainer, "save_checkpoint", save), \
            patch.object(protocol, "FIELDS", (*protocol.FIELDS, *v3.FIELDS)):
        protocol.train(PILOT, resume=resume)


def register_v3_pair(entries):
    methods = (("e2", previous.BASE), ("candidate", PILOT))
    if len(entries) != len(methods):
        raise ValueError("Expected exactly two paired results")
    pair = []
    for entry, (key, experiment) in zip(entries, methods):
        if entry["checkpoint"] != experiment.final.as_posix() or entry["result_dir"] != (PILOT.result / key).as_posix():
            raise ValueError("Registration refers to the wrong run")
        pair.append(dict(entry, id=f"{GROUP}-{key}", comparable_group=GROUP))
    register_many(pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "screen", "train-val", "resume-val", "val"),
                        nargs="?", default="check")
    action = parser.parse_args().action
    os.chdir(previous.v1.ROOT)
    if action == "screen":
        from diagnose_e2_v3_conflict_screen import main as screen
        screen()
        return
    manifest, baseline, lock = preflight()
    if action == "check":
        print(f"V3: train{len(manifest['splits']['train'])}/val{len(baseline)}, CUDA={torch.cuda.is_available()}")
    elif action == "smoke":
        with patch.object(previous.v1, "PILOT", PILOT), patch.object(previous.v1, "correction", v3):
            # Smoke on the same single scene as the frozen screen.
            if "4880" not in manifest["splits"]["val"]:
                raise ValueError("Missing fixed validation scene 4880")
            previous.v1.smoke(dict(manifest, splits=dict(manifest["splits"], val=["4880"])))
    else:
        if action != "val":
            train(manifest, lock, resume=action == "resume-val")
        # The historical paired evaluator has a fixed registration group. Select
        # the new pilot and its group here without changing V2's locked sources.
        with patch.object(previous, "PILOT", PILOT), patch.object(previous, "register_many", register_v3_pair):
            previous.validate(manifest, baseline, lock)


if __name__ == "__main__":
    main()
