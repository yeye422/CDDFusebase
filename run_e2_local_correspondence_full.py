"""Train the local Detail exchange model from scratch on the full QXS split."""
import argparse
import os
from pathlib import Path

import torch
from filelock import FileLock

from e2_local_correspondence import LocalCorrespondenceE2
from e2_residual_local_exchange import ResidualLocalExchangeE2
from maintain_experiment_index import sha256_file
from run_e2_cross_scale import check_training_settings
from run_e2_local_correspondence import LocalCorrespondencePilot
from run_e2_scratch_twostage import BASE_ANCHOR, STAGE_LEARNING_RATES, TOTAL_EPOCHS, config_values, train, training_args
from run_e2_source_interval_val import RCCR, read_metric_rows, validate_against_e2
from run_qxs_reference_test1000 import read_json
from utils.dataset_os import validate_manifest_splits
from utils.fusion_metrics import DEFAULT_PROFILE


class FullLocalExchange(LocalCorrespondencePilot):
    save_continuation = True


FULL = FullLocalExchange(
    "e2_local_correspondence_full_v1", "E2-LOCAL-CORRESPONDENCE-FULL-V1",
    LocalCorrespondenceE2.VARIANT, "test_OS.py", "光学 Base E2＋可拒配的局部 Detail 交换",
    "固定 train8000/val1000，seed42，随机初始化，4 轮重建＋12 轮原融合损失。",
    train_count=8000, val_count=1000,
)
GROUP = "QXS-E2-LOCAL-CORRESPONDENCE-FULL-V1-SEED42-VAL1000-REFERENCE-FMI"


class FullResidualLocalExchange(FullLocalExchange):
    def build_model(self):
        return ResidualLocalExchangeE2()


RESIDUAL = FullResidualLocalExchange(
    "e2_residual_local_exchange_full_v1", "E2-RESIDUAL-LOCAL-EXCHANGE-FULL-V1",
    ResidualLocalExchangeE2.VARIANT, "test_OS.py", "光学 Base E2＋第三尺度残差式局部 Detail 交换",
    "完整三条卷积分支；交换以零初始化投影加至第三分支。从头训练 4+12 轮，沿用 E2 原损失。",
    train_count=8000, val_count=1000,
)
RESIDUAL_GROUP = "QXS-E2-RESIDUAL-LOCAL-EXCHANGE-FULL-V1-SEED42-VAL1000-REFERENCE-FMI"


def check(experiment=FULL):
    manifest = read_json(experiment.manifest)
    validate_manifest_splits(manifest)
    splits = manifest["splits"]
    if (len(splits["train"]), len(splits["val"])) != (experiment.train_count, experiment.val_count):
        raise ValueError("Expected full train8000/val1000 split")
    config = read_json(BASE_ANCHOR.run_dir / "config.json")
    check_training_settings(config_values(training_args(experiment)), config)
    if config["total_epochs"] != TOTAL_EPOCHS or config["stage_learning_rates"] != STAGE_LEARNING_RATES:
        raise ValueError("Complete E2 baseline has a different schedule")
    baseline = read_json(RCCR.result / "run_metadata.json")
    if (baseline["split_manifest_sha256"] != sha256_file(experiment.manifest)
            or baseline["checkpoint_hashes"]["e2"] != sha256_file(BASE_ANCHOR.final)
            or baseline["metric_profile"] != DEFAULT_PROFILE or baseline["output_policy"] != "clamp_then_uint8"
            or baseline["scene_ids"] != splits["val"] or baseline["split"] != "val" or baseline["offset"] != 0):
        raise ValueError("Archived full E2 validation protocol differs")
    read_metric_rows(RCCR.result / "e2/metrics.csv", splits["val"])
    print(f"{experiment.variant}: train8000/val1000; seed42; random initialization; 4+12 epochs; CUDA={torch.cuda.is_available()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "train-val", "resume-val", "val"))
    parser.add_argument("--residual", action="store_true", help="Retain dilation 3 and add zero-initialized exchange")
    args = parser.parse_args()
    action = args.action
    experiment, group = (RESIDUAL, RESIDUAL_GROUP) if args.residual else (FULL, GROUP)
    os.chdir(Path(__file__).resolve().parent)
    check(experiment)
    if action == "check":
        return
    experiment.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(experiment.run_dir.parent / ".seed42.lock"), timeout=0):
        if action != "val":
            train(experiment, resume=action == "resume-val")
        if action != "train":
            validate_against_e2(experiment, group, "candidate", experiment.method)


if __name__ == "__main__":
    main()
