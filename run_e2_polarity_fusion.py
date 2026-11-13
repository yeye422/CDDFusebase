"""Train the polarity-aware E2 from scratch and evaluate the fixed QXS validation split."""

import argparse
import os
from pathlib import Path

from e2_polarity_fusion import PolarityFusionE2
from run_e2_scratch_twostage import Experiment, train
from run_e2_source_interval_val import validate_against_e2


EXPERIMENT = Experiment(
    name="e2_scratch_polarity_v1",
    experiment_id="E2-SCRATCH-POLARITY-V1",
    variant=PolarityFusionE2.VARIANT,
    evaluator="test_OS.py",
    method="LWPD E2 + 亮暗极性解耦边界融合",
    notes="随机初始化，4轮源图重建与12轮融合。极性解耦梯度场与低频亮度共同生成融合图。",
)
GROUP = "QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "train-val"))
    parser.add_argument("--resume", action="store_true", help="Continue the same run from its latest committed epoch.")
    args = parser.parse_args()
    if args.resume and args.action == "val":
        parser.error("--resume requires train or train-val")
    os.chdir(Path(__file__).resolve().parent)
    if args.action in ("train", "train-val"):
        run = EXPERIMENT.run_dir
        run.parent.mkdir(parents=True, exist_ok=True)
        lock = run.parent / f".{run.name}.training.lock"
        with lock.open("x", encoding="utf-8") as file:
            file.write(str(os.getpid()))
        try:
            train(EXPERIMENT, resume=args.resume)
        finally:
            lock.unlink()
    if args.action in ("val", "train-val"):
        validate_against_e2(EXPERIMENT, GROUP, "polarity", EXPERIMENT.method, report_nabf_strata=True)


if __name__ == "__main__":
    main()
