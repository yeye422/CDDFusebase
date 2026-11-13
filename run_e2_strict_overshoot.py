"""Train optical Base E2 with only a strict gradient overshoot penalty added."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from run_e2_scratch_twostage import BASE_ANCHOR, train
from run_e2_source_interval_val import validate_against_e2


EXPERIMENT = replace(
    BASE_ANCHOR, name="e2_scratch_strict_overshoot_v1", experiment_id="E2-SCRATCH-STRICT-OVERSHOOT-V1",
    method="E2 光学 Base + 严格过冲惩罚",
    notes="从头训练4轮重建和12轮融合。融合阶段只在原E2损失上加权重2.5的无容差L2 Sobel过冲项。",
    uses_strict_overshoot=True,
)
GROUP = "QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "train-val"))
    parser.add_argument("--resume", action="store_true", help="Continue from the latest completed epoch in this run.")
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
        validate_against_e2(EXPERIMENT, GROUP, "strict_overshoot", EXPERIMENT.method,
                            report_nabf_strata=True)


if __name__ == "__main__":
    main()
