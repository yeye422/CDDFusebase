"""Train the directional SAR Detail E2 from scratch or compare saved weights."""

import argparse
import os
from pathlib import Path

from e2_sar_directional import SARDirectionalE2
from run_e2_scratch_twostage import Experiment, train
from run_e2_source_interval_val import validate_against_e2


EXPERIMENT = Experiment(
    name="e2_scratch_sar_directional_v1",
    experiment_id="E2-SCRATCH-SAR-DIRECTIONAL-V1",
    variant=SARDirectionalE2.VARIANT,
    evaluator="test_OS.py",
    method="LWPD E2 + SAR 方向子带收缩",
    notes="从头训练4轮重建和12轮融合。按SAR自身方向支持有界收缩高频，仅作用于融合支路。",
)
GROUP = "QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "val", "train-val"))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if args.action in ("train", "train-val"):
        train(EXPERIMENT)
    if args.action in ("val", "train-val"):
        validate_against_e2(EXPERIMENT, GROUP, "directional", EXPERIMENT.method,
                            report_nabf_strata=True, sar_detail_ablation=True)


if __name__ == "__main__":
    main()
