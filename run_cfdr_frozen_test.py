"""Evaluate a fixed frozen CFDR milestone on the held-out QXS test1000."""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import cfdr_evaluation as evaluation
from e2_hcfdr import HCFDRE2, VARIANT
from hcfdr_comparison import EVALUATOR_FILES, METHODS, reuse_metrics
from run_cfdr_frozen import PARENT, load_backbone
from run_cfdr_frozen_study import ARMS, MILESTONES, checkpoint, comparison_dir, read_state
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, MANIFEST, read_json, verify_images
from train_direct_fusion import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


OUTPUT = Path("test_result/e2_hcfdr_v1_frozen/test1000")
CONTRASTS = (("CF0", "CF1"), ("CF0", "CF2"), ("CF0", "H2"), ("CF1", "CF2"), ("CF1", "H2"), ("CF2", "H2"))


def infer(arm, path, folder, split, ids, digest, device):
    if arm != "H2":
        return evaluation.infer(arm, path, folder, split, ids, digest, device)
    import test_OS

    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST), "--split", split,
            "--limit", str(len(ids)), "--variant", VARIANT, "--checkpoint", str(path), "--output-dir", str(folder),
            "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, {VARIANT: HCFDRE2}), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    metadata = read_json(folder / "run_metadata.json")
    expected = dict(split=split, offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids), variant=VARIANT,
                    metric_profile=DEFAULT_PROFILE, split_manifest_sha256=digest, output_range_policy="clamp",
                    checkpoint_sha256=sha256_file(path), metrics_only=False)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected H2 evaluation metadata: {folder}")
    for kind in ("gray", "rgb", "preview"):
        verify_images(folder / kind, ids)
    raw = (folder / "metrics.csv").rename(folder / "evaluator_metrics.csv")
    return read_metric_rows(raw, ids)


def evaluate(epoch, device):
    backbone, args = load_backbone()
    for arm in ARMS:
        read_state(arm, epoch, backbone, args)
    del backbone
    manifest = read_json(MANIFEST)
    ids = manifest["splits"]["test"]
    if len(ids) != 1000:
        raise ValueError("Expected the original locked test1000 split")
    digest = sha256_file(MANIFEST)
    checkpoints = {"CF0": PARENT, **{arm: checkpoint(arm, epoch) for arm in ARMS}}
    hashes = {arm: sha256_file(path) for arm, path in checkpoints.items()}
    validation = reuse_metrics(comparison_dir(epoch), manifest["splits"]["val"], digest, hashes, epoch)
    baseline, archive = evaluation.archived_cf0("test", 12, ids, digest, PARENT)
    protocol = dict(split="test", offset=0, frozen_epochs=epoch, scene_ids=ids, device=str(device),
                    metric_profile=DEFAULT_PROFILE, output_policy=evaluation.POLICY,
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                    checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()},
                    checkpoint_hashes=hashes, archive_cf0=archive,
                    validation_metadata=validation.as_posix(), validation_metadata_sha256=sha256_file(validation),
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "run_cfdr_frozen_test.py", "cfdr_evaluation.py", "e2_cfdr.py", "e2_hcfdr.py", *EVALUATOR_FILES,
                    )})
    output = OUTPUT / f"epoch_{epoch:03d}_joint_v1"
    group = f"QXS-CFDR-STUDY-FROZEN-SEED42-TEST1000-EPOCH{epoch:02d}-JOINT-REF-FMI"
    evaluation.evaluate_group(output, group, "test", device, checkpoints, protocol, baseline,
                              methods=METHODS, contrasts=CONTRASTS, infer_method=infer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, choices=MILESTONES, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CFDR test")
    evaluate(args.epoch, torch.device(args.device))


if __name__ == "__main__":
    main()
