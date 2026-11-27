"""Evaluate the epoch-4 frozen ASCD-V1 checkpoint on the held-out QXS test1000."""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

import torch

from build_five_method_test1000_comparison import main as refresh_five_methods
from cfdr_evaluation import archived_cf0, infer as infer_control
from e2_ascd import ASCDE2
from e2_cfdr import VARIANTS
from hcfdr_comparison import EVALUATOR_FILES
from run_ascd_frozen import (
    CANDIDATE, DATA, DEFAULT_PROFILE, EPOCHS, GROUP, MANIFEST, METHODS, PARENT, POLICY, ROOT,
    cfdr_checkpoint, checkpoint, read_checkpoint, training_inputs,
)
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import read_json, verify_images, write_json
from train_direct_fusion import sha256_file


OUTPUT = Path("test_result/e2_ascd_v1_frozen/test1000/epoch_004")
CONTROL_ARCHIVE = Path("test_result/e2_hcfdr_v1_frozen/test1000/epoch_004_joint_v1")
IMAGE_TYPES = ("gray", "rgb", "preview")
CONTRASTS = (("CF0", "CF1"), ("CF0", CANDIDATE), ("CF1", CANDIDATE))


def verify_run(path, ids, digest, checkpoint_hash, variant):
    metadata = read_json(path)
    expected = dict(split="test", offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids),
                    variant=variant, metric_profile=DEFAULT_PROFILE, split_manifest_sha256=digest,
                    output_range_policy="clamp", checkpoint_sha256=checkpoint_hash, metrics_only=False)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected test metadata: {path}")


def reuse_cf1(ids, digest, checkpoint_hash):
    group_path = CONTROL_ARCHIVE / "run_metadata.json"
    group = read_json(group_path)
    expected = dict(split="test", offset=0, scene_ids=ids, frozen_epochs=EPOCHS,
                    split_manifest_sha256=digest, metric_profile=DEFAULT_PROFILE, output_policy=POLICY)
    if (any(group.get(key) != value for key, value in expected.items())
            or group["checkpoint_hashes"]["CF1"] != checkpoint_hash
            or any(group["source_sha256"].get(name) != sha256_file(Path(name)) for name in EVALUATOR_FILES)):
        raise ValueError(f"Archived CF1 test protocol differs: {group_path}")
    folder = CONTROL_ARCHIVE / "CF1"
    metadata = folder / "run_metadata.json"
    verify_run(metadata, ids, digest, checkpoint_hash, VARIANTS["CF1"])
    for kind in IMAGE_TYPES:
        verify_images(folder / kind, ids)
    raw = folder / "evaluator_metrics.csv"
    return read_metric_rows(raw, ids), dict(
        archived_metrics=raw.as_posix(), archived_metrics_sha256=sha256_file(raw),
        archived_metadata=metadata.as_posix(), archived_metadata_sha256=sha256_file(metadata),
        group_metadata=group_path.as_posix(), group_metadata_sha256=sha256_file(group_path),
        archived_images={kind: (folder / kind).as_posix() for kind in IMAGE_TYPES},
    )


def infer_candidate(path, folder, ids, digest, device):
    import test_OS

    variant = ASCDE2.VARIANT
    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST), "--split", "test",
            "--limit", str(len(ids)), "--variant", variant, "--checkpoint", str(path), "--output-dir", str(folder),
            "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, {variant: ASCDE2}), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    verify_run(folder / "run_metadata.json", ids, digest, sha256_file(path), variant)
    for kind in IMAGE_TYPES:
        verify_images(folder / kind, ids)
    raw = (folder / "metrics.csv").rename(folder / "evaluator_metrics.csv")
    return read_metric_rows(raw, ids)


def load_controls(ids, digest, cf1_hash):
    results, archives = {}, {}
    readers = {"CF0": lambda: archived_cf0("test", 12, ids, digest, PARENT),
               "CF1": lambda: reuse_cf1(ids, digest, cf1_hash)}
    for arm, reader in readers.items():
        try:
            rows, provenance = reader()
            if rows is not None:
                results[arm], archives[arm] = rows, provenance
                continue
        except (OSError, KeyError, ValueError) as error:
            print(f"{arm} archive cannot be reused: {error}", flush=True)
        print(f"{arm}: evaluating the existing checkpoint on test1000", flush=True)
    return results, archives


def evaluate(device):
    frontend, _, training = training_inputs()
    config = ROOT / "config.json"
    if read_json(config) != training:
        raise ValueError(f"Unexpected ASCD training protocol: {config}")
    read_checkpoint(EPOCHS, sha256_file(config), frontend)
    ids = read_json(MANIFEST)["splits"]["test"]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected the locked QXS test1000 split")
    digest = training["split_manifest_sha256"]
    paths = {"CF0": PARENT, "CF1": cfdr_checkpoint("CF1", EPOCHS), CANDIDATE: checkpoint(EPOCHS)}
    hashes = {arm: sha256_file(path) for arm, path in paths.items()}
    results, archives = load_controls(ids, digest, hashes["CF1"])
    sources = ("run_ascd_frozen_test.py", "cfdr_evaluation.py", "e2_cfdr.py", "run_e2_aspr_pilot.py",
               *EVALUATOR_FILES)
    protocol = dict(split="test", offset=0, scene_ids=ids, device=str(device), frozen_epochs=EPOCHS,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, control_archives=archives,
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                    checkpoints={arm: path.as_posix() for arm, path in paths.items()}, checkpoint_hashes=hashes,
                    training_config=config.as_posix(), training_config_sha256=sha256_file(config),
                    source_sha256={name: sha256_file(Path(name)) for name in sources})
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve previous ASCD test results at {OUTPUT}")
    else:
        work = OUTPUT.parent / f".{OUTPUT.name}-{uuid4().hex}"
        work.mkdir(parents=True)
        for arm, path in paths.items():
            folder = work / arm
            if arm in archives:
                folder.mkdir()
            elif arm == CANDIDATE:
                results[arm] = infer_candidate(path, folder, ids, digest, device)
            else:
                results[arm] = infer_control(arm, path, folder, "test", ids, digest, device)
        summaries = summarize(work, ids, results, CONTRASTS, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        work.rename(OUTPUT)
    for arm in paths.keys() - archives.keys():
        for kind in IMAGE_TYPES:
            verify_images(OUTPUT / arm / kind, ids)
    refresh_five_methods()
    summary_path = OUTPUT / "registry_metrics.json"
    summaries = read_json(summary_path)
    group = f"{GROUP}-TEST1000-EPOCH04-REF-FMI"
    register_many([dict(
        id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split="test", offset=0,
        sample_count=len(ids), task="fusion", method=label, status="formal_test",
        result_dir=(OUTPUT / arm).as_posix(), metadata_path=(OUTPUT / "run_metadata.json").as_posix(),
        checkpoint=paths[arm].as_posix(), checkpoint_sha256=hashes[arm], split_manifest=MANIFEST.as_posix(),
        split_manifest_sha256=digest, metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=summaries[arm],
        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                           paths={key: [arm, key] for key in summaries[arm]}),
        notes=("复用已核验的 test1000 逐图指标和全部图像。每项指标使用三组共同有限场景。"
               if arm in archives else "保存 test1000 全部三类图像。每项指标使用三组共同有限场景。"),
    ) for arm, label in METHODS.items()])
    print(f"Paired comparison: {OUTPUT / 'paired_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ASCD test")
    evaluate(torch.device(args.device))


if __name__ == "__main__":
    main()
