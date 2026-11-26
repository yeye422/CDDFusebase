"""Compare all three frozen CFDR variants after exactly four epochs."""

from pathlib import Path

import torch

from cfdr_evaluation import evaluate_group, infer as infer_cfdr
from e2_cfdr import VARIANTS
from run_cfdr_frozen import checkpoint_path
from run_e2_source_interval_val import read_metric_rows
from run_hcfdr_frozen import (
    ARCHIVE, EPOCHS, EXTENDED_EPOCHS, FINAL, MANIFEST, PARENT, POLICY,
    DEFAULT_PROFILE, archived_controls, h2_checkpoint, infer, load_backbone,
    read_h2_checkpoint, read_json, sha256_file, training_protocol, two_epoch_protocol,
)


OUTPUT = Path("test_result/e2_hcfdr_v1_frozen/val1000/epoch_004_joint_v1")
METHODS = {"CF0": "光学 Base 锚定 E2", "CF1": "CF1 原位加权", "CF2": "CF2 动态邻域", "H2": "H-CFDR"}
EVALUATOR_FILES = ("test_OS.py", "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py")


def verify_candidate(arm, path, previous, backbone, training):
    metadata = path.parent / "config.json"
    saved = read_json(metadata)
    expected = {key: training[key] for key in ("args", "lr", "parent_checkpoint_sha256", "split_manifest_sha256")}
    expected.update(epochs=EXTENDED_EPOCHS, arm=arm, continuation_checkpoint=previous.as_posix(),
                    continuation_checkpoint_sha256=sha256_file(previous))
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected continuation settings: {metadata}")
    if arm == "H2":
        state = read_h2_checkpoint(path, metadata, EXTENDED_EPOCHS)
        policy = "hcfdr_only_from_converged_cf0"
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["arm"], state["epoch"], state["variant"], state["config_sha256"]) != (
            arm, EXTENDED_EPOCHS, VARIANTS[arm], sha256_file(metadata)
        ):
            raise ValueError(f"Unexpected checkpoint: {path}")
        policy = "cfdr_only_from_converged_cf0"
    if (saved.get("training_policy") != policy
            or any(not torch.equal(value, state["model"][key]) for key, value in backbone.items())):
        raise ValueError(f"The run did not preserve the frozen backbone: {path}")


def reuse_metrics(folder, ids, digest, hashes, epochs):
    metadata = folder / "run_metadata.json"
    saved = read_json(metadata)
    expected = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=epochs, split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY)
    if (any(saved.get(key) != value for key, value in expected.items())
            or any(saved["checkpoint_hashes"].get(arm) != checksum for arm, checksum in hashes.items())
            or any(saved["source_sha256"].get(name) != sha256_file(Path(name)) for name in EVALUATOR_FILES)):
        raise ValueError(f"Archived CFDR evaluation differs: {folder}")
    return metadata


def evaluate_epoch_four(device):
    backbone, args = load_backbone()
    training = two_epoch_protocol(training_protocol(args))
    previous = {"CF1": checkpoint_path("CF1"), "CF2": checkpoint_path("CF2"), "H2": FINAL}
    candidates = {"CF1": checkpoint_path("CF1", EXTENDED_EPOCHS),
                  "CF2": checkpoint_path("CF2", EXTENDED_EPOCHS), "H2": h2_checkpoint(EXTENDED_EPOCHS)}
    for arm, path in candidates.items():
        verify_candidate(arm, path, previous[arm], backbone, training)
    ids = read_json(MANIFEST)["splits"]["val"]
    if len(ids) != 1000:
        raise ValueError("Expected the locked val1000 split")
    digest = sha256_file(MANIFEST)
    original, archive = archived_controls(ids, digest, training)
    checkpoints = {"CF0": PARENT, **candidates, **{f"{arm}_E02": path for arm, path in previous.items()}}
    hashes = {arm: sha256_file(path) for arm, path in checkpoints.items()}
    cf_metadata = reuse_metrics(ARCHIVE, ids, digest, {"CF1": hashes["CF1_E02"]}, EPOCHS)
    h2_folder = OUTPUT.parent / "epoch_002"
    h2_metadata = h2_folder / "run_metadata.json"
    h2_saved = read_json(h2_metadata)
    expected = dict(split="val", scene_ids=ids, frozen_epochs=EPOCHS, split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, archive=archive)
    if (any(h2_saved.get(key) != value for key, value in expected.items())
            or h2_saved["checkpoint_hashes"]["H2"] != hashes["H2_E02"]):
        raise ValueError("Archived H2 epoch-2 evaluation differs")
    cached_paths = {"CF1_E02": (ARCHIVE / "CF1/evaluator_metrics.csv", cf_metadata),
                    "CF2_E02": (ARCHIVE / "CF2/evaluator_metrics.csv", cf_metadata),
                    "H2_E02": (h2_folder / "H2/evaluator_metrics.csv", h2_metadata)}
    cf_four = ARCHIVE.parent / f"epoch_{EXTENDED_EPOCHS:03d}"
    if cf_four.exists() and (not OUTPUT.exists() or "CF1" in read_json(OUTPUT / "run_metadata.json")["cached"]):
        meta = reuse_metrics(cf_four, ids, digest, {arm: hashes[arm] for arm in ("CF0", "CF1", "CF2")}, EXTENDED_EPOCHS)
        cached_paths.update({arm: (cf_four / arm / "evaluator_metrics.csv", meta) for arm in ("CF1", "CF2")})
    cached = {arm: read_metric_rows(raw, ids) for arm, (raw, _) in cached_paths.items()}
    sources = {arm: dict(metrics=raw.as_posix(), metrics_sha256=sha256_file(raw), metadata=meta.as_posix(),
                         metadata_sha256=sha256_file(meta)) for arm, (raw, meta) in cached_paths.items()}
    protocol = dict(split="val", offset=0, frozen_epochs=EXTENDED_EPOCHS, scene_ids=ids, device=str(device),
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, split_manifest=MANIFEST.as_posix(),
                    split_manifest_sha256=digest, checkpoint_hashes=hashes, archive_cf0=archive, cached=sources,
                    checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()},
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "hcfdr_comparison.py", "cfdr_evaluation.py", "run_hcfdr_frozen.py",
                        "e2_hcfdr.py", "e2_cfdr.py", *EVALUATOR_FILES,
                    )})
    methods = {**METHODS, **{f"{arm}_E02": f"{METHODS[arm]} 第 2 轮" for arm in previous}}
    contrasts = (("CF0", "CF1"), ("CF0", "CF2"), ("CF0", "H2"), ("CF1", "CF2"),
                 ("CF1", "H2"), ("CF2", "H2"), *((f"{arm}_E02", arm) for arm in previous))

    def infer_arm(arm, checkpoint, folder, split, scene_ids, manifest_hash, device):
        if arm == "H2":
            return infer(checkpoint, folder, scene_ids, manifest_hash, device)
        return infer_cfdr(arm, checkpoint, folder, split, scene_ids, manifest_hash, device)

    evaluate_group(OUTPUT, "QXS-HCFDR-V1-FROZEN-SEED42-VAL1000-EPOCH04-JOINT-REF-FMI", "val", device,
                   checkpoints, protocol, original["H0"], methods=methods, contrasts=contrasts,
                   cached=cached, infer_method=infer_arm)
