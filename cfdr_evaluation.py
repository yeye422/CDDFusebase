"""Matched full-image CFDR evaluation and checkpoint-specific comparisons."""

import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cfdr import ARMS, CFDRE2, VARIANTS
from run_cfdr import PARENT, checkpoint_path
from run_e2_aspr_pilot import summarize
from run_e2_matched_no_pac import register_many
from run_e2_source_interval_val import read_metric_rows
from run_qxs_reference_test1000 import DATA, LEGACY_FMI_OUTPUT, MANIFEST, read_json, verify_images, write_json
from run_spie_analytic import load_p0
from train_direct_fusion import sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


OUTPUT = Path("test_result/e2_cfdr_v21")
METHODS = {"CF0": "光学 Base 锚定 E2", "CF1": "CFDR Scalar", "CF2": "CFDR V2.1 动态滤波"}
CONTRASTS = (("CF0", "CF1"), ("CF1", "CF2"), ("CF0", "CF2"))
POLICY = "clamp_then_uint8"


def archived_cf0(split, epoch, ids, digest, checkpoint):
    if epoch != 12:
        return None, None
    if split == "val":
        return load_p0(ids, digest)
    archive = LEGACY_FMI_OUTPUT / "lwpd_e2_scratch"
    metadata = archive / "reference_metadata.json"
    metrics = archive / "metrics.csv"
    if not (metadata.is_file() and metrics.is_file()):
        return None, None
    meta = read_json(metadata)
    expected = dict(split="test", scene_ids=ids, offset=0, split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, source_output_policy=POLICY,
                    checkpoint_sha256=sha256_file(checkpoint),
                    evaluator_sha256=sha256_file(Path("utils/reference_metrics.py")),
                    fmi_evaluator_sha256=sha256_file(Path("utils/Evaluator.py")))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived CF0 test1000 differs from the matched protocol")
    for folder in meta["images"].values():
        verify_images(Path(folder), ids)
    return read_metric_rows(metrics, ids), dict(archived_metrics=metrics.as_posix(),
        archived_metrics_sha256=sha256_file(metrics), archived_metadata=metadata.as_posix(),
        archived_metadata_sha256=sha256_file(metadata), archived_images=meta["images"])


def infer(arm, checkpoint, folder, split, ids, digest, device):
    import test_OS

    types = {variant: (lambda key=key: CFDRE2(key)) for key, variant in VARIANTS.items()}
    types[OpticalBaseAnchoredE2.VARIANT] = OpticalBaseAnchoredE2
    variant = VARIANTS.get(arm, OpticalBaseAnchoredE2.VARIANT)
    argv = ["test_OS.py", "--data-root", str(DATA), "--split-manifest", str(MANIFEST),
            "--split", split, "--limit", str(len(ids)), "--variant", variant,
            "--checkpoint", str(checkpoint), "--output-dir", str(folder),
            "--metric-profile", DEFAULT_PROFILE, "--output-range-policy", "clamp"]
    with patch.dict(test_OS.DIRECT_MODEL_TYPES, types), patch.object(sys, "argv", argv):
        test_OS.main(device=str(device))
    meta = read_json(folder / "run_metadata.json")
    expected = dict(split=split, offset=0, selected_scene_ids=ids, evaluated_pairs=len(ids), variant=variant,
                    metric_profile=DEFAULT_PROFILE, split_manifest_sha256=digest, output_range_policy="clamp",
                    checkpoint_sha256=sha256_file(checkpoint), metrics_only=False)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected evaluation metadata: {folder}")
    for kind in ("gray", "rgb", "preview"):
        verify_images(folder / kind, ids)
    raw = (folder / "metrics.csv").rename(folder / "evaluator_metrics.csv")
    return read_metric_rows(raw, ids)


def evaluate(split, epoch, device):
    ids = read_json(MANIFEST)["splits"][split]
    if len(ids) != 1000:
        raise ValueError(f"Expected the fixed {split}1000")
    manifest_hash = sha256_file(MANIFEST)
    checkpoints = {arm: checkpoint_path(arm, epoch) for arm in METHODS}
    hashes = {arm: sha256_file(checkpoint) for arm, checkpoint in checkpoints.items()}
    for arm in ARMS:
        checkpoint = checkpoints[arm]
        config = checkpoint.parent / "config.json"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved = read_json(config)
        if ((state["arm"], state["epoch"], state["stage"], state["variant"]) != (arm, epoch + 3, "fusion", VARIANTS[arm])
                or state["config_sha256"] != sha256_file(config) or saved["split_manifest_sha256"] != manifest_hash
                or saved["parent_checkpoint_sha256"] != sha256_file(PARENT)):
            raise ValueError(f"Unexpected matched CFDR checkpoint: {checkpoint}")
    archived, archive_details = archived_cf0(split, epoch, ids, manifest_hash, checkpoints["CF0"])
    protocol = dict(split=split, offset=0, scene_ids=ids, stage_ii_epoch=epoch, device=str(device),
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, archive_cf0=archive_details,
                    checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()}, checkpoint_hashes=hashes,
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "e2_cfdr.py", "cfdr_evaluation.py", "test_OS.py", "utils/fusion_metrics.py",
                        "utils/reference_metrics.py", "utils/Evaluator.py",
                    )})
    output = OUTPUT / f"{split}1000" / f"fusion_epoch_{epoch:02d}"
    if output.exists():
        if read_json(output / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve existing results with different provenance: {output}")
    else:
        work = output.parent / f".{output.name}-{uuid4().hex}"
        work.mkdir(parents=True)
        print(f"Writing evaluation to {work}", flush=True)
        results = {}
        for arm, path in checkpoints.items():
            if arm == "CF0" and archived is not None:
                (work / arm).mkdir()
                results[arm] = archived
            else:
                results[arm] = infer(arm, path, work / arm, split, ids, manifest_hash, device)
        summaries = summarize(work, ids, results, CONTRASTS, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        work.rename(output)

    if split == "test":
        from build_five_method_test1000_comparison import main as refresh_five_methods

        refresh_five_methods()
    summary_path = output / "registry_metrics.json"
    summaries = read_json(summary_path)
    group = f"QXS-CFDR-V21-SEED42-{split.upper()}1000-FUSION{epoch:02d}-REF-FMI"
    for arm in METHODS:
        if arm != "CF0" or archived is None:
            for kind in ("gray", "rgb", "preview"):
                verify_images(output / arm / kind, ids)
    register_many([dict(id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split=split, offset=0,
                        sample_count=len(ids), task="fusion", method=label,
                        status="formal_test" if split == "test" else "development_validation",
                        result_dir=(output / arm).as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
                        checkpoint=checkpoints[arm].as_posix(), checkpoint_sha256=hashes[arm],
                        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash,
                        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=summaries[arm],
                        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                                           paths={key: [arm, key] for key in summaries[arm]}),
                        notes="同组使用共同有限场景。CF0 可复用已核验的整套图像和逐图指标。")
                   for arm, label in METHODS.items()])
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)
