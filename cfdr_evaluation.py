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
INITIAL_EVALUATOR_SHA256 = "a1cfa68ea4c644be7d70ef38239b181c44a1ac217fa34e3357238782da9fda3d"


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
    group = f"QXS-CFDR-V21-SEED42-{split.upper()}1000-FUSION{epoch:02d}-REF-FMI"
    evaluate_group(output, group, split, device, checkpoints, protocol, archived)


def evaluate_frozen(parent, candidates, epochs, device, *, previous=None):
    ids = read_json(MANIFEST)["splits"]["val"]
    if len(ids) != 1000:
        raise ValueError("Expected the fixed val1000 split")
    digest = sha256_file(MANIFEST)
    checkpoints = {"CF0": parent, **candidates}
    hashes = {arm: sha256_file(path) for arm, path in checkpoints.items()}
    backbone = torch.load(parent, map_location="cpu", weights_only=False)["model"]
    for arm, path in candidates.items():
        metadata = path.parent / "config.json"
        saved = read_json(metadata)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if ((state["arm"], state["epoch"], state["variant"]) != (arm, epochs, VARIANTS[arm])
                or state["config_sha256"] != sha256_file(metadata) or saved["split_manifest_sha256"] != digest
                or saved["parent_checkpoint_sha256"] != hashes["CF0"]
                or saved["training_policy"] != "cfdr_only_from_converged_cf0"
                or any(not torch.equal(value, state["model"][key]) for key, value in backbone.items())):
            raise ValueError(f"Unexpected frozen CFDR checkpoint: {path}")
        if previous and (Path(saved["continuation_checkpoint"]) != previous[arm]
                         or saved["continuation_checkpoint_sha256"] != sha256_file(previous[arm])):
            raise ValueError(f"Unexpected continuation parent: {path}")
    archived, archive_details = archived_cf0("val", 12, ids, digest, parent)
    protocol = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=epochs, device=str(device),
                    split_manifest=MANIFEST.as_posix(), split_manifest_sha256=digest,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, archive_cf0=archive_details,
                    checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()}, checkpoint_hashes=hashes,
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "e2_cfdr.py", "cfdr_evaluation.py", "test_OS.py", "utils/fusion_metrics.py",
                        "utils/reference_metrics.py", "utils/Evaluator.py",
                    )})
    output = OUTPUT.parent / "e2_cfdr_v21_frozen" / "val1000" / f"epoch_{epochs:03d}"
    group = f"QXS-CFDR-V21-FROZEN-SEED42-VAL1000-EPOCH{epochs:02d}-REF-FMI"
    if not previous:
        if output.exists() and epochs == 2:
            # The archived two-epoch result used the evaluator before the extension.
            saved = read_json(output / "run_metadata.json")
            old_sources = {**protocol["source_sha256"], "cfdr_evaluation.py": INITIAL_EVALUATOR_SHA256}
            if saved == dict(protocol, source_sha256=old_sources):
                protocol = saved
        evaluate_group(output, group, "val", device, checkpoints, protocol, archived)
        return

    previous_output = output.parent / "epoch_002"
    previous_metadata = previous_output / "run_metadata.json"
    archive = read_json(previous_metadata)
    expected = dict(split="val", offset=0, scene_ids=ids, frozen_epochs=2,
                    split_manifest_sha256=digest, metric_profile=DEFAULT_PROFILE, output_policy=POLICY)
    if (any(archive.get(key) != value for key, value in expected.items())
            or archive["checkpoint_hashes"]["CF0"] != hashes["CF0"]
            or any(archive["source_sha256"][name] != protocol["source_sha256"][name]
                   for name in ("test_OS.py", "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py"))):
        raise ValueError("Archived epoch-2 evaluation differs from the matched protocol")
    methods = dict(METHODS)
    cached = {}
    provenance = {"metadata": previous_metadata.as_posix(), "metadata_sha256": sha256_file(previous_metadata)}
    for arm, path in previous.items():
        key = f"{arm}_E02"
        digest = sha256_file(path)
        if archive["checkpoint_hashes"][arm] != digest:
            raise ValueError(f"Archived epoch-2 metrics do not match {path}")
        metrics = previous_output / arm / "metrics.csv"
        checkpoints[key], hashes[key] = path, digest
        methods[key] = f"{METHODS[arm]} 第 2 轮"
        cached[key] = read_metric_rows(metrics, ids)
        provenance[key] = dict(metrics=metrics.as_posix(), metrics_sha256=sha256_file(metrics))
    protocol.update(archived_epoch_002=provenance, checkpoints={arm: path.as_posix() for arm, path in checkpoints.items()})
    contrasts = (*CONTRASTS, *((f"{arm}_E02", arm) for arm in previous))
    evaluate_group(output, group, "val", device, checkpoints, protocol, archived,
                   methods=methods, contrasts=contrasts, cached=cached)


def evaluate_group(output, group, split, device, checkpoints, protocol, archived, *,
                   methods=METHODS, contrasts=CONTRASTS, cached=None, infer_method=None):
    ids = protocol["scene_ids"]
    manifest_hash = protocol["split_manifest_sha256"]
    hashes = protocol["checkpoint_hashes"]
    reused = dict(cached or {})
    if archived is not None:
        reused["CF0"] = archived
    if output.exists():
        if read_json(output / "run_metadata.json") != protocol:
            raise FileExistsError(f"Preserve existing results with different provenance: {output}")
    else:
        work = output.parent / f".{output.name}-{uuid4().hex}"
        work.mkdir(parents=True)
        print(f"Writing evaluation to {work}", flush=True)
        results = {}
        for arm, path in checkpoints.items():
            if arm in reused:
                (work / arm).mkdir()
                results[arm] = reused[arm]
            else:
                evaluate_arm = infer if infer_method is None else infer_method
                results[arm] = evaluate_arm(arm, path, work / arm, split, ids, manifest_hash, device)
        summaries = summarize(work, ids, results, contrasts, diagnostic_fields=(), source_metrics=())
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        work.rename(output)

    if split == "test":
        from build_five_method_test1000_comparison import main as refresh_five_methods

        refresh_five_methods()
    summary_path = output / "registry_metrics.json"
    summaries = read_json(summary_path)
    for arm in methods:
        if arm not in reused:
            for kind in ("gray", "rgb", "preview"):
                verify_images(output / arm / kind, ids)
    register_many([dict(id=f"{group}-{arm}", comparable_group=group, dataset="QXS-SAROPT", split=split, offset=0,
                        sample_count=len(ids), task="fusion",
                        method=f"冻结 CF0 + {label}" if "frozen_epochs" in protocol and arm != "CF0" else label,
                        status="formal_test" if split == "test" else "development_validation",
                        result_dir=(output / arm).as_posix(), metadata_path=(output / "run_metadata.json").as_posix(),
                        checkpoint=checkpoints[arm].as_posix(), checkpoint_sha256=hashes[arm],
                        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash,
                        metric_profile=DEFAULT_PROFILE, output_policy=POLICY, metrics=summaries[arm],
                        metric_source=dict(type="json_paths", path=summary_path.as_posix(),
                                           paths={key: [arm, key] for key in summaries[arm]}),
                        notes=("同组使用共同有限场景。CF0 和第 2 轮组复用已核验的逐图指标。" if cached else
                               "同组使用共同有限场景。CF0 可复用已核验的整套图像和逐图指标。"))
                   for arm, label in methods.items()])
    print(f"Paired comparison: {output / 'paired_comparison.csv'}", flush=True)
