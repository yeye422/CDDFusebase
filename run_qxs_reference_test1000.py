"""Re-score archived QXS outputs and infer only the missing test images."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from maintain_experiment_index import REGISTRY_PATH, sha256_file, validate
from utils.Evaluator import PaperEvaluator
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator
from utils.reference_metrics import PROFILE, STATUS, ReferenceEvaluator


DATA = Path("../datasets/QXSLAB_SAROPT")
MANIFEST = DATA / "splits/cddfuse_8000_1000_1000_seed20260924.json"
BASELINES = Path("../comparation/results/qxslab_paper_1000_manual")
LWPD = Path("test_result/lwpd_qxs_full_v1")
BASE_ANCHOR = Path("test_result/e2_scratch_base_anchor_v1/val1000/seed42")
GROUP = "QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1"
OUTPUT = Path("test_result/qxs_reference_v2_candidate_test1000_v1")
LEGACY_FMI_OUTPUT = Path("test_result/qxs_reference_v2_legacy_fmi_test1000_v1")
LEGACY_FMI_GROUP = "QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1"
B0_ARCHIVE = Path("test_result/reliability_v4/B0_clamp")
B0_OUTPUT = Path("test_result/qxs_reference_v2_with_trained_b0_test1000_v1")
B0_GROUP = "QXS-TEST1000-REFERENCE-V2-WITH-TRAINED-B0-v1"
METRICS = ReferenceEvaluator.METRIC_NAMES
METHODS = (
    ("wavelet", "Wavelet"),
    ("u2fusion", "U2Fusion（多曝光权重，零样本）"),
    ("laplacian", "Laplacian"),
    ("cddfuse_zero_shot", "CDDFuse 零样本（clamp）"),
    ("lwpd_e4", "LWPD E4"),
    ("lwpd_e2", "LWPD E2"),
    ("lwpd_e2_scratch", "LWPD E2 从头训练（光学 Base，零初始化修正）"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_image(path, mode):
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def verify_images(folder, scene_ids):
    if {image.stem for image in folder.glob("*.png")} != set(scene_ids):
        raise ValueError(f"Unexpected scene IDs in {folder}")


def prepare_sources(scene_ids, manifest, manifest_hash):
    protocol_path = BASELINES / "protocol.json"
    protocol = read_json(protocol_path)
    if (protocol["image_ids"] != scene_ids or protocol["offset"] != 0
            or protocol["split_manifest_sha256"] != manifest["splits_sha256"]):
        raise ValueError("Archived comparison split differs from the fixed test split")
    u2 = Path("../comparation/U2Fusion-main/model/model.pth")
    sources = {
        key: dict(archive=(BASELINES / key).as_posix(), source_metadata=protocol_path.as_posix(),
                  source_output_policy="saved_uint8_rgb_luma")
        for key in ("wavelet", "u2fusion", "laplacian")
    }
    sources["u2fusion"].update(checkpoint=u2.as_posix(), checkpoint_sha256=sha256_file(u2),
                               checkpoint_provenance="local weight; historical protocol has no weight hash")
    for key, folder, variant in (
        ("lwpd_e4", LWPD / "e4_seed42_test1000", "lwpd_e4"),
        ("lwpd_e2", LWPD / "e2_seed42_test1000", "lwpd_e2"),
        ("lwpd_e2_scratch", BASE_ANCHOR, "lwpd_e2_base_anchor_v1"),
    ):
        metadata_path = folder / "run_metadata.json"
        meta = read_json(metadata_path)
        is_scratch = key == "lwpd_e2_scratch"
        checkpoint = Path(meta["checkpoint"])
        if (meta["split"] != ("val" if is_scratch else "test")
                or meta["selected_scene_ids"] != (manifest["splits"]["val"] if is_scratch else scene_ids)
                or meta["variant"] != variant or meta["offset"] != 0
                or meta["output_range_policy"] != "clamp" or meta["split_manifest_sha256"] != manifest_hash
                or sha256_file(checkpoint) != meta["checkpoint_sha256"]):
            raise ValueError(f"Unexpected source settings in {metadata_path}")
        sources[key] = dict(
            source_metadata=metadata_path.as_posix(), checkpoint=checkpoint.as_posix(),
            checkpoint_sha256=meta["checkpoint_sha256"], variant=variant, source_output_policy="clamp_then_uint8",
        )
        if is_scratch:
            sources[key]["entry_point"] = "e2_base_anchor.py"
        else:
            sources[key]["archive"] = folder.as_posix()
            verify_images(folder / "preview", scene_ids)
    cddfuse = Path("models/CDDFuse_IVF.pth")
    sources["cddfuse_zero_shot"] = dict(
        entry_point="test_OS.py", variant="original", checkpoint=cddfuse.as_posix(),
        checkpoint_sha256=sha256_file(cddfuse), source_output_policy="clamp_then_uint8",
    )
    for source in sources.values():
        if "archive" in source:
            for kind in ("gray", "rgb"):
                verify_images(Path(source["archive"]) / kind, scene_ids)
    return sources


def score_saved(source, selected, work, evaluator=ReferenceEvaluator):
    archive = Path(source["archive"])
    preview_dir = work / "preview"
    make_preview = not (archive / "preview").is_dir()
    if make_preview:
        preview_dir.mkdir()
    rows = []
    for index, (optical_path, sar_path) in enumerate(selected, start=1):
        stem = optical_path.stem
        optical = read_image(optical_path, cv2.IMREAD_COLOR)
        sar = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
        fused = read_image(archive / "gray" / f"{stem}.png", cv2.IMREAD_UNCHANGED)
        optical_y = cv2.cvtColor(optical, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        if fused.dtype != np.uint8 or fused.shape != sar.shape or fused.shape != optical_y.shape:
            raise ValueError(f"{archive}: invalid grayscale image {stem}")
        rows.append((stem, *evaluator.metric_vector(fused, optical_y, sar)))
        if make_preview:
            rgb = read_image(archive / "rgb" / f"{stem}.png", cv2.IMREAD_COLOR)
            sar_bgr = cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR)
            preview = np.concatenate((optical, sar_bgr, rgb), axis=1)
            if not cv2.imwrite(str(preview_dir / f"{stem}.png"), preview):
                raise OSError(f"Cannot write preview for {stem}")
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"{archive.name}: {index}/{len(selected)}", flush=True)
    values = np.array([row[1:] for row in rows])
    with (work / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", *METRICS))
        writer.writerows(rows)
        writer.writerow(("mean", *values.mean(axis=0)))
        writer.writerow(("std", *values.std(axis=0)))


def infer(source, selected, work):
    subprocess.run([
        sys.executable, source["entry_point"], "--data-root", str(DATA), "--split-manifest", str(MANIFEST),
        "--split", "test", "--offset", "0", "--limit", str(len(selected)), "--checkpoint", source["checkpoint"],
        "--variant", source["variant"], "--metric-profile", PROFILE, "--output-range-policy", "clamp",
        "--timing-warmup", "0", "--output-dir", str(work),
    ], check=True)
    meta = read_json(work / "run_metadata.json")
    if meta["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed during inference: {source['checkpoint']}")


def summarize(path, scene_ids):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if [row["image"] for row in rows] != [*scene_ids, "mean", "std"]:
        raise ValueError(f"Unexpected metrics scene IDs in {path}")
    summary = {}
    for metric in METRICS:
        count = sum(np.isfinite(float(row[metric])) for row in rows[:-2])
        mean, std = (float(row[metric]) for row in rows[-2:])
        summary[metric] = dict(
            mean=mean if np.isfinite(mean) else None,
            std=std if np.isfinite(std) else None,
            finite_count=int(count),
        )
    return summary


def evaluate_method(key, source, selected, protocol, output, evaluator=ReferenceEvaluator):
    folder = output / key
    scene_ids = protocol["scene_ids"]
    if "archive" in source:
        archive = Path(source["archive"])
        preview = archive / "preview" if (archive / "preview").is_dir() else folder / "preview"
        images = dict(gray=(archive / "gray").as_posix(), rgb=(archive / "rgb").as_posix(),
                      preview=preview.as_posix())
    else:
        images = {kind: (folder / kind).as_posix() for kind in ("gray", "rgb", "preview")}
    provenance = dict(protocol, **source, images=images)
    if not folder.exists():
        work = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=output))
        if "archive" in source:
            score_saved(source, selected, work, evaluator=evaluator)
        else:
            infer(source, selected, work)
        summarize(work / "metrics.csv", scene_ids)
        for image_dir in map(Path, images.values()):
            if image_dir.parent == folder:
                verify_images(work / image_dir.name, scene_ids)
        write_json(work / "reference_metadata.json", provenance)
        work.rename(folder)
    elif read_json(folder / "reference_metadata.json") != provenance:
        raise ValueError(f"Existing output has a different protocol: {folder}")
    for image_dir in images.values():
        if Path(image_dir).parent == folder:
            verify_images(Path(image_dir), scene_ids)
    return summarize(folder / "metrics.csv", scene_ids)


def write_comparison(output, full_results, protocol, methods=METHODS):
    scene_ids = protocol["scene_ids"]
    matrices = []
    for key, _ in methods:
        with (output / key / "metrics.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))[:-2]
        matrices.append([[float(row[metric]) for metric in METRICS] for row in rows])
    scores = np.asarray(matrices)
    common = np.isfinite(scores).all(axis=0)
    excluded = {
        metric: [scene_id for scene_id, valid in zip(scene_ids, common[:, col]) if not valid]
        for col, metric in enumerate(METRICS)
    }
    results = {key: {} for key, _ in methods}
    for col, metric in enumerate(METRICS):
        values = scores[:, common[:, col], col]
        count = values.shape[1]
        for row, (key, _) in enumerate(methods):
            results[key][metric] = dict(
                mean=float(values[row].mean()) if count else None,
                std=float(values[row].std()) if count else None,
                common_count=count,
                finite_count=full_results[key][metric]["finite_count"],
            )
    write_json(output / "comparison_14_metrics.json", dict(
        protocol=protocol, methods=results, full_test_methods=full_results, excluded_scene_ids=excluded,
    ))
    with (output / "comparison_14_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        fields = ("mean", "std", "common_count", "finite_count")
        writer.writerow(["method", "images", *(f"{metric}_{field}" for metric in METRICS for field in fields)])
        for key, label in methods:
            writer.writerow([label, len(scene_ids),
                             *(results[key][metric][field] for metric in METRICS for field in fields)])
    fmi = protocol.get("fmi_description", "FAST-FMI DCT")
    method_count = "七种" if methods == METHODS else f"{len(methods)} 种"
    title = "七方法复评" if methods == METHODS else f"{len(methods)} 种方法复评"
    lines = [f"# QXS-SAROPT {title}", "",
             f"评价器 `{protocol['metric_profile']}`，共 {len(scene_ids)} 个测试场景。",
             f"NAB/F 为 Kumar 改版，FMI 为 {fmi}。参考指标的 MATLAB 一致性待核验。",
             f"各指标取{method_count}方法共同有效场景的均值，无共同有效场景记作 —。标准差和样本数见 CSV，排除场景见 JSON。",
             "", "| 方法 | " + " | ".join(name + ("↓" if name == "Nabf" else "↑") for name in METRICS) + " |",
             "|---|" + "---:|" * len(METRICS)]
    for key, label in methods:
        formatted = []
        for metric in METRICS:
            value = results[key][metric]["mean"]
            values = [summary[metric]["mean"] for summary in results.values()]
            if value is None:
                formatted.append("—")
                continue
            text = f"{value:.6g}" if metric == "Nabf" else f"{value:.4f}"
            if None not in values and value == (min(values) if metric == "Nabf" else max(values)):
                text = f"**{text}**"
            formatted.append(text)
        lines.append("| " + " | ".join((label, *formatted)) + " |")
    lines.append("| 共同有效场景数 | " + " | ".join(
        str(len(scene_ids) - len(excluded[metric])) for metric in METRICS
    ) + " |")
    (output / "comparison_14_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_results(output, sources, results, protocol, group=GROUP, methods=METHODS):
    registry = read_json(REGISTRY_PATH)
    by_id = {entry["id"]: entry for entry in registry["experiments"]}
    for key, label in methods:
        folder = output / key
        metrics = {name: value["mean"] for name, value in results[key].items() if value["mean"] is not None}
        entry = dict(
            id=f"{group}-{key}", method=label, task="fusion", status="reference_candidate_test",
            dataset="QXS-SAROPT", split="test", offset=0, sample_count=len(protocol["scene_ids"]),
            comparable_group=group, metric_profile=protocol["metric_profile"], output_policy=protocol["output_policy"],
            split_manifest=MANIFEST.as_posix(), split_manifest_sha256=protocol["split_manifest_sha256"],
            result_dir=folder.as_posix(), metadata_path=(folder / "reference_metadata.json").as_posix(),
            metrics=metrics, metric_source=dict(type="csv_mean", path=(folder / "metrics.csv").as_posix()),
        )
        for field in ("checkpoint", "checkpoint_sha256"):
            if field in sources[key]:
                entry[field] = sources[key][field]
        previous = by_id.get(entry["id"])
        if previous is None:
            registry["experiments"].append(entry)
        elif previous != entry:
            raise ValueError(f"Registry already has a different record for {entry['id']}")
    errors = validate(registry)
    if errors:
        raise ValueError("\n".join(errors))
    if len(registry["experiments"]) != len(by_id):
        registry["updated_at"] = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        temp = REGISTRY_PATH.with_suffix(".json.tmp")
        write_json(temp, registry)
        temp.replace(REGISTRY_PATH)
    subprocess.run([sys.executable, "maintain_experiment_index.py", "--check", "--render"], check=True)


def read_test_rows(path, scene_ids):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if [row["image"] for row in rows] != [*scene_ids, "mean", "std"]:
        raise ValueError(f"Unexpected scene IDs in {path}")
    return rows


def replace_fmi(pairs, manifest_hash, smoke):
    """Reuse the completed reference run and compute only unavailable paper FMI."""
    scene_ids = [optical.stem for optical, _ in pairs]
    reference = read_json(OUTPUT / "comparison_14_metrics.json")["protocol"]
    if (reference["scene_ids"] != scene_ids or reference["split_manifest_sha256"] != manifest_hash
            or reference["metric_profile"] != PROFILE):
        raise ValueError("The completed reference results use a different test protocol")
    if smoke:
        Path("tmp").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="qxs_legacy_fmi_smoke_", dir="tmp"))
        pairs = pairs[:1]
    else:
        output = LEGACY_FMI_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        reference, scene_ids=scene_ids[:len(pairs)], metric_profile=DEFAULT_PROFILE,
        fmi_description="PaperEvaluator 的全图 DCT 互信息之和",
        fmi_evaluator_sha256=sha256_file(Path("utils/Evaluator.py")),
    )
    sources = {}
    results = {}
    for key, _ in METHODS:
        source = OUTPUT / key
        metrics_path = source / "metrics.csv"
        meta_path = source / "reference_metadata.json"
        meta = read_json(meta_path)
        if any(meta.get(field) != value for field, value in reference.items()):
            raise ValueError(f"Unexpected source protocol in {meta_path}")
        reference_rows = read_test_rows(metrics_path, scene_ids)
        rows = reference_rows[:len(pairs)]
        archived = Path(meta["archive"]) / "metrics.csv" if "archive" in meta else None
        gray = Path(meta["images"]["gray"])
        if archived is not None and archived.parent / "gray" != gray:
            raise ValueError(f"Legacy FMI does not correspond to the saved gray images: {archived}")
        provenance = dict(meta)
        provenance.update(protocol, reference_metadata=meta_path.as_posix(),
                          reference_metrics=metrics_path.as_posix(), reference_metrics_sha256=sha256_file(metrics_path),
                          legacy_fmi_source=(archived or gray).as_posix(),
                          legacy_fmi_source_sha256=sha256_file(archived) if archived else None)
        folder = output / key
        if not folder.exists():
            if archived is not None:
                legacy = read_test_rows(archived, scene_ids)
                fmi = [float(row["FMI"]) for row in legacy[:len(pairs)]]
            else:
                fmi = []
                for optical_path, sar_path in pairs:
                    optical = read_image(optical_path, cv2.IMREAD_COLOR)
                    optical_y = cv2.cvtColor(optical, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                    sar = read_image(sar_path, cv2.IMREAD_GRAYSCALE)
                    fused = read_image(gray / f"{optical_path.stem}.png", cv2.IMREAD_UNCHANGED)
                    if fused.dtype != np.uint8 or fused.shape != sar.shape or fused.shape != optical_y.shape:
                        raise ValueError(f"Invalid saved gray image for {key}, scene {optical_path.stem}")
                    fmi.append(PaperEvaluator.FMI(fused, optical_y, sar))
            if not np.isfinite(fmi).all():
                raise ValueError(f"Nonfinite paper FMI for {key}")
            work = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=output))
            with (work / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("image", *METRICS))
                writer.writeheader()
                for row, value in zip(rows, fmi):
                    writer.writerow(dict(row, FMI=value))
                for label, value, reference_row in zip(("mean", "std"), (np.mean(fmi), np.std(fmi)),
                                                       reference_rows[-2:]):
                    if smoke:
                        values = {
                            metric: rows[0][metric] if label == "mean" or not np.isfinite(float(rows[0][metric])) else 0.0
                            for metric in METRICS
                        }
                    else:
                        values = reference_row
                    writer.writerow(dict(values, image=label, FMI=value))
            write_json(work / "reference_metadata.json", provenance)
            work.rename(folder)
        elif read_json(folder / "reference_metadata.json") != provenance:
            raise ValueError(f"Existing output has different sources: {folder}")
        sources[key] = meta
        results[key] = summarize(folder / "metrics.csv", protocol["scene_ids"])
        print(f"{key}: FMI={results[key]['FMI']['mean']:.6f}", flush=True)
    write_comparison(output, results, protocol)
    if not smoke:
        register_results(output, sources, results, protocol, group=LEGACY_FMI_GROUP)
    print((output / "comparison_14_metrics.md").resolve())


def trained_b0_source(scene_ids, manifest_hash):
    metadata_path = B0_ARCHIVE / "run_metadata.json"
    meta = read_json(metadata_path)
    expected = dict(selected_scene_ids=scene_ids, split_manifest_sha256=manifest_hash,
                    offset=0, evaluated_pairs=len(scene_ids), output_range_policy="clamp",
                    metric_profile="paper")
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Archived trained CDDFuse B0 uses a different split or output protocol")
    checkpoint = Path(meta["checkpoint"])
    if sha256_file(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Archived B0 checkpoint hash has changed")
    for kind in ("gray", "rgb", "preview"):
        verify_images(B0_ARCHIVE / kind, scene_ids)
    return dict(archive=B0_ARCHIVE.as_posix(), source_metadata=metadata_path.as_posix(),
                source_metadata_sha256=sha256_file(metadata_path), variant="original",
                checkpoint=checkpoint.as_posix(), checkpoint_sha256=meta["checkpoint_sha256"],
                source_output_policy="clamp_then_uint8")


def compare_with_trained_b0(pairs, manifest_hash, smoke):
    scene_ids = [optical.stem for optical, _ in pairs]
    reference = read_json(LEGACY_FMI_OUTPUT / "comparison_14_metrics.json")["protocol"]
    expected = dict(scene_ids=scene_ids, split="test", offset=0, split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy="uint8_grayscale_no_renormalization",
                    evaluator_sha256=sha256_file(Path("utils/reference_metrics.py")),
                    fmi_evaluator_sha256=sha256_file(Path("utils/Evaluator.py")))
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed seven-method results use a different test split or evaluator")
    b0 = trained_b0_source(scene_ids, manifest_hash)
    if smoke:
        with tempfile.TemporaryDirectory() as directory:
            score_saved(b0, pairs[:1], Path(directory), evaluator=FusionEvaluator)
            metrics = summarize(Path(directory) / "metrics.csv", scene_ids[:1])
            archived = read_test_rows(B0_ARCHIVE / "metrics.csv", scene_ids)
            if not np.isclose(metrics["FMI"]["mean"], float(archived[0]["FMI"]), atol=5e-7, rtol=0):
                raise ValueError("Single-scene paper FMI differs from the archived B0 result")
            print(f"B0 scene {scene_ids[0]}: FMI={metrics['FMI']['mean']:.6f}, NAB/F={metrics['Nabf']['mean']:.6f}")
        return

    methods = (*METHODS, ("cddfuse_b0", "CDDFuse B0（QXS 训练后）"))
    protocol = dict(reference, comparison_source=LEGACY_FMI_OUTPUT.as_posix())
    B0_OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {"cddfuse_b0": b0}
    results = {}
    for key, _ in METHODS:
        archive = LEGACY_FMI_OUTPUT / key
        meta_path = archive / "reference_metadata.json"
        metrics_path = archive / "metrics.csv"
        meta = read_json(meta_path)
        if any(meta.get(field) != value for field, value in reference.items()):
            raise ValueError(f"Existing method uses a different reference protocol: {key}")
        read_test_rows(metrics_path, scene_ids)
        for image_dir in meta["images"].values():
            verify_images(Path(image_dir), scene_ids)
        provenance = dict(meta, **protocol, reused_metrics=metrics_path.as_posix(),
                          reused_metrics_sha256=sha256_file(metrics_path),
                          reused_metadata=meta_path.as_posix(), reused_metadata_sha256=sha256_file(meta_path))
        folder = B0_OUTPUT / key
        if not folder.exists():
            work = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=B0_OUTPUT))
            shutil.copyfile(metrics_path, work / "metrics.csv")
            write_json(work / "reference_metadata.json", provenance)
            work.rename(folder)
        elif (read_json(folder / "reference_metadata.json") != provenance
              or sha256_file(folder / "metrics.csv") != provenance["reused_metrics_sha256"]):
            raise ValueError(f"Existing extended comparison has different sources: {folder}")
        sources[key] = meta
        results[key] = summarize(folder / "metrics.csv", scene_ids)
    results["cddfuse_b0"] = evaluate_method("cddfuse_b0", b0, pairs, protocol, B0_OUTPUT,
                                           evaluator=FusionEvaluator)
    write_comparison(B0_OUTPUT, results, protocol, methods=methods)
    register_results(B0_OUTPUT, sources, results, protocol, group=B0_GROUP, methods=methods)
    print((B0_OUTPUT / "comparison_14_metrics.md").resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One test scene per method, without registration")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--legacy-fmi", action="store_true", help="Reuse the reference run with paper FMI (default)")
    profile.add_argument("--reference-fmi", action="store_true", help="Reproduce the original FAST-FMI reference run")
    profile.add_argument("--with-trained-b0", action="store_true",
                         help="Rescore saved trained B0 images and reuse the existing seven-method results")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    splits, manifest = manifest_pairs(DATA, MANIFEST)
    pairs = splits["test"]
    if len(pairs) != 1000:
        raise ValueError("Expected exactly 1000 test scenes")
    manifest_hash = sha256_file(MANIFEST)
    if args.with_trained_b0:
        compare_with_trained_b0(pairs, manifest_hash, args.smoke)
        return
    if not args.reference_fmi:
        replace_fmi(pairs, manifest_hash, args.smoke)
        return
    sources = prepare_sources([optical.stem for optical, _ in pairs], manifest, manifest_hash)
    output = OUTPUT
    if args.smoke:
        Path("tmp").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="qxs_reference_smoke_", dir="tmp"))
        pairs = pairs[:1]
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        dataset="QXS-SAROPT", split="test", offset=0, scene_ids=[optical.stem for optical, _ in pairs],
        split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash, metric_profile=PROFILE,
        metric_status=STATUS, output_policy="uint8_grayscale_no_renormalization",
        evaluator_sha256=sha256_file(Path("utils/reference_metrics.py")),
        evaluator_base_sha256=sha256_file(Path("utils/Evaluator.py")),
    )
    results = {key: evaluate_method(key, sources[key], pairs, protocol, output) for key, _ in METHODS}
    write_comparison(output, results, protocol)
    if not args.smoke:
        register_results(output, sources, results, protocol)
    print((output / "comparison_14_metrics.md").resolve())


if __name__ == "__main__":
    main()
