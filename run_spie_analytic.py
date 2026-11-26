"""Compare SPIE screened-Poisson fusion with fixed val1000 controls."""

import argparse
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

from compare_lwpd_prl import read_metrics
from run_e2_aspr_pilot import METRICS, summarize, write_csv
from run_e2_matched_no_pac import register_many
from run_qxs_reference_test1000 import DATA, MANIFEST, read_image, read_json, verify_images, write_json
from spie_analytic import SPEC, fuse
from train_direct_fusion import sha256_file
from utils.dataset_os import manifest_pairs
from utils.fusion_metrics import DEFAULT_PROFILE, FusionEvaluator


P0_ARCHIVE = Path("test_result/e2_scratch_rccr_l2_v1/val1000/seed42")
P0_CHECKPOINT = Path("models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth")
OUTPUT = Path("test_result/spie_analytic_v1/val1000")
POLICY = "clamp_then_uint8"
METHODS = {"P0": "LWPD E2 光学 Base 锚定", "P1": "有效区域双源平均",
           "P2": "SPIE 强度参考场", "P3": "SPIE 筛泊松解析融合"}
CONTRASTS = (("P2", "P3"), ("P0", "P3"), ("P1", "P2"), ("P1", "P3"))


def load_p0(scene_ids, manifest_hash):
    metadata = P0_ARCHIVE / "run_metadata.json"
    metrics = P0_ARCHIVE / "e2/metrics.csv"
    meta = read_json(metadata)
    checkpoint_hash = sha256_file(P0_CHECKPOINT)
    expected = dict(split="val", offset=0, scene_ids=scene_ids, split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY)
    if (any(meta.get(key) != value for key, value in expected.items())
            or Path(meta["checkpoints"]["e2"]) != P0_CHECKPOINT
            or meta["checkpoint_hashes"]["e2"] != checkpoint_hash):
        raise ValueError("Archived E2 differs from the fixed P0 protocol")
    original_metadata = Path(meta["baseline_metadata"])
    original = read_json(original_metadata)
    if (sha256_file(original_metadata) != meta["baseline_metadata_sha256"]
            or original["selected_scene_ids"] != scene_ids or original["split_manifest_sha256"] != manifest_hash
            or original["checkpoint_sha256"] != checkpoint_hash or original["output_range_policy"] != "clamp"):
        raise ValueError("P0 image archive provenance differs")
    raw, _ = read_metrics(metrics)
    if list(raw) != scene_ids or tuple(next(iter(raw.values()))) != ("image", *METRICS):
        raise ValueError("P0 metrics do not match the locked validation scene order and columns")
    details = dict(checkpoint=P0_CHECKPOINT.as_posix(), checkpoint_sha256=checkpoint_hash,
                   archived_metrics=metrics.as_posix(), archived_metrics_sha256=sha256_file(metrics),
                   archived_metadata=metadata.as_posix(), archived_metadata_sha256=sha256_file(metadata),
                   archived_images=Path(meta["baseline_gray"]).parent.as_posix())
    return [(tuple(float(raw[scene][name]) for name in METRICS), (), ()) for scene in scene_ids], details


def evaluate_scene(pair, device, output=None):
    optical_bgr = read_image(pair[0], cv2.IMREAD_COLOR)
    optical_ycc = cv2.cvtColor(optical_bgr, cv2.COLOR_BGR2YCrCb)
    optical = optical_ycc[:, :, 0].copy()
    sar = read_image(pair[1], cv2.IMREAD_GRAYSCALE)
    sources = [torch.from_numpy(image).to(device=device, dtype=torch.float32)[None, None] / 255
               for image in (optical, sar)]
    images, diagnostics = fuse(*sources)
    if diagnostics["relative_residual"] > 1e-4 or not all(torch.isfinite(image).all() for image in images.values()):
        raise FloatingPointError(f"Invalid screened-Poisson solution: {pair[0].stem}/{diagnostics}")
    scores = {}
    for method, image in images.items():
        gray = image[0, 0].clamp(0, 1).mul(255).round().byte().cpu().numpy()
        scores[method] = (FusionEvaluator.metric_vector(gray, optical, sar), (), ())
        if output is not None:
            ycc = optical_ycc.copy()
            ycc[:, :, 0] = gray
            bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
            preview = np.concatenate((optical_bgr, cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR), bgr), axis=1)
            for kind, asset in (("gray", gray), ("rgb", bgr), ("preview", preview)):
                path = output / method / kind / f"{pair[0].stem}.png"
                if not cv2.imwrite(str(path), asset):
                    raise OSError(f"Cannot save {path}")
    return scores, diagnostics


def run(device, action):
    splits, _ = manifest_pairs(DATA, MANIFEST)
    pairs = splits["val"]
    if len(pairs) != 1000:
        raise ValueError("Expected the locked QXS-SAROPT val1000")
    ids = [pair[0].stem for pair in pairs]
    manifest_hash = sha256_file(MANIFEST)
    p0, p0_details = load_p0(ids, manifest_hash)
    if action == "check":
        print(f"P0 reused: {len(p0)} archived scenes. Split SHA-256: {manifest_hash}")
        return
    if action == "smoke":
        scores, diagnostics = evaluate_scene(pairs[0], device)
        assert all(len(row[0]) == 14 for row in scores.values())
        print(dict(scene=ids[0], **diagnostics))
        return

    protocol = dict(dataset="QXS-SAROPT", split="val", offset=0, scene_ids=ids, device=str(device),
                    spec=SPEC, split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash,
                    metric_profile=DEFAULT_PROFILE, output_policy=POLICY, p0=p0_details,
                    source_sha256={name: sha256_file(Path(name)) for name in (
                        "run_spie_analytic.py", "spie_analytic.py", "e2_aspr.py",
                        "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py",
                    )})
    if OUTPUT.exists():
        if read_json(OUTPUT / "run_metadata.json") != protocol:
            raise FileExistsError(f"Existing results have different provenance: {OUTPUT}")
    else:
        work = OUTPUT.parent / f".{OUTPUT.name}-{uuid4().hex}"
        print(f"Writing evaluation to {work}", flush=True)
        (work / "P0").mkdir(parents=True)
        results = {"P0": p0}
        for method in ("P1", "P2", "P3"):
            results[method] = []
            for kind in ("gray", "rgb", "preview"):
                (work / method / kind).mkdir(parents=True)
        diagnostics = []
        for index, pair in enumerate(pairs, start=1):
            scores, stats = evaluate_scene(pair, device, work)
            for method, row in scores.items():
                results[method].append(row)
            diagnostics.append((pair[0].stem, *stats.values()))
            if index == 1 or index % 25 == 0:
                print(f"SPIE {index}/{len(ids)}: {pair[0].stem}", flush=True)
        summaries = summarize(work, ids, results, CONTRASTS, diagnostic_fields=(), source_metrics=())
        write_csv(work / "p3_solver_diagnostics.csv", ("image", *stats), diagnostics)
        write_json(work / "registry_metrics.json", summaries)
        write_json(work / "run_metadata.json", protocol)
        work.rename(OUTPUT)

    summaries = read_json(OUTPUT / "registry_metrics.json")
    summary = OUTPUT / "registry_metrics.json"
    group = "QXS-SPIE-ANALYTIC-V1-VAL1000-REFERENCE-FMI"
    entries = []
    for method, label in METHODS.items():
        folder = OUTPUT / method
        if method != "P0":
            for kind in ("gray", "rgb", "preview"):
                verify_images(folder / kind, ids)
        means = summaries[method]
        entries.append(dict(id=f"{group}-{method}", comparable_group=group, method=label,
                            dataset="QXS-SAROPT", split="val", offset=0, sample_count=len(ids),
                            task="fusion", status="development_validation", result_dir=folder.as_posix(),
                            metric_profile=DEFAULT_PROFILE, output_policy=POLICY,
                            metadata_path=(OUTPUT / "run_metadata.json").as_posix(),
                            split_manifest=MANIFEST.as_posix(), split_manifest_sha256=manifest_hash,
                            metrics=means, metric_source=dict(type="json_paths", path=summary.as_posix(),
                                paths={metric: [method, metric] for metric in means}),
                            notes="P0 复用归档结果，P1–P3 为无参数解析方法。比较使用四组共同有限场景。"))
        if method == "P0":
            entries[-1].update({key: p0_details[key] for key in ("checkpoint", "checkpoint_sha256")})
    print(f"Paired comparisons: {OUTPUT / 'paired_comparison.csv'}", flush=True)
    register_many(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "val"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not torch.cuda.is_available():
        raise RuntimeError("SPIE requires CUDA in this validation protocol")
    run(torch.device(args.device), args.action)


if __name__ == "__main__":
    main()
