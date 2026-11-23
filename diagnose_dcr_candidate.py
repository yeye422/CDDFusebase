"""Audit D1 proposal coverage and run fixed-position decision controls."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from dcr_synth.config import ROOT, load_config, read_json, write_json
from dcr_synth.decision import SOURCES, choose, inspect, proposal_trace, render_target
from dcr_synth.evaluation import metrics, pair_metrics
from diagnose_dcr_d1 import archive_oracle, evaluate_probes, implementation_files, line_coverage, review_protocol
from diagnose_dcr_v12 import csv_rows, register, save_image
from diagnose_e2_boundary_supervision import BoundaryProbe
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import PAPER_METRICS, sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


STAGES = ("raw", "geometry_valid", "score_threshold", "post_nms")
C1 = {"C1-O": "D1-O", "C1-S": "D1-S", "C1-Max": "D1-Max",
      "C1-C": "D1-C", "C1-Shuffle": "D1-C-shuffle"}
FIELDS = ("x1", "y1", "x2", "y2")


def masks_and_rows(traced, image, config):
    rows, masks = [], {}
    for records in traced.values():
        for item in records:
            line = dict(zip(FIELDS, item["segment"]))
            probe = item["line"].probe if "line" in item else BoundaryProbe(image, line)
            mask = (probe.taper > 0) & (probe.across.abs() < 7)
            value = mask.cpu().numpy()
            masks[item["id"]] = value
            ys, xs = np.nonzero(value)
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None
            rank = item["nms_rank"]
            rows.append(dict({key: value for key, value in item.items() if key != "line"},
                             line=line, mask_key=item["id"], window_bbox_xyxy=bounds,
                             window_pixels=int(mask.sum()), in_original_top_k=(
                                 rank is not None and rank <= config["max_candidates_per_source"])))
    return rows, masks


def stages(records):
    return {
        "raw": records,
        "geometry_valid": [row for row in records if row["status"] != "border"],
        "score_threshold": [row for row in records if row["score_rank"] is not None],
        "post_nms": sorted((row for row in records if row["nms_rank"] is not None),
                           key=lambda row: row["nms_rank"]),
    }


def top_k(pools, source, k):
    if source == "union":
        # With K=8 and full pools this is the original four-per-source allocation.
        optical, sar = (pools[name]["post_nms"] for name in SOURCES)
        return optical[:(k + 1) // 2] + sar[:k // 2]
    return pools[source]["post_nms"][:k]


def coverage(reader, records, masks, cutoff):
    centers = reader.center.cpu().numpy().round().astype(int)
    tangent = reader.normal.cpu().numpy()[[1, 0]] * [1, -1]
    union = np.zeros(len(centers), bool)
    ranked = []
    for entry in records:
        a, b, c, d = entry["segment"]
        direction = np.array([c - a, d - b], dtype=np.float64)
        if abs(tangent @ direction) < cutoff * np.linalg.norm(direction):
            continue
        covered = masks[entry["id"]][centers[:, 1], centers[:, 0]]
        union |= covered
        if covered.any():
            ranked.append(dict(id=entry["id"], window_fraction=float(covered.mean()), score=entry["score"],
                               raw_score_rank=entry.get("raw_score_rank"), score_rank=entry["score_rank"],
                               nms_rank=entry["nms_rank"], suppressed_by=entry["suppressed_by"],
                               status=entry["status"]))
    return float(union.mean()), ranked


def development_probes(protocol, config):
    grouped = {}
    for case in protocol["cases"]:
        kind = "protection" if case["role"] == "preserve" else "positive"
        grouped.setdefault(case["scene"], []).append(dict(id=case["id"], kind=kind, source="optical",
                                                         line=case["target"]))
    path = ROOT / config["development_negative_probes"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scene"] in grouped and row["label"] == "disconnected":
                grouped[row["scene"]].append(dict(id=f"{row['scene']}-{row['id']}", kind="negative",
                                                  source="optical", line={key: int(row[key]) for key in FIELDS}))
    return grouped


def c0_scene(optical, sar, probes, config, d1, folder):
    traced = proposal_trace((optical, sar), d1)
    rows, masks = masks_and_rows(traced, optical, d1)
    pools = {source: stages([row for row in rows if row["id"].startswith(source + "-")])
             for source in SOURCES}
    np.savez_compressed(folder / "window_masks.npz", **masks)
    cutoff = math.cos(math.radians(config["orientation_deg"]))
    floor = config["window_coverage_floor"]
    reports, flat = [], []
    counts = []
    available = {}
    for source in (*SOURCES, "union"):
        current = pools[source] if source != "union" else {
            stage: pools["optical"][stage] + pools["sar"][stage] for stage in STAGES}
        for k in config["top_k"]:
            current[f"top_{k}"] = top_k(pools, source, k)
        available[source] = current
        counts.extend(dict(source=source, stage=stage, candidates=len(items)) for stage, items in current.items())
    for label in probes:
        reader = BoundaryProbe(optical if label["source"] == "optical" else sar, label["line"])
        region = {}
        for source, pool in available.items():
            region[source] = {}
            for stage, records in pool.items():
                fraction, nearby = coverage(reader, records, masks, cutoff)
                details = dict(window_fraction=fraction, detected=fraction >= floor,
                               centerline_fraction=line_coverage(reader, [row["line"] for row in records]),
                               candidates=len(records))
                region[source][stage] = details
                flat.append(dict(probe=label["id"], kind=label["kind"], source=source, stage=stage, **details))
                if stage == "raw":
                    details["nearby_candidates"] = nearby
        union = region["union"]
        missed = next((key for key in (*STAGES, "top_8") if not union[key]["detected"]), None)
        cause = {"raw": "detection", "geometry_valid": "border", "score_threshold": "threshold",
                 "post_nms": "nms", "top_8": "top_k", None: "covered"}
        reports.append(dict(label, first_loss=cause[missed], stages=region))
    write_json(folder / "proposal_trace.json", dict(candidates=rows, counts=counts, probes=reports,
                                                   coordinate_frame="native_256x256_xy_zero_based"))
    csv_rows(folder / "coverage.csv", flat, ["probe", "kind", "source", "stage", "candidates", "detected",
                                            "window_fraction", "centerline_fraction"])
    return dict(counts=counts, probes=reports)


def c0_summary(results, config, blind):
    totals = []
    records = [probe for result in results.values() for probe in result["probes"]]
    for source in (*SOURCES, "union"):
        for stage in (*STAGES, *(f"top_{k}" for k in config["top_k"])):
            for kind in sorted({probe["kind"] for probe in records}):
                probes = [probe for probe in records if probe["kind"] == kind]
                hits = sum(probe["stages"][source][stage]["detected"] for probe in probes)
                totals.append(dict(source=source, stage=stage, kind=kind, covered=hits, total=len(probes),
                                   recall=hits / len(probes)))
    eligible = []
    for k in config["top_k"]:
        rates = {row["kind"]: row["recall"] for row in totals
                 if row["source"] == "union" and row["stage"] == f"top_{k}"}
        if rates.get("positive", 0) >= config["blind_positive_recall_floor"] and (
                rates.get("negative", 1) <= config["blind_negative_recall_ceiling"]):
            eligible.append(k)
    decision = ("exploratory_only" if not blind else
                "coverage_supported" if eligible else "coverage_not_demonstrated")
    return dict(decision=decision, recall=totals, eligible_k=eligible if blind else [])


def c1_scene(model, optical, sar, label, negatives, d1, folder):
    baseline, detail, decode, context = inspect(model, optical, sar, d1, fixed=label["line"])
    focus = context[0]
    eligible = [index for index, other in enumerate(context[1:], 1)
                if other.relation != focus.relation and not (other.support & focus.support).any()]
    rng = np.random.default_rng(d1["shuffle_seed"] + int(label["id"].split("-")[0]))
    donor = int(rng.choice(eligible)) if eligible else None
    selections = {name: choose(context, method, d1, {0: donor}, only_index=0)
                  for name, method in C1.items() if name != "C1-Shuffle" or donor is not None}
    write_json(folder / "fixed_candidate.json", dict(line=focus.geometry, evidence=focus.evidence,
               relation=focus.relation, shuffle_status="different_nonoverlapping_donor" if donor else "unavailable",
               donor=dict(line=context[donor].geometry, relation=context[donor].relation) if donor else None))
    oracle = archive_oracle(d1, label["id"].split("-")[0], baseline)
    reports = {}
    for dose in d1["doses_gl"]:
        images = {"D0": baseline}
        rows = {"D0": dict(status="baseline", rms_gl=0., peak_gl=0.)}
        cached = {}
        for name, selection in selections.items():
            key = selection[1] if selection else None
            if key not in cached:
                cached[key] = render_target(detail, decode, baseline, context, selection, dose, d1)
            image, measurement = cached[key]
            images[name], rows[name] = image, dict(measurement, decision_score=selection[2] if selection else None)
        if oracle and dose == 0.5:
            images["Oracle"], rows["Oracle"] = oracle
        for name, image in images.items():
            save_image(folder / f"{name}_{dose}.png", image)
            rows[name]["metrics"] = metrics(image, optical, sar)
        if donor is None:
            rows["C1-Shuffle"] = dict(status="no_distinct_donor", rms_gl=None, peak_gl=None)
        probe = dict(category="fixed_candidate_history", source="optical", line=focus.geometry)
        evaluations = evaluate_probes(images, {"probes": [probe, *negatives]}, optical, sar, [focus], selections)
        readout = evaluations[0]["methods"]
        side_effects = []
        for entry in evaluations[1:]:
            baseline_hits = entry["methods"]["D0"]["native_hits"]
            additions = {name: sum(value and not old for value, old in zip(item["native_hits"], baseline_hits))
                         for name, item in entry["methods"].items()}
            side_effects.append(dict(line=entry["line"], new_negative_hits=additions))
        comparisons = []
        conditional = rows["C1-C"]
        for name in ("C1-O", "C1-S", "C1-Max", "C1-Shuffle"):
            other = rows[name]
            matched = (conditional["status"] == other["status"] == "edited"
                       and abs(conditional["rms_gl"] - other["rms_gl"]) <= d1["dose_tolerance_gl"])
            changes = {key: readout["C1-C"][key] - readout[name][key] for key in ("points", "longest")} if name in readout else None
            comparisons.append(dict(reference=name, matched=matched, changes=changes))
        reports[str(dose)] = dict(methods=rows, fixed_probe=readout, comparisons=comparisons,
                                  negative_side_effects=side_effects)
    return dict(uses_manual_position=True, shuffle_donor=donor, doses=reports)


def run(args):
    os.chdir(ROOT)
    config = read_json(args.config)
    d1_path = ROOT / config["d1"]
    d1, protocol = load_config(d1_path)
    if config["top_k"] != [8, 16, 32, 64, 128] or not 0 < config["window_coverage_floor"] <= 1:
        raise ValueError("Use the registered C0 sweep and a valid window coverage threshold")
    if args.smoke and not args.case:
        raise ValueError("A smoke test requires one case")
    if args.stage == "c1" and args.split != "development":
        raise ValueError("C1 is restricted to archived manually supplied development windows")
    manifest_path = ROOT / protocol["split_manifest"]
    manifest = read_json(manifest_path)
    if (sha256_file(manifest_path) != protocol["split_manifest_sha256"]
            or sha256_file(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]):
        raise ValueError("E2 checkpoint or evaluation split differs from the registered assets")
    blind = args.split == "blind"
    gate = review_protocol(d1, protocol, manifest)
    prerequisite = None
    if blind:
        if args.case or args.smoke or not args.output or gate["status"] != "ready" or not args.lock:
            raise SystemExit("独立 C0 需先登记完整地理证据、冻结审阅协议，并使用独立输出目录。")
        locked = read_json(args.lock)
        for key, value in gate.items():
            if locked[key] != value:
                raise ValueError("Blind review changed after freezing")
        if locked["config_sha256"] != sha256_file(d1_path):
            raise ValueError("D1 config changed after blind review freeze")
        probes = {scene: [dict(id=f"{scene}-{i}", source=p["source"], line=p["line"], category=p["category"],
                              kind="negative" if p["category"] in {"parallel", "misaligned"} else "positive")
                          for i, p in enumerate(item["probes"])]
                  for scene, item in gate["blind"].items()}
    elif args.stage == "c0":
        probes = development_probes(protocol, config)
    else:
        source = (args.c0_report or (ROOT / config["c0_output"] / "c0_report.json"))
        if not source.is_file():
            raise SystemExit("请先完成 C0，或用 --c0-report 指定已有的 C0 报告。")
        audit = read_json(source)
        prerequisite = sha256_file(source)
        probes = {case["id"]: [dict(id=case["id"], source="optical", line=case["target"])]
                  for case in protocol["cases"] if case["id"] in config["c1_cases"]}
        archived_probes = development_probes(protocol, config)
        for case in probes:
            probes[case].extend(dict(item, category="archived_disconnected") for item in archived_probes[case.split("-")[0]]
                                if item["kind"] == "negative")
        required = {case.split("-")[0] for case in ([args.case] if args.case else probes)}
        if audit.get("split") != "development" or not required <= audit["results"].keys():
            raise ValueError("Complete the development C0 audit for the selected C1 scenes first")
    keys = [name for name in probes if not args.case or name == args.case]
    if not keys:
        raise ValueError("Unknown case in this stage")
    output = (args.output or Path(config[f"{args.stage}_output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Specify a new output directory inside the repository")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请使用 lightglue_env。")
    device = torch.device(args.device)
    if args.stage == "c1":
        model, checkpoint = frozen_checkpoint(device)
        if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
            raise ValueError("Unexpected frozen checkpoint")
        if args.smoke:
            d1.update(linear_steps=2, nonlinear_steps=1)
    paths = {key: locate_scene(Path(protocol["data"]), manifest, key.split("-")[0]) for key in keys}
    input_hashes = {key: {file.as_posix(): sha256_file(file) for file in pair} for key, pair in paths.items()}
    if blind and locked["blind_sources"] != input_hashes:
        raise ValueError("Blind source files changed after freezing")
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), *implementation_files(), args.config.resolve(), d1_path]
    digests = {}
    for file in files:
        name = file.relative_to(ROOT)
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        digests[name.as_posix()] = sha256_file(file)
    metadata = dict(stage=args.stage, device=str(device), torch_version=str(torch.__version__),
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
                    metric_profile=DEFAULT_PROFILE, smoke=args.smoke, case_ids=keys, c0_sha256=prerequisite,
                    inputs=input_hashes, code_sha256=digests)
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "protocol.json", dict(stage=args.stage, split=args.split, audit=config,
                                             d1=d1, probes=probes, review=gate))
    results = {}
    for key in keys:
        folder = output / key
        folder.mkdir()
        print(f"{args.stage.upper()} {key}", flush=True)
        optical, sar = (to_tensor(image, device) for image in read_pair(paths[key]))
        if args.stage == "c0":
            results[key] = c0_scene(optical, sar, probes[key], config, d1, folder)
        else:
            label, *negatives = probes[key]
            results[key] = c1_scene(model, optical, sar, label, negatives, d1, folder)
    if args.stage == "c0":
        summary = c0_summary(results, config, blind)
        csv_rows(output / "recall.csv", summary["recall"], ["source", "stage", "kind", "covered", "total", "recall"])
    else:
        quality = {}
        flat = []
        for dose in d1["doses_gl"]:
            key = str(dose)
            arms = ("D0", *C1.keys())
            scores = []
            for case, item in results.items():
                group = item["doses"][key]
                scores.append({name: row["metrics"] for name, row in group["methods"].items() if "metrics" in row})
                for name, row in group["methods"].items():
                    flat.append(dict(case=case, dose_gl=dose, method=name, **row, **row.get("metrics", {}),
                                     points=group["fixed_probe"].get(name, {}).get("points"),
                                     longest=group["fixed_probe"].get(name, {}).get("longest")))
            quality[key] = pair_metrics(scores, arms)
        csv_rows(output / "fixed_comparison.csv", flat, ["case", "dose_gl", "method", "status", "source",
                 "rms_gl", "peak_gl", "target_error_gl", "points", "longest", *PAPER_METRICS])
        summary = dict(decision="fixed_position_diagnostic_only", common_finite_metrics=quality)
    counts = dict(cases=len(keys), scenes=len({key.split("-")[0] for key in keys}))
    filename = f"{args.stage}_report.json"
    write_json(output / filename, dict(results=results, registry=counts, split=args.split, **summary))
    print(f"{output / filename} [{summary['decision']}]", flush=True)
    try:
        register(output, metadata, counts["scenes"], args.stage, filename, counts, version="D1")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。旧实验索引校验失败，请查看上方错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("c0", "c1"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_candidate_audit.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--split", choices=("development", "blind"), default="development")
    parser.add_argument("--case")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--c0-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
