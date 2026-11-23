"""Frozen E2 source-conditional decision experiment and blind-review intake."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from dcr_synth.atoms import rms
from dcr_synth.config import ROOT, load_config, read_json, write_json
from dcr_synth.decision import METHODS, inspect, choose, render_target
from dcr_synth.evaluation import metrics, pair_metrics
from diagnose_dcr_v12 import csv_rows, register, save_image
from diagnose_e2_boundary_supervision import BoundaryProbe
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import PAPER_METRICS, sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


CATEGORIES = {"common", "opposite_polarity", "optical_only", "sar_only", "misaligned", "parallel"}
NEGATIVES = {"misaligned", "parallel"}


def review_protocol(config, protocol, manifest):
    review_path = ROOT / config["review"]
    review = read_json(review_path)
    reasons = []
    blocks, hashes = {}, {}
    for entry in review["blocks"]:
        key = entry["id"]
        if key in blocks or entry["role"] not in ("development", "blind"):
            raise ValueError(f"Invalid geographic block: {key}")
        evidence = ROOT / entry["provenance_file"]
        if not evidence.is_file() or not entry["parent_scene_id"]:
            raise ValueError(f"Missing geographic evidence for {key}")
        hashes[evidence.as_posix()] = sha256_file(evidence)
        blocks[key] = entry
    parent_ids = [entry["parent_scene_id"] for entry in blocks.values()]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("Parent scenes must occur in only one geographic block")
    inspected = set(review["historically_inspected_scenes"]) | set(config["development"])
    assigned = {}
    categories = set()
    for sample in review["samples"]:
        scene = sample["scene"]
        block = blocks[sample["geo_block"]]
        if scene not in manifest["splits"]["val"] or scene in assigned:
            raise ValueError(f"Scene must be unique in the frozen pilot validation split: {scene}")
        if block["role"] == "blind" and scene in inspected:
            raise ValueError(f"Previously inspected scene cannot enter the blind set: {scene}")
        if block["role"] == "blind" and not sample["source_only_review"]:
            raise ValueError(f"Blind review inspected a fused output: {scene}")
        if not sample["verified_by"] or (block["role"] == "blind" and not sample["probes"]):
            raise ValueError(f"Missing source review or probes: {scene}")
        for probe in sample["probes"]:
            if probe["category"] not in CATEGORIES or probe["source"] not in ("optical", "sar"):
                raise ValueError(f"Invalid probe category or source: {scene}")
            if probe["category"] == "sar_only" and (
                    probe["source"] != "sar" or not probe.get("verification_evidence")):
                raise ValueError(f"Verified SAR-exclusive probe needs supporting evidence: {scene}")
            line = probe["line"]
            points = [line[key] for key in ("x1", "y1", "x2", "y2")]
            if not all(6 <= value <= 249 for value in points) or points[:2] == points[2:]:
                raise ValueError(f"Invalid review line in {scene}")
            if block["role"] == "blind":
                categories.add(probe["category"])
        assigned[scene] = sample
    blind = {key: item for key, item in assigned.items() if blocks[item["geo_block"]]["role"] == "blind"}
    blind_blocks = {item["geo_block"] for item in blind.values()}
    if len(blind_blocks) < config["minimum_blind_blocks"]:
        reasons.append("insufficient_blind_geographic_blocks")
    documented_dev = {scene for scene, sample in assigned.items()
                      if blocks[sample["geo_block"]]["role"] == "development"}
    if not inspected <= documented_dev:
        reasons.append("development_parent_scene_mapping_incomplete")
    missing_categories = CATEGORIES - categories
    if missing_categories:
        reasons.append("missing_blind_categories")
    return dict(status="ready" if not reasons else "exploratory_only",
                reasons=reasons, review_sha256=sha256_file(review_path), provenance_sha256=hashes,
                missing_categories=sorted(missing_categories), blind_blocks=sorted(blind_blocks),
                blind=blind, samples=assigned, blocks=blocks, inspected_scenes=sorted(inspected))


@torch.no_grad()
def line_coverage(reader, geometries):
    center = reader.center
    hits = torch.zeros(len(center), dtype=torch.bool, device=center.device)
    tangent = torch.stack((reader.normal[1], -reader.normal[0]))
    for g in geometries:
        start = center.new_tensor([g["x1"], g["y1"]])
        end = center.new_tensor([g["x2"], g["y2"]])
        direction = end - start
        length = direction.norm()
        direction = direction / length
        if abs(float(tangent @ direction)) >= 0.966:
            projection = ((center - start) @ direction).clamp_min(0).minimum(length)
            nearest = start + projection[:, None] * direction
            hits |= (center - nearest).norm(dim=1) <= 2
    return float(hits.float().mean())


@torch.no_grad()
def evaluate_probes(images, sample, optical, sar, proposals, selections):
    """Manual records are consumed here, after all automatic images are rendered."""
    if not sample:
        return []
    rows = []
    for probe in sample["probes"]:
        source = sar if probe["source"] == "sar" else optical
        reader = BoundaryProbe(source, probe["line"])
        offsets = torch.arange(-6, 7, device=source.device, dtype=source.dtype)
        points = reader.center[None] + offsets[:, None, None] * reader.normal
        plus = (2 * (points + 0.5 * reader.normal) / reader.scale - 1)[None]
        minus = (2 * (points - 0.5 * reader.normal) / reader.scale - 1)[None]
        profile = {}
        for name, image in images.items():
            center = 255 * reader.contrast(image)
            curve = 255 * (F.grid_sample(image, plus, align_corners=True)
                           - F.grid_sample(image, minus, align_corners=True))[0, 0]
            index = curve.abs().argmax(dim=0)
            signed_peak = curve.gather(0, index[None])[0] * reader.sign
            if probe["category"] in NEGATIVES or probe.get("kind") == "negative":
                hit = (center.abs() >= 3) & ((index - 6).abs() <= 2)
            else:
                hit = (center * reader.sign >= 3) & ((index - 6).abs() <= 2) & (signed_peak > 0)
            values = hit.tolist()
            longest = current = 0
            for value in values:
                current = current + 1 if value else 0
                longest = max(longest, current)
            profile[name] = dict(points=sum(values), longest=longest, native_hits=values,
                                 signed_gl=(center * reader.sign).tolist(), offsets_px=(index - 6).tolist())
        coverage = {name: line_coverage(reader, [pick[0].geometry]) if pick else 0.
                    for name, pick in selections.items()}
        rows.append(dict(category=probe["category"], source=probe["source"], line=probe["line"],
                         proposal_coverage=line_coverage(reader, [item.geometry for item in proposals]),
                         selected_coverage=coverage, methods=profile))
    return rows


def archive_oracle(config, scene, baseline):
    root = ROOT / config["f05"]
    run = read_json(root / "f05_attribution.json")
    case = {"546": "546-B", "4880": "4880-B"}.get(scene)
    if case not in run["results"]:
        return None
    row = run["results"][case]["arms"]["R1"]["fits"]["strict"]
    if not row["witness_reproduced"]:
        return None
    path = root / case / "R1_strict.npz"
    target_path = root / case / "fixed_witness_float.npz"
    with np.load(target_path) as target:
        archived_base = torch.from_numpy(target["baseline"]).to(baseline)
    if not torch.allclose(baseline, archived_base, atol=1e-5, rtol=0):
        raise ValueError("Archived oracle uses a different E2 baseline")
    with np.load(path) as data:
        image = torch.from_numpy(data["image"]).to(baseline)
    change = 255 * (image - baseline)
    return image, dict(status="archived_oracle", path=path.as_posix(), sha256=sha256_file(path),
                       rms_gl=float(rms(change)), peak_gl=float(change.abs().max()))


def run_scene(model, paths, scene, folder, config, device, sample, allow_oracle):
    optical, sar = (to_tensor(image, device) for image in read_pair(paths))
    baseline, detail, decode, candidates = inspect(model, optical, sar, config)
    save_image(folder / "D0.png", baseline)
    order = np.random.default_rng(config["shuffle_seed"] + int(scene)).permutation(len(candidates))
    shuffle = dict(zip(order.tolist(), np.roll(order, 1).tolist()))
    shifts = sum(candidates[index].relation != candidates[other].relation for index, other in shuffle.items())
    evidence = [dict(line=proposal.geometry, origin=proposal.origin, evidence=proposal.evidence,
                     relation=proposal.relation, shuffled_relation_from=shuffle[index])
                for index, proposal in enumerate(candidates)]
    write_json(folder / "decision_evidence.json", dict(candidates=evidence, relation_rows_changed=shifts))
    selections = {name: choose(candidates, name, config, shuffle) for name in METHODS}
    base_metrics = metrics(baseline, optical, sar)
    oracle = archive_oracle(config, scene, baseline) if allow_oracle else None
    doses = {}
    for dose in config["doses_gl"]:
        images = {"D0": baseline}
        rows = {"D0": dict(status="baseline", rms_gl=0., peak_gl=0., metrics=base_metrics)}
        cache = {}
        for method, selection in selections.items():
            print(f"  {dose:g} {method}", flush=True)
            key = None if selection is None else (id(selection[0]), selection[1])
            if key not in cache:
                cache[key] = render_target(detail, decode, baseline, candidates, selection, dose, config)
            image, row = cache[key]
            if selection is not None:
                row = dict(row, decision_score=selection[2])
            save_image(folder / f"{method}_{dose}.png", image)
            images[method] = image
            rows[method] = dict(row, metrics=metrics(image, optical, sar))
        if oracle and dose == 0.5:
            image, row = oracle
            save_image(folder / f"Oracle_{dose}.png", image)
            images["Oracle"] = image
            rows["Oracle"] = dict(row, metrics=metrics(image, optical, sar))
        doses[str(dose)] = dict(methods=rows, independent_readouts=evaluate_probes(
            images, sample, optical, sar, candidates, selections))
    return dict(proposals=len(candidates), relation_rows_changed=shifts, doses=doses)


def compare(results, gate, config, blind):
    comparisons = {}
    methods = ("D0", *METHODS)
    wins = set()
    tested = set()
    for dose in config["doses_gl"]:
        key = str(dose)
        scores, paired = [], []
        for scene, result in results.items():
            rows = result["doses"][key]["methods"]
            scores.append({name: rows[name]["metrics"] for name in methods})
            block = gate["samples"].get(scene, {}).get("geo_block")
            c = rows["D1-C"]
            for name in ("D1-O", "D1-S", "D1-Max", "D1-Gap", "D1-C-shuffle"):
                r = rows[name]
                matched = (c["status"] == r["status"] == "edited" and
                           abs(c["rms_gl"] - r["rms_gl"]) <= config["dose_tolerance_gl"])
                comparisons_row = dict(scene=scene, geo_block=block, reference=name, matched=matched,
                                       conditional_rms_gl=c["rms_gl"], reference_rms_gl=r["rms_gl"])
                differences = []
                for probe in result["doses"][key]["independent_readouts"]:
                    left, right = [probe["methods"][method] for method in ("D1-C", name)]
                    differences.append(dict(category=probe["category"], points=left["points"] - right["points"],
                                            longest=left["longest"] - right["longest"]))
                comparisons_row["differences"] = differences
                paired.append(comparisons_row)
        if blind and dose == 0.245:
            for scene in results:
                scene_rows = [row for row in paired if row["scene"] == scene and row["matched"]]
                required = {"D1-O", "D1-S", "D1-Max", "D1-C-shuffle"}
                if not required <= {row["reference"] for row in scene_rows}:
                    continue
                block = gate["blind"][scene]["geo_block"]
                tested.add(block)
                preserved = all(diff["points"] <= 0 if diff["category"] in NEGATIVES else diff["points"] >= 0
                                for row in scene_rows for diff in row["differences"]
                                if diff["category"] in NEGATIVES | {"sar_only"})
                arms = results[scene]["doses"][key]["methods"]
                baseline_nabf, c_nabf = (arms[name]["metrics"]["Nabf"] for name in ("D0", "D1-C"))
                actual = results[scene]["doses"][key]["independent_readouts"]
                references = ["D0", *sorted(required)]
                positive = any(
                    all(probe["methods"]["D1-C"]["points"] > probe["methods"][name]["points"]
                        and probe["methods"]["D1-C"]["longest"] > probe["methods"][name]["longest"]
                        for name in references)
                    for probe in actual if probe["category"] not in NEGATIVES
                )
                no_regression = all(
                    probe["methods"]["D1-C"]["points"] <= probe["methods"]["D0"]["points"]
                    if probe["category"] in NEGATIVES else
                    probe["methods"]["D1-C"]["points"] >= probe["methods"]["D0"]["points"]
                    for probe in actual if probe["category"] in NEGATIVES | {"sar_only"})
                if (positive and preserved and no_regression and results[scene]["relation_rows_changed"] > 0
                        and None not in (c_nabf, baseline_nabf) and c_nabf <= baseline_nabf + 5e-6):
                    wins.add(block)
        comparisons[key] = dict(pairs=paired, common_finite_metrics=pair_metrics(scores, methods),
                                edit_counts={method: sum(result["doses"][key]["methods"][method]["status"] == "edited"
                                                         for result in results.values()) for method in METHODS})
    needed = config["minimum_blind_blocks"]
    if not blind:
        status = "exploratory_only"
    elif len(wins) >= needed:
        status = "blind_decision_benefit_supported"
    elif len(tested) >= needed:
        status = "blind_decision_benefit_not_demonstrated"
    else:
        status = "insufficient_dose_matched_blind_blocks"
    return dict(decision=status, matched_blind_blocks=sorted(tested), winning_blind_blocks=sorted(wins),
                comparisons=comparisons)


def implementation_files():
    return [Path(__file__).resolve(), ROOT / "diagnose_dcr_v2_f05.py", ROOT / "diagnose_dcr_v12.py",
            ROOT / "net.py", ROOT / "direct_fusion.py", ROOT / "lwpd.py", ROOT / "e2_base_anchor.py",
            ROOT / "diagnose_e2_subspace_gradients.py", ROOT / "diagnose_e2_boundary_supervision.py",
            ROOT / "diagnose_e2_information_path.py", ROOT / "utils/fusion_metrics.py",
            ROOT / "utils/reference_metrics.py", ROOT / "utils/Evaluator.py",
            *sorted((ROOT / "dcr_synth").glob("*.py"))]


def source_hashes(protocol, manifest, scenes):
    paths = {scene: locate_scene(Path(protocol["data"]), manifest, scene) for scene in scenes}
    digests = {scene: {path.as_posix(): sha256_file(path) for path in paths[scene]} for scene in scenes}
    return paths, digests


def source_review_packet(directory, candidates, protocol, manifest, requested):
    output = directory.resolve()
    scenes = [requested] if requested else candidates
    if ROOT not in output.parents or output.exists() or not set(scenes) <= set(candidates):
        raise ValueError("Use a new packet directory and only previously unreviewed pilot validation scenes")
    paths, digests = source_hashes(protocol, manifest, scenes)
    output.mkdir(parents=True, exist_ok=False)
    for scene, pair in paths.items():
        for source, path in zip(("optical", "sar"), pair):
            shutil.copyfile(path, output / f"{scene}_{source}{path.suffix}")
    write_json(output / "intake.json", dict(scenes=scenes, inputs=digests, source_only=True))


def run(args):
    os.chdir(ROOT)
    config, protocol = load_config(args.config)
    if config["doses_gl"] != [0.245, 0.5] or min(
            config[key] for key in ("max_candidates_per_source", "linear_steps", "nonlinear_steps",
                                    "ridge", "peak_gl", "minimum_blind_blocks")) <= 0:
        raise ValueError("D1 requires both registered dose caps and positive solver settings")
    manifest = read_json(ROOT / protocol["split_manifest"])
    gate = review_protocol(config, protocol, manifest)
    files = implementation_files()
    digests = {file.relative_to(ROOT).as_posix(): sha256_file(file) for file in files}
    frozen = dict(config_sha256=sha256_file(args.config), code_sha256=digests,
                  e2_protocol_sha256=sha256_file(ROOT / config["protocol"]),
                  checkpoint_sha256=protocol["checkpoint_sha256"],
                  split_manifest_sha256=sha256_file(ROOT / protocol["split_manifest"]))
    if args.stage == "protocol":
        candidates = [scene for scene in manifest["splits"]["val"] if scene not in gate["inspected_scenes"]]
        _, blind_sources = source_hashes(protocol, manifest, gate["blind"])
        payload = dict(gate, **frozen, source_review_candidates=candidates, blind_sources=blind_sources)
        if args.output:
            if ROOT not in args.output.resolve().parents:
                raise ValueError("Use an output path inside the repository")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, payload)
        if args.review_packet:
            source_review_packet(args.review_packet, candidates, protocol, manifest, args.case)
        print(f"D1-0 {gate['status']}: {len(gate['blind_blocks'])} independent blocks, "
              f"{len(candidates)} unreviewed source pairs from the pilot validation split", flush=True)
        return
    blind = args.stage == "blind"
    if blind:
        if args.case or args.smoke or not args.lock or gate["status"] != "ready":
            raise SystemExit("正式盲测需要已冻结的 D1-0 协议、完整地理证据和全部盲测样本。")
        locked = read_json(args.lock)
        if any(locked[key] != value for key, value in dict(gate, **frozen).items()):
            raise ValueError("D1-0 review or config changed after freezing")
    scenes = list(gate["blind"]) if blind else config["development"]
    if args.case:
        scenes = [scene for scene in scenes if scene == args.case]
    if not scenes or (args.smoke and len(scenes) != 1):
        raise ValueError("Smoke requires one registered development scene")
    if args.smoke:
        config.update(linear_steps=2, nonlinear_steps=1, max_candidates_per_source=2)
    if sha256_file(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen E2 checkpoint hash differs")
    if sha256_file(ROOT / protocol["split_manifest"]) != protocol["split_manifest_sha256"]:
        raise ValueError("Frozen validation split hash differs")
    output = (args.output or Path(config["output"])).resolve()
    if ROOT not in output.parents or output.exists() or (blind and args.output is None):
        raise ValueError("Use a new output path inside the repository, distinct for each phase")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。请使用已验证的 lightglue_env 解释器。")
    device = torch.device(args.device)
    model, checkpoint = frozen_checkpoint(device)
    if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("Unexpected checkpoint")
    paths, inputs = source_hashes(protocol, manifest, scenes)
    if blind and locked["blind_sources"] != inputs:
        raise ValueError("Blind source files changed after freezing")
    output.mkdir(parents=True, exist_ok=False)
    for file in files:
        name = file.relative_to(ROOT)
        saved = output / "source_snapshot" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(file.read_bytes())
    write_json(output / "protocol.json", dict(settings=config, review=gate, inputs=inputs))
    metadata = dict(stage=args.stage, device=str(device), torch_version=str(torch.__version__),
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
                    metric_profile=DEFAULT_PROFILE, smoke=args.smoke, scene_ids=scenes, code_sha256=digests,
                    config_sha256=sha256_file(args.config), protocol_sha256=sha256_file(output / "protocol.json"))
    write_json(output / "run_metadata.json", metadata)
    results = {}
    for scene in scenes:
        print(f"D1 {scene}", flush=True)
        folder = output / scene
        folder.mkdir()
        sample = gate["samples"].get(scene)
        if not blind and (sample is None or not sample["probes"]):
            probes = [dict(category="historic_optical_probe", source="optical", line=case["target"])
                      for case in protocol["cases"] if case["scene"] == scene]
            sample = dict(probes=probes)
        results[scene] = run_scene(model, paths[scene], scene, folder, config, device,
                                   sample, allow_oracle=not blind)
    report = compare(results, gate, config, blind)
    counts = dict(scenes=len(scenes), proposals=sum(value["proposals"] for value in results.values()),
                  matched_blind_blocks=len(report["matched_blind_blocks"]),
                  winning_blind_blocks=len(report["winning_blind_blocks"]))
    write_json(output / "d1_report.json", dict(report, results=results, registry=counts))
    rows = [dict(scene=scene, dose_gl=float(dose), method=name, **row, **row["metrics"])
            for scene, result in results.items() for dose, block in result["doses"].items()
            for name, row in block["methods"].items()]
    csv_rows(output / "d1_comparison.csv", rows, ["scene", "dose_gl", "method", "status", "rms_gl",
             "peak_gl", "source", "decision_score", "target_error_gl", "initial_target_error_gl",
             "protected_rms_gl", "far_rms_gl", "max_protection_change_gl", *PAPER_METRICS])
    print(f"{output / 'd1_report.json'} [{report['decision']}]", flush=True)
    try:
        register(output, metadata, len(scenes), "d1", "d1_report.json", counts, version="D1")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。实验索引校验失败，请查看上方错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_d1.json"))
    parser.add_argument("--stage", choices=("protocol", "dev", "blind"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--review-packet", type=Path, help="Export source pairs only during --stage protocol")
    parser.add_argument("--case")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
