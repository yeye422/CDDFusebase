"""Cross the frozen D1 and R1 solvers with the same archived targets."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from dcr_synth.atoms import rms
from dcr_synth.config import ROOT, load_config, read_json, write_json
from dcr_synth.decision import OutputGuard, choose, inspect, render_target
from dcr_synth.evaluation import gray8, metrics
from diagnose_dcr_candidate import development_probes
from diagnose_dcr_d1 import evaluate_probes, implementation_files
from diagnose_dcr_v12 import csv_rows, register, save_image
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import PAPER_METRICS, sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


SIGNALS = ("rms_gl", "peak_gl", "protected_rms_gl", "far_rms_gl", "max_protection_change_gl")
PAIRS = (
    ("forced_execution", "C2-A", "C2-0"),
    ("target_in_d1", "C2-B", "C2-A"),
    ("solver_on_oracle_projection", "C2-C", "C2-B"),
    ("solver_on_automatic_target", "C2-D", "C2-A"),
    ("target_in_r1", "C2-C", "C2-D"),
    ("full_vs_projected", "C2-E", "C2-C"),
)


def archived_reference(root, case, baseline, protocol):
    meta = read_json(root / "run_metadata.json")
    if meta["checkpoint_sha256"] != protocol["checkpoint_sha256"] or meta["smoke"]:
        raise ValueError("Expected the complete F0.5 run on the same checkpoint")
    folder = root / case
    with np.load(folder / "fixed_witness_float.npz") as reference:
        base = torch.from_numpy(reference["baseline"]).to(baseline)
        witness = torch.from_numpy(reference["witness"]).to(baseline)
    if not torch.allclose(baseline, base, atol=1e-5, rtol=0):
        raise ValueError(f"Archived F0.5 baseline differs on {case}")
    with np.load(folder / "R1_strict.npz") as result:
        archived = torch.from_numpy(result["image"]).to(baseline)
    return witness, archived


def c2_scene(model, optical, sar, case, protocol, d1, config, negatives, folder, smoke):
    baseline, detail, decode, candidates = inspect(model, optical, sar, d1, fixed=case["target"])
    name = case["id"]
    focus = candidates[0]
    accepted = choose(candidates, "D1-C", d1, {}, only_index=0)
    forced = accepted or choose(candidates, "D1-C", d1, {}, only_index=0, force=True)
    if forced is None:
        raise ValueError(f"No nonzero-polarity source hypothesis for {name}")
    witness, archived = archived_reference(ROOT / d1["f05"], name, baseline, protocol)
    full = 255 * (witness - baseline)
    projected = 255 * focus.contrast(witness - baseline)
    automatic = detail.new_tensor(focus.evidence[forced[1]]["target_change_gl"])
    guard = OutputGuard(candidates, focus, baseline, d1)
    save_image(folder / "E2.png", baseline)
    save_image(folder / "fixed_witness.png", witness)
    write_json(folder / "targets.json", dict(fixed_window=case["target"], automatic_gl=automatic.tolist(),
               oracle_projection_gl=projected.tolist(), forced_source=forced[1],
               originally_accepted=accepted is not None, evidence=focus.evidence,
               automatic_protection_lines=[line.geometry for line in guard.lines],
               witness_sha256=sha256_file(ROOT / d1["f05"] / name / "fixed_witness_float.npz"),
               archived_r1_sha256=sha256_file(ROOT / d1["f05"] / name / "R1_strict.npz")))
    target_probe = dict(category="fixed_history", source="optical", line=case["target"])
    doses = {}
    for dose in d1["doses_gl"]:
        images, arms = {"D0": baseline}, {"D0": dict(status="baseline")}
        bounds = detail.new_tensor((dose, d1["peak_gl"], d1["protection_rms_gl"],
                                   d1["far_rms_gl"], d1["protect_readout_gl"]))
        for arm, selection, desired, solver in (
            ("C2-0", accepted, automatic, "d1"), ("C2-A", forced, automatic, "d1"),
            ("C2-B", forced, projected, "d1"), ("C2-C", forced, projected, "r1"),
            ("C2-D", forced, automatic, "r1"),
        ):
            print(f"  {dose:g} {arm}", flush=True)
            if arm == "C2-A" and accepted is not None:
                image, row = images["C2-0"], arms["C2-0"]
            else:
                image, row = render_target(detail, decode, baseline, candidates, selection, dose, d1,
                                           desired_override=desired, solver=solver)
            images[arm], arms[arm] = image, dict(row)
        print(f"  {dose:g} C2-E", flush=True)
        if dose == 0.5:
            images["C2-E"], arms["C2-E"] = archived, dict(status="archived_F05_R1_strict")
        else:
            image, row = render_target(detail, decode, baseline, candidates, forced, dose, d1,
                                       desired_override=full, solver="r1", full_image=True)
            images["C2-E"], arms["C2-E"] = image, dict(row, status="full_target_R1_main_cap_" + row["status"])
        for arm, image in images.items():
            save_image(folder / f"{arm}_{dose}.png", image)
            signal = guard.signals(image)
            violations = [key for key, fail in zip(SIGNALS, (signal > bounds + 1e-5).tolist()) if fail]
            achieved = 255 * focus.contrast(image - baseline)
            arms[arm].update(zip(SIGNALS, signal.tolist()))
            arms[arm].update(constraint_violations=violations,
                             automatic_target_rmse_gl=float(rms(achieved - automatic)),
                             oracle_projection_rmse_gl=float(rms(achieved - projected)),
                             full_witness_rmse_gl=float(rms(255 * (image - witness))),
                             achieved_profile_gl=achieved.tolist(), metrics=metrics(image, optical, sar))
        readings = evaluate_probes(images, {"probes": [target_probe, *negatives]}, optical, sar, [], {})
        primary = readings[0]["methods"]
        for arm, row in arms.items():
            points = primary[arm]["points"] - primary["D0"]["points"]
            longest = primary[arm]["longest"] - primary["D0"]["longest"]
            row.update(points=primary[arm]["points"], longest=primary[arm]["longest"], hit_gain=points,
                       run_gain=longest, recovered=not row["constraint_violations"] and
                       points >= config["min_hit_gain"] and longest >= config["min_run_gain"])
            row["new_negative_hits"] = sum(
                current and not original for record in readings[1:] for current, original in zip(
                    record["methods"][arm]["native_hits"], record["methods"]["D0"]["native_hits"]))
        pairs = []
        for hypothesis, left, right in PAIRS:
            a, b = arms[left], arms[right]
            difference = a["rms_gl"] - b["rms_gl"]
            pairs.append(dict(hypothesis=hypothesis, method=left, reference=right,
                              meets_common_constraints=not a["constraint_violations"] and not b["constraint_violations"],
                              matched_actual_dose=abs(difference) <= d1["dose_tolerance_gl"],
                              rms_difference_gl=difference, points_gain=a["points"] - b["points"],
                              longest_gain=a["longest"] - b["longest"],
                              oracle_projection_error_gain_gl=b["oracle_projection_rmse_gl"] - a["oracle_projection_rmse_gl"],
                              full_witness_error_gain_gl=b["full_witness_rmse_gl"] - a["full_witness_rmse_gl"]))
        c1 = ROOT / config["c1"] / name / f"C1-C_{dose}.png"
        preview = cv2.imread(str(c1), cv2.IMREAD_GRAYSCALE) if c1.is_file() else None
        doses[str(dose)] = dict(methods=arms, comparisons=pairs, structure=readings,
                               c2_0_matches_archived_c1_preview=(
                                   bool(np.array_equal(preview, gray8(images["C2-0"]))) if not smoke and preview is not None
                                   else None))
    return dict(originally_accepted=accepted is not None, forced_source=forced[1], doses=doses)


def run(args):
    os.chdir(ROOT)
    config = read_json(args.config)
    output = (args.output or Path(config["review_output" if args.source_review else "output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("请指定仓库内未使用的输出目录。")
    if args.source_review:
        from dcr_synth.source_review import review_packet
        review_packet(ROOT / config["source_packet"], output, config, args.manifest_only)
        return
    if args.manifest_only:
        raise ValueError("--manifest-only is only valid with --source-review")
    audit = read_json(ROOT / "configs/dcr_candidate_audit.json")
    d1, protocol = load_config(ROOT / config["d1"])
    manifest = read_json(ROOT / protocol["split_manifest"])
    case_map = {case["id"]: case for case in protocol["cases"]}
    keys = [case for case in config["cases"] if not args.case or case == args.case]
    if not keys or (args.smoke and len(keys) != 1):
        raise ValueError("Smoke requires one registered case")
    if args.smoke:
        d1.update(linear_steps=2, nonlinear_steps=1)
    if (sha256_file(ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]
            or sha256_file(ROOT / protocol["split_manifest"]) != protocol["split_manifest_sha256"]):
        raise ValueError("Registered checkpoint or split changed")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请使用已验证的 lightglue_env。")
    device = torch.device(args.device)
    model, path = frozen_checkpoint(device)
    if path.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("Unexpected checkpoint")
    archived_labels = development_probes(protocol, audit)
    output.mkdir(parents=True, exist_ok=False)
    paths = {key: locate_scene(Path(protocol["data"]), manifest, case_map[key]["scene"]) for key in keys}
    inputs = {key: {file.as_posix(): sha256_file(file) for file in pair} for key, pair in paths.items()}
    files = [Path(__file__).resolve(), ROOT / "diagnose_dcr_candidate.py", *implementation_files(), args.config.resolve()]
    hashes = {}
    for file in files:
        relative = file.relative_to(ROOT)
        saved = output / "source_snapshot" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(file.read_bytes())
        hashes[relative.as_posix()] = sha256_file(file)
    write_json(output / "protocol.json", dict(c2=config, d1=d1, inputs=inputs))
    metadata = dict(stage="c2", device=str(device), torch_version=str(torch.__version__), smoke=args.smoke,
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
                    metric_profile=DEFAULT_PROFILE, case_ids=keys, source_sha256=hashes,
                    protocol_sha256=sha256_file(output / "protocol.json"))
    write_json(output / "run_metadata.json", metadata)
    results = {}
    for key in keys:
        print(f"C2 {key}", flush=True)
        folder = output / "outputs" / key
        folder.mkdir(parents=True)
        case = case_map[key]
        negatives = [dict(item, category="archived_disconnected") for item in archived_labels[case["scene"]]
                     if item["kind"] == "negative"]
        optical, sar = (to_tensor(image, device) for image in read_pair(paths[key]))
        results[key] = c2_scene(model, optical, sar, case, protocol, d1, config, negatives, folder, args.smoke)
    counts = Counter(cases=len(keys))
    rows, comparisons = [], []
    for case, result in results.items():
        for dose, group in result["doses"].items():
            for arm, row in group["methods"].items():
                rows.append(dict(case=case, dose_gl=dose, method=arm, **row, **row["metrics"]))
                if row["recovered"]:
                    counts[f"recovered_{arm}_{dose}"] += 1
            comparisons.extend(dict(case=case, dose_gl=dose, **pair) for pair in group["comparisons"])
    write_json(output / "attribution.json", dict(results=results, registry=dict(counts),
               decision="fixed_position_diagnostic_only", uses_manual_position=True))
    csv_rows(output / "comparison.csv", rows, ["case", "dose_gl", "method", "status", *SIGNALS, "points", "longest",
             "recovered", "new_negative_hits", "automatic_target_rmse_gl", "oracle_projection_rmse_gl",
             "full_witness_rmse_gl", "constraint_violations", *PAPER_METRICS])
    csv_rows(output / "attribution_pairs.csv", comparisons, ["case", "dose_gl", "hypothesis", "method", "reference",
             "meets_common_constraints", "matched_actual_dose", "rms_difference_gl", "points_gain", "longest_gain",
             "oracle_projection_error_gain_gl", "full_witness_error_gain_gl"])
    print(f"{output / 'attribution.json'} [fixed_position_diagnostic_only]", flush=True)
    try:
        register(output, metadata, len({case_map[key]["scene"] for key in keys}),
                 "c2", "attribution.json", dict(counts), version="D1")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。旧实验索引校验失败，请查看上方错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_c2.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-review", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
