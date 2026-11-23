"""Run the frozen E2 DCR-Synth V1.2 P0–P2 diagnostics."""

import argparse
import csv
import hashlib
import os
import subprocess
from dataclasses import replace
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F

from dcr_synth.atoms import Window, rms, source_atoms
from dcr_synth.config import ROOT, freeze, load_config, read_json, write_json
from dcr_synth.evaluation import gray8, linearization, measure, metrics, pair_metrics, replay, violations
from dcr_synth.readouts import Readouts
from dcr_synth.response import combine, finite_differences, matrices, responses
from dcr_synth.solver import METHODS, affine_system, solve
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import sha256_file


def save_image(path, image):
    if not cv2.imwrite(str(path), gray8(image)):
        raise OSError(f"Could not save {path}")


class Scene:
    def __init__(self, model, case, paths, settings, device):
        self.case = case
        self.optical, self.sar = (to_tensor(image, device) for image in read_pair(paths))
        with torch.no_grad():
            encoded = model.encode_pair(self.optical, self.sar)
            self.base, self.detail = model.fuse_features(*encoded)
            self.source_bases = (encoded[0], encoded[2])
            self.source_details = (encoded[1], encoded[3])
            self.fuse = lambda optical, sar: model.detail_fusion(optical, sar, self.base)
            self.decode = lambda detail: model.reconstruct(self.optical, self.base, detail)
            self.baseline = model(self.optical, self.sar)
            replayed = self.decode(combine(self.detail, [], []))
            if self.detail.ndim != 4 or self.detail.shape[1] != 64:
                raise ValueError("Expected 64-channel NCHW fused Detail")
            if not torch.equal(self.baseline, replayed):
                raise ValueError("Zero-action replay differs from native E2 forward")
        self.readouts = Readouts(self.optical, self.sar, self.baseline, case, settings)
        self.windows = [Window(x, y, settings["window_size"], group) for x, y, group in case["windows"]]

    def row(self, image):
        return measure(image, self.baseline, self.readouts)

    def score(self, image):
        return metrics(image, self.optical, self.sar)

    def atoms(self, floor, simple=False):
        profiles = None
        if simple:
            probe = self.readouts.target
            extent = probe.across / 7
            shape = extent * (1 - extent.square()).clamp_min(0) * probe.taper
            coords = probe.center[None] + probe.center.new_tensor([3, -3])[:, None, None] * probe.normal
            grid = (2 * coords / probe.scale - 1)[None]
            profiles = []
            for detail in self.source_details:
                flanks = F.grid_sample(detail, grid, align_corners=True)
                contrast = flanks[:, :, 0] - flanks[:, :, 1]
                profiles.append(contrast.mean(dim=2)[..., None, None] * shape)
        return source_atoms(self.fuse, *self.source_details, self.windows, floor, profiles)


def trial(scene, atoms, data, dose, method, settings, relinearize=False):
    center = np.zeros(len(atoms))
    baseline = scene.baseline
    readouts = scene.readouts
    actual = torch.zeros_like(baseline)
    attempts = []
    for _ in range(2 if relinearize else 1):
        affine = affine_system(data, center, actual, readouts)
        candidates, audit = solve(data, affine, readouts, method, settings, center)
        scales = []
        for coefficients in candidates:
            predicted = data["full"] @ coefficients
            signal = np.linalg.norm(predicted) / np.sqrt(len(predicted))
            scales.append(min(1., dose / max(signal, 1e-12)))
        selection, trials = replay(scene.decode, scene.detail, atoms, candidates,
                                   baseline, readouts, dose, settings, scales)
        audit["replays"] = trials
        audit["center"] = center.tolist()
        attempts.append(audit)
        if selection is None:
            break
        image, row = selection
        coefficients = np.asarray(row["coefficients"])
        prediction = baseline.new_tensor(data["full"] @ coefficients + affine["image"]).view_as(baseline)
        row["linearity"] = linearization(image, baseline, prediction, readouts, settings)
        row["linear_ok"] = all(value["passed"] for value in row["linearity"].values())
        audit["selected"] = row
        if row["linear_ok"]:
            break
        if len(attempts) == 1 and relinearize:
            center = coefficients
            actual = 255 * (image - baseline)
            local, seconds = responses(scene.decode, combine(scene.detail, atoms, center), atoms)
            audit["relinearization_jvp_seconds"] = seconds
            data = matrices(local, atoms, readouts, settings["permutation_seed"])
    selected = attempts[-1].get("selected")
    if selected is None:
        image = baseline
        selected = scene.row(baseline)
        selected.update(coefficients=np.zeros(len(atoms)).tolist(), linear_ok=False)
    selected = dict(selected, matched=selected["rms_gl"] >= dose - settings["dose_tolerance_gl"],
                    fallback="selected" not in attempts[-1],
                    budget_rejections=sum(bool(row["violations"]) for audit in attempts
                                          for row in audit["replays"]))
    again = scene.decode(combine(scene.detail, atoms, selected["coefficients"]))
    if not torch.equal(image, again):
        raise ValueError("Non-reproducible replay")
    return image, selected, attempts


def free_detail(scene, dose, settings):
    """Bounded optimization over free Detail values on the registered target strip."""
    region = scene.readouts.region
    reference = scene.detail[..., region]
    delta = torch.zeros_like(reference, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(rms(reference)) * 0.006)
    radius = 0.2 * float(rms(reference))
    best_image, best = scene.baseline, scene.row(scene.baseline)
    rejected = 0
    for step in range(settings["free_steps"] + 1):
        detail = scene.detail.clone()
        detail[..., region] = reference + delta
        image = scene.decode(detail)
        with torch.no_grad():
            row = scene.row(image)
            failed = violations(row, dose, settings)
            rejected += bool(failed)
            if not failed and row["target_error_gl"] < best["target_error_gl"]:
                best_image, best = image.detach(), row
        if step == settings["free_steps"]:
            break
        change = 255 * (image - scene.baseline)
        readouts = scene.readouts
        target_loss = (readouts.target_values(change) - readouts.b).square().mean()
        protection = change[..., readouts.protected].square().mean()
        spill = change[..., ~region].square().mean()
        dose_excess = (change.square().mean() / dose**2 - 1).clamp_min(0)
        peak_excess = (change.abs().amax() - settings["peak_gl"]).clamp_min(0)
        objective = target_loss + 30 * protection + 30 * spill + 50 * dose_excess.square() + peak_excess.square()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            delta.mul_((radius / rms(delta).clamp_min(1e-12)).clamp_max(1))
    best.update(matched=best["rms_gl"] >= dose - settings["dose_tolerance_gl"],
                budget_rejections=rejected)
    return best_image, best


def baseline_stage(scene, folder, protocol, historical):
    result = scene.row(scene.baseline)
    result["metrics"] = scene.score(scene.baseline)
    result["zero_action_exact"] = True
    save_image(folder / "A0.png", scene.baseline)
    if scene.case["id"] == "546-A":
        archive = read_json(ROOT / protocol["archive"])
        old = archive["results"]["546-A"]["baseline"]
        if archive["checkpoint_sha256"].lower() != protocol["checkpoint_sha256"]:
            raise ValueError("546-A archive uses a different E2 checkpoint")
        result["archive_baseline"] = dict(contrast_gl=old["contrast"], aligned=old["aligned"],
                                          max_point_error_gl=max(abs(a - b) for a, b in
                                                                 zip(old["points"], result["points_gl"])))
        result["archive_matched"] = (result["archive_baseline"]["max_point_error_gl"] <= 0.02
                                     and old["aligned"] == result["aligned"])
        if historical:
            from diagnose_e2_boundary_reachability import Case, feature_search
            from diagnose_e2_boundary_supervision import ORIGINAL_LOSS_PILOT, training_args
            from diagnose_e2_subspace_gradients import frozen_checkpoint
            model, _ = frozen_checkpoint(scene.detail.device)
            labels = {}
            for item in protocol["cases"]:
                if item["scene"] == "546":
                    labels[("546", item["id"].split("-")[-1])] = item["target"]
            for item in scene.case["protections"]:
                labels[("546", item["label"])] = item["line"]
            case = Case(model, read_json(ROOT / protocol["split_manifest"]), labels, ("546", "A"),
                        scene.detail.device, training_args(ORIGINAL_LOSS_PILOT))
            fresh = feature_search(model, case, archive["steps"])
            write_json(folder / "historical_replay.json", fresh)
            key = "maximum_contrast_at_nonincreased_loss"
            expected = archive["results"]["546-A"]["arms"]["imitate_output"]["budgets"]["0.245"].get(key)
            observed = fresh["arms"]["imitate_output"]["budgets"]["0.245"].get(key)
            matched = bool(expected and observed and abs(expected["rms"] - observed["rms"]) <= 0.012
                           and abs(expected["contrast"] - observed["contrast"]) <= 0.25)
            result["historical_replay"] = dict(archive=expected, reproduced=observed, matched=matched)
    return result


def p1_scene(scene, folder, settings):
    arms = {}
    atom_records = {}
    atoms, rejected = scene.atoms(settings["atom_floor"])
    simple, rejected_simple = scene.atoms(settings["atom_floor"], simple=True)
    for arm, atom_set, rejects in (("A1", atoms, rejected), ("A2", simple, rejected_simple),
                                  ("A1_truncated", [replace(atom, direction=atom.direction *
                                                            atom.window.mask(atom.direction),
                                                            support="output_truncated") for atom in atoms], [])):
        arm_data = dict(rejected=rejects, atoms=[], status="no_atoms")
        atom_records[arm] = arm_data
        arms[arm] = {}
        if not atom_set:
            continue
        arm_data["status"] = "evaluated"
        response, seconds = responses(scene.decode, scene.detail, atom_set)
        arm_data["jvp_seconds"] = seconds
        data = matrices(response, atom_set, scene.readouts, settings["permutation_seed"])
        for index, atom in enumerate(atom_set):
            coeff = np.zeros(len(atom_set))
            coeff[index] = settings["u"] / 4
            image = scene.decode(combine(scene.detail, atom_set, coeff))
            save_image(folder / f"{arm}_{atom.name}.png", image)
            np.save(folder / f"{arm}_{atom.name}_response_gl.npy", response[index].cpu().numpy())
            arm_data["atoms"].append(dict(name=atom.name, source=atom.source, band=atom.band,
                                          window=vars(atom.window), input_rms=atom.input_rms,
                                          support=atom.support, fused_direction_rms=float(rms(atom.direction)),
                                          fused_outside_window_rms=float(
                                              rms(atom.direction[~atom.window.mask(atom.direction)])),
                                          derivative_target_gl=data["h"][:, index].tolist(),
                                          actual=scene.row(image), coefficient=float(coeff[index])))
        for dose in settings["doses_gl"]:
            image, row, audit = trial(scene, atom_set, data, dose, "J2", settings)
            save_image(folder / f"{arm}_{dose}.png", image)
            arms[arm][str(dose)] = dict(row, metrics=scene.score(image), audit=audit)
    arms["A3"] = {}
    for dose in settings["doses_gl"]:
        image, row = free_detail(scene, dose, settings)
        save_image(folder / f"A3_{dose}.png", image)
        arms["A3"][str(dose)] = dict(row, metrics=scene.score(image))
    return arms, atom_records


def screen_weights(data, readouts, baseline, settings):
    """Apply the same exploratory grid to every registered solver on 546-A."""
    center = np.zeros(data["h"].shape[1])
    affine = affine_system(data, center, torch.zeros_like(baseline), readouts)
    target = readouts.b.cpu().double().numpy()
    protection = readouts.tau.cpu().double().numpy()
    for weights in product(settings["weight_grid"], repeat=3):
        config = dict(settings, **dict(zip(("lambda_p", "lambda_l", "eta"), weights)))
        for method in METHODS:
            candidates, audit = solve(data, affine, readouts, method, config, center)
            for coefficients in candidates or [None]:
                row = dict(method=method, seconds=audit["seconds"],
                           **{key: config[key] for key in ("lambda_p", "lambda_l", "eta")})
                if coefficients is not None:
                    estimate = data["full"] @ coefficients
                    scale = min(1., 0.245 * np.sqrt(len(estimate)) / max(np.linalg.norm(estimate), 1e-12))
                    coefficients = scale * coefficients
                    estimate *= scale
                    projected_protection = data["q"] @ coefficients
                    dose_gl = np.linalg.norm(estimate) / np.sqrt(len(estimate))
                    row.update(target_error_gl=float(np.linalg.norm(data["h"] @ coefficients - target)
                                                     / np.sqrt(len(target))),
                               rms_gl=float(dose_gl),
                               peak_gl=float(np.abs(estimate).max()),
                               protected_readout_ok=bool((np.abs(projected_protection) <= protection + 1e-5).all()),
                               coefficients=coefficients.tolist())
                yield row


def p2_scene(scene, folder, settings, weight_sweep=False):
    atoms, rejected = scene.atoms(settings["atom_floor"])
    if not atoms:
        return [], dict(rejected=rejected, status="no_atoms", fd_pass=False)
    readouts = scene.readouts
    response, seconds = responses(scene.decode, scene.detail, atoms)
    fd = finite_differences(scene.decode, scene.detail, atoms, response, settings)
    data = matrices(response, atoms, readouts, settings["permutation_seed"])
    np.savez_compressed(folder / "linear_response.npz", R=response.cpu().numpy(), H=data["h"], Q=data["q"],
                        G_P=data["grams"]["J4"], G_L=np.diag(data["leakage"]),
                        target_change_gl=readouts.b.cpu().numpy(),
                        protection_tolerance_gl=readouts.tau.cpu().numpy(),
                        target_mask=readouts.region.cpu().numpy(),
                        protection_mask=readouts.protected.cpu().numpy(), **data["grams"])
    report = dict(atoms=[atom.name for atom in atoms], rejected=rejected, fd=fd,
                  fd_pass=all(row["passed"] for row in fd), jvp_seconds=seconds,
                  gram_min_eigenvalue={name: float(np.linalg.eigvalsh(g)[0])
                                       for name, g in data["grams"].items()}, doses={})
    if weight_sweep:
        csv_rows(folder / "weight_screen.csv", screen_weights(data, readouts, scene.baseline, settings),
                 ["method", "lambda_p", "lambda_l", "eta", "rms_gl", "peak_gl", "protected_readout_ok",
                  "target_error_gl", "coefficients", "seconds"])
    methods = [(method, method, settings) for method in METHODS]
    methods.extend((name, "J4", dict(settings, **{key: 0.0})) for name, key in
                   (("J4_no_GP", "lambda_p"), ("J4_no_GL", "lambda_l")))
    rows = []
    for dose in settings["doses_gl"]:
        report["doses"][str(dose)] = {}
        for name, method, config in methods:
            image, row, audit = trial(scene, atoms, data, dose, method, config, relinearize=True)
            save_image(folder / f"{name}_{dose}.png", image)
            row.update(case=scene.case["id"], role=scene.case["role"], dose_gl=dose, method=name,
                       fd_pass=report["fd_pass"], metrics=scene.score(image))
            rows.append(row)
            report["doses"][str(dose)][name] = dict(selected=row, attempts=audit)
    return rows, report


def csv_rows(path, rows, fields):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def viable(row):
    return bool(row and row["matched"] and not row.get("fallback") and row.get("linear_ok", True))


def comparable(left, right, tolerance):
    return viable(left) and viable(right) and abs(left["rms_gl"] - right["rms_gl"]) <= tolerance


def capacity_report(results, lock):
    settings = lock["protocol"]["settings"]
    cases = {entry["id"]: entry for entry in lock["protocol"]["cases"]}
    rows = []
    for key, result in results.items():
        if result["case"]["role"] == "preserve":
            continue
        baseline = result["baseline"]
        for dose, free in result["arms"]["A3"].items():
            simple = result["arms"]["A2"].get(dose)
            method = result["arms"]["A1"].get(dose)
            delta = (method["valid_points"] - baseline["valid_points"]) if method else 0
            gain = (baseline["target_error_gl"] - method["target_error_gl"]) if method else 0
            matched = comparable(method, simple, settings["dose_tolerance_gl"])
            rows.append(dict(case=key, geo_block=cases[key].get("geo_block"), dose_gl=float(dose),
                             role=cases[key]["role"], matched_to_A2=bool(matched), gain_gl=gain,
                             valid_point_gain=delta, free_gain_gl=baseline["target_error_gl"] - free["target_error_gl"],
                             beats_A2=bool(matched and delta >= 1 and gain >= settings["win_gl"]
                                           and simple["target_error_gl"] - method["target_error_gl"]
                                           >= settings["win_gl"])))
    wins = {row["geo_block"] for row in rows if row["role"] == "holdout" and row["geo_block"]
            and row["beats_A2"] and cases[row["case"]].get("verified_natural_missing")
            and row["dose_gl"] == 0.245}
    enough = lock["confirmed_holdout_blocks"] >= settings["minimum_holdout_blocks"]
    passed = enough and len(wins) >= settings["minimum_holdout_blocks"] and settings["calibrated"]
    free_wins = {row["geo_block"] for row in rows if row["role"] == "holdout" and row["geo_block"]
                 and row["dose_gl"] == 0.245 and row["free_gain_gl"] >= settings["win_gl"]}
    if passed:
        decision = "pass"
    elif enough and settings["calibrated"]:
        decision = ("redesign_eligible" if len(free_wins) >= 2
                    else "insufficient_detail_reachability")
    else:
        decision = "exploratory_only"
    return dict(decision=decision, paired=rows, winning_holdout_blocks=sorted(wins),
                reference_reachable_blocks=sorted(free_wins),
                confirmed_holdout_blocks=lock["confirmed_holdout_blocks"])


def compare_solvers(rows, capacity, lock):
    settings = lock["protocol"]["settings"]
    cases = {case["id"]: case for case in lock["protocol"]["cases"]}
    lookup = {(row["case"], row["dose_gl"], row["method"]): row for row in rows}
    paired = []
    for candidate in (row for row in rows if row["method"] == "J4"):
        case_id, dose = candidate["case"], candidate["dose_gl"]
        for method in ("J0", "J1", "J2", "J3", "J6", "A2", "J4_no_GP", "J4_no_GL"):
            if method == "A2":
                reference = capacity["results"][case_id]["arms"]["A2"].get(str(dose))
            else:
                reference = lookup.get((case_id, dose, method))
            matched = bool(candidate["fd_pass"] and (reference or {}).get("fd_pass", True)
                           and comparable(candidate, reference, settings["dose_tolerance_gl"]))
            row = dict(case=case_id, dose_gl=dose, method=method, matched=matched,
                       role=candidate["role"], geo_block=cases[case_id].get("geo_block"))
            if matched:
                gain = reference["target_error_gl"] - candidate["target_error_gl"]
                points = candidate["valid_points"] - reference["valid_points"]
                protected_gain = ((reference["protected_rms_gl"] or 0)
                                  - (candidate["protected_rms_gl"] or 0))
                row.update(target_gain_gl=gain, valid_point_gain=points,
                           protected_rms_gain_gl=protected_gain,
                           spill_energy_gain_gl2=reference["spill_energy_gl2"] - candidate["spill_energy_gl2"],
                           rms_difference_gl=candidate["rms_gl"] - reference["rms_gl"],
                           win=(gain >= settings["win_gl"] and points >= 1
                                and protected_gain >= -settings["protection_win_gl"]) or (
                               points >= 0 and gain >= -settings["win_gl"]
                               and protected_gain >= settings["protection_win_gl"]))
            paired.append(row)
    required = {"J1", "J2", "J3", "J6", "A2"}
    confirmed = [row for row in paired if row["role"] == "holdout" and row["geo_block"]
                 and cases[row["case"]].get("verified_natural_missing") and row["dose_gl"] == 0.245]
    winning_blocks = set()
    for case_id in {row["case"] for row in confirmed}:
        if {row["method"] for row in confirmed if row["case"] == case_id and row.get("win")} >= required:
            winning_blocks.add(cases[case_id]["geo_block"])
    preserve_or_negative = {case["id"] for case in cases.values()
                            if case["role"] in ("preserve", "negative")}
    main_rows = {row["case"]: row for row in rows if row["method"] == "J4" and row["dose_gl"] == 0.245}
    sar_cases = {case["id"] for case in cases.values() if any(
        probe["role"] == "sar_only" and probe.get("verified") for probe in case["protections"])}
    tested_sar_cases = {key for key in sar_cases if viable(main_rows.get(key)) and main_rows[key]["fd_pass"]}
    safety_coverage = bool(tested_sar_cases and preserve_or_negative <= main_rows.keys())
    safety_ok = safety_coverage and all(
        not main_rows[key]["fallback"] and main_rows[key]["protection_ok"] for key in preserve_or_negative)
    enough = (settings["calibrated"] and safety_coverage
              and lock["confirmed_holdout_blocks"] >= settings["minimum_holdout_blocks"])
    coverage = {method: len({row["geo_block"] for row in confirmed
                             if row["method"] == method and row["matched"]}) for method in sorted(required)}
    if enough and safety_ok and len(winning_blocks) >= settings["minimum_holdout_blocks"]:
        decision = "pass"
    elif enough and min(coverage.values()) >= settings["minimum_holdout_blocks"]:
        decision = "stop_response_route"
    else:
        decision = "insufficient_evidence"
    by_dose = {}
    methods = [*METHODS, "J4_no_GP", "J4_no_GL"]
    for dose in settings["doses_gl"]:
        groups = {}
        for case_id, case in cases.items():
            actual_rows = [lookup.get((case_id, dose, method)) for method in methods]
            if not all(map(viable, actual_rows)):
                continue
            actual_doses = [row["rms_gl"] for row in actual_rows]
            if max(actual_doses) - min(actual_doses) <= settings["dose_tolerance_gl"]:
                groups.setdefault(case["role"], []).append({row["method"]: row["metrics"] for row in actual_rows})
        by_dose[str(dose)] = {role: pair_metrics(groups.get(role, []), methods)
                             for role in sorted({case["role"] for case in cases.values()})}
    return dict(decision=decision, comparisons=paired, paired_metric_summary=by_dose,
                matched_holdout_blocks=coverage, winning_holdout_blocks=sorted(winning_blocks),
                confirmed_holdout_blocks=lock["confirmed_holdout_blocks"],
                verified_sar_only_probes=lock["sar_only_probes"],
                sar_only_cases_with_matched_replay=sorted(tested_sar_cases), safety_checks_passed=safety_ok)


def register(output, metadata, count, stage, report, summary, version="V1.2"):
    from run_e2_matched_no_pac import register_many

    relative = output.relative_to(ROOT).as_posix()
    identity = f"QXS-DCR-{version.replace('.', '')}-" + hashlib.sha256(relative.encode()).hexdigest()[:12].upper()
    register_many([dict(id=identity, comparable_group=identity, task="engineering",
                        status="diagnostic_fixed_subset", method=f"DCR-Synth {version} {stage.upper()}",
                        dataset="QXS-SAROPT", split="val", offset=0, sample_count=count,
                        checkpoint=metadata["checkpoint"], checkpoint_sha256=metadata["checkpoint_sha256"],
                        split_manifest=metadata["split_manifest"],
                        split_manifest_sha256=metadata["split_manifest_sha256"],
                        metric_profile=metadata["metric_profile"],
                        output_policy="float_gl_with_uint8_gray_previews", result_dir=relative,
                        metadata_path=relative + "/run_metadata.json",
                        metrics=summary, metric_source=dict(type="json_paths", path=f"{relative}/{report}",
                                                            paths={key: ["registry", key] for key in summary}))])


def run(args):
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    stage_config, protocol = load_config(args.config)
    if stage_config["stage"] != args.stage:
        raise ValueError("Config stage does not match --stage")
    if args.weight_sweep and (args.stage != "p2" or args.case != "546-A" or not args.exploratory):
        raise ValueError("The weight screen is limited to exploratory P2 on 546-A")
    settings = protocol["settings"]
    if args.stage != "p0":
        prior = args.protocol_lock or Path(stage_config["protocol_lock"])
        baseline_path = prior.parent / "baseline.json"
        missing = [str(path) for path in (prior, baseline_path) if not path.is_file()]
        if missing:
            raise SystemExit(
                f"P0 尚未完成，缺少 {', '.join(missing)}。"
                "请先运行 --stage p0。若已将完整结果保存到其他目录，"
                "请用 --protocol-lock 指向该目录的 protocol.json。"
            )
    if args.stage == "p2":
        capacity_path = args.p1_result or Path(stage_config["p1_result"])
        if not capacity_path.is_file():
            raise SystemExit(
                f"P1 尚未完成，缺少 {capacity_path}。"
                "请先运行 --stage p1，或用 --p1-result 指向已有的 reachability.json。"
            )
    manifest = read_json(ROOT / protocol["split_manifest"])
    paths = {case["scene"]: locate_scene(Path(protocol["data"]), manifest, case["scene"])
             for case in protocol["cases"]}
    inputs = {scene: [{"path": path.as_posix(), "sha256": sha256_file(path)} for path in pair]
              for scene, pair in paths.items()}
    lock = freeze(protocol, manifest, inputs)
    output = (args.output or Path(stage_config["output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside the repository")
    cases = protocol["cases"]
    if args.case:
        if args.stage != "p0" and any(case["id"] == args.case and case["role"] == "holdout" for case in cases):
            raise ValueError("Single-case runs are for development and safety checks")
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise ValueError(f"Unknown case {args.case}")
    protocol_hash = None
    if args.stage != "p0":
        protocol_hash = sha256_file(prior)
        if read_json(prior) != lock:
            raise ValueError("P0 protocol, checkpoint, split or source data changed")
        baseline = read_json(baseline_path)
        development = baseline["results"].get("546-A", {})
        if not development.get("archive_matched"):
            raise ValueError("P0 must reproduce the archived 546-A baseline")
        if settings["calibrated"] and not development.get("historical_replay", {}).get("matched"):
            raise ValueError("A calibrated P0 protocol requires a matched historical 546-A intervention")
    if args.stage == "p2":
        capacity = read_json(capacity_path)
        if capacity["protocol_lock_sha256"] != protocol_hash:
            raise ValueError("P1 result belongs to a different P0 protocol")
        if capacity["decision"] != "pass":
            if not args.exploratory:
                raise ValueError("P1 has not passed. P2 development-only replay requires --exploratory")
            cases = [case for case in cases if case["role"] == "development"]
            if not cases:
                raise ValueError("Exploratory P2 requires a development case")
        if any(case["id"] not in capacity["results"] for case in cases):
            raise ValueError("Run P1 for every selected P2 case first")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = frozen_checkpoint(device)
    if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("The frozen loader resolved a different checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", lock)
    dependencies = ("direct_fusion.py", "net.py", "lwpd.py", "e2_base_anchor.py",
                    "diagnose_e2_subspace_gradients.py", "diagnose_e2_information_path.py",
                    "diagnose_e2_boundary_supervision.py", "diagnose_e2_boundary_reachability.py",
                    "utils/fusion_metrics.py", "utils/reference_metrics.py", "utils/Evaluator.py")
    code = [Path(__file__).resolve(), *sorted((ROOT / "dcr_synth").glob("*.py")),
            *(ROOT / path for path in dependencies)]
    source = output / "source_snapshot"
    source.mkdir()
    for path in code:
        destination = source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (output / "stage_config.json").write_bytes(args.config.read_bytes())
    metadata = dict(stage=args.stage, device=str(device), torch_version=str(torch.__version__),
                    numpy_version=np.__version__, scipy_version=scipy.__version__,
                    deterministic_algorithms="warn_only",
                    config_sha256=sha256_file(args.config), protocol_sha256=sha256_file(output / "protocol.json"),
                    p0_protocol_sha256=protocol_hash, metric_profile=lock["metric_profile"],
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=lock["sha256"]["checkpoint"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=lock["sha256"]["split_manifest"],
                    source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in code},
                    case_ids=[case["id"] for case in cases], exploratory=args.exploratory)
    write_json(output / "run_metadata.json", metadata)
    result, atom_records, rows = {}, {}, []
    for case in cases:
        case_id = case["id"]
        print(f"{args.stage} {case_id}", flush=True)
        folder = output / case_id
        folder.mkdir()
        scene = Scene(model, case, paths[case["scene"]], settings, device)
        if args.stage == "p0":
            result[case_id] = baseline_stage(scene, folder, protocol, args.historical_replay)
        elif args.stage == "p1":
            save_image(folder / "A0.png", scene.baseline)
            arms, atom_records[case_id] = p1_scene(scene, folder, settings)
            result[case_id] = dict(case=case, baseline=dict(scene.row(scene.baseline),
                                                         metrics=scene.score(scene.baseline)), arms=arms)
        else:
            save_image(folder / "A0.png", scene.baseline)
            cases_rows, audit = p2_scene(scene, folder, settings, args.weight_sweep)
            rows.extend(cases_rows)
            result[case_id] = dict(baseline=dict(scene.row(scene.baseline), metrics=scene.score(scene.baseline)),
                                   audit=audit)
    summary = dict(cases=len(cases), scenes=len({case["scene"] for case in cases}))
    payload = dict(results=result, registry=summary, protocol_lock_sha256=protocol_hash)
    if args.stage == "p0":
        filename = "baseline.json"
        summary["zero_action_passes"] = sum(row["zero_action_exact"] for row in result.values())
        summary["archive_baseline_passes"] = sum(row.get("archive_matched", False) for row in result.values())
        summary["archive_intervention_passes"] = sum(row.get("historical_replay", {}).get("matched", False)
                                                    for row in result.values())
    elif args.stage == "p1":
        filename = "reachability.json"
        write_json(output / "atoms.json", atom_records)
        payload.update(capacity_report(result, lock))
        summary["winning_holdout_blocks"] = len(payload["winning_holdout_blocks"])
    else:
        filename = "response_audit.json"
        payload.update(compare_solvers(rows, capacity, lock))
        payload["p1_result_sha256"] = sha256_file(capacity_path)
        summary["finite_difference_passes"] = sum(row["audit"]["fd_pass"] for row in result.values())
        flattened = [dict(row, **row["metrics"]) for row in rows]
        from maintain_experiment_index import PAPER_METRICS
        csv_rows(output / "solver_comparison.csv", flattened,
                 ["case", "role", "method", "dose_gl", "matched", "rms_gl", "peak_gl", "target_rms_gl",
                  "protected_rms_gl", "target_error_gl", "valid_points", "longest", "spill_energy_gl2",
                  "linear_ok", "fd_pass", "fallback", "budget_rejections", *PAPER_METRICS])
        csv_rows(output / "paired_comparison.csv", payload["comparisons"],
                 ["case", "role", "geo_block", "dose_gl", "method", "matched", "win", "target_gain_gl",
                  "protected_rms_gain_gl", "valid_point_gain", "spill_energy_gain_gl2", "rms_difference_gl"])
    write_json(output / filename, payload)
    print(f"{output / filename} [{payload.get('decision', 'baseline_audited')}]", flush=True)
    try:
        register(output, metadata, summary["scenes"], args.stage, filename, summary)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。实验索引校验失败，请检查上方注册表错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("p0", "p1", "p2"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", help="Run only one registered case")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--protocol-lock", type=Path, help="P0 protocol.json")
    parser.add_argument("--p1-result", type=Path, help="P1 reachability.json")
    parser.add_argument("--historical-replay", action="store_true", help="Re-run the archived 546-A intervention in P0")
    parser.add_argument("--exploratory", action="store_true", help="Allow a development-only P2 run before a P1 pass")
    parser.add_argument("--weight-sweep", action="store_true", help="Screen the registered weight grid on 546-A")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
