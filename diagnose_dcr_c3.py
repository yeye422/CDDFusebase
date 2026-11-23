"""Compare spatial descriptions of one archived witness at matched real Decoder doses."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import os
from pathlib import Path
import subprocess

import torch

from dcr_synth.atoms import rms
from dcr_synth.config import ROOT, load_config, read_json, write_json
from dcr_synth.decision import OutputGuard, inspect
from dcr_synth.evaluation import metrics
from diagnose_dcr_c2 import SIGNALS, archived_reference
from diagnose_dcr_candidate import development_probes
from diagnose_dcr_d1 import evaluate_probes, implementation_files
from diagnose_dcr_v12 import csv_rows, register, save_image
from diagnose_dcr_v2_f05 import LocalSpace, RAY, linear_direction
from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import PAPER_METRICS, sha256_file
from utils.fusion_metrics import DEFAULT_PROFILE


METHODS = ("T0", "T1", "T2", "T3", "T4")
PAIR_NAMES = tuple(combinations(METHODS, 2))


class SpatialReadouts:
    def __init__(self, focus, baseline, config):
        self.focus = focus
        self.baseline = baseline
        height, width = baseline.shape[-2:]
        offsets = baseline.new_tensor(config["normal_offsets_px"])
        xy = focus.probe.center[:, None] + offsets[None, :, None] * focus.probe.normal
        low = xy.floor()
        frac = xy - low
        x, y = low.long().unbind(dim=-1)
        alpha, beta = frac.unbind(dim=-1)
        if not ((x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)).all():
            raise ValueError("Registered normal samples touch the image boundary")
        self.indices = torch.stack((y * width + x, y * width + x + 1,
                                    (y + 1) * width + x, (y + 1) * width + x + 1))
        self.weights = torch.stack(((1 - alpha) * (1 - beta), alpha * (1 - beta),
                                    (1 - alpha) * beta, alpha * beta))
        rows, cols = focus.support.nonzero(as_tuple=True)
        padding = config["local_padding_px"]
        self.roi = (slice(max(0, int(rows.min()) - padding), min(height, int(rows.max()) + padding + 1)),
                    slice(max(0, int(cols.min()) - padding), min(width, int(cols.max()) + padding + 1)))
        self.continuity_weight = config["continuity_weight"] * len(focus.probe.center) / (len(focus.probe.center) - 1)

    def profiles(self, image):
        return (image.flatten()[self.indices] * self.weights).sum(0)

    def read(self, image, method):
        delta = image - self.baseline
        if method == "T0":
            return self.focus.contrast(delta)
        if method in ("T1", "T2"):
            profile = self.profiles(delta)
            if method == "T1":
                return profile.flatten()
            continuity = (profile[1:] - profile[:-1]) * self.continuity_weight
            return torch.cat((profile.flatten(), continuity.flatten()))
        if method == "T3":
            return delta[..., self.roi[0], self.roi[1]].flatten()
        return delta.flatten()


@dataclass
class Candidate:
    vector: torch.Tensor
    image: torch.Tensor
    rms_gl: float
    loss_gl: float


class Search:
    def __init__(self, space, reader, witness, method, guard, dose, settings):
        self.space, self.reader, self.method, self.guard = space, reader, method, guard
        self.goal = 255 * reader.read(witness, method)
        self.scale = self.goal.abs().mean().clamp_min(1e-4)
        self.bounds = space.zero.new_tensor((dose, settings["peak_gl"], settings["protection_rms_gl"],
                                            settings["far_rms_gl"], settings["protect_readout_gl"]))
        self.settings = settings
        self.attempts = 0
        self.rejections = Counter()

    @torch.no_grad()
    def test(self, vector):
        self.attempts += 1
        image = self.space(vector)
        values = self.guard.signals(image)
        issues = (values > self.bounds + 1e-5).tolist()
        if any(issues):
            self.rejections.update(name for name, failed in zip(SIGNALS, issues) if failed)
        loss = float((255 * self.reader.read(image, self.method) - self.goal).abs().mean())
        entry = Candidate(vector.detach().clone(), image.detach(), float(values[0]), loss)
        return entry, not any(issues)

    def solve(self):
        def observe(vector):
            return self.reader.read(self.space(vector), self.method)

        observe.zero, observe.project = self.space.zero, self.space.project
        start, linear = linear_direction(observe, self.goal, self.settings["linear_steps"], self.settings["ridge"])
        best = widest = self.test(self.space.zero)[0]

        def retain(vector):
            nonlocal best, widest
            entry, feasible = self.test(vector)
            if feasible and entry.loss_gl < best.loss_gl:
                best = entry
            if feasible and entry.rms_gl > widest.rms_gl:
                widest = entry

        for gain in RAY[1:]:
            retain(gain * start)
        param = best.vector.clone().requires_grad_()
        optimizer = torch.optim.Adam([param], lr=self.settings["learning_rate"])
        for _ in range(self.settings["nonlinear_steps"]):
            image = self.space(param)
            residual = 255 * self.reader.read(image, self.method) - self.goal
            excess = (self.guard.signals(image) / self.bounds - 1).clamp_min(0)
            loss = residual.abs().mean() / self.scale + 10 * excess.square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.space.project(param)
            retain(param.detach())
        return best, widest, dict(linear=linear, fit_attempts=self.attempts, rejections=dict(self.rejections),
                                  highest_observed_rms_gl=widest.rms_gl, dimensions=self.goal.numel())

    def match(self, dose, best, widest, config):
        tolerance = config["dose_target_tolerance_gl"]
        found = []
        for entry in (best, widest) if not torch.equal(best.vector, widest.vector) else (best,):
            if abs(entry.rms_gl - dose) <= tolerance:
                found.append(entry)
                continue
            if entry.rms_gl < dose:
                continue
            low, high = 0., 1.
            # Brackets use measured endpoint doses. Only feasible tested frames can enter the comparison.
            for index in range(config["interpolation_steps"]):
                scale = dose / entry.rms_gl if index == 0 else (low + high) / 2
                candidate, feasible = self.test(scale * entry.vector)
                if feasible and abs(candidate.rms_gl - dose) <= tolerance:
                    found.append(candidate)
                    break
                if candidate.rms_gl < dose:
                    low = scale
                else:
                    high = scale
        return min(found, key=lambda row: row.loss_gl) if found else None


def structure(images, case, negatives, optical, sar, config):
    probe = dict(category="fixed_history", source="optical", line=case["target"])
    readings = evaluate_probes(images, {"probes": [probe, *negatives]}, optical, sar, [], {})
    baseline = readings[0]["methods"]["D0"]
    report = {}
    for method, row in readings[0]["methods"].items():
        gain = row["points"] - baseline["points"]
        run_gain = row["longest"] - baseline["longest"]
        negatives_count = sum(current and not original for entry in readings[1:] for current, original in
                              zip(entry["methods"][method]["native_hits"], entry["methods"]["D0"]["native_hits"]))
        report[method] = dict(row, hit_gain=gain, run_gain=run_gain, new_negative_hits=negatives_count,
                              recovered=gain >= config["min_hit_gain"] and run_gain >= config["min_run_gain"])
    return report


def evaluate_group(case, negatives, sources, baseline, witness, guard, chosen, output, config, full_quality=False):
    images = {"D0": baseline, **{name: row.image for name, row in chosen.items() if row is not None}}
    rows = structure(images, case, negatives, *sources, config)
    output.mkdir(parents=True)
    for method, entry in chosen.items():
        if entry is None:
            continue
        file = output / f"{method}.png"
        save_image(file, entry.image)
        rows[method].update(zip(SIGNALS, guard.signals(entry.image).tolist()))
        rows[method].update(target_mae_gl=entry.loss_gl, witness_rmse_gl=float(rms(255 * (entry.image - witness))),
                            image=file.relative_to(output.parents[3]).as_posix())
        if full_quality:
            rows[method]["metrics"] = metrics(entry.image, *sources)
    return {name: rows.get(name) for name in chosen}


def compare(rows, tolerance):
    pairs = []
    for lower, higher in PAIR_NAMES:
        if lower not in rows or higher not in rows:
            continue
        a, b = rows.get(lower), rows.get(higher)
        row = dict(method=higher, reference=lower, matched=False)
        if a is not None and b is not None:
            dose_delta = b["rms_gl"] - a["rms_gl"]
            row.update(matched=abs(dose_delta) <= tolerance, actual_rms_difference_gl=dose_delta,
                       points_gain=b["points"] - a["points"], longest_gain=b["longest"] - a["longest"],
                       witness_rmse_gain_gl=a["witness_rmse_gl"] - b["witness_rmse_gl"])
        pairs.append(row)
    return pairs


def run_case(model, optical, sar, case, negatives, protocol, settings, config, output):
    baseline, detail, decode, candidates = inspect(model, optical, sar, settings, fixed=case["target"])
    witness, _ = archived_reference(ROOT / settings["f05"], case["id"], baseline, protocol)
    focus = candidates[0]
    reader = SpatialReadouts(focus, baseline, config)
    index = config["normal_offsets_px"]
    profile = reader.profiles(witness - baseline)
    if not torch.allclose(profile[:, index.index(3)] - profile[:, index.index(-3)],
                          reader.read(witness, "T0"), atol=1e-7, rtol=1e-5):
        raise ValueError("T0 and normal profile coordinates disagree")
    guard = OutputGuard(candidates, focus, baseline, settings)
    space = LocalSpace((detail,), decode, focus.support, settings["feature_radius_fraction"])
    output.mkdir(parents=True)
    save_image(output / "E2.png", baseline)
    save_image(output / "fixed_witness.png", witness)
    archive = ROOT / settings["f05"] / case["id"] / "fixed_witness_float.npz"
    write_json(output / "targets.json", dict(witness_sha256=sha256_file(archive), witness_rms_gl=float(rms(255 * (
        witness - baseline))), window=focus.geometry, roi_yx=[[s.start, s.stop] for s in reader.roi],
        projection_gl=(255 * reader.read(witness, "T0")).tolist(),
        normal_profiles_gl=(255 * profile).tolist(), continuity_weight=config["continuity_weight"],
        automatic_protection_lines=[item.geometry for item in guard.lines]))
    groups = {}
    for dose in settings["doses_gl"]:
        key = str(dose)
        folder = output / key
        searches, fits, widest, audit = {}, {}, {}, {}
        for method in METHODS:
            print(f"  {dose:g} {method}", flush=True)
            search = Search(space, reader, witness, method, guard, dose, settings)
            fits[method], widest[method], audit[method] = search.solve()
            searches[method] = search
        sources = (optical, sar)
        caps = evaluate_group(case, negatives, sources, baseline, witness, guard, fits, folder / "cap", config, True)
        requests = [(f"fraction_{fraction:g}", dose * fraction, METHODS) for fraction in config["dose_fractions"]]
        minimum = min(widest[name].rms_gl for name in METHODS)
        requests.append(("observed_common", minimum, METHODS))
        requests.append(("observed_T3_T4", min(widest[name].rms_gl for name in ("T3", "T4")), ("T3", "T4")))
        matched = {}
        for label, target, methods in requests:
            if target <= config["dose_target_tolerance_gl"]:
                continue
            chosen = {name: searches[name].match(target, fits[name], widest[name], config) for name in methods}
            rows = evaluate_group(case, negatives, sources, baseline, witness, guard, chosen, folder / label, config)
            matched[label] = dict(target_rms_gl=target, requested_methods=list(methods), methods=rows,
                                  comparisons=compare(rows, config["dose_pair_tolerance_gl"]))
        for method, search in searches.items():
            audit[method]["matching_attempts"] = search.attempts - audit[method]["fit_attempts"]
        groups[key] = dict(cap=caps, solver=audit, matched=matched)
    return groups


def write_report(output, results, metadata):
    table, pairs = [], []
    counts = Counter(cases=len(results))
    for case, result in results.items():
        for cap, group in result.items():
            all_groups = {"cap": {"methods": group["cap"], "target_rms_gl": None}, **group["matched"]}
            for label, entry in all_groups.items():
                for method, row in entry["methods"].items():
                    row = row or {}
                    table.append(dict(case=case, cap_gl=cap, group=label, method=method,
                                      target_rms_gl=entry["target_rms_gl"], available=bool(row),
                                      **row, **row.get("metrics", {})))
                for row in entry.get("comparisons", []):
                    pairs.append(dict(case=case, cap_gl=cap, group=label, target_rms_gl=entry["target_rms_gl"], **row))
                    counts["matched_pairs"] += row["matched"]
    write_json(output / "c3_report.json", dict(results=results, registry=dict(counts),
               decision="oracle_representation_diagnostic_only", uses_manual_position=True))
    csv_rows(output / "comparison.csv", table, ["case", "cap_gl", "group", "method", "target_rms_gl", "available",
             *SIGNALS, "points", "longest", "recovered", "new_negative_hits", "target_mae_gl",
             "witness_rmse_gl", "image", *PAPER_METRICS])
    csv_rows(output / "matched_pairs.csv", pairs, ["case", "cap_gl", "group", "target_rms_gl", "method", "reference",
             "matched", "actual_rms_difference_gl", "points_gain", "longest_gain", "witness_rmse_gain_gl"])
    print(f"{output / 'c3_report.json'} [oracle_representation_diagnostic_only]", flush=True)
    try:
        register(output, metadata, len({name.split("-")[0] for name in results}), "c3", "c3_report.json",
                 dict(counts), version="D1")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。旧实验索引校验失败，请查看上方错误。") from error


def run(args):
    os.chdir(ROOT)
    config = read_json(args.config)
    settings, protocol = load_config(ROOT / config["d1"])
    audit = read_json(ROOT / config["candidate_audit"])
    if len(set(config["normal_offsets_px"])) != len(config["normal_offsets_px"]) or not {-3, 3} <= set(
            config["normal_offsets_px"]):
        raise ValueError("Normal profile offsets must be unique and include ±3")
    if not (0 < config["dose_target_tolerance_gl"] <= config["dose_pair_tolerance_gl"] / 2):
        raise ValueError("Requested-dose tolerance must be at most half of pair tolerance")
    case_map = {case["id"]: case for case in protocol["cases"]}
    keys = [key for key in config["cases"] if not args.case or key == args.case]
    if not keys or (args.smoke and not args.case):
        raise ValueError("Smoke requires one registered case")
    if args.smoke:
        settings.update(linear_steps=2, nonlinear_steps=1)
    output = (args.output or Path(config["output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("请指定仓库内未使用的输出目录。")
    manifest_path = ROOT / protocol["split_manifest"]
    if sha256_file(manifest_path) != protocol["split_manifest_sha256"] or sha256_file(
            ROOT / protocol["checkpoint"]) != protocol["checkpoint_sha256"]:
        raise ValueError("Registered E2 checkpoint or validation split changed")
    manifest = read_json(manifest_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请使用已验证的 lightglue_env。")
    device = torch.device(args.device)
    model, checkpoint = frozen_checkpoint(device)
    if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("Unexpected checkpoint")
    at = development_probes(protocol, audit)
    paths = {key: locate_scene(Path(protocol["data"]), manifest, case_map[key]["scene"]) for key in keys}
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), ROOT / "diagnose_dcr_c2.py", ROOT / "diagnose_dcr_candidate.py",
             args.config.resolve(), ROOT / config["d1"], *implementation_files()]
    hashes = {}
    for file in files:
        relative = file.relative_to(ROOT)
        saved = output / "source_snapshot" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(file.read_bytes())
        hashes[relative.as_posix()] = sha256_file(file)
    inputs = {key: {file.as_posix(): sha256_file(file) for file in pair} for key, pair in paths.items()}
    write_json(output / "protocol.json", dict(c3=config, d1=settings, inputs=inputs))
    metadata = dict(stage="c3", device=str(device), torch_version=str(torch.__version__), smoke=args.smoke,
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=protocol["checkpoint_sha256"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=protocol["split_manifest_sha256"],
                    metric_profile=DEFAULT_PROFILE, case_ids=keys, source_sha256=hashes,
                    protocol_sha256=sha256_file(output / "protocol.json"))
    write_json(output / "run_metadata.json", metadata)
    results = {}
    for key in keys:
        print(f"C3 {key}", flush=True)
        optical, sar = (to_tensor(image, device) for image in read_pair(paths[key]))
        case = case_map[key]
        negatives = [dict(probe, category="archived_disconnected") for probe in at[case["scene"]]
                     if probe["kind"] == "negative"]
        results[key] = run_case(model, optical, sar, case, negatives, protocol, settings, config, output / "outputs" / key)
    write_report(output, results, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_c3.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--case")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
