"""Locate the frozen E2 control space for the fixed F0 image witnesses."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from torch.func import jvp, vjp

from dcr_synth.atoms import rms
from dcr_synth.config import ROOT, freeze, read_json, write_json
from dcr_synth.evaluation import gray8
from diagnose_dcr_v12 import Scene, csv_rows, register, save_image
from diagnose_dcr_v2_f0 import Endpoint, optimize
from diagnose_e2_information_path import locate_scene
from diagnose_e2_subspace_gradients import frozen_checkpoint
from maintain_experiment_index import sha256_file


ARMS = ("R1", "R2", "R3", "R4")
RAY = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def restore_witness(scene, endpoint, archive, recorded, settings):
    """Replay the archived image-only path and require identity of its saved readouts."""
    dose = settings["dose_gl"]
    root = archive / scene.case["id"]
    for path in (root / "E2.png", root / f"output_free_{dose}.png"):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_png = cv2.imread(str(root / "E2.png"), cv2.IMREAD_GRAYSCALE)
    if not np.array_equal(gray8(scene.baseline), baseline_png):
        raise ValueError(f"{scene.case['id']} E2 preview differs from F0")
    image, replay = optimize(endpoint, "output_free", dose)
    preview_path = root / f"output_free_{dose}.png"
    png = cv2.imread(str(preview_path), cv2.IMREAD_GRAYSCALE)
    observed = gray8(image)
    changed = int(np.count_nonzero(observed != png))
    point_error = max(abs(a - b) for a, b in zip(replay["points_gl"], recorded["points_gl"]))
    peak_error = max(abs(a - b) for a, b in zip(replay["signed_peak_gl"], recorded["signed_peak_gl"]))
    delta_error = max(abs(replay[key] - recorded[key])
                      for key in ("rms_gl", "peak_gl", "protected_rms_gl", "far_rms_gl"))
    if (changed or max(point_error, peak_error, delta_error) > settings["archive_tolerance_gl"]
            or any(replay[key] != recorded[key] for key in
                   ("native_hits", "peak_offsets_px", "longest", "structure_recovered", "violations"))):
        raise ValueError(f"{scene.case['id']} F0 replay failed identity checks "
                         f"(PNG pixels={changed}, point error={point_error:g}, peak error={peak_error:g}, "
                         f"budget error={delta_error:g})")
    return image.detach(), dict(source=preview_path.as_posix(), sha256=sha256_file(preview_path),
                                preview_changed_pixels=changed, points_error_gl=point_error,
                                peak_profile_error_gl=peak_error, budget_error_gl=delta_error,
                                archived=recorded, replay=replay)


def layer_spaces(model, scene):
    """Return exact suffixes of the existing frozen Decoder."""
    decoder = model.decoder
    skip = scene.optical if model.uses_image_skip else None
    halfway = len(decoder.encoder_level2) // 2
    early = decoder.encoder_level2[:halfway]
    late = decoder.encoder_level2[halfway:]

    def finish(logits):
        if skip is not None:
            logits = logits + skip
        return decoder.sigmoid(logits)

    with torch.no_grad():
        reduction = decoder.reduce_channel(torch.cat((scene.base, scene.detail), dim=1))
        mid = early(reduction)
        before_last = decoder.output[1](decoder.output[0](late(mid)))
    return {
        "R1": ((scene.detail,), scene.decode),
        "R2": ((scene.base, scene.detail), lambda base, detail: model.reconstruct(scene.optical, base, detail)),
        "R3": ((mid,), lambda value: finish(decoder.output(late(value)))),
        "R4": ((before_last,), lambda value: finish(decoder.output[2](value))),
    }


class LocalSpace:
    def __init__(self, values, render, mask, fraction):
        indices = mask.flatten().nonzero().flatten()
        self.reference = tuple(value.flatten(2) for value in values)
        self.shapes = tuple(value.shape for value in values)
        self.local = tuple(value.index_select(2, indices) for value in self.reference)
        self.radii = tuple(max(float(rms(value)), 1e-7) * fraction for value in self.local)
        self.sizes = tuple(value.numel() for value in self.local)
        self.zero = values[0].new_zeros(sum(self.sizes))
        self.indices, self.render = indices, render

    def __call__(self, vector):
        edited = [ref.index_copy(2, self.indices, local + radius * item.view_as(local)).view(shape)
                  for ref, local, radius, item, shape in
                  zip(self.reference, self.local, self.radii, vector.split(self.sizes), self.shapes)]
        return self.render(*edited)

    @torch.no_grad()
    def project(self, vector):
        for item in vector.split(self.sizes):
            item.mul_((1 / rms(item).clamp_min(1e-12)).clamp_max(1))
        return vector


def linear_direction(space, desired, steps, ridge):
    """Matrix-free conjugate gradients on the regularized JVP/VJP normal equation."""
    def gray(vector):
        return 255 * space(vector)

    zero = space.zero
    count = desired.numel()
    _, pullback = vjp(gray, zero)
    residual = pullback(desired / count)[0].detach()
    direction, estimate = residual.clone(), zero.clone()
    norm2 = torch.dot(residual, residual)
    initial = float(norm2)
    actual_steps = 0
    for _ in range(steps):
        if float(norm2) <= max(initial * 1e-10, 1e-20):
            break
        _, variation = jvp(gray, (zero,), (direction,))
        normal = (pullback(variation / count)[0] + ridge * direction).detach()
        denom = torch.dot(direction, normal)
        if float(denom) <= 0:
            break
        alpha = norm2 / denom
        estimate = estimate + alpha * direction
        residual = residual - alpha * normal
        updated = torch.dot(residual, residual)
        direction = residual + (updated / norm2) * direction
        norm2 = updated
        actual_steps += 1
    del pullback
    space.project(estimate)
    _, linear = jvp(gray, (zero,), (estimate,))
    return estimate, dict(iterations=actual_steps, linear_rmse_gl=float(rms(linear - desired)),
                          predicted_rms_gl=float(rms(linear)), predicted_peak_gl=float(linear.abs().amax()))


class Fit:
    def __init__(self, endpoint, witness, witness_row, settings, relaxed=False):
        self.endpoint, self.witness, self.row = endpoint, witness, witness_row
        self.settings = settings
        self.relaxed = relaxed
        self.rejected = Counter()
        self.tries = 0
        self.best = None

    @torch.no_grad()
    def score(self, image):
        dose = self.settings["dose_gl"]
        row = self.endpoint.read(image, dose)
        error = 255 * (image - self.witness)
        row["fit_rmse_gl"] = float(rms(error))
        row["target_fit_rmse_gl"] = float(rms(error[..., self.endpoint.scene.readouts.region]))
        row["relative_fit_error"] = row["fit_rmse_gl"] / max(self.row["rms_gl"], 1e-12)
        total = self.row["valid_points"]
        overlap = sum(a and b for a, b in zip(row["native_hits"], self.row["native_hits"]))
        row["witness_hit_overlap"] = overlap / total if total else None
        issues = [name for name in row["violations"] if not (self.relaxed and name == "peak")]
        row["reproduced_under_fit_constraints"] = bool(
            self.row["structure_recovered"] and not issues
            and row["hit_gain"] >= self.settings["min_hit_gain"]
            and row["run_gain"] >= self.settings["min_run_gain"]
            and row["relative_fit_error"] <= self.settings["relative_fit_limit"]
            and row["witness_hit_overlap"] is not None
            and row["witness_hit_overlap"] >= self.settings["witness_hit_overlap"]
        )
        row["witness_reproduced"] = row["reproduced_under_fit_constraints"] and not row["violations"]
        return row, issues

    @torch.no_grad()
    def consider(self, image, vector):
        row, issues = self.score(image)
        self.tries += 1
        self.rejected.update(issues)
        if issues:
            return
        key = (not row["reproduced_under_fit_constraints"], row["fit_rmse_gl"])
        if self.best is None or key < self.best[0]:
            self.best = key, image.detach(), vector.detach().clone(), row

    def result(self):
        _, image, vector, row = self.best
        return image, vector, dict(row, candidates=self.tries, rejection_reasons=dict(self.rejected),
                                   peak_relaxed=self.relaxed)


def ray_search(fit, space, direction):
    with torch.no_grad():
        for alpha in RAY:
            local = alpha * direction
            fit.consider(space(local), local)


def optimize_fit(fit, space, start):
    endpoint = fit.endpoint
    scene, settings = endpoint.scene, fit.settings
    readouts = scene.readouts
    param = start.clone().requires_grad_()
    optimizer = torch.optim.Adam([param], lr=settings["learning_rate"])
    dose = settings["dose_gl"]
    for step in range(settings["nonlinear_steps"] + 1):
        image = space(param)
        fit.consider(image.detach(), param)
        if step == settings["nonlinear_steps"]:
            break
        change = 255 * (image - scene.baseline)
        error = 255 * (image - fit.witness)
        masks = ((readouts.region, settings["target_rms_gl"]),
                 (readouts.protected, settings["protected_rms_gl"]),
                 (readouts.far, settings["max_far_rms_gl"]))
        excess = (change.square().mean() / dose**2 - 1).clamp_min(0).square()
        for mask, limit in masks:
            if mask.any():
                excess = excess + (change[..., mask].square().mean() / limit**2 - 1).clamp_min(0).square()
        if not fit.relaxed:
            excess = excess + (change.abs().amax() / settings["peak_gl"] - 1).clamp_min(0).square()
        protection = (readouts.protection_values(change).abs() / readouts.tau - 1).clamp_min(0)
        loss = error.square().mean() / dose**2 + 10 * (excess + protection.square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        space.project(param)


def evaluate_arm(space, endpoint, witness, witness_row, settings, folder, arm):
    if not torch.equal(space(space.zero), endpoint.scene.baseline):
        raise ValueError(f"{arm} zero edit differs from native decoder")
    target = 255 * (witness - endpoint.scene.baseline)
    direction, linear = linear_direction(space, target, settings["linear_steps"], settings["ridge"])
    strict = Fit(endpoint, witness, witness_row, settings)
    relaxed = Fit(endpoint, witness, witness_row, settings, relaxed=True)
    for fit in (strict, relaxed):
        ray_search(fit, space, direction)
        optimize_fit(fit, space, fit.result()[1])
    relaxed_image, relaxed_vector, relaxed_row = relaxed.result()
    restored = Fit(endpoint, witness, witness_row, settings)
    ray_search(restored, space, relaxed_vector)
    optimize_fit(restored, space, restored.result()[1])
    strict_image, strict_vector, strict_row = strict.result()
    restored_image, restored_vector, restored_row = restored.result()
    rows = {"strict": strict_row, "peak_relaxed": relaxed_row, "restored": restored_row}
    for stage, image, vector in (("strict", strict_image, strict_vector),
                                 ("peak_relaxed", relaxed_image, relaxed_vector),
                                 ("restored", restored_image, restored_vector)):
        save_image(folder / f"{arm}_{stage}.png", image)
        np.savez_compressed(folder / f"{arm}_{stage}.npz", image=image.cpu().numpy(),
                            normalized_edit=vector.cpu().numpy())
        rows[stage]["metrics"] = endpoint.scene.score(image)
        rows[stage]["used_fraction_of_feature_radius"] = [float(rms(block)) for block in vector.split(space.sizes)]
    return dict(linear=linear, feature_shapes=[list(shape) for shape in space.shapes], local_radii=space.radii,
                fits=rows)


def first_reachable(arms):
    for arm in ARMS:
        if any(arms[arm]["fits"][mode]["witness_reproduced"] for mode in ("strict", "restored")):
            return arm
    return None


def run(args):
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。请用已验证的 lightglue_env 解释器。")
    device = torch.device(args.device)
    config = read_json(args.config)
    archive = ROOT / config["f0_result"]
    prereg = read_json(archive / "protocol.json")
    report = read_json(archive / "f0_reachability.json")
    meta = read_json(archive / "run_metadata.json")
    if meta["smoke"] or sha256_file(archive / "protocol.json") != meta["protocol_sha256"]:
        raise ValueError("F0 archive must be a complete frozen run")
    for name, digest in meta["code_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Current code differs from the archived F0 dependency: {name}")
    protocol = prereg["protocol"]
    manifest = read_json(ROOT / protocol["split_manifest"])
    paths = {case["scene"]: locate_scene(Path(protocol["data"]), manifest, case["scene"])
             for case in protocol["cases"]}
    inputs = {scene: [{"path": path.as_posix(), "sha256": sha256_file(path)} for path in pair]
              for scene, pair in paths.items()}
    lock = freeze(protocol, manifest, inputs)
    if lock != {name: prereg[name] for name in lock}:
        raise ValueError("Frozen sources or assets differ from F0")
    settings = dict(prereg["f0"], **config)
    if settings["dose_gl"] != 0.5 or min(settings[key] for key in
            ("linear_steps", "nonlinear_steps", "ridge", "feature_radius_fraction", "archive_tolerance_gl")) <= 0:
        raise ValueError("Register positive search settings and the archived 0.5 witness dose")
    for key in ("relative_fit_limit", "witness_hit_overlap"):
        if not 0 < settings[key] <= 1:
            raise ValueError(f"Invalid fraction: {key}")
    if args.smoke:
        if not args.case:
            raise ValueError("--smoke requires one --case")
        settings.update(linear_steps=2, nonlinear_steps=1)
    selected = config["witness_cases"] + config["control_cases"]
    if len(set(selected)) != len(selected) or (args.case and args.case not in selected):
        raise ValueError("Use unique registered F0.5 case IDs")
    cases = [case_id for case_id in selected if not args.case or case_id == args.case]
    output = (args.output or Path(config["output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("请使用仓库内的新输出目录。")
    model, checkpoint = frozen_checkpoint(device)
    if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("Unexpected E2 checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", dict(f0=prereg, f05=settings))
    source = output / "source_snapshot"
    source.mkdir()
    for name in (*meta["code_sha256"], Path(__file__).name):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output / "stage_config.json").write_bytes(args.config.read_bytes())
    metadata = dict(stage="f05", device=str(device), torch_version=str(torch.__version__),
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=lock["sha256"]["checkpoint"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=lock["sha256"]["split_manifest"],
                    metric_profile=lock["metric_profile"], smoke=args.smoke, case_ids=cases,
                    source_f0_sha256=sha256_file(archive / "f0_reachability.json"),
                    source_f0_protocol_sha256=meta["protocol_sha256"],
                    config_sha256=sha256_file(args.config), code_sha256=sha256_file(Path(__file__)),
                    protocol_sha256=sha256_file(output / "protocol.json"))
    write_json(output / "run_metadata.json", metadata)
    all_results, rows = {}, []
    for case_id in cases:
        print(f"F0.5 {case_id} validating archived float target", flush=True)
        archived_case = report["results"][case_id]
        case = archived_case["case"]
        scene = Scene(model, case, paths[case["scene"]], prereg["f0"], device)
        endpoint = Endpoint(scene, prereg["f0"])
        recorded = archived_case["doses"][str(settings["dose_gl"])]["output_free"]
        expect_witness = case_id in config["witness_cases"]
        if recorded["structure_recovered"] != expect_witness:
            raise ValueError(f"Unexpected F0 witness status: {case_id}")
        image, audit = restore_witness(scene, endpoint, archive, recorded, settings)
        folder = output / case_id
        folder.mkdir()
        save_image(folder / "E2.png", scene.baseline)
        save_image(folder / "fixed_witness.png", image)
        np.savez_compressed(folder / "fixed_witness_float.npz",
                            baseline=scene.baseline.cpu().numpy(), witness=image.cpu().numpy(),
                            delta_gl=(255 * (image - scene.baseline)).cpu().numpy())
        arms = {}
        for arm, (value, render) in layer_spaces(model, scene).items():
            print(f"  {arm}", flush=True)
            space = LocalSpace(value, render, scene.readouts.region, settings["feature_radius_fraction"])
            arms[arm] = evaluate_arm(space, endpoint, image, audit["replay"], settings, folder, arm)
            for mode, row in arms[arm]["fits"].items():
                rows.append(dict(case=case_id, arm=arm, fit=mode, **row, **row["metrics"]))
        all_results[case_id] = dict(witness_status="recovered" if expect_witness else "control",
                                   target=audit, first_reachable=first_reachable(arms), arms=arms)
        write_json(folder / "case_result.json", all_results[case_id])
    counts = dict(cases=len(cases), verified_witnesses=sum(key in config["witness_cases"] for key in cases),
                  reproduced_witnesses=sum(row["first_reachable"] is not None for row in all_results.values()))
    write_json(output / "f05_attribution.json", dict(results=all_results, registry=counts))
    from maintain_experiment_index import PAPER_METRICS
    csv_rows(output / "f05_comparison.csv", rows,
             ["case", "arm", "fit", "fit_rmse_gl", "relative_fit_error", "target_fit_rmse_gl",
              "witness_hit_overlap", "witness_reproduced", "reproduced_under_fit_constraints",
              "structure_recovered", "rms_gl", "peak_gl",
              "protected_rms_gl", "far_rms_gl", "valid_points", "longest", "violations", *PAPER_METRICS])
    print(f"F0.5 results: {output / 'f05_attribution.json'}", flush=True)
    try:
        register(output, metadata, len({report["results"][name]["case"]["scene"] for name in cases}),
                 "f0.5", "f05_attribution.json", counts, version="V2")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。实验索引校验失败，请查看上方错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_v2_f05.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--case")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
