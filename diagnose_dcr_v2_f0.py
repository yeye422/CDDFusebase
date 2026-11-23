"""Frozen E2 structure reachability with a common native-position endpoint."""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from dcr_synth.atoms import rms
from dcr_synth.config import ROOT, freeze, load_config, read_json, write_json
from dcr_synth.evaluation import measure, pair_metrics, violations
from diagnose_dcr_v12 import Scene, csv_rows, register, save_image
from diagnose_e2_information_path import locate_scene
from diagnose_e2_subspace_gradients import frozen_checkpoint
from diagnose_fcsr_decoder import gradients, signed_transfer, source_specific_mask
from maintain_experiment_index import sha256_file


FEATURE_ARMS = ("fused_free", "optical_detail", "sar_detail", "paired_detail")
RENDERED_ARMS = ("output_free", *FEATURE_ARMS, "optical_recon_local", "sar_recon_local")


class Endpoint:
    def __init__(self, scene, settings):
        self.scene = scene
        self.settings = settings
        probe = scene.readouts.target
        offsets = torch.arange(-6, 7, device=probe.center.device, dtype=probe.center.dtype)
        positions = probe.center[None] + offsets[:, None, None] * probe.normal
        self.plus = (2 * (positions + 0.5 * probe.normal) / probe.scale - 1)[None]
        self.minus = (2 * (positions - 0.5 * probe.normal) / probe.scale - 1)[None]
        self.native = offsets.abs() <= scene.readouts.offset_limit
        self.valid_sign = probe.sign != 0
        self.sar_grad = gradients(scene.sar)
        self.sar_proxy = source_specific_mask(self.sar_grad, gradients(scene.optical), scene.sar > 0)
        self.base_row = self.measure(scene.baseline)
        self.base_row["sar_gradient_proxy"] = self.proxy(scene.baseline)
        self.verified = [index for index, item in enumerate(scene.case["protections"])
                         if item["role"] == "sar_only" and item.get("verified")]

    def proxy(self, image):
        return signed_transfer(self.sar_grad, gradients(image), self.sar_proxy)

    def normal_profiles(self, gray):
        return (F.grid_sample(gray, self.plus, align_corners=True)
                - F.grid_sample(gray, self.minus, align_corners=True))[0, 0]

    def measure(self, image):
        readouts = self.scene.readouts
        row = measure(image, self.scene.baseline, readouts)
        profile = self.normal_profiles(255 * image)
        peak_index = profile.abs().argmax(dim=0)
        signed_peak = readouts.target.sign * profile.gather(0, peak_index.unsqueeze(0)).squeeze(0)
        contrasts = readouts.target_values(255 * image)
        hits = ((contrasts >= readouts.hit) & self.native[peak_index] & (signed_peak > 0)).tolist()
        longest = current = 0
        for hit in hits:
            current = current + 1 if hit else 0
            longest = max(longest, current)
        row.update(valid_points=sum(hits), longest=longest, native_hits=hits,
                   signed_peak_gl=signed_peak.tolist())
        return row

    def geometry_loss(self, image):
        gray = 255 * image
        readouts = self.scene.readouts
        amplitude = (readouts.hit - readouts.target_values(gray)).clamp_min(0)
        profiles = self.normal_profiles(gray)
        native_signed = (profiles[self.native] * readouts.target.sign).amax(0)
        rival = torch.maximum(profiles[~self.native].abs().amax(0),
                              (-profiles[self.native] * readouts.target.sign).amax(0))
        displaced = (rival - native_signed + 0.25).clamp_min(0)
        deficit = (amplitude + displaced)[self.valid_sign]
        if not deficit.numel():
            return gray.sum() * 0
        loss = deficit.square().mean()
        if deficit.numel() > 1:
            # Adjacent failing samples both contribute to the optimization.
            loss = loss + 0.5 * torch.maximum(deficit[:-1], deficit[1:]).square().mean()
        return loss

    @torch.no_grad()
    def read(self, image, dose):
        row = self.measure(image)
        errors = violations(row, dose, self.settings)
        if row["far_rms_gl"] is not None and row["far_rms_gl"] > self.settings["max_far_rms_gl"] + 1e-5:
            errors.append("far_rms")
        row["violations"] = errors
        row["matched_to_cap"] = not errors and row["rms_gl"] >= dose - self.settings["dose_tolerance_gl"]
        row["hit_gain"] = row["valid_points"] - self.base_row["valid_points"]
        row["run_gain"] = row["longest"] - self.base_row["longest"]
        row["verified_sar_probes"] = len(self.verified)
        row["structure_recovered"] = (not errors and row["hit_gain"] >= self.settings["min_hit_gain"]
                                      and row["run_gain"] >= self.settings["min_run_gain"])
        return row


def rank(row):
    return (row["structure_recovered"], row["longest"], row["valid_points"],
            -row["target_error_gl"], -row["spill_energy_gl2"])


class Frontier:
    def __init__(self, endpoint, dose):
        self.endpoint, self.dose = endpoint, dose
        self.image = endpoint.scene.baseline
        self.row = endpoint.read(self.image, dose)
        self.rejections = Counter()
        self.steps = 0

    def consider(self, image):
        row = self.endpoint.read(image, self.dose)
        self.steps += 1
        self.rejections.update(row["violations"])
        if not row["violations"] and rank(row) > rank(self.row):
            self.image, self.row = image.detach(), row

    def result(self):
        self.row["sar_gradient_proxy"] = self.endpoint.proxy(self.image)
        self.row["candidate_count"] = self.steps
        self.row["rejection_reasons"] = dict(self.rejections)
        return self.image, self.row


def action_space(scene, arm):
    """Use the actual nonlinear source-to-fusion maps with Base frozen."""
    if arm == "output_free":
        return (scene.baseline,), lambda img: img.clamp(0, 1)
    if arm == "fused_free":
        return (scene.detail,), scene.decode
    optical, sar = scene.source_details
    if arm == "optical_detail":
        return (optical,), lambda value: scene.decode(scene.fuse(value, sar))
    if arm == "sar_detail":
        return (sar,), lambda value: scene.decode(scene.fuse(optical, value))
    return (optical, sar), lambda left, right: scene.decode(scene.fuse(left, right))


def optimize(endpoint, arm, dose):
    scene, settings = endpoint.scene, endpoint.settings
    frontier = Frontier(endpoint, dose)
    if scene.case["role"] in ("preserve", "negative") or not endpoint.valid_sign.any():
        return frontier.result()
    references, render = action_space(scene, arm)
    indices = scene.readouts.region.flatten().nonzero().flatten()
    originals = [reference.flatten(2).index_select(2, indices) for reference in references]
    radii = [settings["peak_gl"] / 255 if arm == "output_free" else
             max(float(rms(local)), 1e-7) * settings["feature_radius_fraction"] for local in originals]
    rng = torch.Generator(device=scene.detail.device).manual_seed(settings["seed"])
    ratio = settings["learning_rate"]
    for start in range(settings["starts"]):
        params = []
        for value in originals:
            initial = torch.randn(value.shape, device=value.device, generator=rng, dtype=value.dtype) if start else (
                torch.zeros_like(value))
            params.append(initial.mul_(0.1).requires_grad_())
        opt = torch.optim.Adam(params, lr=ratio)
        for step in range(settings["steps"] + 1):
            edited = [ref.flatten(2).index_copy(2, indices, local + radius * param).view_as(ref)
                      for ref, local, radius, param in zip(references, originals, radii, params)]
            image = render(*edited)
            frontier.consider(image.detach())
            if step == settings["steps"]:
                break
            change = 255 * (image - scene.baseline)
            readouts = scene.readouts
            protected = change[..., readouts.protected].square().mean()
            outside = change[..., readouts.far].square().mean()
            excess = (change.square().mean() / dose**2 - 1).clamp_min(0)
            peak = (change.abs().amax() / settings["peak_gl"] - 1).clamp_min(0)
            contrast = (readouts.protection_values(change).abs() / readouts.tau - 1).clamp_min(0)
            loss = (endpoint.geometry_loss(image) + 1000 * (excess.square() + peak.square())
                    + 50 * protected + 50 * outside + 100 * contrast.square().mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for param in params:
                    if arm == "output_free":
                        param.clamp_(-1, 1)
                    else:
                        param.mul_((1 / rms(param).clamp_min(1e-12)).clamp_max(1))
    return frontier.result()


@torch.no_grad()
def reconstruction_control(endpoint, reconstructed, dose):
    """Dose-match a local image blend with the raw, native single-source render."""
    scene = endpoint.scene
    frontier = Frontier(endpoint, dose)
    if scene.case["role"] in ("preserve", "negative"):
        return frontier.result()
    probe = scene.readouts.target
    taper = (1 - probe.across.abs() / 7).clamp(0, 1) * probe.taper
    delta = (reconstructed - scene.baseline) * taper
    gray = 255 * delta
    settings = endpoint.settings
    scale = min(1., dose / max(float(rms(gray)), 1e-12),
                settings["peak_gl"] / max(float(gray.abs().max()), 1e-12))
    for fraction in (0.25, 0.5, 0.75, 0.9, 1.0):
        frontier.consider(scene.baseline + fraction * scale * delta)
    return frontier.result()


def evaluate_scene(model, scene, folder, settings):
    endpoint = Endpoint(scene, settings)
    save_image(folder / "E2.png", scene.baseline)
    references = {}
    raw = {}
    with torch.no_grad():
        for index, source in enumerate(("optical", "sar")):
            image = getattr(scene, source)
            decoded = model.reconstruct(image, scene.source_bases[index], scene.source_details[index])
            raw[source] = decoded
            save_image(folder / f"{source}_recon_raw.png", decoded)
            references[source] = dict(endpoint.read(decoded, settings["doses_gl"][-1]),
                                      sar_gradient_proxy=endpoint.proxy(decoded),
                                      metrics=scene.score(decoded))
    results = {}
    for dose in settings["doses_gl"]:
        results[str(dose)] = {}
        for arm in RENDERED_ARMS:
            print(f"  {dose:g} {arm}", flush=True)
            if arm.endswith("_recon_local"):
                image, row = reconstruction_control(endpoint, raw[arm.split("_")[0]], dose)
            else:
                image, row = optimize(endpoint, arm, dose)
            save_image(folder / f"{arm}_{dose}.png", image)
            results[str(dose)][arm] = dict(row, metrics=scene.score(image))
    main = results[str(settings["doses_gl"][0])]
    if scene.case["role"] in ("preserve", "negative"):
        finding = "abstention_control"
    elif any(main[arm]["structure_recovered"] for arm in FEATURE_ARMS[1:]):
        finding = "source_detail_recovery_observed"
    elif main["fused_free"]["structure_recovered"]:
        finding = "fused_detail_only_recovery_observed"
    elif main["output_free"]["structure_recovered"]:
        finding = "output_witness_without_detail_recovery"
    else:
        finding = "no_output_witness_found_within_search"
    return dict(case=scene.case, baseline=dict(endpoint.base_row, metrics=scene.score(scene.baseline)),
                regional_sar_proxy_pixels=int(endpoint.sar_proxy.sum()),
                raw_source_reconstruction=references, doses=results, primary_finding=finding)


def summarize(results, settings, lock):
    primary = str(settings["doses_gl"][0])
    flattened = []
    confirmed, recovered = set(), set()
    sar_only = 0
    for case_id, result in results.items():
        case = result["case"]
        sar_only += sum(item["role"] == "sar_only" and bool(item.get("verified"))
                        for item in case["protections"])
        block = case.get("geo_block")
        independent = (case["role"] == "holdout" and block is not None
                       and case.get("verified_natural_missing") and
                       not any(other.get("geo_block") == block for other in lock["protocol"]["cases"]
                               if other["role"] != "holdout"))
        if independent:
            confirmed.add(block)
        for dose, arms in result["doses"].items():
            for arm, row in arms.items():
                flattened.append(dict(case=case_id, role=case["role"], arm=arm, dose_gl=float(dose),
                                      **row, **row["metrics"]))
            if independent and dose == primary and arms["fused_free"]["structure_recovered"]:
                recovered.add(block)
    required = settings["min_holdout_blocks"]
    if len(confirmed) < required or not sar_only:
        status = "exploratory_only"
    elif len(recovered) >= required:
        status = "fused_detail_reachability_supported"
    else:
        status = "fused_detail_reachability_not_demonstrated"
    return flattened, dict(decision=status, evaluated_holdout_blocks=len(confirmed),
                           recovered_holdout_blocks=len(recovered),
                           verified_sar_only_probes=sar_only)


def matched_metrics(results, settings):
    comparisons = {}
    for dose in settings["doses_gl"]:
        key = str(dose)
        comparisons[key] = {}
        for arm in RENDERED_ARMS:
            if arm == "fused_free":
                continue
            groups = {}
            for result in results.values():
                arms = result["doses"][key]
                lhs, rhs = arms["fused_free"], arms[arm]
                if lhs["matched_to_cap"] and rhs["matched_to_cap"] and (
                        abs(lhs["rms_gl"] - rhs["rms_gl"]) <= settings["dose_tolerance_gl"]):
                    groups.setdefault(result["case"]["role"], []).append(
                        {"fused_free": lhs["metrics"], arm: rhs["metrics"]})
            roles = {result["case"]["role"] for result in results.values()}
            comparisons[key][arm] = {role: pair_metrics(groups.get(role, []), ("fused_free", arm))
                                     for role in sorted(roles)}
    return comparisons


def run(args):
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。请使用已验证的 lightglue_env 解释器。")
    device = torch.device(args.device)
    config, protocol = load_config(args.config)
    settings = dict(protocol["settings"], **config["f0"])
    if settings["doses_gl"] != [0.245, 0.5] or min(
            settings[key] for key in ("steps", "starts", "feature_radius_fraction", "learning_rate",
                                      "max_far_rms_gl", "min_hit_gain", "min_run_gain", "min_holdout_blocks")) <= 0:
        raise SystemExit("F0 需要使用注册的主预算和扩展预算，搜索及达标参数必须为正。")
    if args.steps is not None:
        if not args.case or args.steps < 1:
            raise SystemExit("--steps 只用于单案例 smoke test，且必须为正。")
        settings["steps"] = args.steps
    output = (args.output or Path(config["output"])).resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("请指定仓库内尚未使用的输出目录。")
    cases = [case for case in protocol["cases"] if not args.case or case["id"] == args.case]
    if not cases or (args.case and cases[0]["role"] == "holdout"):
        raise SystemExit("单案例诊断需要指定已注册的开发、困难或保护案例。")
    manifest = read_json(ROOT / protocol["split_manifest"])
    paths = {case["scene"]: locate_scene(Path(protocol["data"]), manifest, case["scene"])
             for case in protocol["cases"]}
    inputs = {scene: [{"path": path.as_posix(), "sha256": sha256_file(path)} for path in pair]
              for scene, pair in paths.items()}
    lock = freeze(protocol, manifest, inputs)
    model, checkpoint = frozen_checkpoint(device)
    if checkpoint.resolve() != (ROOT / protocol["checkpoint"]).resolve():
        raise ValueError("Checkpoint differs from the registered E2 checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", dict(lock, f0=settings))
    source_paths = [Path(__file__).resolve(), ROOT / "diagnose_dcr_v12.py",
                    ROOT / "diagnose_fcsr_decoder.py", ROOT / "diagnose_e2_subspace_gradients.py",
                    ROOT / "diagnose_e2_boundary_supervision.py", ROOT / "diagnose_e2_information_path.py",
                    *(ROOT / name for name in ("direct_fusion.py", "net.py", "lwpd.py", "e2_base_anchor.py",
                                               "utils/fusion_metrics.py", "utils/reference_metrics.py",
                                               "utils/Evaluator.py")), *sorted((ROOT / "dcr_synth").glob("*.py"))]
    code = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        code[relative.as_posix()] = sha256_file(path)
    metadata = dict(stage="f0", device=str(device), torch_version=str(torch.__version__),
                    checkpoint=protocol["checkpoint"], checkpoint_sha256=lock["sha256"]["checkpoint"],
                    split_manifest=protocol["split_manifest"], split_manifest_sha256=lock["sha256"]["split_manifest"],
                    metric_profile=lock["metric_profile"], config_sha256=sha256_file(args.config),
                    protocol_sha256=sha256_file(output / "protocol.json"), case_ids=[case["id"] for case in cases],
                    smoke=args.steps is not None, code_sha256=code)
    write_json(output / "run_metadata.json", metadata)
    results = {}
    for case in cases:
        print(f"F0 {case['id']}", flush=True)
        folder = output / case["id"]
        folder.mkdir()
        scene = Scene(model, case, paths[case["scene"]], settings, device)
        results[case["id"]] = evaluate_scene(model, scene, folder, settings)
    rows, gate = summarize(results, settings, lock)
    counts = dict(cases=len(cases), evaluated_holdout_blocks=gate["evaluated_holdout_blocks"],
                  recovered_holdout_blocks=gate["recovered_holdout_blocks"])
    write_json(output / "f0_reachability.json", dict(results=results, registry=counts,
                                                   paired_metrics=matched_metrics(results, settings), **gate))
    from maintain_experiment_index import PAPER_METRICS
    csv_rows(output / "f0_comparison.csv", rows, ["case", "role", "arm", "dose_gl", "rms_gl", "target_rms_gl",
             "peak_gl", "far_rms_gl", "protected_rms_gl", "spill_energy_gl2", "target_error_gl", "contrast_gl",
             "valid_points", "longest", "hit_gain", "run_gain", "structure_recovered", "matched_to_cap",
             "sar_gradient_proxy", *PAPER_METRICS])
    print(f"{output / 'f0_reachability.json'} [{gate['decision']}]", flush=True)
    try:
        register(output, metadata, len({case["scene"] for case in cases}), "f0", "f0_reachability.json",
                 counts, version="V2")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"结果已保存至 {output}。实验索引校验失败，请查看上方错误。") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dcr_v2_f0.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--case")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
