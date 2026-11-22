"""One fixed-scene image-target probe; optional bounded frozen-Decoder fit."""

from pathlib import Path
import tempfile

import torch

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_source_objective import SourceContrastLosses, SPEC
from maintain_experiment_index import sha256_file
from run_e2_aspr_pilot import summarize
from run_qxs_reference_test1000 import DATA, write_json
from train_direct_fusion import DirectFusionLosses
from utils.fusion_metrics import DEFAULT_PROFILE


TARGET_MAE_GRAY = 0.25
MAX_PIXEL_CHANGE_GRAY = 4.0
PROJECTION_STEPS = 8
DETAIL_RMS_FRACTION = 0.04


def bounded_image_step(image, gradient, target_mae, max_change=MAX_PIXEL_CHANGE_GRAY / 255):
    """Match output change using pixels only; never consult quality metrics."""
    if not torch.isfinite(gradient).all() or gradient.abs().max() <= 1e-12:
        raise ValueError("Image objective has no finite nonzero descent direction")
    direction = -gradient.detach() / gradient.detach().abs().max()
    limit = (image + max_change * direction).clamp(0, 1)
    available = float((limit - image).abs().mean())
    if not 0 < target_mae <= available:
        raise ValueError("Requested image effect is not reachable within the fixed pixel bound")
    lo, hi = 0.0, max_change
    for _ in range(32):
        step = (lo + hi) / 2
        result = (image + step * direction).clamp(0, 1)
        effect = float((result - image).abs().mean())
        if effect < target_mae:
            lo = step
        else:
            hi = step
    result = (image + (lo + hi) / 2 * direction).clamp(0, 1)
    effect = float((result - image).abs().mean())
    if abs(effect - target_mae) / target_mae > 0.02:
        raise ValueError("Image-effect matching failed; do not compare unmatched conditions")
    return result.detach()


def project_detail(model, base, detail, target):
    """Fit a transient feature only; no model parameter or optimizer changes."""
    base, detail, target = (value.detach() for value in (base, detail, target))
    bound = float(detail.float().square().mean().sqrt()) * DETAIL_RMS_FRACTION
    if not bound > 1e-12:
        raise ValueError("Cannot define a relative Detail perturbation on zero features")
    with torch.no_grad():
        original = model.decoder(None, base, detail)[0]
        best_image = original.detach()
        best_error = float((original - target).square().mean())
    initial_error = best_error
    delta = torch.zeros_like(detail)
    best_delta = delta.clone()
    for _ in range(PROJECTION_STEPS):
        variable = delta.detach().requires_grad_(True)
        image = model.decoder(None, base, detail + variable)[0]
        error = (image - target).square().mean()
        grad = torch.autograd.grad(error, variable)[0]
        norm = grad.float().square().mean().sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite latent-fit gradient")
        if norm <= 1e-12:
            break
        delta = (variable - (bound / 4) * grad / norm).detach()
        rms = delta.float().square().mean().sqrt()
        delta *= torch.clamp(torch.as_tensor(bound, device=delta.device) / rms.clamp_min(1e-12), max=1)
        with torch.no_grad():
            fitted = model.decoder(None, base, detail + delta)[0]
            fitted_error = float((fitted - target).square().mean())
            if fitted_error < best_error:
                best_error, best_image, best_delta = fitted_error, fitted.detach(), delta.clone()
    return best_image, dict(initial_target_mse=initial_error, fitted_target_mse=best_error,
                            detail_delta_rms=float(best_delta.float().square().mean().sqrt()),
                            detail_rms_bound=bound,
                            image_mae_gray=float((best_image - original).abs().mean()) * 255)


def probe(scene_id, output=None, *, project=False):
    from run_e2_source_objective import (
        ROOT, OUTPUT_POLICY, STATS, experiment, source_manifest, image_scores, save_images,
        register_results, read_pair, locate_scene, to_tensor,
        write_artifacts,
        archived_baseline,
    )
    from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT

    manifest = source_manifest(experiment("objective"))
    archived_baseline(manifest["splits"]["val"])
    if scene_id not in manifest["splits"]["val"]:
        raise ValueError("The probe scene must belong to the fixed val64 manifest")
    suffix = "detail" if project else "image"
    output = output or Path(f"test_result/e2_source_objective_probe_v1/scene{scene_id}_{suffix}")
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic; choose --output-dir: {output}")
    checkpoint = ORIGINAL_LOSS_PILOT.final
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("variant") != OpticalBaseAnchoredE2.VARIANT:
        raise ValueError("The diagnostic requires the archived optical-Base E2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpticalBaseAnchoredE2().to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    pair = locate_scene(DATA, manifest, scene_id)
    optical, sar = read_pair(pair)
    o, s = (to_tensor(im, device) for im in (optical, sar))
    valid = (s > 0).float()
    with torch.no_grad():
        encoded = model.encode_pair(o, s)
        base, detail = model.fuse_features(*encoded)
        initial = model.decoder(None, base, detail)[0].detach()
    criteria = {"old_target": DirectFusionLosses().to(device), "new_target": SourceContrastLosses().to(device)}
    gradients, reachable = {}, {}
    for key, criterion in criteria.items():
        variable = initial.clone().requires_grad_(True)
        value = criterion.fusion(o, s, variable, valid, 10.0)[0]
        gradient = torch.autograd.grad(value, variable)[0].detach()
        if not torch.isfinite(gradient).all() or gradient.abs().max() <= 1e-12:
            raise ValueError(f"Cannot compare a zero or nonfinite {key} direction")
        gradients[key] = gradient
        largest = (initial - MAX_PIXEL_CHANGE_GRAY / 255 * gradient / gradient.abs().max()).clamp(0, 1)
        reachable[key] = float((largest - initial).abs().mean())
    common_effect = min(TARGET_MAE_GRAY / 255, 0.9 * min(reachable.values()))
    if common_effect <= 1e-10:
        raise ValueError("Common image effect is too small for a meaningful matched probe")
    images = {"e2": initial}
    images.update({key: bounded_image_step(initial, grad, common_effect) for key, grad in gradients.items()})
    fit = None
    if project:
        images["projected_target"], fit = project_detail(model, base, detail, images["new_target"])
    metadata = dict(dataset="QXS-SAROPT", split="val", scene_ids=[scene_id], seed=42,
                    checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256_file(checkpoint),
                    split_manifest=ORIGINAL_LOSS_PILOT.manifest.as_posix(),
                    split_manifest_sha256=sha256_file(ORIGINAL_LOSS_PILOT.manifest),
                    metric_profile=DEFAULT_PROFILE, output_policy=OUTPUT_POLICY,
                    inputs_sha256=[sha256_file(path) for path in pair],
                    candidate=SPEC, common_image_mae_gray=common_effect * 255,
                    max_pixel_change_gray=MAX_PIXEL_CHANGE_GRAY, projection=fit,
                    script_sha256=sha256_file(Path(__file__)),
                    module_sha256=sha256_file(ROOT / "e2_source_objective.py"),
                    projection_steps=PROJECTION_STEPS if project else 0,
                    caveats=["No independent structure truth or automatic quality admission.",
                             "Target loss decrease is not fusion success.",
                             "Latent fit is bounded and may not match direct image-effect magnitude."])
    results, losses = {}, {}
    with torch.no_grad():
        for key, image in images.items():
            if not torch.isfinite(image).all():
                raise FloatingPointError("Non-finite diagnostic output")
            losses[key] = {name: float(criterion.fusion(o, s, image, valid, 10.0)[0])
                           for name, criterion in criteria.items()}
            losses[key]["image_mae_gray"] = float((image - initial).abs().mean()) * 255
    metadata["losses_and_effects"] = losses
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-target-probe-", dir=output.parent) as temporary:
        pending = Path(temporary) / "result"
        for key, image in images.items():
            gray, scores, sources = image_scores(image, optical, sar)
            save_images(pending / key, pair, gray, optical, sar)
            results[key] = [(scores, sources, [None] * len(STATS))]
        contrasts = (("e2", "old_target"), ("e2", "new_target"), ("old_target", "new_target"))
        if project:
            contrasts += (("e2", "projected_target"),)
        summaries = summarize(pending, [scene_id], results, contrasts, diagnostic_fields=STATS)
        write_json(pending / "run_metadata.json", metadata)
        write_json(pending / "registry_metrics.json", summaries)
        write_artifacts(pending)
        pending.rename(output)
    # Output paths distinguish repeated diagnostics with different explicit directories.
    import hashlib
    tag = hashlib.sha256(output.resolve().as_posix().encode("utf-8")).hexdigest()[:10].upper()
    register_results(output, metadata, summaries, {key: checkpoint for key in images},
                     f"E2-SOURCE-OBJECTIVE-PROBE-V1-{scene_id}-{tag}", task="engineering")
    print(f"Frozen one-scene probe: {output}; matched direct MAE={common_effect * 255:.6f} gray levels")
    print("No automatic admission: inspect independent structures and SAR preservation before training.")


if __name__ == "__main__":
    from run_e2_source_objective import main
    main()
