"""Frozen E2 diagnostic: paired optical/SAR Detail subspaces and source-loss gradients.

No parameter updates. Only locked train1024/val64 validation scenes, never test.
Subspace test: same-pixel paired vs same-scene rolled vs different-scene SAR.
Gradient test: EXACTLY partition the original maximum-intensity/maximum-Sobel
fusion supervision by the modality that supplies the winner (ties: optical).
This is not a new loss or a PCGrad training implementation.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from diagnose_e2_information_path import locate_scene, read_pair, to_tensor
from e2_base_anchor import OpticalBaseAnchoredE2
from maintain_experiment_index import sha256_file
from run_e2_local_correspondence import source_manifest
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, training_args
from run_qxs_reference_test1000 import DATA
from train_direct_fusion import DirectFusionLosses


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("test_result/e2_subspace_gradient_diagnosis_v1/val16_seed42")
GROUPS = ("optical_projection", "sar_projection", "scale_branches",
          "base_projection", "output")
RIDGE = 0.01
GRID_STEP = 4


def rms(x):
    return float(x.detach().float().square().mean().sqrt())


def feature_grid(feature, step=GRID_STEP):
    """Subsample spatial positions in SAME coordinates; no resizing."""
    return (feature[0, :, ::step, ::step].permute(1, 2, 0)
            .reshape(-1, feature.shape[1]).detach().cpu().double())


def valid_grid(sar, step=GRID_STEP):
    return ((sar[0, 0, ::step, ::step] > 0).flatten()
            .detach().cpu().bool())


def roll_grid(values, height, width):
    return values.reshape(height, width, -1).roll(
        (height // 2, width // 2), dims=(0, 1)).reshape_as(values)


def cca_spectrum(x, y, mask=None, ridge=RIDGE):
    """Ridge-whitened CCA, condition on one shared set of spatial positions.

    Spectrum is a diagnostic correlation statistic, not structure correctness.
    """
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("Expected paired [positions, channels] features")
    if mask is not None:
        if mask.shape != (x.shape[0],):
            raise ValueError("Mask length differs from sampled positions")
        x, y = x[mask], y[mask]
    if x.shape[0] < max(32, 2 * x.shape[1]):
        raise ValueError("Too few valid pixels for CCA")
    if not 0 < ridge <= 1:
        raise ValueError("Invalid ridge")
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    n = x.shape[0]
    sx = x.T @ x / n
    sy = y.T @ y / n
    cross = x.T @ y / n
    def inv_root(a):
        lam, q = torch.linalg.eigh(a)
        floor = ridge * torch.trace(a).clamp_min(1e-12) / a.shape[0]
        return (q * (lam + floor).clamp_min(1e-12).rsqrt()[None, :]) @ q.T
    values = torch.linalg.svdvals(inv_root(sx) @ cross @ inv_root(sy))
    values = values.clamp(0, 1)
    return values.detach().cpu()


def spectrum_stats(x, y, mask):
    spectra = cca_spectrum(x, y, mask)
    top = min(8, len(spectra))
    return {
        "top1": float(spectra[0]),
        "top8_mean": float(spectra[:top].mean()),
        "top8_min": float(spectra[top - 1]),
        "above_0_5": int((spectra > 0.5).sum()),
        "spectrum": [float(v) for v in spectra],
        "valid_positions": int(mask.sum()),
    }


def loss_partition(optical, sar, fused, valid, losses, gradient_weight):
    """Exact hard-max original E2 fusion objective split by source winner.

    L_o+L_s equals DirectFusionLosses.fusion(...)[0]; no alternative
    source-specific gradients or new weighting are introduced.
    """
    o, s, f = optical.float(), sar.float(), fused.float()
    m = valid.float() > 0.5
    intensity_sar = m & (s > o)
    intensity_opt = ~intensity_sar
    grad_o, grad_s = losses.sobel(o), losses.sobel(s)
    gradient_sar = m & (grad_s > grad_o)
    gradient_opt = ~gradient_sar
    intensity_target = torch.where(intensity_sar, s, o)
    gradient_target = torch.where(gradient_sar, grad_s, grad_o)
    intensity_error = (f - intensity_target).abs()
    gradient_error = (losses.sobel(f) - gradient_target).abs()
    if intensity_error.shape != intensity_opt.shape or gradient_error.shape != gradient_opt.shape:
        raise ValueError("Unmatched original loss / winner mask shapes")
    o_int = (intensity_error * intensity_opt).mean()
    s_int = (intensity_error * intensity_sar).mean()
    o_grad = (gradient_error * gradient_opt).mean()
    s_grad = (gradient_error * gradient_sar).mean()
    optical_loss = o_int + gradient_weight * o_grad
    sar_loss = s_int + gradient_weight * s_grad
    diagnostics = dict(
        optical_intensity=float(o_int.detach()),
        sar_intensity=float(s_int.detach()),
        optical_weighted_gradient=float((gradient_weight * o_grad).detach()),
        sar_weighted_gradient=float((gradient_weight * s_grad).detach()),
        sar_intensity_winner_fraction=float(intensity_sar.float().mean()),
        sar_gradient_winner_fraction=float(gradient_sar.float().mean()),
        valid_sar_fraction=float(m.float().mean()),
    )
    return optical_loss, sar_loss, diagnostics


def pair_grads(params, loss_o, loss_s):
    a = torch.autograd.grad(loss_o, params, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(loss_s, params, allow_unused=True)
    return a, b


def grouped_gradient_stats(named_params, gradients_o, gradients_s, group):
    if group == "all_detail":
        positions = list(range(len(named_params)))
    else:
        positions = [j for j, (name, _) in enumerate(named_params)
                     if name.startswith(group + ".")]
    if not positions:
        raise ValueError("Empty parameter group: " + group)
    dot, a2, b2, present = 0.0, 0.0, 0.0, 0
    for j in positions:
        ga, gb = gradients_o[j], gradients_s[j]
        if ga is None and gb is None:
            continue
        ga = (torch.zeros_like(named_params[j][1]) if ga is None else ga.detach())
        gb = (torch.zeros_like(named_params[j][1]) if gb is None else gb.detach())
        dot += float((ga.double() * gb.double()).sum())
        a2 += float(ga.double().square().sum())
        b2 += float(gb.double().square().sum())
        present += 1
    na, nb = math.sqrt(a2), math.sqrt(b2)
    return dict(group=group, active_tensors=present,
                grad_optical_norm=na, grad_sar_norm=nb,
                grad_sar_to_optical=nb / max(na, 1e-15),
                grad_cosine=dot / max(na * nb, 1e-30),
                grad_conflict=bool(dot < 0 and na > 1e-12 and nb > 1e-12))


def frozen_checkpoint(device):
    path = ORIGINAL_LOSS_PILOT.final
    if not path.is_file():
        raise FileNotFoundError(f"Missing E2 pilot checkpoint: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("variant") != OpticalBaseAnchoredE2.VARIANT or
            saved.get("epoch") != 15 or saved.get("stage") != "fusion"):
        raise ValueError("Expected archived E2 optical-Base epoch 016 fusion checkpoint")
    model = OpticalBaseAnchoredE2()
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, path


def sample_features(model, manifest, scenes, device):
    observations = []
    for index, scene in enumerate(scenes):
        optical, sar = (
            to_tensor(im, device) for im in
            read_pair(locate_scene(DATA, manifest, scene)))
        with torch.inference_mode():
            bo, detail_o, bs, detail_s = model.encode_pair(optical, sar)
        observations.append(dict(scene=str(scene),
                                 optical=feature_grid(detail_o),
                                 sar=feature_grid(detail_s),
                                 valid=valid_grid(sar),
                                 shape=tuple(detail_o.shape[-2:])))
        print(f"subspace {index + 1}/{len(scenes)}: {scene}", flush=True)
    return observations


def subspace_diagnostic(observations, step=GRID_STEP):
    rows, spectra = [], {}
    count = len(observations)
    for i, current in enumerate(observations):
        nheight, nwidth = (n // step for n in current["shape"])
        if nheight * nwidth != len(current["valid"]):
            raise ValueError("Unexpected sampled feature grid")
        other = observations[(i + 1) % count]
        for control in ("paired", "spatial_roll", "different_scene"):
            if control == "paired":
                sar, mask = current["sar"], current["valid"]
            elif control == "spatial_roll":
                sar = roll_grid(current["sar"], nheight, nwidth)
                mask = roll_grid(current["valid"].unsqueeze(-1),
                                 nheight, nwidth).squeeze(-1).bool()
            else:
                sar, mask = other["sar"], other["valid"]
            if sar.shape != current["optical"].shape:
                raise ValueError("Mismatch cross-scene sample shapes")
            result = spectrum_stats(current["optical"], sar, mask)
            spectra[f'{current["scene"]}:{control}'] = result.pop("spectrum")
            rows.append(dict(scene=current["scene"], control=control, **result))
    return rows, spectra


def gradient_diagnostic(model, manifest, scenes, device, crop, weight):
    losses = DirectFusionLosses().to(device).eval()
    rows = []
    model.detail_fusion.requires_grad_(True)
    named_params = list(model.detail_fusion.named_parameters())
    params = [p for _, p in named_params]
    for index, scene in enumerate(scenes):
        original_o, original_s = read_pair(locate_scene(DATA, manifest, scene))
        o, s = (to_tensor(im, device) for im in (original_o, original_s))
        if crop and crop < o.shape[-1]:
            top, left = (o.shape[-2] - crop) // 2, (o.shape[-1] - crop) // 2
            o = o[..., top:top + crop, left:left + crop]
            s = s[..., top:top + crop, left:left + crop]
        valid = (s > 0).float()
        with torch.no_grad():
            bo, do, bs, ds = model.encode_pair(o, s)
            fb = bo + model.base_fusion(bo, bs)
        with torch.enable_grad():
            fd = model.detail_fusion(do, ds, fb)
            fused = model.reconstruct(o, fb, fd)
            lo, ls, winner = loss_partition(
                o, s, fused, valid, losses, weight)
            total_reference = losses.fusion(o, s, fused, valid, weight)[0]
            if not torch.allclose(
                    lo + ls, total_reference, atol=1e-5, rtol=1e-5):
                raise AssertionError(
                    f"Source-partitioned losses do not equal E2 fusion loss: {scene}")
            grads_o, grads_s = pair_grads(params, lo, ls)
        residual = abs(float((lo + ls - total_reference).detach()))
        for group in ("all_detail", *GROUPS):
            stats = grouped_gradient_stats(
                named_params, grads_o, grads_s, group)
            rows.append(dict(
                scene=str(scene), patch=int(o.shape[-1]),
                original_gradient_weight=weight,
                optical_partition_loss=float(lo.detach()),
                sar_partition_loss=float(ls.detach()),
                partition_residual=residual,
                **winner, **stats))
        del fd, fused, grads_o, grads_s, lo, ls
        print(f"gradient {index + 1}/{len(scenes)}: {scene}", flush=True)
    model.detail_fusion.requires_grad_(False)
    return rows


def summarize_rows(subspace, gradient):
    stats = {}
    for field in ("top1", "top8_mean"):
        by = {name: [row[field] for row in subspace
                      if row["control"] == name]
              for name in ("paired", "spatial_roll", "different_scene")}
        stats[field] = {name: sum(vals)/len(vals) for name, vals in by.items()}
        stats[field]["paired_minus_roll"] = (
            stats[field]["paired"] - stats[field]["spatial_roll"])
        stats[field]["paired_minus_different"] = (
            stats[field]["paired"] - stats[field]["different_scene"])
    for group in ("all_detail", *GROUPS):
        rows = [row for row in gradient if row["group"] == group]
        stats["gradient_" + group] = dict(
            mean_cosine=sum(r["grad_cosine"] for r in rows) / len(rows),
            negative_cosine_count=sum(r["grad_conflict"] for r in rows),
            sample_count=len(rows),
            mean_sar_norm_ratio=sum(r["grad_sar_to_optical"] for r in rows)/len(rows))
    return stats


def write_csv(path, rows):
    if not rows:
        raise ValueError("No output rows")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    os.chdir(ROOT)
    if args.scenes < 3 or args.gradient_scenes < 1:
        raise ValueError("Need >=3 subspace scenes and >=1 gradient scene")
    if args.gradient_scenes > args.scenes:
        raise ValueError("Gradient subset must be contained in subspace subset")
    if args.grad_crop < 32 or args.grad_crop % 8:
        raise ValueError("Gradient crop must be a multiple of 8 and >=32")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = source_manifest()
    ids = [str(x) for x in manifest["splits"]["val"][:args.scenes]]
    if len(ids) != args.scenes:
        raise ValueError("Requested more than val64")
    output = args.output_dir or DEFAULT_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = frozen_checkpoint(device)
    examples = sample_features(model, manifest, ids, device)
    subspace, spectra = subspace_diagnostic(examples)
    weight = float(training_args(ORIGINAL_LOSS_PILOT).gradient_weight)
    gradient = gradient_diagnostic(
        model, manifest, ids[:args.gradient_scenes],
        device, args.grad_crop, weight)
    metrics = summarize_rows(subspace, gradient)
    provenance = dict(
        status="frozen_e2_no_optimizer_step",
        dataset="QXS-SAROPT", split="matched_pilot_val64",
        scene_ids=ids, gradient_scene_ids=ids[:args.gradient_scenes],
        input_shape=[256, 256], gradient_crop=args.grad_crop,
        checkpoint=checkpoint.as_posix(),
        checkpoint_sha256=sha256_file(checkpoint),
        script_sha256=sha256_file(Path(__file__)),
        ridge=RIDGE, grid_step=GRID_STEP, device=str(device),
        gradient_weight=weight,
        gradient_partition=(
            "original intensity and Sobel hard-max targets partitioned "
            "by strict SAR winner, ties and invalid SAR assigned optical; "
            "partitions add exactly to the original weighted fusion loss"),
        cca_note=("ridge-regularized cross-modal CCA; relative matched-vs-"
                  "shifted-vs-other statistics only; no semantic labels"),
        gradient_note=("parameter gradients of the original fusion loss only; "
                       "reconstruction loss contributes no detail_fusion gradient"),
        results=metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".subspace-gradient-", dir=output.parent) as temp:
        pending = Path(temp) / "output"
        pending.mkdir()
        write_csv(pending / "subspace.csv", subspace)
        write_csv(pending / "gradient_conflict.csv", gradient)
        (pending / "spectra.json").write_text(
            json.dumps(spectra, indent=2), encoding="utf-8")
        (pending / "summary.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2),
            encoding="utf-8")
        pending.rename(output)
    print(f"Saved: {output.resolve()}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--gradient-scenes", type=int, default=8)
    parser.add_argument("--grad-crop", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
