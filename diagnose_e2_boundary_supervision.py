"""Measure baseline E2 loss preference for the archived 546-A boundary."""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnose_e2_detail_route_localization import (
    DATA, frozen_checkpoint, locate_scene, read_pair, source_manifest, to_tensor,
)
from maintain_experiment_index import sha256_file
from run_e2_scratch_twostage import ORIGINAL_LOSS_PILOT, training_args
from train_direct_fusion import DirectFusionLosses


ROOT = Path(__file__).resolve().parent
LABELS = Path("test_result/e2_contour_evidence_v1/holdout_labels/source_labels.csv")
DOSES = (-4, -2, -1, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125,
         0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32)


class BoundaryProbe:
    def __init__(self, image, label):
        start = image.new_tensor([float(label["x1"]), float(label["y1"])])
        end = image.new_tensor([float(label["x2"]), float(label["y2"])])
        segment = end - start
        length = segment.norm()
        tangent = segment / length
        self.normal = torch.stack((-tangent[1], tangent[0]))
        height, width = image.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, device=image.device),
                                torch.arange(width, device=image.device), indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).to(image.dtype) - start
        along = coords @ tangent
        self.across = coords @ self.normal
        self.taper = (along / 6).clamp(0, 1) * ((length - along) / 6).clamp(0, 1)
        self.sample_position = ((along / length - 0.08) / 0.84 * 16).clamp(0, 16)
        self.center = start + torch.linspace(0.08, 0.92, 17, device=image.device)[:, None] * segment
        self.scale = image.new_tensor([width - 1, height - 1])
        self.sign = self.contrast(image).sign()

    def contrast(self, image, offset=0):
        points = self.center[None] + image.new_tensor([offset + 3, offset - 3])[:, None, None] * self.normal
        grid = (2 * points / self.scale - 1)[None]
        profiles = F.grid_sample(image, grid, align_corners=True)[0, 0]
        return profiles[0] - profiles[1]

    def score(self, image):
        return (255 * self.sign * self.contrast(image)).mean()

    def direction(self, baseline, offset, half_width, local_polarity):
        across = (self.across - offset) / half_width
        footprint = across * (1 - across.square()).clamp_min(0) * self.taper
        if local_polarity:
            low = self.sample_position.floor().long()
            high = (low + 1).clamp_max(16)
            polarity = torch.lerp(self.sign[low], self.sign[high], self.sample_position - low)
        else:
            polarity = self.sign.sum().sign()
        # Use symmetric headroom so both signs of the declared sweep stay in range.
        headroom = torch.minimum(baseline, 1 - baseline)[0, 0] * 255 / max(DOSES)
        raw = polarity * footprint
        raw = raw / raw.abs().max().clamp_min(1e-12)
        positive = torch.minimum(raw.clamp_min(0), headroom)
        negative = torch.minimum((-raw).clamp_min(0), headroom)
        mass = torch.minimum(positive.sum(), negative.sum())
        if not mass.item():
            raise ValueError("The perturbation has no feasible zero-mean direction")
        return ((positive / positive.sum() - negative / negative.sum()) * mass)[None, None]

    def optical_input_probe(self, optical):
        axis = torch.arange(-6, 7, device=optical.device, dtype=optical.dtype)
        kernel = torch.exp(-0.5 * (axis / 2).square())
        kernel /= kernel.sum()
        blurred = F.conv2d(F.pad(optical, (6, 6, 6, 6), mode="reflect"),
                          (kernel[:, None] * kernel[None, :])[None, None])
        footprint = torch.exp(-0.5 * (self.across / 3.5).square()) * self.taper
        headroom = torch.minimum(optical, 1 - optical)
        return (0.25 * footprint * (optical - blurred)).clamp(-headroom, headroom)


def trace(model, optical_detail, sar_detail, base):
    head = model.detail_fusion
    op, sp, bp = head._project_inputs(optical_detail, sar_detail, base)
    joined = torch.cat((op, sp), dim=1)
    aggregate = torch.cat((op, sp, *(branch(joined) for branch in head.scale_branches), bp), dim=1)
    mix = head.output[0](aggregate)
    activated = head.output[1](mix)
    spatial = head.output[2]
    center_weight = torch.zeros_like(spatial.weight)
    center_weight[:, :, 1, 1] = spatial.weight[:, :, 1, 1]
    center = F.conv2d(activated, center_weight, padding=1)
    neighbor = F.conv2d(activated, spatial.weight - center_weight, spatial.bias, padding=1)
    detail = center + neighbor
    decoder = model.decoder
    decoder_input = torch.cat((base, detail), dim=1)
    reduced = decoder.reduce_channel(decoder_input)
    image = decoder.sigmoid(decoder.output(decoder.encoder_level2(reduced)))
    return dict(aggregate_1x1_input=aggregate, aggregate_1x1_output=mix,
                aggregate_gelu_output=activated, center=center, neighbor=neighbor,
                fused_detail=detail, decoder_input=decoder_input,
                decoder_reduce_output=reduced, image=image)


def dot(a, b):
    return torch.dot(a.flatten(), b.flatten()).item()


def gradient_record(gradient, recovery, response):
    scale = (gradient.norm() * recovery.norm()).item()
    return {
        "cosine_loss_descent_vs_recovery_ascent": -dot(gradient, recovery) / scale if scale else None,
        "loss_slope_along_optical_detail_probe": dot(gradient, response),
        "score_slope_along_optical_detail_probe": dot(recovery, response),
    }


def layer_attribution(stages, objectives, recovery, optical_response):
    nodes = list(stages.values())
    recovery_gradients = torch.autograd.grad(recovery, nodes, retain_graph=True)
    results = {}
    for name, loss in objectives.items():
        gradients = torch.autograd.grad(loss, nodes, retain_graph=True)
        results[name] = {
            key: gradient_record(g, r, optical_response[key])
            for key, g, r in zip(stages, gradients, recovery_gradients)
            if key not in ("center", "neighbor")
        }
        # Compare each kernel's pullback to their shared GELU output.
        # The raw center/neighbor outputs have identical downstream derivatives.
        activated = stages["aggregate_gelu_output"]
        for key in ("center", "neighbor"):
            index = list(stages).index(key)
            pullback = torch.autograd.grad(stages[key], activated, gradients[index], retain_graph=True)[0]
            recovery_index = list(stages).index("aggregate_gelu_output")
            results[name][key + "_kernel_pullback"] = gradient_record(
                pullback, recovery_gradients[recovery_index], optical_response["aggregate_gelu_output"])
            results[name][key + "_kernel_pullback"]["score_slope_along_optical_detail_probe"] = dot(
                recovery_gradients[index], optical_response[key])
    return results


def sweep(baseline, direction, probe, loss_maps, output_gradients):
    base_maps = loss_maps(baseline)
    rows = []
    for dose in DOSES:
        delta = dose * direction / 255
        edited = baseline + delta
        if edited.min() < -1e-7 or edited.max() > 1 + 1e-7:
            raise ValueError("Perturbation left the valid image range")
        losses = loss_maps(edited)
        changes = {name: (losses[name] - base_maps[name]).mean().item() for name in losses}
        changes["fusion"] = sum(changes.values())
        rows.append(dict(dose=dose, contrast_gl=probe.score(edited).item(),
                         aligned_sample_count=int(((probe.contrast(edited) * probe.sign) > 0).sum()),
                         mean_change_gl=(255 * (edited - baseline).double().mean()).item(),
                         delta_rms_gl=(255 * delta.square().mean().sqrt()).item(),
                         loss_delta=changes))
    return dict(support_pixels=int(direction.count_nonzero()),
                peak_gl_per_dose=direction.abs().max().item(),
                contrast_slope_gl_per_dose=probe.score(direction / 255).item(),
                loss_slope_per_dose={name: dot(g, direction / 255)
                                     for name, g in output_gradients.items()}, sweep=rows)


def run(device):
    model, checkpoint = frozen_checkpoint(device)
    with LABELS.open(encoding="utf-8-sig", newline="") as file:
        label = next(row for row in csv.DictReader(file) if row["scene"] == "546" and row["id"] == "A")
    optical, sar = [to_tensor(im, device) for im in read_pair(locate_scene(DATA, source_manifest(), "546"))]
    probe = BoundaryProbe(optical, label)
    losses = DirectFusionLosses().to(device)
    weight = training_args(ORIGINAL_LOSS_PILOT).gradient_weight
    valid = sar > 0
    intensity_target = torch.where(valid, torch.maximum(optical, sar), optical).double()
    optical_gradient = losses.sobel(optical)
    gradient_target = torch.where(valid, torch.maximum(optical_gradient, losses.sobel(sar)),
                                  optical_gradient).double()

    def objective(image):
        total, intensity, gradient = losses.fusion(optical, sar, image, valid, weight)
        return dict(intensity=intensity, weighted_gradient=weight * gradient, fusion=total)

    def loss_maps(image):
        return dict(intensity=(image.double() - intensity_target).abs(),
                    weighted_gradient=weight * (losses.sobel(image).double() - gradient_target).abs())

    with torch.no_grad():
        bo, do, bs, ds = model.encode_pair(optical, sar)
        base, detail = model.fuse_features(bo, do, bs, ds)
        baseline = model.reconstruct(optical, base, detail)
        reconstruction = model.reconstruct(optical, bo, do)
        # Two-sided optical input probe for the stagewise Detail path, with Base fixed.
        delta = probe.optical_input_probe(optical)
        _, plus, _, _ = model.encode_pair(optical + delta, sar)
        _, minus, _, _ = model.encode_pair(optical - delta, sar)
        pos = trace(model, plus, ds, base)
        neg = trace(model, minus, ds, base)
        response = {key: (pos[key] - neg[key]) / 2 for key in pos}

    stages = trace(model, do.requires_grad_(), ds, base)
    torch.testing.assert_close(stages["image"], baseline, rtol=0, atol=2e-6)
    torch.testing.assert_close(stages["fused_detail"], detail, rtol=0, atol=2e-6)
    objectives = objective(stages["image"])
    output_gradients = {name: torch.autograd.grad(value, stages["image"], retain_graph=True)[0]
                        for name, value in objectives.items()}
    attribution = layer_attribution(stages, objectives, probe.score(stages["image"]), response)
    with torch.no_grad():
        for name, loss_map in loss_maps(baseline).items():
            torch.testing.assert_close(loss_map.mean().float(), objectives[name], rtol=1e-6, atol=1e-8)
        directions = {
            "aligned": probe.direction(baseline, 0, 6, True),
            "aligned_narrow": probe.direction(baseline, 0, 4.5, True),
            "shifted_12px": probe.direction(baseline, 12, 6, True),
            "majority_polarity": probe.direction(baseline, 0, 6, False),
        }
        aligned = directions["aligned"]
        shifted = directions["shifted_12px"]
        matched_norm = torch.minimum(aligned.norm(), shifted.norm())
        directions["aligned"] = aligned * (matched_norm / aligned.norm())
        directions["shifted_12px"] = shifted * (matched_norm / shifted.norm())
        sweeps = {name: sweep(baseline, direction, probe, loss_maps, output_gradients)
                  for name, direction in directions.items()}
    return dict(scene="546-A", label=label, checkpoint=checkpoint.as_posix(),
                checkpoint_sha256=sha256_file(checkpoint), device=str(device),
                gradient_weight=weight, static_contrast_gl={
                    name: probe.score(image).item() for name, image in
                    (("optical", optical), ("sar", sar), ("optical_reconstruction", reconstruction),
                     ("fused", baseline))},
                base_losses={name: value.item() for name, value in objectives.items()},
                baseline_aligned_sample_count=int(((probe.contrast(baseline) * probe.sign) > 0).sum()),
                samples=int(probe.sign.numel()),
                reconstruction_weight=training_args(ORIGINAL_LOSS_PILOT).reconstruction_weight,
                optical_probe_input_contrast_gl=probe.score(delta).item(),
                output_interventions=sweeps, detail_path_attribution=attribution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
