"""Automatic source proposals, source deletion, and frozen R1 target fitting."""

from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diagnose_dcr_v2_f05 import LocalSpace, RAY, linear_direction
from diagnose_e2_boundary_supervision import BoundaryProbe

from .atoms import rms


SOURCES = ("optical", "sar")
METHODS = ("D1-O", "D1-S", "D1-Max", "D1-Gap", "D1-C", "D1-C-shuffle")


class Line:
    def __init__(self, image, segment, origin):
        self.origin = origin
        self.geometry = dict(zip(("x1", "y1", "x2", "y2"), map(int, segment)))
        self.probe = BoundaryProbe(image, self.geometry)
        self.support = (self.probe.taper > 0) & (self.probe.across.abs() < 7)
        self.evidence = {}
        xy = self.probe.center[None] + image.new_tensor([3, -3])[:, None, None] * self.probe.normal
        lower = xy.floor()
        fraction = xy - lower
        x, y = lower.long().unbind(dim=-1)
        wx, wy = fraction.unbind(dim=-1)
        width = image.shape[-1]
        self.indices = torch.stack((y * width + x, y * width + x + 1,
                                    (y + 1) * width + x, (y + 1) * width + x + 1))
        self.weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy))

    def contrast(self, image, offset=0):
        if offset:
            return self.probe.contrast(image, offset)
        profile = (image.flatten()[self.indices] * self.weights).sum(0)
        return profile[0] - profile[1]


@torch.no_grad()
def proposal_trace(images, settings):
    """Trace the original per-source detector through its full ranked NMS pool."""
    traced = {}
    for source, image in zip(SOURCES, images):
        pixels = (image[0, 0] * 255).round().byte().cpu().numpy()
        smooth = cv2.GaussianBlur(pixels, (0, 0), 1.0)
        edges = cv2.Canny(smooth, 30, 80, L2gradient=True)
        cv2.setRNGSeed(42)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 17, minLineLength=22, maxLineGap=4)
        records, ranked = [], []
        for segment in [] if lines is None else lines[:, 0]:
            record = dict(id=f"{source}-{len(records)}", segment=[int(v) for v in segment],
                          score=None, score_rank=None, nms_rank=None, suppressed_by=None)
            records.append(record)
            if np.any(segment < 10) or np.any(segment > 245):
                record["status"] = "border"
                continue
            line = Line(image, segment, source)
            score, _, _ = source_profile(line, image, settings["source_scale_gl"][source])
            record["score"] = score
            record["line"] = line
            record["status"] = "below_threshold"
            if score >= settings["proposal_threshold"]:
                ranked.append((score, tuple(segment), record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        scored = sorted((row for row in records if row["score"] is not None),
                        key=lambda row: (-row["score"], tuple(row["segment"])))
        for rank, record in enumerate(scored, 1):
            record["raw_score_rank"] = rank
        selected = []
        for rank, (_, _, record) in enumerate(ranked, 1):
            record["score_rank"] = rank
            line = record["line"]
            suppressor = next((other for other in selected
                               if float((line.support & other["line"].support).sum()) >
                               0.65 * min(int(line.support.sum()), int(other["line"].support.sum()))), None)
            if suppressor:
                record.update(status="nms_suppressed", suppressed_by=suppressor["id"])
                continue
            record.update(status="retained", nms_rank=len(selected) + 1)
            selected.append(record)
        traced[source] = records
    return traced


def proposals(images, settings):
    """Take the same per-source Top-K from the traced detector."""
    traced = proposal_trace(images, settings)
    return [record["line"] for source in SOURCES
            for record in sorted((row for row in traced[source] if row["nms_rank"] is not None),
                                 key=lambda row: row["nms_rank"])[:settings["max_candidates_per_source"]]]


def source_profile(line, image, scale):
    smooth = F.avg_pool2d(image, 5, stride=1, padding=2, count_include_pad=False)
    offsets = (-4, -2, 0, 2, 4)
    raw = torch.stack([line.contrast(image, offset) for offset in offsets]) * 255
    coarse = torch.stack([line.contrast(smooth, offset) for offset in offsets]) * 255
    midpoint = raw.shape[1] // 2
    index = int(raw.abs().sort(dim=1).values[:, midpoint].argmax())
    coarse_index = int(coarse.abs().sort(dim=1).values[:, midpoint].argmax())
    vector = raw[2]
    sign = vector.median().sign()
    support = float(((vector * sign) > 2).float().mean())
    stability = 1 / (1 + abs(offsets[index] - offsets[coarse_index]) + abs(offsets[index]) / 2)
    strength = float(vector.abs().median()) / scale
    return min(strength, 2.) * support * stability, vector, offsets[index]


@torch.no_grad()
def inspect(model, optical, sar, settings, fixed=None):
    encoded = model.encode_pair(optical, sar)
    base, detail = model.fuse_features(*encoded)
    sources = (encoded[1], encoded[3])
    decode = lambda value: model.reconstruct(optical, base, value)
    baseline = decode(detail)
    if not torch.equal(baseline, model(optical, sar)):
        raise ValueError("Frozen native E2 differs from zero-edit replay")
    candidates = proposals((optical, sar), settings)
    if fixed is not None:
        candidates.insert(0, Line(optical, [fixed[key] for key in ("x1", "y1", "x2", "y2")], "fixed"))
    smooth_details = [F.avg_pool2d(value, 7, stride=1, padding=3, count_include_pad=False) for value in sources]
    for line in candidates:
        original = 255 * line.contrast(baseline)
        taper = line.probe.taper * (1 - line.probe.across.abs() / 7).clamp(0, 1)
        for index, (name, image) in enumerate(zip(SOURCES, (optical, sar))):
            scale = settings["source_scale_gl"][name]
            strength, readout, offset = source_profile(line, image, scale)
            desired = 24 * readout / scale
            target = (desired - original).clamp(-settings["max_target_change_gl"],
                                                settings["max_target_change_gl"])
            removal = (smooth_details[index] - sources[index]) * taper
            contribution = []
            for alpha in settings["deletion_strengths"]:
                inputs = list(sources)
                inputs[index] = inputs[index] + alpha * removal
                deleted = decode(model.detail_fusion(*inputs, base))
                delta = original - 255 * line.contrast(deleted)
                contribution.append(dict(alpha=alpha, signed_gl=float((delta * readout.sign()).mean()),
                                         contrast_rms_gl=float(rms(delta)),
                                         local_feature_relative_rms=float(
                                             rms((alpha * removal)[..., line.support]) /
                                             rms(sources[index][..., line.support]).clamp_min(1e-8))))
            line.evidence[name] = dict(strength=strength, offset_px=offset, sign=float(readout.median().sign()),
                                       native_gl=readout.tolist(), target_change_gl=target.tolist(),
                                       gap=float(rms(target)) / settings["max_target_change_gl"],
                                       contribution=contribution)
        left, right = (line.evidence[source] for source in SOURCES)
        distance = abs(left["offset_px"] - right["offset_px"])
        confirmed = min(left["strength"], right["strength"]) >= settings["proposal_threshold"]
        line.relation = dict(distance_px=distance, compatibility=(1 - min(distance / 4, 1)) if confirmed else 0.,
                             relative_polarity=left["sign"] * right["sign"], both_supported=confirmed)
    return baseline, detail, decode, candidates


def choose(candidates, method, settings, shuffled, only_index=None, force=False):
    scored = []
    for index, line in enumerate(candidates):
        if only_index is not None and index != only_index:
            continue
        relation = candidates[shuffled[index]].relation if method == "D1-C-shuffle" else line.relation
        local = []
        for source in SOURCES:
            if method in ("D1-O", "D1-S") and source != SOURCES[method == "D1-S"]:
                continue
            e = line.evidence[source]
            if (not force and e["strength"] < settings["proposal_threshold"]) or not e["sign"]:
                continue
            score = e["strength"]
            if method == "D1-Gap":
                score = e["gap"]
            if method in ("D1-C", "D1-C-shuffle"):
                contributions = [trial["signed_gl"] / trial["alpha"] for trial in e["contribution"]]
                coherent = all(value > 0 for value in contributions)
                already_used = min(1, min(contributions) / 24) if coherent else 0
                compatibility = relation["compatibility"]
                geometry = 1 + 0.35 * compatibility if relation["both_supported"] else 1
                if relation["both_supported"] and relation["distance_px"] >= 4:
                    geometry *= 0.65
                score = e["strength"] * e["gap"] * geometry * (1 - 0.5 * already_used)
            local.append((score, -index, source, line))
        local.sort(key=lambda x: x[:3], reverse=True)
        # Keep opposite-polarity hypotheses distinct. Low-margin ambiguity abstains.
        if (not force and method in ("D1-C", "D1-C-shuffle") and len(local) == 2 and relation["both_supported"]
                and relation["relative_polarity"] < 0 and local[0][0] - local[1][0] < settings["conflict_margin"]):
            continue
        scored.extend(local[:1])
    if not scored:
        return None
    score, _, source, line = max(scored, key=lambda item: item[:3])
    return (line, source, score) if force or score >= settings["decision_floor"] else None


class OutputGuard:
    def __init__(self, candidates, selected, baseline, settings):
        self.lines = [line for line in candidates if line is not selected and
                      not (line.support & selected.support).any()]
        self.mask = torch.zeros_like(selected.support)
        for line in self.lines:
            self.mask |= line.support
        self.far = ~(selected.support | self.mask)
        self.baseline, self.settings = baseline, settings

    def signals(self, image):
        delta = image - self.baseline
        gray = 255 * delta
        peak = gray.abs().amax()
        def stable_rms(tensor):
            return tensor.square().mean().clamp_min(1e-16).sqrt()

        protected = stable_rms(gray[..., self.mask]) if self.mask.any() else peak * 0
        far = stable_rms(gray[..., self.far]) if self.far.any() else peak * 0
        reads = [255 * line.contrast(delta).abs().amax() for line in self.lines]
        readout = torch.stack(reads).amax() if reads else peak * 0
        return torch.stack((stable_rms(gray), peak, protected, far, readout))


def render_target(detail, decode, baseline, candidates, selection, dose, settings,
                  desired_override=None, solver="d1", full_image=False):
    if selection is None:
        return baseline, dict(status="abstained", rms_gl=0., peak_gl=0., rejections={})
    line, source, score = selection
    desired = (detail.new_tensor(line.evidence[source]["target_change_gl"]) if desired_override is None
               else desired_override)
    space = LocalSpace((detail,), decode, line.support, settings["feature_radius_fraction"])
    guard = OutputGuard(candidates, line, baseline, settings)
    bounds = detail.new_tensor((dose, settings["peak_gl"], settings["protection_rms_gl"],
                               settings["far_rms_gl"], settings["protect_readout_gl"]))
    # The F0.5 normal-equation solver targets the selected 17-point readout.
    def read_delta(image):
        delta = image - baseline
        return delta if full_image else line.contrast(delta)

    def read_space(vector):
        return read_delta(space(vector))

    read_space.zero = space.zero
    read_space.project = space.project
    start, linear = linear_direction(read_space, desired, settings["linear_steps"], settings["ridge"])
    baseline_error = float(rms(desired))
    rejected = Counter()
    names = ("rms_gl", "peak_gl", "protected_rms_gl", "far_rms_gl", "max_protection_change_gl")

    def replay(direction, limits):
        best = (baseline_error, baseline, space.zero, guard.signals(baseline).tolist())

        @torch.no_grad()
        def record(image, vector):
            nonlocal best
            signal = guard.signals(image)
            failed = signal > limits + 1e-5
            rejected.update(name for name, flag in zip(names, failed.tolist()) if flag)
            error = float(rms(255 * read_delta(image) - desired))
            if not failed.any() and error < best[0] - 1e-5:
                best = error, image.detach(), vector.detach().clone(), signal.tolist()

        for alpha in RAY:
            with torch.no_grad():
                vector = alpha * direction
                record(space(vector), vector)
        param = best[2].clone().requires_grad_()
        opt = torch.optim.Adam([param], lr=settings["learning_rate"])
        for _ in range(settings["nonlinear_steps"]):
            image = space(param)
            error = 255 * read_delta(image) - desired
            ratio = guard.signals(image) / limits
            if solver == "r1":
                # F0.5's squared fit and squared RMS constraint ratios.
                over = (torch.stack((ratio[0].square(), ratio[1], ratio[2].square(),
                                     ratio[3].square(), ratio[4])) - 1).clamp_min(0)
                objective = error.square().mean() / dose**2
            else:
                over = (ratio - 1).clamp_min(0)
                objective = error.abs().mean() / settings["max_target_change_gl"]
            loss = objective + 10 * over.square().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            space.project(param)
            with torch.no_grad():
                record(space(param), param)
        return best

    best = replay(start, bounds)
    passes = {"strict": best[0]}
    if solver == "r1":
        relaxed_limits = bounds.clone()
        relaxed_limits[1] = torch.inf
        relaxed = replay(start, relaxed_limits)
        restored = replay(relaxed[2], bounds)
        passes.update(peak_relaxed=relaxed[0], restored=restored[0])
        best = min(best, restored, key=lambda item: item[0])
    return best[1], dict(status="edited" if best[0] < baseline_error else "no_feasible_gain",
                         source=source, line=line.geometry, decision_score=score,
                         target_error_gl=best[0], initial_target_error_gl=baseline_error,
                         automatic_protection_lines=[item.geometry for item in guard.lines],
                         rejections=dict(rejected), linear=linear, solver=solver,
                         optimizer_fit_rmse_by_pass=passes, **dict(zip(names, best[3])))
