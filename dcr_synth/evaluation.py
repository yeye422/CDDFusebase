"""Measurement, physical budgets, nonlinear replay, and paired comparisons."""

import numpy as np
import torch

from maintain_experiment_index import PAPER_METRICS
from utils.fusion_metrics import FusionEvaluator

from .atoms import rms
from .response import combine


def gray8(image):
    return (255 * image[0, 0].clamp(0, 1)).round().byte().cpu().numpy()


def metrics(image, optical, sar):
    fused, left, right = map(gray8, (image, optical, sar))
    values = FusionEvaluator.metric_vector(fused, left, right)
    return {name: float(value) if np.isfinite(value) else None
            for name, value in zip(PAPER_METRICS, values)}


def region_rms(change, mask):
    return float(rms(change[..., mask])) if mask.any() else None


def measure(image, baseline, readouts):
    delta = 255 * (image - baseline)
    protection = readouts.protection_values(delta)
    absolute_protection = readouts.base_protection + protection
    report = dict(readouts.points(255 * image), rms_gl=float(rms(delta)),
                  target_rms_gl=region_rms(delta, readouts.region),
                  protected_rms_gl=region_rms(delta, readouts.protected),
                  far_rms_gl=region_rms(delta, readouts.far), peak_gl=float(delta.abs().max()),
                  spill_energy_gl2=float(delta[..., ~readouts.region].square().sum()),
                  spill_pixels=int((~readouts.region).sum()),
                  target_pixels=int(readouts.region.sum()),
                  protected_pixels=int(readouts.protected.sum()),
                  far_pixels=int(readouts.far.sum()),
                  target_error_gl=float(rms(readouts.target_values(delta) - readouts.b)),
                  max_protection_readout_gl=float(protection.abs().max()) if protection.numel() else None,
                  protection_ok=bool((protection.abs() <= readouts.tau + 1e-5).all()),
                  protections=[])
    for index, (role, _, tolerance) in enumerate(readouts.protections):
        sl = slice(17 * index, 17 * (index + 1))
        report["protections"].append(dict(role=role, tolerance_gl=tolerance,
                                          baseline_gl=readouts.base_protection[sl].tolist(),
                                          values_gl=absolute_protection[sl].tolist(),
                                          changes_gl=protection[sl].tolist()))
    return report


def violations(record, dose, settings):
    return [name for name, failed in (
        ("global_rms", record["rms_gl"] > dose + 1e-5),
        ("peak", record["peak_gl"] > settings["peak_gl"] + 1e-5),
        ("target_rms", record["target_rms_gl"] > settings["target_rms_gl"] + 1e-5),
        ("protection_readout", not record["protection_ok"]),
        ("protected_rms", record["protected_rms_gl"] is not None
         and record["protected_rms_gl"] > settings["protected_rms_gl"] + 1e-5),
    ) if failed]


def linearization(image, baseline, predicted, readouts, settings):
    actual = 255 * (image - baseline)
    result = {}
    for name, mask in (("target", readouts.region), ("protection", readouts.protected),
                       ("far", readouts.far)):
        signal = region_rms(actual, mask)
        if signal is None:
            continue
        error = region_rms(predicted - actual, mask)
        relative = error / signal if signal else None
        abs_limit = settings["linear_abs_gl"][name]
        limit = settings["linear_relative"][name]
        passed = error <= abs_limit if signal <= abs_limit / limit else relative <= limit
        result[name] = dict(error_gl=error, actual_gl=signal, relative=relative, passed=passed)
    return result


@torch.no_grad()
def replay(decode, detail, atoms, candidates, baseline, readouts, dose, settings, scales=None):
    trials, best, best_row = [], None, None

    def evaluate(coefficients, scale):
        nonlocal best, best_row
        scaled = coefficients * scale
        if np.max(np.abs(scaled)) > settings["u"] + 1e-8:
            return None
        image = decode(combine(detail, atoms, scaled))
        row = measure(image, baseline, readouts)
        issues = violations(row, dose, settings)
        row.update(coefficients=scaled.tolist(), scale=scale, violations=issues)
        trials.append(row)
        if not issues:
            reaches_dose = row["rms_gl"] >= dose - settings["dose_tolerance_gl"]
            key = (reaches_dose, -row["target_error_gl"], -row["spill_energy_gl2"])
            if best is None or key > best:
                best, best_row = key, (image, row)
        return row

    if scales is None:
        scales = [1.] * len(candidates)
    for coefficients, central_scale in zip(candidates, scales):
        if not np.any(coefficients):
            evaluate(coefficients, 1)
            continue
        sampled = [evaluate(coefficients, central_scale * scale) for scale in settings["ray_scales"]]
        lower = [row["scale"] for row in sampled if row and not row["violations"]]
        if not lower:
            continue
        low = max(lower)
        upper = [row["scale"] for row in sampled if row and row["scale"] > low and row["violations"]]
        if not upper:
            continue
        high = min(upper)
        for _ in range(4):
            middle = (low + high) / 2
            row = evaluate(coefficients, middle)
            if row and not row["violations"]:
                low = middle
                if row["rms_gl"] >= dose - settings["dose_tolerance_gl"]:
                    break
            else:
                high = middle
    return best_row, trials


def pair_metrics(rows, methods):
    result = {}
    for name in PAPER_METRICS:
        common = [row for row in rows if all(row.get(method, {}).get(name) is not None
                                             for method in methods)]
        result[name] = dict(common_n=len(common), methods={
            method: dict(mean=float(np.mean(values)), std=float(np.std(values)))
            for method in methods
            if (values := [row[method][name] for row in common])
        })
    return result
