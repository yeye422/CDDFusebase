"""Deterministic, equal-displacement controls for two-task GCO-E2 gradients.

Does not change training losses, use an optimizer, or require CUDA.
"""

import math
import torch

POLICIES = ("raw", "symmetric_pcgrad", "sar_one_sided", "norm_balance")


def _check_pair(optical, sar):
    if len(optical) != len(sar) or not optical:
        raise ValueError("Gradient lists must be nonempty and of equal length")
    if any(a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all()
           for a, b in zip(optical, sar)):
        raise ValueError("Gradient shapes or values are invalid")


def squared_norm(values):
    return sum(float(t.detach().double().square().sum()) for t in values)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Mismatched vector lengths")
    return sum(float((u.detach().double() * v.detach().double()).sum())
               for u, v in zip(a, b))


def vector_norm(values):
    return math.sqrt(squared_norm(values))


def generate_directions(optical, sar):
    """All policies share exactly the same two source-partitioned gradients.

    Symmetric PCGrad uses simultaneous projections of the ORIGINAL task
    gradients (not an ordered in-place mutation). SAR one-sided modifies only
    the SAR gradient; norm_balance is a reweighting-only negative control.
    """
    _check_pair(optical, sar)
    no = squared_norm(optical)
    ns = squared_norm(sar)
    if no <= 1e-24 or ns <= 1e-24:
        raise ValueError("Both source gradient norms must be nonzero")
    cross = dot(optical, sar)
    raw = [a + b for a, b in zip(optical, sar)]
    if vector_norm(raw) <= 1e-12:
        raise ValueError("Original total gradient canceled to zero")
    if cross < 0:
        o_pcgrad = [a - cross / ns * b for a, b in zip(optical, sar)]
        s_pcgrad = [b - cross / no * a for a, b in zip(optical, sar)]
        sar_corrected = s_pcgrad
    else:
        o_pcgrad = list(optical)
        s_pcgrad = list(sar)
        sar_corrected = list(sar)
    policies = {
        "raw": raw,
        "symmetric_pcgrad": [a + b for a, b in zip(o_pcgrad, s_pcgrad)],
        "sar_one_sided": [a + b for a, b in zip(optical, sar_corrected)],
        "norm_balance": [a + math.sqrt(no / ns) * b for a, b in zip(optical, sar)],
    }
    return policies


def equal_displacement(optical, sar, parameters, fraction):
    """Return directions with identical global L2 norm (= 1).

    For each direction d, update parameters as theta -= delta_norm * d,
    delta_norm=fraction*||theta||_2. No per-layer renorm, optimizer or clipping.
    """
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 0.05:
        raise ValueError("fraction must be finite in (0, 0.05]")
    if len(parameters) != len(optical):
        raise ValueError("Parameter list differs from gradient list")
    directions = generate_directions(optical, sar)
    param_norm = vector_norm(parameters)
    if param_norm <= 0:
        raise ValueError("Parameter vector is zero")
    scale = fraction * param_norm
    normalized = {}
    for name, direction in directions.items():
        length = vector_norm(direction)
        if length <= 1e-12:
            raise ValueError(f"Zero direction for policy {name}")
        normalized[name] = [v / length for v in direction]
    return scale, normalized


def direction_metadata(optical, sar, normalized):
    cosine = dot(optical, sar) / max(vector_norm(optical) * vector_norm(sar), 1e-30)
    return {
        "raw_grad_cosine": cosine,
        "sar_to_optical_gradient_norm": vector_norm(sar) / max(vector_norm(optical), 1e-30),
        "policies": {
            name: {
                "cos_to_optical": dot(vec, optical) / max(vector_norm(optical), 1e-30),
                "cos_to_sar": dot(vec, sar) / max(vector_norm(sar), 1e-30),
                "normalized_norm": vector_norm(vec),
            }
            for name, vec in normalized.items()
        },
    }


def apply_delta(parameters, normalized_direction, magnitude, original=None):
    """Apply ONE temporary virtual step. Caller restores full originals.

    No automatic restoration: use try/finally around evaluation.
    """
    if len(parameters) != len(normalized_direction):
        raise ValueError("Direction length differs from parameters")
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("Update magnitude must be finite and nonnegative")
    with torch.no_grad():
        for idx, (parameter, delta) in enumerate(zip(parameters, normalized_direction)):
            source = parameter if original is None else original[idx]
            parameter.copy_(source - magnitude * delta.to(parameter.dtype))
