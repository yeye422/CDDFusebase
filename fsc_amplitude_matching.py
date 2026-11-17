"""Deterministic gain-matching helpers for untrained FSC conditioning probes.

The reference is a *change in SSM output*, not a target image or quality
metric. The gain is fixed on one calibration scene and reused on holdouts.
"""

import math


def rms(tensor):
    return tensor.float().square().mean().sqrt().item()


def response_stats(correct, shuffled, zero):
    """Return output response and mismatch sensitivity, without quality claims."""
    for value in (correct, shuffled, zero):
        if not value.isfinite().all():
            raise ValueError("Non-finite SSM output")
    effect = rms(correct - zero)
    mismatch = rms(correct - shuffled)
    zero_rms = rms(zero)
    return dict(correct_vs_zero_rms=effect,
                shuffled_vs_zero_rms=rms(shuffled - zero),
                correct_vs_shuffled_rms=mismatch,
                mismatch_over_effect=mismatch / max(effect, 1e-12),
                effect_over_zero=effect / max(zero_rms, 1e-12),
                zero_rms=zero_rms,
                correct_rms=rms(correct),
                max_abs_correct=float(correct.float().abs().max()))


def calibrate_gain(effect_at_gain, target, *, min_gain=0.01,
                   max_gain=512.0, tolerance=0.05, iterations=8):
    """Match RMS effect by a fixed scalar, without optimizing an SSM weight.

    effect_at_gain(g) returns RMS(output(correct_LL*g) - output(zero_LL)).
    Response may be nonlinear after the SSM's GELU merge. Iteration is
    deterministic, multiplicatively step-limited, and not presumed to converge.
    Results include whether the target was actually matched.
    """
    if not (math.isfinite(target) and target > 1e-10):
        raise ValueError("Reference LL feature effect is too small")
    if not (0 < min_gain <= 1 <= max_gain and 0 < tolerance < 1):
        raise ValueError("Invalid gain bounds or tolerance")
    base = float(effect_at_gain(1.0))
    if not (math.isfinite(base) and base > 1e-12):
        raise ValueError("Baseline state response is too weak to calibrate")
    gain = min(max(target / base, min_gain), max_gain)
    history = [(1.0, base)]
    response = base
    for _ in range(iterations):
        response = float(effect_at_gain(gain))
        if not math.isfinite(response):
            raise ValueError("Non-finite response after gain adjustment")
        history.append((gain, response))
        if abs(response - target) / target <= tolerance:
            break
        if response <= 1e-12:
            raise ValueError("State effect collapsed after gain adjustment")
        correction = min(2.0, max(0.5, target / response))
        new_gain = min(max(gain * correction, min_gain), max_gain)
        if new_gain == gain:
            break
        gain = new_gain
    return dict(gain=gain, target_rms=target, state_gain_one_rms=base,
                matched_state_rms=response, relative_error=abs(response-target)/target,
                matched=abs(response-target)/target <= tolerance,
                bounds=[min_gain, max_gain], tolerance=tolerance,
                observations=[dict(gain=g, rms=v) for g,v in history])
