"""Pure primitives for a train-only structural prediction / frozen transfer audit.

This is a linear predictive probe and a fixed source-Detail interpolation, NOT
a trained RCT fusion model. Predictors never see E2's globally mixed features.
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


PREDICTORS = ("self", "self_wide", "cross", "rolled", "swapped")
CONDITIONS = ("predictive", "rolled", "swapped", "shuffled", "native", "uniform")
SOURCES = ("optical", "sar")
CONTEXT_OFFSETS = tuple((y, x) for y in (-3, 0, 3) for x in (-3, 0, 3) if (y, x) != (0, 0))
WIDE_OFFSETS = ((-6, 0), (6, 0), (0, -6), (0, 6))
DESCRIPTOR_CHANNELS = 9
GRID_STRIDE = 4
# Gaussian5 -> Sobel3 -> histogram pooling3 -> pooling4: a 12px support.
# Context centers are separated by 12px, so their raw supports do not overlap.
RAW_SUPPORT_WIDTH = 12
RULES = dict(min_prediction_scenes=16, prediction_relative_gain=0.01,
             scene_win_fraction=0.60, min_transfer_scenes=8,
             min_manual_scenes=3, source_transfer_tolerance=0.005,
             negative_path_tolerance=0.02, min_changed_pixel_fraction=0.01,
             bootstrap_replicates=2000)


def structure_descriptor(image, valid=None):
    """Fixed unsigned, softly binned dense HOG, plus magnitude; no learned weights.

    Returns one descriptor per 4x4 source cell. Invalid support is propagated
    through the full 12px receptive field, including image boundaries.
    """
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("Expected one BCHW grayscale image")
    if min(image.shape[-2:]) < 64 or any(n % 4 for n in image.shape[-2:]):
        raise ValueError("Source dimensions must be multiples of four and >=64")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("Nonfinite source")
    if valid is None:
        valid = torch.ones_like(image, dtype=torch.bool)
    if valid.shape != image.shape:
        raise ValueError("Validity shape differs from the source")
    x = image.float()
    axis = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    gaussian = torch.exp(-axis.square()/2)
    gaussian /= gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)[None, None]
    smooth = F.conv2d(F.pad(x, (2,)*4, mode="reflect"), kernel)
    sobel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
    gradient = F.conv2d(F.pad(smooth, (1,)*4, mode="reflect"),
                        torch.stack((sobel, sobel.T))[:, None])
    magnitude = gradient.square().sum(1, keepdim=True).sqrt()
    angle = torch.remainder(torch.atan2(gradient[:, 1:2], gradient[:, :1]), math.pi)
    position = angle * (DESCRIPTOR_CHANNELS/math.pi)
    bins = torch.arange(DESCRIPTOR_CHANNELS, device=x.device, dtype=x.dtype)[None, :, None, None]
    distance = torch.abs(position-bins)
    distance = torch.minimum(distance, DESCRIPTOR_CHANNELS-distance)
    votes = magnitude * (1-distance).clamp_min(0)
    votes = F.avg_pool2d(F.pad(votes, (1,)*4, mode="reflect"), 3, stride=1)
    votes = F.avg_pool2d(votes, GRID_STRIDE)
    descriptor = votes / votes.square().sum(1, keepdim=True).sqrt().clamp_min(1e-4)
    mag = F.avg_pool2d(magnitude, GRID_STRIDE)
    invalid = (~valid.bool()).float()
    invalid = F.max_pool2d(F.pad(invalid, (4,)*4, value=1), 9, stride=1)
    permitted = F.max_pool2d(invalid, GRID_STRIDE) == 0
    return (descriptor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float64),
            mag[0, 0].detach().cpu().numpy().astype(np.float64),
            permitted[0, 0].detach().cpu().numpy())


def sample_at_offset(values, offset):
    """At p, return values[p+offset]; zero outside, never wrap a context."""
    dy, dx = offset
    h, w = values.shape[:2]
    result = np.zeros_like(values)
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    result[y0:y1, x0:x1] = values[y0+dy:y1+dy, x0+dx:x1+dx]
    return result


def prediction_design(own, other, own_valid, other_valid, swapped, swapped_valid):
    """Same target/context, equal column counts, train-matched input controls.

    self_wide adds distant OWN context. rolled/swapped get their own heads fit
    with the same control on training, avoiding a solely OOD mismatch test.
    """
    if own.shape != other.shape or own.shape != swapped.shape:
        raise ValueError("Source descriptor shapes differ")
    if own.ndim != 3 or own.shape[2] != DESCRIPTOR_CHANNELS:
        raise ValueError("Unexpected descriptor channels")
    h, w = own.shape[:2]
    if any(mask.shape != (h, w) for mask in (own_valid, other_valid, swapped_valid)):
        raise ValueError("Invalid descriptor support shape")
    context = np.concatenate([sample_at_offset(own, d) for d in CONTEXT_OFFSETS], axis=2)
    wide = np.mean([sample_at_offset(own, d) for d in WIDE_OFFSETS], axis=0)
    rolled = np.roll(other, (h//2, w//2), axis=(0, 1))
    rolled_valid = np.roll(other_valid, (h//2, w//2), axis=(0, 1))
    mask = own_valid & other_valid & swapped_valid & rolled_valid
    for offset in CONTEXT_OFFSETS + WIDE_OFFSETS:
        mask = mask & sample_at_offset(own_valid, offset)
    # Every predictor uses the EXACT same eligible positions on a scene.
    features = dict(self=np.concatenate((context, np.zeros_like(other)), axis=2),
                    self_wide=np.concatenate((context, wide), axis=2),
                    cross=np.concatenate((context, other), axis=2),
                    rolled=np.concatenate((context, rolled), axis=2),
                    swapped=np.concatenate((context, swapped), axis=2))
    return features, own, mask


class RidgeMoments:
    """Scene-streamed float64 sufficient statistics; no validation fitting."""

    def __init__(self, inputs, outputs):
        self.n = 0
        self.sx = np.zeros(inputs)
        self.sy = np.zeros(outputs)
        self.xx = np.zeros((inputs, inputs))
        self.xy = np.zeros((inputs, outputs))
        self.yy = np.zeros(outputs)

    def add(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("Expected nonempty paired training rows")
        if x.shape[1] != len(self.sx) or y.shape[1] != len(self.sy):
            raise ValueError("Ridge dimension mismatch")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite fitting rows")
        self.n += len(x)
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx += x.T@x
        self.xy += x.T@y
        self.yy += (y*y).sum(0)

    def finish(self, alpha=0.01, context_columns=None):
        if self.n <= len(self.sx) or not 0 < alpha <= 1:
            raise ValueError("Insufficient rows or invalid ridge penalty")
        xm, ym = self.sx/self.n, self.sy/self.n
        covariance = self.xx/self.n-np.outer(xm, xm)
        xs = np.sqrt(np.maximum(np.diag(covariance), 0)).clip(1e-5)
        ys = np.sqrt(np.maximum(self.yy/self.n-ym*ym, 0)).clip(0.01)
        gram = covariance / np.outer(xs, xs)
        cross = (self.xy/self.n-np.outer(xm, ym))/np.outer(xs, ys)
        transform = np.eye(len(xs))
        if context_columns is not None:
            if not 0 < context_columns < len(xs):
                raise ValueError("Invalid common-context width")
            # Unlabelled TRAIN-only residualization. Duplicating a context
            # feature must not reduce Ridge shrinkage and masquerade as new
            # cross-source information. Fold the transform into the saved head.
            c = context_columns
            transform[:c, c:] = -np.linalg.pinv(gram[:c, :c], rcond=1e-8)@gram[:c, c:]
        transformed = transform.T@gram@transform
        transformed = (transformed+transformed.T)/2
        coef = transform@np.linalg.solve(transformed+alpha*np.eye(len(xs)), transform.T@cross)
        if not np.isfinite(coef).all():
            raise FloatingPointError("Nonfinite ridge solution")
        c = context_columns if context_columns is not None else len(xs)
        extra_rank = int(np.sum(np.linalg.eigvalsh(transformed[c:, c:]) > 1e-8)) if c < len(xs) else 0
        return dict(x_mean=xm, x_scale=xs, y_mean=ym, y_scale=ys, coef=coef,
                    train_rows=np.array(self.n), effective_rank=np.array(np.sum(np.linalg.eigvalsh(transformed) > 1e-8)),
                    context_columns=np.array(c), extra_residual_rank=np.array(extra_rank))


def predict(head, x):
    return head["y_mean"] + (((x-head["x_mean"])/head["x_scale"])@head["coef"])*head["y_scale"]


def prediction_errors(heads, features, target):
    errors = {}
    scale = heads["self"]["y_scale"]
    for name in PREDICTORS:
        if not np.allclose(heads[name]["y_scale"], scale, atol=1e-10):
            raise ValueError("Predictors were not fit on identical target rows")
        estimate = predict(heads[name], features[name])
        errors[name] = np.mean(((estimate-target)/scale)**2, axis=-1)
    errors["constant"] = np.mean(((target-heads["self"]["y_mean"])/scale)**2, axis=-1)
    return errors


def relationship_maps(errors, magnitude, valid, magnitude_scale, gain_scale):
    if magnitude_scale <= 0 or gain_scale <= 0:
        raise ValueError("Condition normalizers must be positive")
    native = np.clip(magnitude/magnitude_scale, 0, 1)*valid
    maps = {"native": native, "uniform": valid.astype(np.float64)}
    for condition, predictor in (("predictive", "cross"), ("rolled", "rolled"), ("swapped", "swapped")):
        increment = errors["self"]-errors[predictor]
        maps[condition] = native*np.clip(increment/gain_scale, 0, 1)
    # Exact value/histogram preservation ON THE COMMON ELIGIBLE GRID.
    # This cyclic valid-index shift is not a guarantee of equal spatial smoothness.
    shuffled = np.zeros_like(native)
    values = maps["predictive"][valid]
    if len(values):
        shuffled[valid] = np.roll(values, len(values)//2)
    maps["shuffled"] = shuffled
    return maps


def sample_rows(mask, count, seed):
    indices = np.flatnonzero(mask.ravel())
    if len(indices) < 32:
        raise ValueError("Fewer than 32 common eligible prediction positions")
    return np.sort(np.random.default_rng(seed).choice(indices, min(count, len(indices)), replace=False))


def scene_seed(scene, salt=0):
    return int(hashlib.sha256((str(scene)+":"+str(salt)).encode()).hexdigest()[:8], 16)


def mismatched_partners(scenes, seed):
    if len(scenes) < 2 or len(set(scenes)) != len(scenes):
        raise ValueError("Need at least two distinct scenes")
    order = np.random.default_rng(seed).permutation(len(scenes))
    return {str(scenes[int(order[i])]): str(scenes[int(order[(i+1) % len(order)])]) for i in range(len(order))}


def paired_stats(candidate, reference, lower_is_better=False, seed=42, replicates=2000):
    a, b = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Expected aligned scene values")
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if not len(a):
        return dict(n=0, candidate_mean=None, reference_mean=None, delta=None,
                    favorable_gain=None, median_gain=None, relative_gain=None,
                    improved=0, ci_low=None, ci_high=None)
    delta = a-b
    gain = -delta if lower_is_better else delta
    boot = np.random.default_rng(seed).choice(gain, (replicates, len(gain)), replace=True).mean(1)
    ci = np.quantile(boot, [0.025, 0.975])
    return dict(n=len(a), candidate_mean=float(a.mean()), reference_mean=float(b.mean()),
                delta=float(delta.mean()), favorable_gain=float(gain.mean()),
                median_gain=float(np.median(gain)), relative_gain=float(gain.mean()/max(abs(b.mean()), 1e-12)),
                improved=int((gain > 0).sum()), ci_low=float(ci[0]), ci_high=float(ci[1]))


def prediction_admission(comparisons):
    results = {}
    for source in SOURCES:
        selected = [x for x in comparisons if x["source"] == source and x["cohort"] == "unaudited"
                    and x["reference"] in ("self", "self_wide", "rolled", "swapped")]
        enough = len(selected) == 4 and all(x["n"] >= RULES["min_prediction_scenes"] for x in selected)
        supported = enough and all(x["relative_gain"] >= RULES["prediction_relative_gain"]
                                  and x["median_gain"] > 0 and x["ci_low"] > 0
                                  and x["improved"]/x["n"] >= RULES["scene_win_fraction"] for x in selected)
        results[source] = dict(status="supported" if supported else ("not_established" if enough else "insufficient_data"),
                               comparisons=len(selected))
    return results


def interpolation_direction(source_detail, fused_detail, condition):
    """Fixed diagnostic operator: condition * (source Detail - fused Detail).

    It neither predicts pixels nor removes the untouched original path.
    No inference about trainable channel transformations follows automatically.
    """
    if source_detail.shape != fused_detail.shape or condition.shape != source_detail[:, :1].shape:
        raise ValueError("Interpolation tensor shape mismatch")
    if not bool(torch.isfinite(condition).all()) or float(condition.min()) < 0 or float(condition.max()) > 1.000001:
        raise ValueError("Condition must be finite and within [0,1]")
    return condition*(source_detail-fused_detail)


def calibrate_effect(measure, target, upper=0.04, tolerance=0.02, steps=16):
    """Bracketed root finding; no quality/edge metric may be passed in."""
    if not 0 < target or not 0 < upper <= 0.1 or not 0 < tolerance <= 0.1:
        raise ValueError("Invalid effect budget")
    lo, low, hi, high = 0., 0., upper, float(measure(upper))
    if not math.isfinite(high) or high < target:
        raise ValueError("Effect unreachable within fixed maximum input")
    best = (hi, high)
    for _ in range(steps):
        weight = np.clip((target-low)/max(high-low, 1e-30), 0.05, 0.95)
        mid = lo+float(weight)*(hi-lo)
        value = float(measure(mid))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid measured effect")
        if abs(value-target) < abs(best[1]-target):
            best = mid, value
        if abs(value-target)/target <= tolerance:
            return mid, value, abs(value-target)/target
        if value < target:
            lo, low = mid, value
        else:
            hi, high = mid, value
    error = abs(best[1]-target)/target
    if error > tolerance:
        raise ValueError("Final image MAE calibration did not converge")
    return best[0], best[1], error


def common_effect_target(nominal, fraction=0.9):
    if not nominal or not 0 < fraction < 1:
        raise ValueError("Missing nominal effects or invalid fraction")
    values = list(nominal.values())
    if any(not math.isfinite(x) or x <= 1e-8 for x in values):
        return None  # A zero condition is insufficient excitation, not a safe success.
    return min(values)*fraction
