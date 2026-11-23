"""Frozen nonlinear Decoder replay and full-image forward derivatives."""

import time

import numpy as np
import torch
from torch.func import jvp

from .atoms import rms


@torch.no_grad()
def responses(decode, detail, atoms):
    start = time.perf_counter()
    result = []
    for atom in atoms:
        _, tangent = jvp(decode, (detail,), (atom.direction,))
        result.append(255 * tangent)
    matrix = torch.stack(result)
    if not torch.isfinite(matrix).all():
        raise ValueError("Decoder JVP produced non-finite values")
    if detail.is_cuda:
        torch.cuda.synchronize(detail.device)
    return matrix, time.perf_counter() - start


def combine(baseline, atoms, coefficients):
    result = baseline
    if len(atoms) != len(coefficients):
        raise ValueError("One coefficient is required per atom")
    for atom, value in zip(atoms, coefficients):
        if value:
            result = result + float(value) * atom.direction
    return result


@torch.no_grad()
def finite_differences(decode, detail, atoms, matrix, settings):
    records = []
    for index in dict.fromkeys((0, len(atoms) // 2, len(atoms) - 1)):
        atom, tangent = atoms[index], matrix[index]
        trials = []
        for epsilon in settings["fd_steps"]:
            perturbation = epsilon * atom.direction
            estimated = 255 * (decode(detail + perturbation) - decode(detail - perturbation)) / (2 * epsilon)
            scale = float(rms(tangent))
            error = float(rms(estimated - tangent))
            lower, upper = estimated.double().flatten(), tangent.double().flatten()
            norm_product = lower.norm() * upper.norm()
            cosine = float((torch.dot(lower, upper) / norm_product).clamp(-1, 1)) if norm_product > 0 else None
            low_signal = scale <= settings["fd_abs_gl"]
            passed = error <= settings["fd_abs_gl"] if low_signal else (
                error / scale <= settings["fd_rel"] and cosine is not None
                and cosine >= settings["fd_cosine"])
            trials.append(dict(epsilon=epsilon, rms_error_gl=error, cosine=cosine, passed=passed))
        records.append(dict(atom=atom.name, response_rms_gl=float(rms(tangent)), trials=trials,
                            passed=sum(trial["passed"] for trial in trials) >= 2))
    return records


def matrices(responses, atoms, readouts, seed):
    h = torch.stack([readouts.target_values(response) for response in responses], dim=1)
    q = torch.stack([readouts.protection_values(response) for response in responses], dim=1)
    protected = responses[..., readouts.protected].reshape(len(atoms), -1).T.cpu().double().numpy()
    leakage = np.array([
        float(response[~atom.window.mask(response)].square().mean())
        for response, atom in zip(responses, atoms)
    ])
    full = responses.reshape(len(atoms), -1).T.cpu().double().numpy()
    groups = np.array([atom.window.group for atom in atoms])
    rng = np.random.default_rng(seed)
    shuffled = np.column_stack([rng.permutation(col) for col in protected.T])
    norm = max(len(protected), 1)
    gram = protected.T @ protected / norm
    grams = {"J2": np.diag(np.diag(gram)), "J3": gram * (groups[:, None] == groups),
             "J4": gram, "J6": shuffled.T @ shuffled / norm}
    for name, value in grams.items():
        if np.linalg.eigvalsh(value)[0] < -1e-10 * max(np.linalg.norm(value), 1):
            raise ValueError(f"{name} protection Gram is not positive semidefinite")
    if not np.allclose(np.diag(grams["J6"]), np.diag(gram)):
        raise ValueError("J6 changed protection column energy")
    return dict(h=h.cpu().double().numpy(), q=q.cpu().double().numpy(), grams=grams,
                leakage=leakage, protected=protected, shuffled=shuffled, full=full)
