"""Small convex coefficient problems with common physical feasibility checks."""

import time

import numpy as np
from scipy.optimize import minimize


METHODS = ("J0", "J1", "J2", "J3", "J4", "J6")


def affine_system(data, center, actual, readouts):
    """Represent the local model in absolute coefficients: R a + c."""
    intercept = actual.flatten().cpu().double().numpy() - data["full"] @ center
    c = actual.new_tensor(intercept).reshape_as(actual)
    cp = c[..., readouts.protected].flatten().cpu().double().numpy()
    return dict(image=intercept, target=readouts.target_values(c).cpu().double().numpy(),
                readout=readouts.protection_values(c).cpu().double().numpy(), protected=cp)


def solve(data, affine, readouts, method, settings, center):
    start = time.perf_counter()
    h, q, full = (data[key] for key in ("h", "q", "full"))
    b = readouts.b.cpu().double().numpy() - affine["target"]
    tau = readouts.tau.cpu().double().numpy()
    u = settings["u"]
    lam_p, lam_l, eta = (settings[key] for key in ("lambda_p", "lambda_l", "eta"))
    gram = data["grams"].get(method, data["grams"]["J2"])
    protection_response = data["shuffled"] if method == "J6" else data["protected"]
    linear_protect = protection_response.T @ affine["protected"] / max(len(protection_response), 1)
    # Normalize by readout count, pixel count, and typical per-action response energy.
    energy = max(float(np.mean(full * full)), 1e-12)
    penalty = (lam_p * gram + lam_l * np.diag(data["leakage"])) / energy
    curvature = h.T @ h / len(b) + penalty + eta * np.eye(len(center))
    linear = -h.T @ b / len(b) + lam_p * linear_protect / energy
    audits = []

    def subset(indices):
        indices = np.asarray(indices)
        a = curvature[np.ix_(indices, indices)]
        g = linear[indices]
        read = q[:, indices]
        scale = max(np.linalg.norm(a, ord=np.inf), np.linalg.norm(g, ord=np.inf), 1)

        def objective(x):
            return (x @ a @ x + 2 * g @ x) / scale

        def objective_jac(x):
            return (2 / scale) * (a @ x + g)

        def constraint(x):
            prediction = read @ x + affine["readout"]
            return np.r_[tau - prediction, tau + prediction]

        limits = {"type": "ineq", "fun": constraint, "jac": lambda x: np.vstack((-read, read))}
        result = minimize(objective, np.clip(center[indices], -u, u), jac=objective_jac,
                          method="SLSQP", bounds=[(-u, u)] * len(indices),
                          constraints=[limits] if tau.size else [],
                          options={"ftol": 1e-10, "maxiter": 150})
        feasible = bool(np.isfinite(result.x).all() and np.all(constraint(result.x) >= -1e-6))
        audits.append(dict(indices=indices.tolist(), success=bool(result.success),
                           feasible=feasible, message=str(result.message)))
        if not feasible or not result.success:
            return None
        coefficients = np.zeros_like(center)
        coefficients[indices] = result.x
        return coefficients

    if method in ("J0", "J1"):
        singles = [subset([index]) for index in range(len(center))]
        valid = [value for value in singles if value is not None]
        if method == "J0":
            candidates = valid
        else:
            candidates = [np.sum(valid, axis=0)] if len(valid) == len(singles) else []
    else:
        joint = subset(np.arange(len(center)))
        candidates = [joint] if joint is not None else []
    return candidates, dict(seconds=time.perf_counter() - start, problems=audits)
