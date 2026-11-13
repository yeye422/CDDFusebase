"""Fixed-weight asymmetric proximal reconstruction for optical–SAR fusion."""

import kornia
import torch
import torch.nn.functional as F


SPEC = dict(mu=1.0, alpha=0.25, beta=0.50, gamma=0.50, steps=6,
            pool_sizes=(2, 4), optical_sigma=0.8, sar_sigmas=(0.8, 1.6),
            tensor_sigma=1.0, gradient_scale=0.05, optical_weight=0.75,
            sar_weight=0.50, relative_tolerance=1e-7)


def gradient(image):
    """Forward horizontal and vertical differences with zero exterior flux."""
    dx = F.pad(image[..., 1:] - image[..., :-1], (0, 1))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.cat((dx, dy), dim=1)


def gradient_adjoint(field):
    dx, dy = field.split(1, dim=1)
    dx, dy = dx[..., :-1], dy[..., :-1, :]
    return F.pad(dx, (1, 0)) - F.pad(dx, (0, 1)) + (
        F.pad(dy, (0, 0, 1, 0)) - F.pad(dy, (0, 0, 0, 1))
    )


def _pool_projection(image, size):
    # size**2 * P.T @ P replicates each block mean. Incomplete blocks contribute zero.
    height, width = image.shape[-2:]
    blocks = F.avg_pool2d(image, size).repeat_interleave(size, -2).repeat_interleave(size, -1)
    return F.pad(blocks, (0, width % size, 0, height % size))


def _smooth(image, sigma):
    size = 2 * int(3 * sigma + 0.5) + 1
    return kornia.filters.gaussian_blur2d(image, (size, size), (sigma, sigma), border_type="replicate")


def _magnitude(field):
    return torch.linalg.vector_norm(field, dim=1, keepdim=True)


def _alignment(a, b):
    cosine = (a * b).sum(dim=1, keepdim=True).abs() / (_magnitude(a) * _magnitude(b)).clamp_min(1e-8)
    return cosine.clamp_max(1.0)


def _coherence(field):
    x, y = field.split(1, dim=1)
    jxx, jxy, jyy = _smooth(torch.cat((x.square(), x * y, y.square()), dim=1), SPEC["tensor_sigma"]).split(1, dim=1)
    return ((jxx - jyy).square() + 4 * jxy.square()).sqrt() / (jxx + jyy + 1e-8)


@torch.no_grad()
def fixed_weights(optical, sar, initial, valid):
    """Use stable display-domain gradients as evidence for the two constraints."""
    go = gradient(_smooth(optical, SPEC["optical_sigma"]))
    gf = gradient(_smooth(initial, SPEC["optical_sigma"]))
    sar_fields = []
    for sigma in SPEC["sar_sigmas"]:
        smoothed = _smooth(sar * valid, sigma) / _smooth(valid, sigma).clamp_min(1e-8)
        sar_fields.append(gradient(smoothed))
    g1, g2 = sar_fields
    m1, m2 = _magnitude(g1), _magnitude(g2)
    scale = SPEC["gradient_scale"]
    qo = (_magnitude(go) / scale).clamp_max(1.0) * (0.5 + 0.5 * _coherence(go))
    stability = _alignment(g1, g2) * torch.minimum(m1, m2) / torch.maximum(m1, m2).clamp_min(1e-8)
    extra = ((_magnitude(gf) - _magnitude(go)) / scale).clamp(0.0, 1.0)
    qs = (m1 / scale).clamp_max(1.0) * _coherence(g1) * stability
    qs = qs * (1.0 - qo.clamp(0.0, 1.0)) * extra * _alignment(gf - go, g1)
    valid_x = F.pad(valid[..., 1:], (0, 1), value=1.0)
    valid_y = F.pad(valid[..., 1:, :], (0, 0, 0, 1), value=1.0)
    return (SPEC["optical_weight"] * qo.clamp(0.0, 1.0),
            SPEC["sar_weight"] * qs.clamp(0.0, 1.0) * valid * valid_x * valid_y)


def solve(initial, optical, wo, ws, *, steps=SPEC["steps"], diagnostics=False):
    """Solve for the correction with matrix-free, batch-wise Jacobi PCG in FP32."""
    with torch.autocast(initial.device.type, enabled=False):
        initial, optical, wo, ws = (x.float() for x in (initial, optical, wo, ws))
        mu, alpha = SPEC["mu"], SPEC["alpha"]
        optical_edge = SPEC["beta"] * wo
        edge = optical_edge + SPEC["gamma"] * ws
        height, width = initial.shape[-2:]
        sizes = tuple(size for size in SPEC["pool_sizes"] if size <= min(height, width))

        def lowpass(image):
            return alpha * sum(_pool_projection(image, size) for size in sizes)

        def apply(image):
            return mu * image + lowpass(image) + gradient_adjoint(edge * gradient(image))

        outgoing_x = edge[..., :-1]
        outgoing_y = edge[..., :-1, :]
        diagonal = mu + F.pad(outgoing_x, (0, 1)) + F.pad(outgoing_x, (1, 0))
        diagonal = diagonal + F.pad(outgoing_y, (0, 0, 0, 1)) + F.pad(outgoing_y, (0, 0, 1, 0))
        for size in sizes:
            mask = F.pad(torch.ones_like(initial[..., :height - height % size, :width - width % size]),
                         (0, width % size, 0, height % size))
            diagonal = diagonal + (alpha / size**2) * mask

        def dot(a, b):
            return (a * b).sum(dim=(1, 2, 3), keepdim=True)

        difference = optical - initial
        rhs = lowpass(difference) + gradient_adjoint(optical_edge * gradient(difference))
        correction = torch.zeros_like(initial)
        residual = rhs
        threshold = dot(rhs, rhs) * SPEC["relative_tolerance"] ** 2
        direction = residual / diagonal
        rz = dot(residual, direction)
        for _ in range(steps):
            product = apply(direction)
            denominator = dot(direction, product)
            active = (dot(residual, residual) > threshold) & (denominator > 0)
            step = torch.where(active, rz, 0.0) / torch.where(active, denominator, 1.0)
            correction = correction + step * direction
            residual = residual - step * product
            z = residual / diagonal
            next_rz = dot(residual, z)
            active = active & (rz > 0)
            ratio = torch.where(active, next_rz, 0.0) / torch.where(active, rz, 1.0)
            direction = z + ratio * direction
            rz = next_rz

        output = initial + correction
        if diagnostics:
            true_residual = rhs - apply(correction)
            return output, dot(true_residual, true_residual).sqrt() / dot(rhs, rhs).sqrt().clamp_min(1e-12)
        return output


def refine(optical, sar, initial, valid):
    with torch.autocast(initial.device.type, enabled=False):
        wo, ws = fixed_weights(optical.float(), sar.float(), initial.float(), valid.float())
        return solve(initial, optical, wo, ws)
