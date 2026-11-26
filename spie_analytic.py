"""Fixed screened-Poisson fusion for the SPIE analytical comparison."""

from functools import lru_cache
import math

from kornia.filters import gaussian_blur2d
import torch
from torch.nn import functional as F

from e2_aspr import gradient, gradient_adjoint


SPEC = dict(alpha=0.25, intensity_sar_weight=0.25, gradient_sar_weight=0.6,
            sar_sigma_fine=0.8, sar_sigma_coarse=1.6, valid_support=0.95,
            polarity_cosine=math.cos(math.pi / 6), optical_polarity_min=2 / 255)


@lru_cache(maxsize=16)
def neumann_basis(size, device):
    positions = torch.arange(size, device=device, dtype=torch.float64)
    basis = (positions[:, None] * (positions[None, :] + 0.5) * (math.pi / size)).cos()
    basis.mul_(math.sqrt(2 / size))
    basis[0].div_(math.sqrt(2))
    eigenvalues = 4 * (positions * (math.pi / (2 * size))).sin().square()
    return basis.float(), eigenvalues.float()


def screened_poisson(reference, field, alpha):
    """Solve (alpha I + D^T D)u = alpha B + D^T v using the DCT-II basis."""
    if alpha <= 0:
        raise ValueError("Screening must be positive")
    rhs = alpha * reference.float() + gradient_adjoint(field.float())
    height, width = rhs.shape[-2:]
    ch, eh = neumann_basis(height, rhs.device)
    cw, ew = neumann_basis(width, rhs.device)
    spectrum = (ch @ rhs @ cw.T) / (alpha + eh[:, None] + ew[None, :])
    solved = ch.T @ spectrum @ cw
    return solved, rhs


def masked_smooth(image, valid, sigma):
    size = 2 * math.ceil(3 * sigma) + 1
    filtered = gaussian_blur2d(torch.cat((image * valid, valid), dim=1), (size, size),
                               (sigma, sigma), border_type="replicate")
    signal, support = filtered.split(1, dim=1)
    return signal / support.clamp_min(1e-8), support


@torch.inference_mode()
def fuse(optical, sar, spec=SPEC):
    """Return the three analytical outputs from registered luminance tensors."""
    if optical.shape != sar.shape or optical.ndim != 4 or optical.shape[1] != 1:
        raise ValueError("SPIE expects aligned N x 1 x H x W inputs")
    optical, sar = optical.float(), sar.float()
    valid = sar > 0
    average = torch.where(valid, 0.5 * (optical + sar), optical)
    fine, fine_support = masked_smooth(sar, valid.float(), spec["sar_sigma_fine"])
    coarse, coarse_support = masked_smooth(sar, valid.float(), spec["sar_sigma_coarse"])
    use_intensity = valid & (fine_support >= spec["valid_support"])
    reference = optical.lerp(torch.where(use_intensity, fine, optical), spec["intensity_sar_weight"])

    supported = use_intensity & (coarse_support >= spec["valid_support"])
    supported = (supported & F.pad(supported[..., 1:], (0, 1), value=True)
                 & F.pad(supported[..., 1:, :], (0, 0, 0, 1), value=True))
    go, gs, gc = (gradient(image) for image in (optical, fine, coarse))
    mo, ms, mc = (field.norm(dim=1, keepdim=True) for field in (go, gs, gc))
    stability = ((gs * gc).sum(dim=1, keepdim=True) / (ms * mc).clamp_min(1e-8)).clamp(0, 1)
    stability = stability * torch.minimum(ms, mc) / torch.maximum(ms, mc).clamp_min(1e-8)
    alignment = (go * gs).sum(dim=1, keepdim=True) / (mo * ms).clamp_min(1e-8)
    flip = (mo >= spec["optical_polarity_min"]) & (alignment <= -spec["polarity_cosine"])
    gs = torch.where(flip, -gs, gs)
    weight = spec["gradient_sar_weight"] * supported * stability * ms / (mo + ms).clamp_min(1e-8)
    field = go.lerp(gs, weight)
    alpha = spec["alpha"]
    fused, rhs = screened_poisson(reference, field, alpha)
    error = alpha * fused + gradient_adjoint(gradient(fused)) - rhs
    relative = error.norm() / rhs.norm().clamp_min(1e-8)
    diagnostics = dict(relative_residual=float(relative), mean_sar_gradient_weight=float(weight.mean()),
                       mean_abs_p3_minus_p2=float((fused - reference).abs().mean()),
                       clipped_fraction=float(((fused < -1e-6) | (fused > 1 + 1e-6)).float().mean()))
    return {"P1": average, "P2": reference, "P3": fused}, diagnostics
