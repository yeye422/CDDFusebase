"""Diagnostic-only BGC1 targets. Never imported by the training path."""
import torch
import torch.nn.functional as F

from utils.loss import Sobelxy


def direction_evidence(sar, valid):
    """5x5 tensor of 3x3 Sobel: require all 7x7 support, including image border."""
    mask = (valid.detach() > .5).float()
    if not torch.isfinite(sar[mask.bool()]).all():
        raise ValueError('Non-finite valid SAR')
    safe = torch.where(mask.bool(), sar.detach().float(), torch.zeros_like(sar).float())
    sobel = Sobelxy().to(safe.device)
    gx = F.conv2d(safe, sobel.weightx, padding=1)
    gy = F.conv2d(safe, sobel.weighty, padding=1)
    t = torch.arange(-2, 3, device=safe.device, dtype=torch.float32)
    kernel = torch.exp(-t.square() / 2)
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    smooth = lambda x: F.conv2d(x, kernel, padding=2)
    xx, yy, xy = smooth(gx.square()), smooth(gy.square()), smooth(gx * gy)
    c = ((xx - yy).square() + 4 * xy.square()).sqrt() / (xx + yy + 1e-6)
    # Explicit zero validity outside the image; max-pool's implicit padding differs.
    padded_invalid = F.pad(1 - mask, (3, 3, 3, 3), value=1)
    support = 1 - F.max_pool2d(padded_invalid, 7, stride=1)
    return c.clamp(0, 1).detach(), support.detach()


def make_targets(optical, sar, valid, eta=.10):
    if not 0 <= eta <= 1:
        raise ValueError('eta must be in [0,1]')
    if optical.shape != sar.shape or valid.shape != sar.shape or sar.ndim != 4 or sar.shape[1] != 1:
        raise ValueError('Expected matching Bx1xHxW tensors')
    if not torch.isfinite(optical).all() or not torch.isfinite(sar).all():
        raise ValueError('Baseline target requires finite source images')
    with torch.no_grad():
        optical, sar = optical.float(), sar.float()
        sobel = Sobelxy().to(sar.device)
        ao, ass = sobel(optical), sobel(sar)
        active = valid > .5
        intensity = torch.where(active, torch.maximum(optical, sar), optical)
        original = torch.where(active, torch.maximum(ao, ass), ao)
        c, support = direction_evidence(sar, valid)
        excess = (ass - ao).clamp_min(0)
        correction = float(eta) * support * (1 - c) * excess
        corrected = original - correction
        return dict(intensity=intensity, original=original, corrected=corrected,
                    coherence=c, support=support, excess=excess, correction=correction,
                    target_gap=sobel(intensity) - original)


def output_diagnostics(image, optical, sar, valid, targets):
    """Gradients are with respect to saved uint8/255 output, not model parameters."""
    f = image.detach().float().clone().requires_grad_(True)
    sobel = Sobelxy().to(f.device)
    amplitude = sobel(f)
    li = (f - targets['intensity']).abs().mean()
    lg = (amplitude - targets['original']).abs().mean()
    ln = (amplitude - targets['corrected']).abs().mean()
    gi = torch.autograd.grad(li, f, retain_graph=True)[0]
    gg = torch.autograd.grad(lg, f, retain_graph=True)[0]
    gn = torch.autograd.grad(ln, f)[0]

    def cosine(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        denom = a.norm() * b.norm()
        return float(a.dot(b) / denom) if denom > 0 else None

    conflict = (gi * gg < 0).float()
    changed = ((amplitude.detach() - targets['original']).sign() !=
               (amplitude.detach() - targets['corrected']).sign()).float()
    interior = torch.ones_like(f, dtype=torch.bool)
    interior[..., 0, :] = False
    interior[..., -1, :] = False
    interior[..., :, 0] = False
    interior[..., :, -1] = False
    # Diagnostic Euclidean amplitude with explicit pi-periodic source direction.
    def field(x):
        gx = F.conv2d(x.detach(), sobel.weightx, padding=1)
        gy = F.conv2d(x.detach(), sobel.weighty, padding=1)
        return (gx.square() + gy.square()).sqrt(), torch.atan2(gy, gx)
    go, ao = field(optical)
    gs, ass = field(sar)
    gf, af = field(f)
    source = torch.where(valid > .5, torch.maximum(go, gs), go)
    angle = torch.where((valid > .5) & (gs > go), ass, ao)
    delta = torch.remainder(af - angle + torch.pi / 2, torch.pi) - torch.pi / 2
    overshoot = (gf - source).clamp_min(0)
    stats = dict(intensity_loss=float(li.detach()), old_gradient_loss=float(lg.detach()),
                 new_gradient_loss=float(ln.detach()), old_fusion_loss=float((li + 10 * lg).detach()),
                 new_fusion_loss=float((li + 10 * ln).detach()),
                 intensity_gradient_cosine=cosine(gi, gg), old_new_gradient_cosine=cosine(gg, gn),
                 intensity_grad_norm=float(gi.double().norm()), old_grad_norm=float(gg.double().norm()),
                 new_grad_norm=float(gn.double().norm()),
                 output_gradient_change_norm=float((10 * (gn - gg)).double().norm()),
                 conflict_fraction=float(conflict.mean()), target_derivative_change_fraction=float(changed.mean()),
                 optical_mae=float((f.detach() - optical).abs().mean()),
                 sar_valid_mae=float((f.detach() - sar).abs()[valid > .5].mean()) if (valid > .5).any() else None)
    for region, mask in [('interior', interior), ('border', ~interior)]:
        stats[region + '_overshoot_fraction'] = float((overshoot[mask] > 0).float().mean())
        stats[region + '_overshoot_mean'] = float(overshoot[mask].mean())
        weights = source[mask]
        stats[region + '_weighted_angle_error'] = float((delta.abs()[mask] * weights).sum() / weights.sum()) if weights.sum() > 0 else None
    maps = dict(conflict=conflict.detach(), derivative_changed=changed.detach(),
                intensity_output_gradient=gi.detach(), old_output_gradient=gg.detach(),
                new_output_gradient=gn.detach(), euclidean_overshoot=overshoot.detach())
    return stats, maps
