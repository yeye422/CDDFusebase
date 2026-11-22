"""The V1 inference graph with direct, source-specific supervision for its two candidates."""

import torch
from torch.nn import functional as F

from e2_local_correction import LocalCorrectionE2, SPEC as V1_SPEC, local_error


SPEC = dict(V1_SPEC, tile_size=16, contrast_span=4, contrast_pool=3,
            reconstruction_candidate_weight=0.25, fusion_candidate_weight=0.10)
FIELDS = ("correction_reconstruction", "ranking", "coarse_fusion", "correction_mae",
          "optical_contrast", "sar_contrast", "optical_keep", "sar_keep")


class LocalCorrectionV2(LocalCorrectionE2):
    VARIANT = "lwpd_e2_local_correction_v2"


def blur_tiles(clean, valid, training):
    h, w = clean.shape[-2:]
    phase = torch.randint(2, (clean.shape[0], 1, 1, 1), device=clean.device) if training else 0
    tile = SPEC["tile_size"]
    active = ((torch.arange(h, device=clean.device)[:, None] // tile
               + torch.arange(w, device=clean.device)[None, :] // tile + phase) % 2).bool()
    size = SPEC["blur_size"]
    def pool(x):
        return F.avg_pool2d(x, size, 1, size // 2, count_include_pad=False)
    blur = pool(clean * valid) / pool(valid).clamp_min(1e-6)
    active = active & valid.bool()
    return torch.where(active, blur, clean), active


def masked_average(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def candidate_loss(candidate, source, clean, valid, degraded):
    """Fit signed x/y flank contrasts inside degraded regions; keep intact pixels unchanged."""
    # Require every pixel in each flank window to be valid and degraded.
    # This excludes padding, checkerboard seams and SAR nodata from the target.
    k, span = SPEC["contrast_pool"], SPEC["contrast_span"]
    usable = F.avg_pool2d((degraded & valid.bool()).float(), k, 1, k // 2) == 1
    error = F.avg_pool2d(candidate.float() - source.float(), k, 1, k // 2)
    horizontal = masked_average((error[..., span:] - error[..., :-span]).abs(),
                                usable[..., span:] & usable[..., :-span])
    vertical = masked_average((error[..., span:, :] - error[..., :-span, :]).abs(),
                              usable[..., span:, :] & usable[..., :-span, :])
    keep = masked_average((candidate.float() - clean).abs(), valid.bool() & ~degraded)
    return (horizontal + vertical) * 0.5, keep


def source_terms(model, losses, optical, sar, valid, encoded, *, reconstruction):
    bo, do, bs, ds = encoded
    direct = optical.new_zeros((), dtype=torch.float32)
    stats = {}
    for index, (name, source, base, detail, mask) in enumerate((
        ("optical", optical, bo, do, torch.ones_like(valid)), ("sar", sar, bs, ds, valid)
    ), start=1):
        coarse, context = model.decoder(None, base, detail)
        stats[name + "_reconstruction"] = losses.reconstruction(source, coarse, mask)
        clean = coarse.detach().float()
        initial, active = blur_tiles(clean, mask, model.training)
        # During fusion the auxiliary task updates candidate heads only.
        # During reconstruction it shares the existing mixed readout and backbone.
        if reconstruction:
            mixed, images, scores = model.decoder.correction(optical, sar, initial, context, do, ds)
            aux = losses.reconstruction(source, mixed, mask)
            error = local_error(images.detach(), source, mask)
            error[:, 2:] = error[:, 2:].masked_fill(sar <= 0, float("inf"))
            best, ids = error.topk(2, dim=1, largest=False)
            decisive = (best[:, 1] - best[:, 0] > SPEC["ranking_margin"]) & mask[:, 0].bool()
            rank = masked_average(F.cross_entropy(scores, ids[:, 0], reduction="none"), decisive)
            for key, value in (("correction_reconstruction", aux), ("ranking", rank)):
                stats[key] = stats.get(key, 0) + 0.5 * value
        else:
            _, images, _ = model.decoder.correction(optical, sar, initial, context.detach(), do.detach(), ds.detach())
        contrast, keep = candidate_loss(images[:, index:index + 1], source, clean, mask, active)
        stats[name + "_contrast"], stats[name + "_keep"] = contrast, keep
        direct = direct + 0.5 * (contrast + keep)
    stats["reconstruction"] = 0.5 * (stats["optical_reconstruction"] + stats["sar_reconstruction"])
    return direct, stats


def reconstruction_forward(model, losses, optical, sar, valid, args):
    direct, stats = source_terms(model, losses, optical, sar, valid, model.encode_pair(optical, sar),
                                 reconstruction=True)
    total = (stats["reconstruction"] + SPEC["reconstruction_aux_weight"] * stats["correction_reconstruction"]
             + SPEC["ranking_weight"] * stats["ranking"] + SPEC["reconstruction_candidate_weight"] * direct)
    return total, stats


def fusion_forward(model, losses, optical, sar, valid, args):
    encoded, diagnostics = model.encode_pair(optical, sar, return_diagnostics=True)
    image, coarse, _, _ = model.corrected(optical, sar, encoded)
    direct, stats = source_terms(model, losses, optical, sar, valid, encoded, reconstruction=False)
    fusion, intensity, gradient = losses.fusion(optical, sar, image, valid, args.gradient_weight)
    initial = losses.fusion(optical, sar, coarse, valid, args.gradient_weight)[0]
    alpha = SPEC["coarse_fusion_weight"]
    total = ((1 - alpha) * fusion + alpha * initial + args.reconstruction_weight * stats["reconstruction"]
             + SPEC["fusion_candidate_weight"] * direct)
    stats.update(fusion=fusion, intensity=intensity, gradient=gradient, coarse_fusion=initial,
                 gate_mean=diagnostics["sar_gates"].mean(), correction_mae=(image.detach() - coarse.detach()).abs().mean())
    return total, stats
