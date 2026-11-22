"""Optical-Base E2 with one decoder pass and source-specific image corrections."""

import torch
from torch import nn
from torch.nn import functional as F

from e2_base_anchor import OpticalBaseAnchoredE2


SPEC = dict(width=16, logit_bound=1.0, ranking_weight=0.05, ranking_margin=1 / 255,
            reconstruction_aux_weight=0.25, coarse_fusion_weight=0.25, blur_size=5)


class LocalCorrection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        width = SPEC["width"]

        def head():
            net = nn.Sequential(nn.Conv2d(channels * 2, width, 1), nn.GELU(),
                                nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 1))
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.optical_head, self.sar_head = head(), head()
        self.score = nn.Sequential(nn.Conv2d(5, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, optical, sar, coarse, context, optical_detail, sar_detail):
        coarse = coarse.float()
        logits = torch.logit(coarse.clamp(1e-6, 1 - 1e-6))
        def candidate(head, detail):
            offset = SPEC["logit_bound"] * head(torch.cat((detail, context), 1)).float().tanh()
            # Delta form preserves the baseline at zero even for saturated AMP pixels.
            return (coarse + (torch.sigmoid(logits + offset) - torch.sigmoid(logits))).clamp(0, 1)

        images = (coarse, candidate(self.optical_head, optical_detail), candidate(self.sar_head, sar_detail))
        scores = torch.cat([self.score(torch.cat((optical, sar, coarse, im, im - coarse), 1))
                            for im in images], 1).float()
        scores = torch.cat((scores[:, :2], scores[:, 2:].masked_fill(sar <= 0, -1e4)), 1)
        weights = scores.softmax(1)
        alternatives = torch.cat(images, 1)
        result = (coarse + ((alternatives - coarse) * weights).sum(1, keepdim=True)).clamp(0, 1)
        return result, alternatives, scores


class LocalCorrectionE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_local_correction_v1"

    def __init__(self):
        super().__init__(variant=self.VARIANT)
        # The existing decoder optimizer group is active in both stages.
        with torch.random.fork_rng(devices=[]):
            self.decoder.correction = LocalCorrection(self.decoder.reduce_channel.out_channels)

    def corrected(self, optical, sar, encoded):
        base, detail = self.fuse_features(*encoded)
        coarse, context = self.decoder(None, base, detail)
        image, _, _ = self.decoder.correction(optical, sar, coarse, context, encoded[1], encoded[3])
        return image, coarse, base, detail

    def forward(self, optical, sar, return_features=False):
        if return_features:
            encoded, diagnostics = self.encode_pair(optical, sar, return_diagnostics=True)
        else:
            encoded = self.encode_pair(optical, sar)
        image, coarse, base, detail = self.corrected(optical, sar, encoded)
        if not return_features:
            return image
        return image, dict(optical_base=encoded[0], optical_detail=encoded[1], sar_base=encoded[2],
                           sar_detail=encoded[3], fused_base=base, fused_detail=detail,
                           coarse_image=coarse, lwpd=diagnostics)


def local_error(images, reference, valid):
    """Average local intensity and signed one-pixel difference errors for each candidate."""
    diff = images.float() - reference.float()
    dx = F.pad((diff[..., 1:] - diff[..., :-1]).abs() * valid[..., :-1] * valid[..., 1:], (0, 1))
    dy = F.pad((diff[..., 1:, :] - diff[..., :-1, :]).abs() * valid[..., :-1, :] * valid[..., 1:, :],
               (0, 0, 0, 1))
    pool = lambda x: F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
    return pool(diff.abs() * valid + 0.5 * (dx + dy)) / pool(valid).clamp_min(1e-6)


def reconstruction_forward(model, losses, optical, sar, valid, args):
    bo, do, bs, ds = model.encode_pair(optical, sar)
    rec_losses, aux_losses, rankings = [], [], []
    for source, base, detail, mask in ((optical, bo, do, torch.ones_like(valid)), (sar, bs, ds, valid)):
        coarse, context = model.decoder(None, base, detail)
        rec_losses.append(losses.reconstruction(source, coarse, mask))
        h, w = source.shape[-2:]
        # Alternate degraded and intact areas. Validation always uses the same phase.
        phase = torch.randint(2, (source.shape[0], 1, 1, 1), device=source.device) if model.training else 0
        tiles = (torch.arange(h, device=source.device)[:, None] // 16
                 + torch.arange(w, device=source.device)[None, :] // 16 + phase) % 2
        clean = coarse.detach().float()
        blurred = F.avg_pool2d(clean, SPEC["blur_size"], 1, SPEC["blur_size"] // 2, count_include_pad=False)
        initial = torch.where(tiles.bool(), blurred, clean)
        corrected, images, scores = model.decoder.correction(optical, sar, initial, context, do, ds)
        error = local_error(images.detach(), source, mask)
        # SAR action is unavailable at invalid SAR pixels, including for optical reconstruction.
        error[:, 2:] = error[:, 2:].masked_fill(sar <= 0, float("inf"))
        best, indices = error.topk(2, dim=1, largest=False)
        decisive = ((best[:, 1] - best[:, 0] > SPEC["ranking_margin"]) & (mask[:, 0] > 0))
        rank = F.cross_entropy(scores, indices[:, 0], reduction="none")
        rankings.append((rank * decisive).sum() / decisive.sum().clamp_min(1))
        aux_losses.append(losses.reconstruction(source, corrected, mask))
    rec = sum(rec_losses) / 2
    aux, rank = sum(aux_losses) / 2, sum(rankings) / 2
    return rec + SPEC["reconstruction_aux_weight"] * aux + SPEC["ranking_weight"] * rank, dict(
        reconstruction=rec, optical_reconstruction=rec_losses[0], sar_reconstruction=rec_losses[1],
        correction_reconstruction=aux, ranking=rank)


def fusion_forward(model, losses, optical, sar, valid, args):
    encoded, diagnostics = model.encode_pair(optical, sar, return_diagnostics=True)
    image, coarse, _, _ = model.corrected(optical, sar, encoded)
    lo = losses.reconstruction(optical, model.reconstruct(optical, *encoded[:2]))
    ls = losses.reconstruction(sar, model.reconstruct(sar, *encoded[2:]), valid)
    rec = (lo + ls) / 2
    fusion, intensity, gradient = losses.fusion(optical, sar, image, valid, args.gradient_weight)
    initial = losses.fusion(optical, sar, coarse, valid, args.gradient_weight)[0]
    alpha = SPEC["coarse_fusion_weight"]
    total = (1 - alpha) * fusion + alpha * initial + args.reconstruction_weight * rec
    return total, dict(fusion=fusion, intensity=intensity, gradient=gradient, coarse_fusion=initial,
                       optical_reconstruction=lo, sar_reconstruction=ls, reconstruction=rec,
                       gate_mean=diagnostics["sar_gates"].mean(),
                       correction_mae=(image.detach() - coarse.detach()).abs().mean())
