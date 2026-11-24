"""Generate two spatially degraded views of a clean optical luminance patch."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ComplementaryViewGenerator:
    def __init__(self, field_size=(4, 8), sigma=(1.0, 3.0)):
        self.field_size = field_size
        self.sigma = sigma

    @staticmethod
    def _coverage(masks):
        first, second = masks.unbind()
        first, second = first >= 0.8, second >= 0.8
        return torch.stack(((first & ~second).float().mean(),
                            (second & ~first).float().mean(),
                            (~first & ~second).float().mean(),
                            (first | second).float().mean()))

    def _masks(self, height, width, generator):
        size = int(torch.randint(self.field_size[0], self.field_size[1] + 1, (), generator=generator))
        fields = torch.randn(1, 2, size, size, generator=generator)
        first, perturbation = F.interpolate(
            fields, size=(height, width), mode="bicubic", align_corners=False
        )[0].unbind()
        quantiles = 0.38 + 0.12 * torch.rand(2, generator=generator)
        # Vary the second field and both thresholds so the masks can overlap.
        for noise_scale in (0.25, 0.125, 0.0):
            second = -first + noise_scale * perturbation
            masks = []
            for field, quantile in zip((first, second), quantiles):
                cutoff = torch.quantile(field.flatten(), quantile)
                transition = (0.15 * field.std()).clamp_min(1e-6)
                masks.append(torch.sigmoid((field - cutoff) / transition + math.log(4)))
            masks = torch.stack(masks)
            coverage = self._coverage(masks)
            if bool((coverage[:2] >= 0.2).all() and coverage[2] <= 0.15 and coverage[3] >= 0.85):
                return masks
        raise RuntimeError("Unable to generate masks that satisfy the TR0 coverage constraints")

    def _blur(self, patches, sigmas):
        count = patches.shape[0]
        radius = math.ceil(3 * self.sigma[1])
        offsets = torch.arange(-radius, radius + 1, device=patches.device, dtype=patches.dtype)
        sigma = sigmas.to(device=patches.device, dtype=patches.dtype)
        weights = torch.exp(-0.5 * (offsets / sigma[:, None]).square())
        weights /= weights.sum(dim=1, keepdim=True)
        mode = "reflect" if min(patches.shape[-2:]) > radius else "replicate"
        horizontal = F.pad(patches.transpose(0, 1), (radius, radius, 0, 0), mode=mode)
        horizontal = F.conv2d(horizontal, weights[:, None, None, :], groups=count)
        vertical = F.pad(horizontal, (0, 0, radius, radius), mode=mode)
        return F.conv2d(vertical, weights[:, None, :, None], groups=count).transpose(0, 1)

    @torch.no_grad()
    def __call__(self, optical, generator):
        if optical.ndim != 4 or optical.shape[1] != 1 or min(optical.shape[-2:]) < 2:
            raise ValueError("Expected N×1 optical luminance patches with both dimensions >= 2")
        count, _, height, width = optical.shape
        masks = torch.stack([self._masks(height, width, generator) for _ in range(count)])
        sigma = torch.empty(count, 2).uniform_(*self.sigma, generator=generator)
        swap = torch.rand(count, generator=generator) < 0.5
        repeat_second = torch.rand(count, generator=generator) < 0.5
        first_indexes = swap.long()
        indices = torch.arange(count)
        permutation = torch.stack((first_indexes, 1 - first_indexes), dim=1)
        masks = masks[indices[:, None], permutation]
        sigma = sigma[indices[:, None], permutation]
        coverage = torch.stack([self._coverage(mask) for mask in masks]).mean(dim=0)
        source = optical.float().repeat_interleave(2, dim=0)
        keep = masks.flatten(0, 1).unsqueeze(1).to(source.device)
        blurred = self._blur(source, sigma.flatten())
        first, second = (keep * source + (1 - keep) * blurred).reshape(
            count, 2, 1, height, width
        ).unbind(dim=1)
        repeated = torch.where(repeat_second.to(source.device)[:, None, None, None], second, first)
        stats = dict(zip(("clear_first_only", "clear_second_only", "both_degraded", "clear_union"), coverage))
        stats.update(sigma=sigma.mean(), swap_rate=swap.float().mean(),
                     degradation_first=(first - optical).abs().mean(),
                     degradation_second=(second - optical).abs().mean())
        return first, second, repeated, stats


class ComplementaryReconstructionLoss(nn.Module):
    def __init__(self, gradient_weight=0.5):
        super().__init__()
        self.gradient_weight = gradient_weight

    def forward(self, prediction, target):
        prediction, target = prediction.float(), target.float()
        pixels = F.l1_loss(prediction, target)
        gradients = 0.5 * sum(
            F.l1_loss(prediction.diff(dim=axis), target.diff(dim=axis)) for axis in (-2, -1)
        )
        return pixels + self.gradient_weight * gradients, pixels, gradients
