"""Training-only appearance-anchored proxy views for C03-H."""

import math

import torch
from torch.nn import functional as F

from tr0_complementary import ComplementaryViewGenerator


class HeterogeneousMaskGenerator(ComplementaryViewGenerator):
    def _masks(self, height, width, generator):
        size = int(torch.randint(self.field_size[0], self.field_size[1] + 1, (), generator=generator))
        coarse = torch.randn(1, 2, size, size, generator=generator)
        fields = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)[0]
        first, second = fields - fields.mean(dim=(-2, -1), keepdim=True)
        first = first / first.std().clamp_min(1e-6)
        # Set a mildly negative spatial correlation before thresholding both fields
        # to about 60% clear. This gives overlap and leaves shared degraded areas.
        second = second - (first * second).sum() / first.square().sum().clamp_min(1e-6) * first
        second = second / second.std().clamp_min(1e-6) - 0.25 * first
        fields = torch.stack((first, second))
        cutoffs = torch.quantile(fields.flatten(1), 0.4, dim=1)[:, None, None]
        transitions = 0.15 * fields.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid((fields - cutoffs) / transitions + math.log(4))


class AppearanceAnchoredViewGenerator:
    def __init__(self, *, field_size, sigma, contrast_min=0.7):
        if not 0 < contrast_min < 1:
            raise ValueError("contrast_min must be in (0, 1)")
        self.v1 = ComplementaryViewGenerator(field_size, sigma)
        self.v2 = HeterogeneousMaskGenerator(field_size, sigma)
        self.contrast_min = contrast_min
        self._base_rng = None
        self._appearance_rng = torch.Generator()

    @torch.no_grad()
    def mixed_views(self, optical, generator):
        first, second, _, _ = self.v1(optical, generator)
        if self._base_rng is not generator:
            self._base_rng = generator
            self._appearance_rng.manual_seed(generator.initial_seed() + 100003)
        rng = self._appearance_rng
        count = len(optical)
        v2_count = count // 2 + (count % 2 and int(torch.randint(2, (), generator=rng)))
        selected = torch.randperm(count, generator=rng)[:v2_count].to(optical.device)
        stats = {"hcsl_fraction": optical.new_tensor(v2_count / count)}
        if v2_count:
            anchor, supplement, _, v2_stats = self.v2(optical.index_select(0, selected), rng)
            gain = torch.empty(v2_count).uniform_(self.contrast_min, 1, generator=rng)
            offset = (1 - gain) * torch.rand(v2_count, generator=rng)
            transformed = gain.to(optical.device)[:, None, None, None] * supplement
            transformed += offset.to(optical.device)[:, None, None, None]
            first = first.index_copy(0, selected, anchor)
            second = second.index_copy(0, selected, transformed)
            stats.update(hcsl_gain=gain.mean(), hcsl_offset=offset.mean(),
                         hcsl_anchor_only=v2_stats["clear_first_only"],
                         hcsl_supplement_only=v2_stats["clear_second_only"],
                         hcsl_both_degraded=v2_stats["both_degraded"],
                         hcsl_both_clear=v2_stats["clear_union"] - v2_stats["clear_first_only"]
                         - v2_stats["clear_second_only"])
        return first, second, stats, selected

    def __call__(self, optical, generator):
        first, second, stats, _ = self.mixed_views(optical, generator)
        return first, second, None, stats
