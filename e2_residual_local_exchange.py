"""Add reciprocal Detail exchange to E2's third dilation branch."""

import torch
from torch import nn

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_local_correspondence import CENTER, LocalCorrespondenceDetailFusion, LocalCorrespondenceE2


class ResidualLocalExchangeDetailFusion(LocalCorrespondenceDetailFusion):
    def __init__(self, e2_head):
        super().__init__()
        # Keep the actual E2 projections, all three branches and the aggregation
        # weights that were initialized alongside this model's encoder/decoder.
        for name, module in e2_head.named_children():
            setattr(self, name, module)
        exchange_channels = self.sar_value.out_channels + self.optical_value.out_channels
        self.residual_projection = nn.Conv2d(exchange_channels, self.optical_projection[0].out_channels,
                                             kernel_size=1, bias=False)
        nn.init.zeros_(self.residual_projection.weight)

    def forward(self, optical_detail, sar_detail, fused_base, *, sar_valid=None, return_diagnostics=False):
        optical, sar, base = self._project_inputs(optical_detail, sar_detail, fused_base)
        if sar_valid is not None and sar_valid.shape != optical_detail[:, :1].shape:
            raise ValueError("Expected a single-channel SAR validity map on the Detail grid")
        joint = torch.cat((optical, sar), dim=1)
        local = [branch(joint) for branch in self.scale_branches]
        o_bands, s_bands, size, (os, so) = self._describe(optical_detail, sar_detail, sar_valid)
        sar_at_optical = self._exchange(s_bands, self.sar_value, os, size)
        optical_at_sar = self._exchange(o_bands, self.optical_value, so, size)
        residual = self.residual_projection(torch.cat((sar_at_optical, optical_at_sar), dim=1))
        local[2] = local[2] + residual
        fused = self.output(torch.cat((optical, sar, *local, base), dim=1))
        if not return_diagnostics:
            return fused
        return fused, dict(match_mass=(os.sum(1).mean() + so.sum(1).mean()) / 2,
                           offcenter_mass=((os.sum(1) - os[:, CENTER]).mean()
                                           + (so.sum(1) - so[:, CENTER]).mean()) / 2,
                           optical_zero_exchange_fraction=(os.sum(1) == 0).float().mean(),
                           sar_zero_exchange_fraction=(so.sum(1) == 0).float().mean(),
                           residual_rms=residual.float().square().mean().sqrt())


class ResidualLocalExchangeE2(LocalCorrespondenceE2):
    VARIANT = "lwpd_e2_residual_local_exchange_v1"

    def __init__(self):
        # The replacement variant discards the original E2 Detail head in its
        # constructor. Start with E2 directly and inherit its SAR-masked exchange
        # variant's prepare_fusion_inputs and fuse_features methods.
        OpticalBaseAnchoredE2.__init__(self, self.VARIANT)
        with torch.random.fork_rng(devices=[]):
            self.detail_fusion = ResidualLocalExchangeDetailFusion(self.detail_fusion)
