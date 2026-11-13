"""E2-G1: detached common-structure PA evidence, bounded directional targets."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from phase_antisymmetry import MonogenicPhaseAntisymmetry


class PADirectionalGradientLoss(nn.Module):
    spec = dict(version='e2_g1_v1', tangent_weight=.25, huber_beta=.1,
                sar_increment_cap=.25, confidence_floor=.3,
                wavelengths=[4., 8., 16.], reflect_padding=16,
                reduction='per-image spatial mean then batch mean')

    def __init__(self):
        super().__init__()
        self.phase = MonogenicPhaseAntisymmetry(wavelengths=(4.,8.,16.), reflect_padding=16)
        # Image coordinates: x points right, y points down (same as PA atan2).
        k = torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])
        self.register_buffer('kernels', torch.stack((k,k.T))[:,None])

    def gradients(self, image):
        return F.conv2d(F.pad(image,(1,1,1,1),mode='reflect'),self.kernels.float())

    @staticmethod
    def agreement(a, b):
        return torch.cos(2*(a-b)).clamp_min(0)

    @torch.no_grad()
    def evidence(self, optical, sar, valid):
        po, ps = self.phase(optical), self.phase(sar)
        # Mean over scales rewards persistence instead of selecting the largest peak.
        co = ((po['fa_mean']-.3)/.7).clamp(0,1)
        cs = ((ps['fa_mean']-.3)/.7).clamp(0,1)
        mask = co*cs*self.agreement(po['orientation'],ps['orientation'])
        mask = F.avg_pool2d(mask,3,1,1)
        # Erode after smoothing: invalid Sobel support must never regain weight.
        support = 1-F.max_pool2d(1-valid,3,1,1)
        return (mask*support).detach(), po['orientation'].detach()

    def forward(self, optical, sar, fused, valid):
        if optical.shape != sar.shape or optical.shape != fused.shape or valid.shape != sar.shape:
            raise ValueError('Expected matching Bx1xHxW tensors')
        if optical.ndim != 4 or optical.shape[1] != 1 or min(optical.shape[-2:]) <= 16:
            raise ValueError('PA loss requires grayscale images larger than 16x16')
        with autocast(enabled=False):
            optical, fused = optical.float(), fused.float()
            valid = (valid.detach()>.5).float()
            # FFT has global support; fill invalid SAR with optical before evidence extraction.
            sar = torch.where(valid.bool(),sar.detach().float(),optical.detach())
            mask, theta = self.evidence(optical.detach(),sar,valid)
            go, gs, gf = [self.gradients(x) for x in (optical.detach(),sar,fused)]
            nx, ny = theta.cos(), theta.sin()
            project = lambda g: (g[:,0:1]*nx+g[:,1:2]*ny, -g[:,0:1]*ny+g[:,1:2]*nx)
            on, ot = project(go); sn, _ = project(gs); fn, ft = project(gf)
            # Unsigned projections avoid requiring matching modality contrast polarity.
            target = torch.minimum(torch.maximum(on.abs(),sn.abs()),on.abs()+.25)
            robust = lambda x,y: F.smooth_l1_loss(x,y,beta=.1,reduction='none')
            directional = robust(fn.abs(),target)+.25*robust(ft.abs(),ot.abs())
            fallback = robust(gf.abs().sum(1,keepdim=True),go.abs().sum(1,keepdim=True))
            return ((1-mask)*fallback+mask*directional).flatten(1).mean(1).mean()
