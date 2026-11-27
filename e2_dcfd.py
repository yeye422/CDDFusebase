"""Dual-source collaborative fusion decoder initialized from CF1."""

from copy import deepcopy
from typing import NamedTuple

import torch
from torch import nn

from e2_base_anchor import OpticalBaseAnchoredE2
from e2_cfdr import CFDRE2, VARIANTS


def zero_projection(in_channels, out_channels):
    layer = nn.Conv2d(in_channels, out_channels, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def depthwise():
    return nn.Conv2d(64, 64, 3, padding=1, groups=64, padding_mode="replicate")


class DualSourceInjection(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(nn.Conv2d(in_channels, 64, 1), nn.GELU(), depthwise(), zero_projection(64, 64))


class BidirectionalPathCollaboration(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_to_d = nn.Sequential(nn.Conv2d(64, 64, 1), nn.GELU(), depthwise(), zero_projection(64, 128))
        self.detail_hint = nn.Sequential(depthwise(), nn.Conv2d(64, 64, 1), nn.Sigmoid())
        self.d_to_s = zero_projection(64, 64)

    def forward(self, structure, detail):
        gamma_raw, beta = self.s_to_d(structure).chunk(2, dim=1)
        gamma = 0.5 * gamma_raw.tanh()
        hint = self.detail_hint(detail)
        return structure + self.d_to_s(hint * detail), detail + gamma * detail + beta


class Reconstruction(NamedTuple):
    base: torch.Tensor
    detail: torch.Tensor
    raw: torch.Tensor
    fused: torch.Tensor
    source_base: torch.Tensor
    gain: torch.Tensor


def reconstruct(output, sigmoid, cfdr, structure, detail, fused_detail, optical, sar, image_skip):
    logits = output(structure)
    if image_skip:
        logits = logits + optical
    prediction = sigmoid(logits)
    context = cfdr.condition(torch.cat((detail, fused_detail), dim=1))
    gain = cfdr.gain(context).softmax(dim=1, dtype=torch.float32)
    with torch.autocast(device_type=optical.device.type, enabled=False):
        inputs = torch.cat((prediction.float(), optical.float(), sar.float()), dim=1)
        base, means = cfdr.smooth(inputs).split((1, 2), dim=1)
        residual = (gain[:, :2] * (inputs[:, 1:] - means)).sum(dim=1, keepdim=True)
        raw = base + residual
    return Reconstruction(base, residual, raw, raw.clamp(0, 1), means, gain)


class DCFDDecoder(nn.Module):
    def __init__(self, cf1):
        super().__init__()
        decoder = cf1.decoder
        first, second, third, fourth = decoder.encoder_level2
        self.shared = nn.Sequential(decoder.reduce_channel, first, second).requires_grad_(False)
        self.structure = nn.ModuleList((deepcopy(third), deepcopy(fourth)))
        self.detail = nn.ModuleList((deepcopy(third), deepcopy(fourth)))
        self.output = deepcopy(decoder.output)
        self.sigmoid = deepcopy(decoder.sigmoid)
        self.cfdr = deepcopy(cf1.cfdr)
        self.image_skip = cf1.uses_image_skip
        self.dsbi = DualSourceInjection(192)
        self.dsdi = DualSourceInjection(128)
        self.bpc = BidirectionalPathCollaboration()

    def forward(self, shared, encoded, fused_detail, optical, sar):
        base_o, detail_o, base_s, detail_s = encoded
        structure = self.structure[0](shared + self.dsbi(torch.cat((base_o, base_s, base_o - base_s), dim=1)))
        detail = self.detail[0](shared + self.dsdi(torch.cat((detail_o, detail_s), dim=1)))
        structure, detail = self.bpc(structure, detail)
        structure, detail = self.structure[1](structure), self.detail[1](detail)
        terms = reconstruct(self.output, self.sigmoid, self.cfdr, structure, detail, fused_detail,
                            optical, sar, self.image_skip)
        return terms, structure, detail


class CF1Teacher(nn.Module):
    def __init__(self, cf1):
        super().__init__()
        self.blocks = deepcopy(cf1.decoder.encoder_level2[2:])
        self.output = deepcopy(cf1.decoder.output)
        self.sigmoid = deepcopy(cf1.decoder.sigmoid)
        self.cfdr = deepcopy(cf1.cfdr)
        self.image_skip = cf1.uses_image_skip
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, shared, fused_detail, optical, sar):
        hidden = self.blocks(shared)
        return reconstruct(self.output, self.sigmoid, self.cfdr, hidden, hidden, fused_detail,
                           optical, sar, self.image_skip)


class DCFDE2(OpticalBaseAnchoredE2):
    VARIANT = "lwpd_e2_dcfd_v1"

    def __init__(self, cf1=None):
        # Take ownership of the initialized front end without allocating a second copy.
        nn.Module.__init__(self)
        cf1 = CFDRE2("CF1") if cf1 is None else cf1
        self.variant = self.VARIANT
        self.encoder = cf1.encoder.requires_grad_(False)
        self.lwpd = cf1.lwpd.requires_grad_(False)
        self.base_fusion = cf1.base_fusion.requires_grad_(False)
        self.detail_fusion = cf1.detail_fusion.requires_grad_(False)
        self.decoder = DCFDDecoder(cf1)
        self.train(False)

    @classmethod
    def initialized(cls, checkpoint):
        if (checkpoint["arm"], checkpoint["epoch"], checkpoint["variant"]) != ("CF1", 4, VARIANTS["CF1"]):
            raise ValueError("DCFD initialization requires CF1 epoch 4")
        cf1 = CFDRE2("CF1")
        cf1.load_state_dict(checkpoint["model"], strict=True)
        return cls(cf1), CF1Teacher(cf1)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.encoder, self.lwpd, self.base_fusion, self.detail_fusion, self.decoder.shared):
            module.eval()
        return self

    def parameter_groups(self):
        inherited = (self.decoder.structure, self.decoder.detail, self.decoder.output, self.decoder.cfdr)
        introduced = (self.decoder.dsbi, self.decoder.dsdi, self.decoder.bpc)
        return [[p for module in modules for p in module.parameters()] for modules in (inherited, introduced)]

    @torch.no_grad()
    def prepare(self, optical, sar):
        encoded = self.encode_pair(optical, sar)
        base, detail = self.fuse_features(*encoded)
        shared = self.decoder.shared(torch.cat((base, detail), dim=1))
        return shared, encoded, detail

    def forward(self, optical, sar):
        terms, _, _ = self.decoder(*self.prepare(optical, sar), optical, sar)
        return terms.fused
