"""BiSHIP and architecture controls at the LWPD-E2 Detail interface."""

import torch
from torch import nn
from torch.nn import functional as F

from direct_fusion import MultiScaleDetailFusion


MAIN_CONTROLS = ("B0", "B1", "B2", "B3", "B7")
METHODS = (*MAIN_CONTROLS, "B4", "B5", "B6", "B8", "K3")


def projection(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 1), nn.GELU())


def depthwise(width):
    return nn.Conv2d(width, width, 3, padding=1, groups=width)


def circular_product(query, key):
    # No conjugation: SHIP uses circular convolution, with wrap-around at the border.
    with torch.autocast(device_type=query.device.type, enabled=False):
        spectrum = torch.fft.rfft2(query.float()) * torch.fft.rfft2(key.float())
        return torch.fft.irfft2(spectrum, s=query.shape[-2:])


class DualStreamProjection(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.optical = projection(channels, width)
        self.sar = projection(channels, width)

    def forward(self, optical, sar):
        return self.optical(optical), self.sar(sar)


class BidirectionalSpatialInteraction(nn.Module):
    def __init__(self, width, local=False):
        super().__init__()
        self.optical_qkv = nn.Conv2d(width, 3 * width, 1)
        self.sar_qkv = nn.Conv2d(width, 3 * width, 1)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2))
        self.local = nn.ModuleList(depthwise(width) for _ in range(2)) if local else None
        self.second = nn.ModuleList(depthwise(width) for _ in range(2))
        self.third = nn.ModuleList(depthwise(width) for _ in range(2))
        self.hidden = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(2))
        self.value = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(2))

    def forward(self, optical, sar):
        qo, ko, vo = self.optical_qkv(optical).chunk(3, dim=1)
        qs, ks, vs = self.sar_qkv(sar).chunk(3, dim=1)
        outputs = []
        for i, (query, key, value) in enumerate(((qo, ks, vs), (qs, ko, vo))):
            relation = circular_product(query, key) if self.local is None else self.local[i](query + key)
            with torch.autocast(device_type=query.device.type, enabled=False):
                relation = self.norms[i](relation.float().movedim(1, -1)).movedim(-1, 1)
            second = self.second[i](relation.tanh().to(value.dtype) * value.tanh())
            third = self.third[i](self.hidden[i](second).tanh() * self.value[i](value).tanh())
            outputs.append(second + third)
        return tuple(outputs)


class HighOrderChannelInteraction(nn.Module):
    def __init__(self, width, inputs=None):
        super().__init__()
        inputs = inputs or width * 4
        self.first = nn.Sequential(projection(inputs, width * 2), nn.Conv2d(width * 2, width * 2, 1))
        self.condition = nn.Conv2d(inputs, width * 2, 1)
        self.second = nn.Conv2d(width * 2, width * 2, 1)

    def forward(self, states):
        pooled = torch.cat(tuple(state.mean(dim=(-2, -1), keepdim=True) for state in states), dim=1)
        code = self.second(self.first(pooled).tanh() * self.condition(pooled)).tanh()
        return code.chunk(2, dim=1)


class SourcePreservingUpdate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Conv2d(width * 2, width, 1)
        self.gamma = nn.Parameter(torch.full((), 0.05))

    def forward(self, state, message, modulation=None):
        adjusted = message if modulation is None else message * (1 + 0.1 * modulation.tanh())
        delta = self.projection(torch.cat((adjusted, message), dim=1))
        return state + self.gamma.to(state.dtype) * delta.tanh()


class DualStreamCNN(nn.Module):
    """Independent readouts of a shared local neighborhood with additive state updates."""

    def __init__(self, width):
        super().__init__()
        hidden = 172  # Matches the default B7 Detail parameter count within 0.2%.
        self.read = nn.ModuleList(nn.Sequential(projection(2 * width, hidden), depthwise(hidden),
                                               nn.GELU(), nn.Conv2d(hidden, width, 1)) for _ in range(2))

    def forward(self, optical, sar):
        return tuple(read(torch.cat((own, other), dim=1))
                     for read, own, other in zip(self.read, (optical, sar), (sar, optical)))


class WindowCrossAttention(nn.Module):
    """Standard QK softmax cross-attention, tiled into 8x8 spatial windows."""

    def __init__(self, width):
        super().__init__()
        self.qkv = nn.ModuleList(nn.Conv2d(width, 3 * width, 1) for _ in range(2))
        hidden = 234
        self.readout = nn.ModuleList(nn.Sequential(projection(width, hidden),
                                                   nn.Conv2d(hidden, width, 1)) for _ in range(2))
        self.heads = 4
        self.window = 8

    def forward(self, optical, sar):
        batch, channels, height, width = optical.shape
        win = self.window
        pad_h, pad_w = (-height) % win, (-width) % win
        rows, cols = (height + pad_h) // win, (width + pad_w) // win
        heads, head_dim = self.heads, channels // self.heads

        def windows(value, nheads, dim):
            value = F.pad(value, (0, pad_w, 0, pad_h))
            value = value.reshape(batch, nheads, dim, rows, win, cols, win)
            return value.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, nheads, win * win, dim)

        mask = None
        if pad_h or pad_w:
            valid = torch.ones_like(optical[:, :1], dtype=torch.bool)
            mask = windows(valid, 1, 1).transpose(-1, -2)
        (qo, ko, vo), (qs, ks, vs) = (
            [windows(value, heads, head_dim) for value in layer(state).chunk(3, dim=1)]
            for layer, state in zip(self.qkv, (optical, sar)))
        messages = []
        for read, (q, k, v) in zip(self.readout, ((qo, ks, vs), (qs, ko, vo))):
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            value = value.reshape(batch, rows, cols, heads, win, win, head_dim)
            value = value.permute(0, 3, 6, 1, 4, 2, 5).reshape(batch, channels, rows * win, cols * win)
            messages.append(read(value[..., :height, :width]))
        return tuple(messages)


class BiSHIPStage(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode = mode
        if mode in ("full", "spatial", "local"):
            self.spatial = BidirectionalSpatialInteraction(width, local=mode == "local")
        elif mode == "cnn":
            self.spatial = DualStreamCNN(width)
        elif mode == "attention":
            self.spatial = WindowCrossAttention(width)
        else:
            self.spatial = None
        self.channel = (HighOrderChannelInteraction(width, 2 * width if mode == "channel" else None)
                        if mode in ("full", "local", "channel") else None)
        self.updates = nn.ModuleList(SourcePreservingUpdate(width) for _ in range(2))

    def forward(self, optical, sar):
        states = (optical, sar)
        messages = states if self.spatial is None else self.spatial(*states)
        modulations = (None, None)
        if self.channel is not None:
            modulations = self.channel(states if self.spatial is None else states + messages)
        updated = tuple(update(state, message, modulation)
                        for update, state, message, modulation in zip(self.updates, states, messages, modulations))
        return updated, messages


def synthesis(inputs, channels):
    return nn.Sequential(projection(inputs, channels), depthwise(channels), nn.GELU(),
                         nn.Conv2d(channels, channels, 1))


class BiSHIPDetailFusion(nn.Module):
    def __init__(self, channels=64, width=32, stages=2, mode="full"):
        super().__init__()
        if stages not in (1, 2, 3) or mode not in ("full", "spatial", "channel", "local", "cnn", "attention"):
            raise ValueError("Unsupported BiSHIP stage count or mode")
        self.source_projection = DualStreamProjection(channels, width)
        self.base_projection = nn.Conv2d(channels, 16, 1)
        self.stages = nn.ModuleList(BiSHIPStage(width, mode) for _ in range(stages))
        self.synthesis = synthesis(6 * width + 16, channels)

    def forward(self, optical, sar, base, *, diagnostics=None):
        initial = states = self.source_projection(optical, sar)
        for i, stage in enumerate(self.stages, start=1):
            previous = states
            states, messages = stage(*states)
            if diagnostics is not None:
                for name, state, old, message, update in zip(("optical", "sar"), states, previous, messages, stage.updates):
                    for key, value in (("message", message), ("state", state), ("update", state - old)):
                        tensor = value.detach().float()
                        diagnostics[f"stage{i}_{name}_{key}_rms"] = tensor.square().mean().sqrt()
                        diagnostics[f"stage{i}_{name}_{key}_absmax"] = tensor.abs().max()
                    diagnostics[f"stage{i}_{name}_gamma"] = update.gamma.detach()
        return self.synthesis(torch.cat(initial + states + messages + (self.base_projection(base),), dim=1))


class AsymmetricSHIPStage(nn.Module):
    """Independent, compact adaptation of SHIP's asymmetric four-order topology."""

    def __init__(self, width):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.refresh = nn.ModuleList(nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                                               nn.ReLU(), nn.Conv2d(width, width, 1))
                                                    for _ in range(3)) for _ in range(2))
        self.channel = nn.ModuleList(nn.Sequential(projection(2 * width, width), nn.Conv2d(width, 2 * width, 1))
                                     for _ in range(4))
        self.mix = nn.ModuleList(nn.Sequential(depthwise(2 * width), projection(2 * width, 2 * width))
                                 for _ in range(3))
        self.merge = nn.Conv2d(2 * width, width, 1)

    def forward(self, optical, sar):
        interaction = circular_product(optical, sar)
        for order, norm in enumerate(self.norms):
            interaction = norm(interaction.float().movedim(1, -1)).movedim(-1, 1).to(sar.dtype) * sar
            if order < len(self.refresh[0]):
                interaction = self.refresh[0][order](interaction)
                sar = self.refresh[1][order](sar)
        optical = optical + interaction
        joint = torch.cat((optical, sar), dim=1)
        descriptor = joint.mean(dim=(-2, -1), keepdim=True)
        for i, project in enumerate(self.channel):
            descriptor = project(descriptor).softmax(dim=1)
            joint = joint * descriptor
            if i < len(self.mix):
                joint = self.mix[i](joint)
        return optical + self.merge(joint), sar


class AsymmetricSHIPDetail(nn.Module):
    def __init__(self, channels=64, width=18):
        super().__init__()
        self.source_projection = DualStreamProjection(channels, width)
        self.stages = nn.ModuleList(AsymmetricSHIPStage(width) for _ in range(3))
        self.base_projection = nn.Conv2d(channels, 16, 1)
        self.synthesis = synthesis(3 * width + 16, channels)

    def forward(self, optical, sar, base):
        optical, sar = self.source_projection(optical, sar)
        states = []
        for stage in self.stages:
            optical, sar = stage(optical, sar)
            states.append(optical)
        return self.synthesis(torch.cat((*states, self.base_projection(base)), dim=1))


def build_detail(method):
    if method == "B0":
        return MultiScaleDetailFusion()
    if method == "B3":
        return AsymmetricSHIPDetail()
    mode = {"B1": "cnn", "B2": "attention", "B4": "spatial", "B5": "channel", "B6": "full",
            "B7": "full", "B8": "local", "K3": "full"}[method]
    return BiSHIPDetailFusion(stages={"B6": 1, "K3": 3}.get(method, 2), mode=mode)
