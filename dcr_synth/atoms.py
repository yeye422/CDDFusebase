"""Source perturbations and their tangents in fused Detail coordinates."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import jvp


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    size: int
    group: int

    def slices(self):
        radius = self.size // 2
        return (slice(self.y - radius, self.y + radius + 1),
                slice(self.x - radius, self.x + radius + 1))

    def mask(self, image):
        mask = torch.zeros_like(image, dtype=torch.bool)
        y, x = self.slices()
        mask[..., y, x] = True
        return mask


@dataclass
class Atom:
    source: str
    band: str
    window: Window
    input_rms: float
    direction: torch.Tensor
    support: str = "input_local"

    @property
    def name(self):
        return f"{self.source}_{self.band}_{self.window.x}_{self.window.y}"


def rms(value):
    return value.square().mean().sqrt()


def bands(detail):
    small = F.avg_pool2d(detail, 3, stride=1, padding=1, count_include_pad=False)
    large = F.avg_pool2d(detail, 7, stride=1, padding=3, count_include_pad=False)
    return {"3": detail - small, "3_7": small - large}


def localized(value, window, floor):
    y, x = window.slices()
    region = value[..., y, x]
    centered = region - region.mean(dim=(-2, -1), keepdim=True)
    energy = float(rms(centered))
    if energy <= floor:
        return None, energy
    perturbation = torch.zeros_like(value)
    perturbation[..., y, x] = centered / energy
    return perturbation, energy


@torch.no_grad()
def source_atoms(fuse, optical, sar, windows, floor, profiles=None):
    if optical.ndim != 4 or optical.shape[1] != 64 or optical.shape != sar.shape:
        raise ValueError("Expected equal NCHW 64-channel Detail features")
    atoms, rejected = [], []
    for index, (name, detail) in enumerate((("optical", optical), ("sar", sar))):
        directions = bands(detail) if profiles is None else {"profile": profiles[index]}
        for window in windows:
            for band, value in directions.items():
                action, energy = localized(value, window, floor)
                if action is not None:
                    if index == 0:
                        _, mapped = jvp(lambda source: fuse(source, sar), (optical,), (action,))
                    else:
                        _, mapped = jvp(lambda source: fuse(optical, source), (sar,), (action,))
                    if mapped.shape != detail.shape or not torch.isfinite(mapped).all():
                        raise ValueError("Invalid fused Detail tangent")
                    if float(rms(mapped)) > floor:
                        atoms.append(Atom(name, band, window, energy, mapped))
                        continue
                rejected.append(dict(source=name, band=band, x=window.x, y=window.y,
                                     input_rms=energy))
    return atoms, rejected
