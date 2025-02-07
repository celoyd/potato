from einops import rearrange
import torch
from torch import nn


def cheap_half(x):
    # Fast 2× downsample
    return (
        x[..., 0::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 0::2]
        + x[..., 1::2, 1::2]
    ) / 4.0


def pile(x, factor):
    return rearrange(
        x, "... c (h f0) (w f1) -> ... (c f0 f1) h w", f0=factor, f1=factor
    )


def tile(x, factor):
    return rearrange(
        x, "... (c f0 f1) h w -> ... c (h f0) (w f1)", f0=factor, f1=factor
    )


"""
As-layers approach (required by JIT):

from einops.layers.torch import Rearrange

class Tiler(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.re = Rearrange("... (c f0 f1) h w -> ... c (h f0) (w f1)", f0=f, f1=f)
    def forward(self, x):
        return self.re(x)

class Piler(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.re = Rearrange("... c (h f0) (w f1) -> ... (c f0 f1) h w", f0=f, f1=f)
    def forward(self, x):
        return self.re(x)

tile2, tile4 = Tiler(2), Tiler(4)
pile2, pile4 = Piler(2), Piler(4)

def pile(x: torch.Tensor, f: int):
    if f == 2:
        return pile2(x)
    elif f == 4:
        return pile4(x)
    else:
        raise ValueError

def tile(x: torch.Tensor, f: int):
    if f == 2:
        return tile2(x)
    elif f == 4:
        return tile4(x)
    else:
        raise ValueError
"""
