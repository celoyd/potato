import torch
from torch import nn
import torch.nn.functional as fn
from torch.nn import Mish

from einops import rearrange, reduce
from inspect import isfunction
from functools import partial
from einops.layers.torch import Rearrange

from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTInverse, DWTForward

from ripple.util import pile, tile
from ripple.color import BandsToOklab

c = Mish()

n = 96


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class WaveDownDWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()
        self.dtx = DWTForward(J=1, wave='db1', mode='zero')
        self.treelevel_to_channels = Rearrange("b c f h w -> b (f c) h w")

        self.remap1 = nn.Conv2d(4*in_count, (4*in_count + out_count)//2, 1)
        self.nl = c
        self.remap2 = nn.Conv2d((4*in_count + out_count)//2, out_count, 1)

    def forward(self, x):

        lr, hr = self.dtx(x)

        hr = self.treelevel_to_channels(hr[0])
        x = torch.cat([lr, hr], dim=1)

        x = self.remap1(x)
        x = self.nl(x)
        x = self.remap2(x)

        return x


class WaveUpDWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()

        self.out_count = out_count
        self.utx = DWTInverse(wave='db1', mode='zero')
        self.channels_to_treelevel = Rearrange(
            "b (c f) h w -> b c f h w", f=3, c=out_count
        )

        self.remap1 = nn.Conv2d(in_count, out_count * 4, 3, padding=1, padding_mode="reflect")

        self.nl = c

    def forward(self, x):
        x = self.nl(x)
        x = self.remap1(x)

        Yl = x[:, :self.out_count] * 4
        Yh = self.channels_to_treelevel(x[:, self.out_count:])

        x = self.utx((Yl, [Yh]))

        return x


class WSConv2d(nn.Conv2d):
    # Based on https://arxiv.org/pdf/2102.06171v1.pdf
    def forward(self, x):
        eps = torch.tensor(1e-6).to(x.device)

        weight = self.weight

        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return fn.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Ya(nn.Module):
    # ConvNeXt-inspired
    def __init__(self, in_depth, out_depth):
        super().__init__()

        self.use_res = in_depth == out_depth
        self.nl = c

        self.norm = nn.GroupNorm(4, out_depth)

        self.A = WSConv2d(
            in_depth, out_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.B = nn.Conv2d(out_depth, out_depth, 1, padding=0)
        self.C = WSConv2d(
            out_depth, out_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.D = nn.Conv2d(out_depth, out_depth, 1, padding=0)

    def forward(self, x):
        if self.use_res:
            r = x

        x = self.A(x)
        x = self.norm(x)

        x = self.B(x)
        x = self.nl(x)

        x = self.C(x)
        x = self.norm(x)

        x = self.D(x)

        if self.use_res:
            x = x + r

        return x


class Join(nn.Module):
    # Lightweight non-normalized merging
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.mixer = nn.Conv2d(in_depth, out_depth, 1)

    def forward(self, x):
        return self.mixer(x)


def concat(*a):
    return torch.cat([*a], dim=1)


class Ripple(nn.Module):
    def __init__(self):
        super().__init__()

        self.oklab = BandsToOklab()

        self.pan_to_half = WaveDownDWT(1, 16)
        self.half1 = Ya(16, n//2)

        self.half_to_quarter = WaveDownDWT(n//2, n)

        self.quarter_front_join = Join(n + 8 + 3, n)

        self.quarter_front = Ya(n, n)
        self.quarter_mid = Ya(n, n)
        self.quarter_end = Ya(n, n)

        self.quarter_end_join = Join(n + 3, n)

        self.quarter_to_half = WaveUpDWT(n, n//2)
        self.half2 = Ya(n, n//2)
        self.the_end = WaveUpDWT(n//2, 3)

    def forward(self, x):
        oklab = self.oklab(x[:, 16:])

        x = torch.pow(torch.clamp(x, 1e-9, None), 1 / 3)
        mul = x[:, 16:]
        pan = tile(x[:, :16], 4)

        half1 = self.pan_to_half(pan)
        half1 = self.half1(half1)

        quarter = self.half_to_quarter(half1)
        quarter = self.quarter_front_join(concat(quarter, mul, oklab))

        quarter = self.quarter_front(quarter)
        quarter = self.quarter_mid(quarter)
        quarter = self.quarter_end(quarter)

        quarter = self.quarter_end_join(concat(quarter, oklab))

        half2 = concat(half1, self.quarter_to_half(quarter))
        half2 = self.half2(half2)

        the_end = self.the_end(half2)

        return the_end
