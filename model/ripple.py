import torch
from torch import nn
import torch.nn.functional as fn
from torch.nn import (
    PixelShuffle,
    PixelUnshuffle,
    Mish,
)
from einops import rearrange, reduce
from inspect import isfunction
from functools import partial
from einops.layers.torch import Rearrange

from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTInverse, DWTForward

# Todo: move these into model
shuf2 = PixelShuffle(4).cuda()
unshuf2 = PixelUnshuffle(4).cuda()

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
        self.dtx = DWTForward(J=1, wave="db1", mode="zero")
        self.treelevel_to_channels = Rearrange("b c f h w -> b (f c) h w")

        self.remap1 = nn.Conv2d(
            4 * in_count, (4 * in_count + out_count) // 2, 1, padding=0
        )
        self.nl = c
        self.remap2 = nn.Conv2d(
            (4 * in_count + out_count) // 2, out_count, 1, padding=0
        )

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
        self.utx = DWTInverse(wave="db1", mode="zero")
        self.channels_to_treelevel = Rearrange(
            "b (c f) h w -> b c f h w", f=3, c=out_count
        )

        self.remap1 = nn.Conv2d(in_count, out_count * 4, 3, padding=1, padding_mode='reflect')
        self.nl = c

    def forward(self, x):
        x = self.nl(x)
        x = self.remap1(x)

        Yl = x[:, : self.out_count] * 4
        Yh = self.channels_to_treelevel(x[:, self.out_count :])

        x = self.utx((Yl, [Yh]))

        return x


"""
class WaveUpDTCWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()

        self.out_count = out_count
        self.utx = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.soft = nn.Upsample(scale_factor=2, mode='bicubic')

        self.channels_to_treelevel = Rearrange(
            "b (f c p) h w -> b c f h w p", f=6, c=out_count, p=2
        )

        self.remap1 = nn.Conv2d(in_count, (in_count + out_count * 13)//2, 1, padding=0)
        self.nl = c
        self.remap2 = nn.Conv2d((in_count + out_count * 13)//2, out_count * 13, 1, padding=0)

    def forward(self, x):
        x = self.remap1(x)
        x = self.nl(x)
        x = self.remap2(x)

        Yl = self.soft(x[:, :self.out_count])
        
        Yh = x[:, self.out_count:]
        
        Yh = self.channels_to_treelevel(Yh)
        x = self.utx((Yl, [Yh]))

        return x

class WaveDownDTCWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()
        self.dtx = DTCWTForward(J=1, biort="near_sym_b", qshift="qshift_b")
        self.treelevel_to_channels = Rearrange("b c f h w p -> b (f c p) h w")
        # self.pad = nn.ZeroPad2d(1)

        self.remap1 = nn.Conv2d(in_count * 16, (in_count * 16 + out_count)//2, 1, padding=0)
        self.nl = c
        self.remap2 = nn.Conv2d((in_count * 16 + out_count)//2, out_count, 1, padding=0)

    def forward(self, x):
        lr, hr = self.dtx(x)
        lr = unshuf(lr)
        hr = self.treelevel_to_channels(hr[0])
        x = torch.cat([lr, hr], dim=1)
        # x = self.pad(x)

        # print(f"{lr.shape = }, {hr.shape = }, {x.shape = }")
        x = self.remap1(x)
        x = self.nl(x)
        x = self.remap2(x)
        return x #.to(dev)
"""


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

        self.A = WSConv2d(in_depth, out_depth, 3, groups=4, padding=1, padding_mode='reflect')
        self.B = nn.Conv2d(out_depth, out_depth, 1, padding=0)
        self.C = WSConv2d(out_depth, out_depth, 3, groups=4, padding=1, padding_mode='reflect')
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
    def __init__(self, out_depth):
        super().__init__()
        self.out_depth = out_depth

    def forward(self, a, b):
        c = torch.zeros((a.shape[0], self.out_depth, a.shape[2], a.shape[3])).to(
            a.device
        )
        c[:, : a.shape[1]] = a
        c = torch.flip(c, dims=(1,))
        c[:, : b.shape[1]] = b
        return c


def concat(a, b):
    return torch.cat([a, b], dim=1)


class Ripple(nn.Module):
    def __init__(self, t_length=16):
        super(Ripple, self).__init__()

        self.join = Join(n)

        self.hpd = WaveDownDWT(1, 16)
        self.hp = Ya(16, n // 2)
        self.qpd = WaveDownDWT(n // 2, n)
        self.qb = Ya(n, n)
        self.t = Ya(n, n)
        self.t2 = Ya(n, n)
        self.t3 = Ya(n, n)
        self.qbu = WaveUpDWT(n, n // 2)

        self.hb = Ya(n, n // 2)
        self.hbu = WaveUpDWT(n // 2, 3)

    def forward(self, x):
        x = x - 0.15
        pan = shuf2(x[:, :16])
        mul = x[:, 16:]

        x = self.hpd(pan)
        hp = self.hp(x)
        qpd = self.qpd(hp)

        qpd[:, :8] = qpd[:, :8] + mul

        qb = self.qb(qpd)
        x = self.t(qb)
        x = self.t2(x)
        x = self.t3(x) + qpd

        x = self.qbu(x)

        x = concat(x, hp)

        hb = self.hb(x)
        hb[:, : n // 2] = hb[:, : n // 2] + hp
        hbu = self.hbu(hb)

        return hbu
