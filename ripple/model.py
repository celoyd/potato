import torch
from torch import nn
import torch.nn.functional as fn
from torch.nn import Mish, LeakyReLU, ELU

from einops import rearrange, reduce
from inspect import isfunction
from functools import partial
from einops.layers.torch import Rearrange

from ripple.util import pile, tile, cheap_half, cheap_db1_dwt, cheap_multilayer_db1_dwt
from ripple.color import BandsToOklab

from torch.nn.utils.parametrizations import weight_norm

c = Mish()

n = 48

def concat(*a):
    return torch.cat([*a], dim=1)

class ConvChunk(nn.Module):
    def __init__(self, in_depth, out_depth, drop=False):
        super().__init__()

        mid_depth = int(
            0.5
            + torch.exp(
                (torch.log(torch.tensor(in_depth)) + torch.log(torch.tensor(out_depth)))
                / 2
            )
        )

        self.should_drop = drop
        if self.should_drop:
            self.drop = nn.Dropout(0.5)

        mid_depth = (mid_depth // 4 + 1) * 4
        self.mid_depth = mid_depth

        self.nl = c

        self.res = nn.Conv2d(in_depth, out_depth, 1)
        self.res.mma_norm = True

        self.front = nn.Conv2d(in_depth, mid_depth, 1)
        self.front.mma_norm = True

        self.A = nn.Conv2d(
            mid_depth, mid_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.A.mma_norm = True

        self.B = nn.Conv2d(mid_depth, mid_depth, 1)
        self.B.mma_norm = True

        self.C = nn.Conv2d(
            mid_depth, mid_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.C.mma_norm = True

        self.D = nn.Conv2d(mid_depth, out_depth, 1)
        self.D.mma_norm = True

    def forward(self, x):

        res = self.res(x)
        x = self.front(x)
        x = self.nl(x)

        x = self.A(x)
        x = self.nl(x)
        x = self.B(x)

        if self.should_drop:
            x = self.drop(x)

        x = self.C(x)
        x = self.D(x) + res

        return x


class Double(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.squeezing = in_depth != out_depth

        if self.squeezing:
            self.squeeze = nn.Conv2d(in_depth, out_depth, 1, padding=0)
            self.squeeze.mma_norm = True

        self.ups = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        if self.squeezing:
            x = self.squeeze(x)
        return self.ups(x)


class Ripple(nn.Module):
    def __init__(self):
        super().__init__()

        self.bili_up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.oklab = BandsToOklab()

        self.bq = ConvChunk(3 + 8 + 16, n)
        self.eq = ConvChunk(n, n // 2, drop=False)
        self.q_to_h = Double(n // 2, n // 2)

        self.bh = ConvChunk(n // 2 + 4 + 3, n)
        self.eh = ConvChunk(n, n // 2, drop=False)
        self.h_to_f = Double(n // 2, n // 2)

        self.bf = ConvChunk(n // 2 + 1 + 3 + 1, 3, drop=False)

    def forward(self, x):
        x = torch.clamp(x, 1e-9, None)
        oklab = self.oklab(x[:, 16:])
        oklab_half = self.bili_up(oklab)
        oklab_full = self.bili_up(oklab_half)
        # return 0, 0, 0, oklab_full

        x = torch.pow(x, 1 / 3)

        pan_quarter = x[:, :16]
        pan_full = tile(pan_quarter, 4)
        pan_half = pile(pan_full, 2)

        pan_detail = pan_full - self.bili_up(
            self.bili_up(pan_quarter.mean(axis=-3, keepdim=True))
        )

        mul = x[:, 16:]

        q = concat(oklab, mul, pan_quarter)
        q = self.bq(q)
        ql = q[:, :3]
        q = self.eq(q)

        h = self.q_to_h(q)

        h = concat(oklab_half, pan_half, h)
        h = self.bh(h)
        hl = h[:, :3]
        h = self.eh(h)

        f = self.h_to_f(h)

        f = concat(f, pan_full, oklab_full, pan_detail)
        f = self.bf(f)

        # oklab_full[:, 0:1] += pan_detail
        # return 0, 0, 0, oklab_full

        return ql, hl, f, f
