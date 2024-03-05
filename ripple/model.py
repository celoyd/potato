import torch
from torch import nn
import torch.nn.functional as fn
from torch.nn import Mish, LeakyReLU, ELU

from einops import rearrange, reduce
from inspect import isfunction
from functools import partial
from einops.layers.torch import Rearrange

from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTInverse, DWTForward

from ripple.util import pile, tile
from ripple.color import BandsToOklab


#c = ELU()
#c = Mish()
#c = LeakyReLU()

n = 96


def squareplus(x):
    return (x + torch.sqrt(x*x+0.02))/2.0

c = squareplus

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class WaveDownDWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()
        self.dtx = DWTForward(J=1, wave="db1", mode="reflect")
        self.treelevel_to_channels = Rearrange("b c f h w -> b (c f) h w")

        # self.remap1 = nn.Conv2d(4*in_count, (4*in_count + out_count)//2, 1)
        # self.nl = c
        # self.remap2 = nn.Conv2d((4*in_count + out_count)//2, out_count, 1)

    def forward(self, x):

        lr, hr = self.dtx(x)

        hr = self.treelevel_to_channels(hr[0])
        x = torch.cat([lr / 2, hr], dim=1)

        # x = self.remap1(x)
        # x = self.nl(x)
        # x = self.remap2(x)

        return x


class WaveUpDWT(nn.Module):
    def __init__(self, in_count, out_count):
        super().__init__()

        self.out_count = out_count
        self.utx = DWTInverse(wave="db1", mode="reflect")
        self.channels_to_treelevel = Rearrange(
            "b (c f) h w -> b c f h w", f=3, c=out_count
        )

        # self.remap1 = nn.Conv2d(in_count, out_count * 4, 3, padding=1, padding_mode="reflect")

        # self.nl = c

    def forward(self, x):
        # x = self.nl(x)
        # x = self.remap1(x)

        Yl = x[:, : self.out_count]
        Yh = self.channels_to_treelevel(x[:, self.out_count :])

        x = self.utx((Yl * 2, [Yh]))

        return x


# class WaveUpDTCWT(nn.Module):
#     def __init__(self, in_count, out_count):
#         super().__init__()

#         self.out_count = out_count
#         self.utx = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

#         self.soft = nn.Upsample(scale_factor=2, mode='bilinear')

#         self.channels_to_treelevel = Rearrange(
#             "b (f c p) h w -> b c f h w p", f=6, c=out_count, p=2
#         )

#         #self.remap1 = nn.Conv2d(in_count, (in_count + out_count * 13)//2, 1, padding=0)
#         #self.nl = c
#         #self.remap2 = nn.Conv2d((in_count + out_count * 13)//2, out_count * 13, 1, padding=0)

#     def forward(self, x):
#         #x = self.remap1(x)
#         #x = self.nl(x)
#         #x = self.remap2(x)

#         Yl = self.soft(x[:, :self.out_count])

#         Yh = x[:, self.out_count:]

#         Yh = self.channels_to_treelevel(Yh)
#         x = self.utx((Yl, [Yh]))

#         return x

# class WaveDownDTCWT(nn.Module):
#     def __init__(self, in_count, out_count):
#         super().__init__()
#         self.dtx = DTCWTForward(J=1, biort="near_sym_b", qshift="qshift_b")
#         self.treelevel_to_channels = Rearrange("b c f h w p -> b (f c p) h w")
#         # self.pad = nn.ZeroPad2d(1)

#         #self.remap1 = nn.Conv2d(in_count * 16, (in_count * 16 + out_count)//2, 1, padding=0)
#         #self.nl = c
#         #self.remap2 = nn.Conv2d((in_count * 16 + out_count)//2, out_count, 1, padding=0)

#     def forward(self, x):
#         lr, hr = self.dtx(x)
#         lr = unshuf(lr)
#         hr = self.treelevel_to_channels(hr[0])
#         x = torch.cat([lr, hr], dim=1)
#         # x = self.pad(x)

#         # print(f"{lr.shape = }, {hr.shape = }, {x.shape = }")
#         #x = self.remap1(x)
#         #x = self.nl(x)
#         #x = self.remap2(x)
#         return x #.to(dev)


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
        mid_depth = ((in_depth + out_depth)//8)*4
        self.nl = c

        self.mid_norm = nn.GroupNorm(4, mid_depth)
        self.out_norm = nn.GroupNorm(4, out_depth)

        self.A = nn.Conv2d(in_depth, mid_depth, 1, padding=0)
        self.B = WSConv2d(
            mid_depth, mid_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.C = nn.Conv2d(mid_depth, out_depth, 1, padding=0)
        self.D = WSConv2d(
            out_depth, out_depth, 3, groups=4, padding=1, padding_mode="reflect"
        )
        self.E = nn.Conv2d(out_depth, out_depth, 1, padding=0)

    def forward(self, x):
        if self.use_res:
            r = x

        x = self.A(x)
        x = self.nl(x)

        x = self.B(x)
        x = self.mid_norm(x)

        x = self.C(x)
        x = self.nl(x)

        x = self.D(x)
        x = self.out_norm(x)

        #x = x / x.square().sum(dim=1, keepdim=True).sqrt()
        #x = self.nl(x)

        x = self.E(x)

        if self.use_res:
            x = x + r

        return x


# class Join(nn.Module):
#     # Lightweight non-normalized merging
#     def __init__(self, in_depth, out_depth):
#         super().__init__()
#         self.mixer = nn.Conv2d(in_depth, out_depth, 1)

#     def forward(self, x):
#         return self.mixer(x)


def concat(*a):
    return torch.cat([*a], dim=1)


class Ripple(nn.Module):
    def __init__(self):
        super().__init__()

        self.drop = nn.Dropout(0.5)
        self.oklab = BandsToOklab()
        self.embiggen = nn.Upsample(scale_factor=(2, 2), mode="bilinear")

        #self.bf = nn.Conv2d(1, n//8, 3, padding=1, padding_mode='reflect')
        self.bf2h = WaveDownDWT(1, 4)

        self.bh = Ya(4, n//4)
        self.bh2q = WaveDownDWT(n//4, n)

        self.q_intro = Ya(n + 8, n)
        self.bq = Ya(n, n)
        self.mq = Ya(n, n)
        self.eq = Ya(n, n)

        #self.bh = Ya(16 + 8 + 3, n)

        self.eq2h = WaveUpDWT(n, n // 4)
        self.eh = Ya(n // 4 + 4, 12)
        self.eh2f = WaveUpDWT(12, 3)
        #self.ef = nn.Conv2d(n // 8, 3, 3, padding=1, padding_mode='reflect')
        #self.the_end = WaveUpDWT(3 * 4, 3)

    def forward(self, x):
        x = torch.clamp(x, 1e-9, None)
        oklab = self.oklab(x[:, 16:])
        original_pan = tile(x[:, :16], 4)

        half_oklab = self.embiggen(oklab)
        full_oklab = self.embiggen(half_oklab)

        x = torch.pow(x, 1 / 3)

        mul = x[:, 16:]
        pan = tile(x[:, :16], 4)

        fake = full_oklab.clone()
        fake[:, 0] = pan[:, 0]

        #bf = self.bf(pan)

        #with torch.no_grad():
        h1 = self.bf2h(pan)

        bh = self.bh(h1)

        q = self.bh2q(bh)

        q_intro = self.q_intro(concat(mul, q))
        #q[:, :8 + 3] += concat(oklab, mul)
        q = self.bq(q_intro)
        q = self.mq(q)
        #q = self.mmq(q)

        #q[:, :8 + 3] += concat(oklab, mul)
        q += q_intro
        q = self.eq(q)
        #qok = q[:, :3]

        hh = self.eq2h(q)
            #h[:, :3] += half_oklab

        #hhh = self.drop(concat(hh, h))
        hhh = concat(hh, h1)
        h = self.eh(hhh)
        #h[:, 0] += h1[:, 0]
        #h[:, 3:6] += h1[:, 1:]
        #h[:, :3] += hh[:, :3]
        #hok = h[:, :3]


        f = self.eh2f(h)
        #f = self.ef(f)
        #return fake

        #return qok, hok, f #+ fake
        return f + fake #fake + f


# class Ripple(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.oklab = BandsToOklab()
#         self.embiggen = nn.Upsample(scale_factor=(2, 2), mode="bilinear")

#         #self.bf = nn.Conv2d(1, n//8, 3, padding=1, padding_mode='reflect')
#         self.bf2h = WaveDownDWT(1, 4)

#         self.bh = Ya(4, n//4)
#         self.bh2q = WaveDownDWT(n//4, n)

#         self.bq = Ya(n, n)
#         self.mq = Ya(n, n)
#         self.mmq = Ya(n, n)
#         self.eq = Ya(n, n)

#         #self.bh = Ya(16 + 8 + 3, n)

#         self.eq2h = WaveUpDWT(n, n // 4)
#         self.eh = Ya(n // 4, n)
#         self.eh2f = WaveUpDWT(n, n // 4)
#         self.ef = nn.Conv2d(n // 4, 3, 3, padding=1, padding_mode='reflect')
#         #self.the_end = WaveUpDWT(3 * 4, 3)

#     def forward(self, x):
#         x = torch.clamp(x, 1e-6, 1)
#         oklab = self.oklab(x[:, 16:])

#         half_oklab = self.embiggen(oklab) # + 2) - 2
#         full_oklab = self.embiggen(half_oklab) # + 2) - 2

#         x = torch.pow(torch.clamp(x, 1e-9, None), 1 / 3)

#         mul = x[:, 16:]
#         pan = tile(x[:, :16], 4)

#         fake = full_oklab.clone()
#         fake[:, 0] = pan[:, 0]

#         #bf = self.bf(pan)
#         h1 = self.bf2h(pan)

#         bh = self.bh(h1)

#         q = self.bh2q(bh)
#         q = self.bq(q)
#         q[:, :8 + 3] += concat(mul, oklab)
#         qq = q
#         q = self.mq(q)
#         q = self.mmq(q)

#         q += qq
#         q = self.eq(q)
#         #qok = q[:, :3]

#         h = self.eq2h(q)
#         h[:, :4] += h1
#         #h[:, :3] += half_oklab
#         h = self.eh(h) # + bh)
#         #hok = h[:, :3]

#         f = self.eh2f(h)
#         f[:, :3] += fake
#         f = self.ef(f) # + bf)

#         #return qok, hok, fake + f
#         return f
