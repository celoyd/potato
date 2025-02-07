import torch
from torch import nn
from torch.nn import Mish

from potato.util import pile, tile
from potato.color import BandsToOklab


class ConvChunk(nn.Module):
    """
    A lightweight, nonspecialized image processing convolutional chunk.
    Takes input and output depths so it can be used to stretch or squeeze.

    The main branch of this block basically does this twice:

    1×1 | dropout | 3×3 (in 8 groups) | mish

    and then ends with a final 1x1.

    """

    def __init__(self, in_depth, out_depth, dropout=True):
        super().__init__()

        # Use an int multiplicatively midway between the in and out depths
        # as the depth of most operations inside the block
        mid_depth = (in_depth * out_depth) ** 0.5

        # round up to next multiple of 8
        mid_depth = int((mid_depth + 8 - 1) // 8) * 8

        self.nl = Mish()

        if dropout:
            self.drop = nn.Dropout(0.5)
        else:
            self.drop = lambda x: x

        self.skip = nn.Conv2d(in_depth, out_depth, 1)
        self.front_pointwise = nn.Conv2d(in_depth, mid_depth, 1)

        self.front_conv = nn.Conv2d(
            mid_depth, mid_depth, 3, groups=8, padding=1, padding_mode="reflect"
        )

        self.middle_pointwise = nn.Conv2d(mid_depth, mid_depth, 1)

        self.back_conv = nn.Conv2d(
            mid_depth, mid_depth, 3, groups=8, padding=1, padding_mode="reflect"
        )

        self.end_pointwise = nn.Conv2d(mid_depth, out_depth, 1)

    def forward(self, x):
        skip = self.skip(x)

        x = self.front_pointwise(x)
        x = self.drop(x)
        x = self.front_conv(x)
        x = self.nl(x)

        x = self.middle_pointwise(x)
        x = self.back_conv(x)
        x = self.nl(x)

        x = self.end_pointwise(x)
        return x + skip


class Potato(nn.Module):
    def __init__(self, n=48):
        super().__init__()

        self.zoom = nn.Upsample(scale_factor=2, mode="bilinear")
        self.oklab = BandsToOklab()

        # Depths given as expressions to help mental math.
        # E.g., here, 3 layers of oklab, 8 of mul, 16 of piled pan:
        self.bq = ConvChunk(3 + 8 + 16, n)
        self.eq = ConvChunk(n, n // 2)

        self.bh = ConvChunk(n // 2 + 4 + 3, n)
        self.eh = ConvChunk(n, n // 2)

        self.f = ConvChunk(n // 2 + 1 + 3 + 1, 3, dropout=False)

    def forward(self, x):
        x = torch.clamp(x, 1e-9, None)
        oklab = self.oklab(x[:, 16:])

        oklab_half = self.zoom(oklab)
        oklab_full = self.zoom(oklab_half)

        pan_quarter = x[:, :16]
        pan_full = tile(pan_quarter, 4)
        pan_half = pile(pan_full, 2)

        pan_detail = pan_full - self.zoom(
            self.zoom(pan_quarter.mean(dim=-3, keepdim=True))
        )

        cheap_sharp = oklab_full.clone()
        cheap_sharp[:, 0:1] += pan_detail

        mul = x[:, 16:]

        q = torch.cat((oklab, mul, pan_quarter), dim=1)
        q = self.bq(q)
        q = self.eq(q)

        h = self.zoom(q)
        h = torch.cat((h, pan_half, oklab_half), dim=1)
        h = self.bh(h)
        h = self.eh(h)

        f = self.zoom(h)
        f = torch.cat((f, pan_full, oklab_full, pan_detail), dim=1)
        f = self.f(f)

        return cheap_sharp + f
