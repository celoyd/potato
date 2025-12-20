"""The core functions of Potato."""

import torch
from torch import nn
from torch.nn import functional as F

from potato.color import BandsToOklab
from potato.util import pile, swish4


class ConvChunk(nn.Module):
    """A lightweight, restoration-oriented convolutional block.

    The main branch of this block is:

        1×1 | 3×3 | fold | 1×1 | 3×3 | fold | 1×1

    And a skip connection (through its own 1×1, to allow for depth changes)
    is added at the end.

    The 3×3 convolutional blocks are in 8 groups.

    The main branch could be seen as basically 2× a sub-block like:

        1×1 | grouped 3×3 | nonlinearity

    with various elaborations. This is very loosely inspired by ConvNeXt
    (https://arxiv.org/abs/2201.03545).
    """

    def __init__(self, in_depth, out_depth):
        """We take in and out depth arguments in order to to stretch or squeeze."""
        super().__init__()

        # For the depth of most operations inside the block, we use an int
        # multiplicatively midway between the in and out depths...
        mid_depth = (in_depth * out_depth) ** (1 / 2)
        # ...rounded up to a multiple of 8.
        mid_depth = int((mid_depth + 8 - 1) // 8) * 8

        self.fold = swish4

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
        """Apply block."""
        skip = self.skip(x)

        x = self.front_pointwise(x)
        x = self.front_conv(x)
        x = self.fold(x)

        x = self.middle_pointwise(x)
        x = self.back_conv(x)
        x = self.fold(x)

        x = self.end_pointwise(x)
        return x + skip


class Potato(nn.Module):
    """The pansharpener itself.

    For details on the actual architecture, see forward()’s docstring.

    Conventions (to help understand variable names):

    - Data is always at a scale of quarter, half, or full. These names are
      relative to the output – or, equivalently, to the input pan(chromatic)
      band. Therefore the mul(tispectral) bands at half scale are of the same
      spatial dimensions, i.e., can be concatenated with, the pan band at
      half scale. Just keep in mind that the scale terms always mean the same
      thing, whatever they’re attached to.
    - Running data, instead of using the x convention, is named q, h, or f
      according to its scale.
    - Where ConvChunks are paired, they’re named b(eginning) and e(nd) plus
      the scale letter. E.g., bh is the first convchunk at half scale. Plane
      f is the lone chunk at full scale.
    """

    def __init__(self, n=48):
        """As simple as possible. The n is a general depth scale."""
        super().__init__()

        self.oklab = BandsToOklab()

        # Depths are given as expressions to match the items in the cat()s in
        # forward(). For example, here we have 16 layers of (piled) pan, 8
        # layers of mul, and 3 of oklab, so:
        self.bq = ConvChunk(16 + 8 + 3, n)
        self.eq = ConvChunk(n, n // 2)

        self.bh = ConvChunk(4 + n // 2 + 3, n)
        self.eh = ConvChunk(n, n // 2)

        self.f = ConvChunk(1 + 1 + n // 2 + 3, 3)

    def _upsamp(self, x):
        return F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False, antialias=True
        )

    def _downsamp(self, x):
        return F.interpolate(
            x, scale_factor=1 / 2, mode="bilinear", align_corners=False, antialias=True
        )

    def forward(self, x, intermediates=False):
        """Welcome to Potato.

        Potato can be thought of as a simple, classic pansharpening method
        with a neural network wired into it. The classic method, here called
        the *cheap*, does the easy 80% of pansharpening. The network is woven
        through, seeing its intermediate values as well as the main input
        data, and provides a small-magnitude residual, diff, or patch that
        tries to do the remaining 20% of pansharpening.

        At a conceptual level, we’re framing pansharpening in predictor-
        corrector terms, with a static algorithm as the predictor and a
        learned algorithm as the corrector.

        Here’s what that means in practice. The cheap is implemented as:
        - Convert the multispectral bands to the oklab color space at their
          original quarter resolution.
        - Upsample this to full resolution. (Upsampling is iterative; we save
          the half-scale version.)
        - High-pass the pan band by taking its depthwise mean at 4× space-to-
          depth (equivalent to downsampling it to quarter resolution), up-
          sampling that back to full resolution, and subtracting it out of
          the full resolution pan band.
        - Adding the high-pass of the pan band to the L channel of the
          upsampled oklab.

        The network, meanwhile, sees the inputs and outputs of the cheap
        method at each resolution. For example, at quarter scale, the network
        has inputs of: the piled-up pan band, the mul bands, and the oklab.
        This goes through learned conv operations and is upsampled. Then the
        half-scale network chunk sees that upsampled output as well as the
        other half-scale data.

        This can be thought of as intertwining learned operations with a
        standard pansharpening method. It could also be thought of as a kind
        of U-net with a non-learned front half. That is, all the downsampling
        is by space-to-depth, and convolutions only happen in the upsampling
        and skip paths.

        We optionally track some intermediate values to apply intermediate
        losses on.
        """
        pan_full, mul = x

        with torch.no_grad():
            # Preparatory steps without any learned operations.
            oklab = self.oklab(mul)

            oklab_half = self._upsamp(oklab)
            oklab_full = self._upsamp(oklab_half)

            pan_half = pile(pan_full, 2)
            pan_quarter = pile(pan_half, 2)

            # High-pass the pan band. We scale by cube root to roughly
            # match the radiance->luma conversion of oklab.
            pan_detail = pan_full.clip(0, None).pow(1 / 3) - self._upsamp(
                self._upsamp(
                    self._downsamp(self._downsamp(pan_full)).clip(0, None).pow(1 / 3)
                )
            )

            cheap = oklab_full.clone()
            cheap[:, 0:1] += pan_detail

            # To get just the cheap as final output, uncomment:
            # return cheap

            if intermediates:
                ints = {}

            q = torch.cat((pan_quarter, mul, oklab), dim=1)

        q = self.bq(q)
        q = self.eq(q)
        if intermediates:
            ints["q"] = q[:, :3]

        h = self._upsamp(q)
        h = torch.cat((pan_half, h, oklab_half), dim=1)
        h = self.bh(h)
        h = self.eh(h)
        if intermediates:
            ints["h"] = h[:, :3]

        f = self._upsamp(h)
        f = torch.cat((pan_full, pan_detail, f, oklab_full), dim=1)
        f = self.f(f) + cheap

        # To visualize just the delta, uncomment:
        # f[:, 0] += 0.5
        # return f

        if intermediates:
            return f, ints
        return f
