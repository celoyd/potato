"""Simulate errors in WV-2/3 images.

Specifically, sample imperfections and band misalignment. Also apply rotations.
"""

import torch
from einops import rearrange
from torch.nn import Conv2d, Module
from torch.nn import functional as F

from potato.util import noisebox


class WV23Misaligner(Module):
    """Misalign bands kind of like how they are in reality.

    We make no guarantee about what the units are, incidentally.
    """

    def __init__(self, side_length):
        """Mostly we just need the band offsets and the base warp grid."""
        super().__init__()

        self.side_length = side_length

        self.upper_bands = (6, 4, 2, 1)
        self.lower_bands = (5, 3, 0, 7)

        ramp = torch.linspace(
            (1 / self.side_length) - 1, 1 - (1 / self.side_length), self.side_length
        )
        grid = torch.stack(torch.meshgrid(ramp, ramp, indexing="xy"), dim=-1)

        self.register_buffer("grid", grid)

        x, y = self.grid.unbind(dim=-1)
        self.register_buffer(
            "center_weighting",
            (1 - (x.square() + y.square()).sqrt()).clamp(0, 1).unsqueeze(-1),
        )

    def _check_shape(self, s):
        """Make sure dimensions are reasonable."""
        if s[-3] != 8:
            raise ValueError(f"Wrong band count: expected 8 but got {s[-3]}.")
        if s[-2] != s[-1]:
            raise ValueError(f"Non-square image: height {s[-2]} is not width {s[-1]}.")
        if s[-2] != self.side_length:
            raise ValueError(
                f"Wrong size: expected {self.side_length} side but got {s[-2]}."
            )

    def _make_warp_field(self, amount, spikiness, device):
        """Return a noise warp field."""
        radius = noisebox(self.side_length, power=1, device=device)
        radius = radius.abs().pow(spikiness)

        angle = noisebox(self.side_length, device=device)

        # Add a random offset to smear out the toward-zero bias.
        angle += 2 * torch.pi * torch.rand(1, device=device)

        x_off = radius * torch.cos(angle)
        y_off = radius * torch.sin(angle)

        offsets = torch.stack([x_off, y_off])

        # grid_sample expects (..., H, W, 2).
        offsets = rearrange(offsets, "... c h w -> ... h w c")

        offsets *= self.center_weighting * amount

        # Translate amount from units of pixels to units of normalized coord.
        offsets /= self.side_length / 2
        return offsets

    def _warp(self, bands, offset):
        """Physically apply an offset field to some bands."""
        B = bands.size(0)

        # grid_warp takes addresses, not offsets, so we have to add our
        # offsets to the coordinate grid.
        warp = self.grid.unsqueeze(0).repeat(B, 1, 1, 1) + offset

        return F.grid_sample(
            bands,
            warp,
            mode="bicubic",
            padding_mode="reflection",
            align_corners=False,
        )

    def forward(self, x, amount, spikiness):
        """Apply that realistic-ish motion/parallax warping.

        x: (B, 8, H, W), H == W
        amount: general scale of offsets near center, in pixels
        spikiness: more means more very small and very large values

        The amount and spikiness values interact and are generally tricky.
        Please decide on them empirically, by directly visualizing what
        different settings do to your data. Don’t just guess.
        """
        with torch.no_grad():
            self._check_shape(x.shape)

            noise = self._make_warp_field(amount, spikiness, x.device)

            x[:, self.upper_bands] = self._warp(x[:, self.upper_bands], noise)
            x[:, self.lower_bands] = self._warp(x[:, self.lower_bands], -noise)
            return x.clip(0, 1)


class HaloMaker(Module):
    """Emulate an imperfect point spread function in resampling.

    The imperfection is modeled as a spectrum of undersharp/blurry <->
    oversharp/ringing. In general, the WV-2/3 data as delivered tends to be
    mildly oversharp, but slight undersharpness also appears, e.g., in imagery
    through light cloud cover.
    """

    def __init__(self, depth):
        """Fun fact: this is some of the oldest code in the project."""
        super().__init__()
        self.register_buffer("cCc", torch.tensor([1.0, 2.0, 1.0], requires_grad=False))
        kernel = torch.outer(self.cCc, self.cCc)
        kernel = kernel / kernel.sum()

        self.blur = Conv2d(
            depth,
            depth,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=False,
            groups=depth,
        )

        self.blur.weight.data = kernel.view(1, 1, 3, 3).repeat(depth, 1, 1, 1)

    def forward(self, x, mean_sharpening=1.0, std=1.0):
        """Apply a random amount of sharpen/blur."""
        with torch.no_grad():
            B = x.shape[0]
            r = torch.normal(mean_sharpening, std, (B,), device=self.cCc.device).view(
                B, 1, 1, 1
            )
            x_blurry = self.blur(x)
            the_blur = x_blurry - x
            return x + (-r * the_blur)


class RandomD4(Module):
    """Rotate and flip one image.

    This is designed to be applied in a data loader, not in a training loop.
    Thus it assumes a single image, not a batch.

    The name for the set of pixel-perfect rotations and flips we can do to a
    square is the dihedral group D₄. It has 8 elements (counting identity),
    and we can reach them all with one random rotation and one random flip. We
    can pick either the n/s or e/w flip equally. See an illustration, e.g.,
    https://larryriddle.agnesscott.org/ifs/symmetric/D4example.htm

    Because we’re applying to only one image at a time, but the image has three
    different tensors (mul, pan, and oklab) as sub-images, this looks different
    from the more usual augmentations.
    """

    def __init__(self):
        """All we need is a starter random state."""
        super().__init__()
        self._randomize()

    def _randomize(self):
        """Set a number of turns and a fliip boolean."""
        self.turns = int(torch.randint(4, (1,)))
        self.flip = int(torch.randint(2, (1,))) == 1

    def _augment(self, x):
        """Apply the (currently chosen) modification to *one* tensor."""
        with torch.no_grad():
            x = x.rot90(dims=(-1, -2), k=self.turns)
            if self.flip:
                x = x.flip(-1)
            return x

    def forward(self, chip):
        """Map the augmentation across tensors and choose a new one."""
        (pan, mul), oklab = chip
        chip = ((self._augment(pan), self._augment(mul)), self._augment(oklab))
        self._randomize()
        return chip
