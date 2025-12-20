"""Things used more than once that do not have a more obvious place.

Some people think util submodules are a code smell. I tend to agree.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from einops import rearrange, reduce
from torch import fft

from potato.color import BandsToOklab, OklabTosRGB
from potato.errors import ImageDimensionError, TooManyNullsError


def timestamp():
    """Make an all-purpose ISO 8601 time string."""
    return datetime.now().astimezone().isoformat()


def cheap_half(x):
    """Downsample sleazy-style."""
    return reduce(x, "... c (h1 h2) (w1 w2) -> ... c h1 w1", "mean", h2=2, w2=2)


def swish4(x):
    """Swish with a beta of 4."""
    return x * torch.sigmoid(4 * x)


def tile(x, factor=2):
    """Depth to space."""
    times = int(torch.round(torch.log2(torch.tensor(factor))))
    for _ in range(times):
        x = rearrange(x, "... (c two dos) h w -> ... c (h two) (w dos)", two=2, dos=2)
    return x


def pile(x, factor=2):
    """Space to depth."""
    times = int(torch.round(torch.log2(torch.tensor(factor))))
    for _ in range(times):
        x = rearrange(x, "... c (h two) (w dos) -> ... (c two dos) h w", two=2, dos=2)
    return x


def valid_fraction(x):
    """Return fraction (0..1) of tensor != 0.

    The 0 means invalid convention is a definition of the ARD format.

    Before you ask, torch.count_nonzero() does not work on CPU uint16.
    """
    pixel_count = x.numel()
    valid_count = (x != 0).sum()
    return valid_count / pixel_count


def noisebox(side, power=2.0, dtype=torch.float32, device="cpu"):
    """Make a square of 1/f^n noise with unit stddev."""
    if side < 4 or side % 2 != 0:
        raise ValueError("Side must be >= 4 and even.")

    h = side
    w_half = (h // 2) + 1

    half_noise = torch.complex(
        torch.randn((1, h, w_half), device=device, dtype=dtype),
        torch.randn((1, h, w_half), device=device, dtype=dtype),
    )

    fy = fft.fftfreq(h, d=1, device=device, dtype=dtype).unsqueeze(1)
    fx = fft.rfftfreq(h, d=1, device=device, dtype=dtype).unsqueeze(0)

    df = (fx.square() + fy.square()).sqrt()
    df[0, 0] = 1.0 / h  # clamp DC

    half_noise /= df.pow(power)

    # keep it real (for rfft2 symmetry)
    half_noise[..., 0, 0].imag = 0.0
    half_noise[..., h // 2, :].imag = 0.0
    half_noise[..., :, -1].imag = 0.0

    noise = fft.irfft2(half_noise, s=(h, h))[0]

    noise -= noise.mean()
    noise /= noise.std()

    return noise


def unit_srgb_to_uint16_tiff(t, f, overwrite_ok=False, compression="zstd"):
    """Write a unit (float 0..1) sRGB image to a uint16 TIFF.

    Only suitable for small images (not sizes where RAM matters).
    """
    f = Path(f)
    if f.exists() and not overwrite_ok:
        raise FileExistsError(f"{f} already exists; will not overwrite")

    if t.shape[0] != 3 or len(t.shape) != 3:
        raise ImageDimensionError(
            f"Expected image data shaped 3, h, w but got {t.shape}"
        )

    if not isinstance(t, np.ndarray):
        t = t.detach().cpu().numpy()

    t = (t * (2**16 - 1)).clip(0, (2**16 - 1)).astype(np.uint16)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
        with rasterio.open(
            f,
            "w",
            driver="gtiff",
            photometric="RGB",
            dtype=np.uint16,
            count=3,
            height=t.shape[1],
            width=t.shape[2],
            compression=compression,
        ) as dst:
            dst.write(t)


class Chip:
    """Wrap a chip consisting of corresponding pan and mul bands.

    We handle adding an oklab representation of the mul bands and
    applying downsampling. This creates a training pair of:

        (pan, mul), oklab

    in a .pt file. The .pt is simple enough that its user does not
    need to instantiate a new Chip to read it. However, we provide the
    ability to do so.

    The .pt is float32. This is storage-hungry but optimized for net
    computation over multiple epochs of training.

    This deserves a rewrite with a more principled (stricter) approach
    to validation when reading a .pt, etc.
    """

    bands_to_oklab = BandsToOklab()
    oklab_to_srgb = OklabTosRGB()

    def __init__(self, pan, mul, extra_downsample=True, minumum_valid=0.75):
        """Make a chip from raw uint16 ARD data."""
        if not (
            (mul.shape[-2] * 4 == pan.shape[-2])
            and (mul.shape[-1] * 4 == pan.shape[-1])
        ):
            raise ImageDimensionError(
                "Mul size must be 1/4 pan size on spatial dimensions "
                f"but got {tuple(mul.shape)} v. {tuple(pan.shape)}."
            )

        if not (pan.dtype == mul.dtype == torch.uint16):
            raise ValueError(
                "My scaling assumes torch.uint16 input pixels, but got "
                f"{pan.dtype} and {mul.dtype}."
            )

        # An interesting empirical question: do we save time skipping
        # downsampling when there are too many nulls, or would the null
        # check be enough faster after downsampling to counterblance?"""
        if (valid_fraction(pan) < minumum_valid) or (
            valid_fraction(mul) < minumum_valid
        ):
            raise TooManyNullsError

        self.pan = self.scale(pan.float())
        self.mul = self.scale(mul.float())

        if extra_downsample:
            # Artifact-reducing step (discussed in documentation).
            self.pan = cheap_half(self.pan)
            self.mul = cheap_half(self.mul)

        self.oklab = self.bands_to_oklab(self.mul)

        self.pan = cheap_half(cheap_half(self.pan))
        self.mul = cheap_half(cheap_half(self.mul))

    def read_pt(self, pt_path):
        """Read a .pt chip without any validation."""
        return torch.load(pt_path, weights_only=False)

    def write_pt(self, pt_path):
        """Write a .pt (or pick you own suffix I guess)."""
        torch.save(((self.pan, self.mul), self.oklab), pt_path)

    def write_tiff(self, tiff_path):
        """Write a TIFF of the oklab for visualization."""
        rgb = self.oklab_to_srgb(self.oklab)
        unit_srgb_to_uint16_tiff(rgb, tiff_path)

    def scale(self, x):
        """Scale by the 10k constant."""
        return x / 10_000.0
