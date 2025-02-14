"""
oklab-vectorscope.py input.image output.image
Plot a vectorscope of the input sRGB image in Oklab color space.

There are many different ways to do vectorscopes, for example
in original color v. monochrome v. at middle brightness. More
or less arbitrarily, and because it’s easy to draw, we simply
pull pixels from the source image in reading order and put
them in the destination image. Later-encountered pixels cover
previous ones. This is somewhat silly but it works.
"""

import rasterio
from rasterio.windows import Window

import numpy as np
import torch

from sys import argv

import tqdm

from potato.color import sRGBToOklab, OklabTosRGB

import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

r_to_l = sRGBToOklab()
l_to_r = OklabTosRGB()

vs_side = 1024

with rasterio.open(argv[1]) as src:
    assert src.profile["count"] == 3

    src_dtype = src.profile["dtype"]
    src_max = np.iinfo(src_dtype).max
    vs = np.zeros((3, vs_side, vs_side), dtype=src_dtype)

    for row in tqdm.tqdm(range(src.profile["height"]), unit=" row"):
        w = Window(0, row, src.profile["width"], 1)
        raw_pixels = src.read(window=w)
        pixels = torch.tensor(raw_pixels.astype(np.float32)).unsqueeze(0)

        oklab = r_to_l(pixels / src_max).numpy()

        a = ((oklab[0, 1] * 2 * vs_side) + vs_side / 2).astype(np.uint16)
        b = ((oklab[0, 2] * 2 * vs_side) + vs_side / 2).astype(np.uint16)

        a_valid = (a >= 0) & (a < vs_side)
        b_valid = (b >= 0) & (b < vs_side)

        valid = a_valid & b_valid

        raw_pixels = raw_pixels[:, 0, :]  # drop the height dimension

        vs[:, a[valid], b[valid]] = raw_pixels[:, valid[0]]

    with rasterio.open(
        argv[2],
        "w",
        driver="gtiff",
        count=3,
        photometric="RGB",
        height=vs_side,
        width=vs_side,
        dtype=src_dtype,
    ) as dst:
        dst.write(vs)
