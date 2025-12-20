"""
chip_to_png.py chip.pt image.png

Put the y (target) of a chip in a PNG. This is meant for quick visualization
only. It introduces quantization (color PNGs are 8 bits per channel), so it’s
not all the information in the chip’s y.

Todo:
- This should probably work in batches. The torch import is slow and doing it
  once per image if you want to look at 100 images is silly.
"""

import torch
import rasterio
from potato.color import OklabTosRGB
from sys import argv

import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

l_to_r = OklabTosRGB()

_, y = torch.load(argv[1])
rgb = l_to_r(y)
rgb8 = (rgb * 255).clip(0, 255).numpy().astype("uint8")

with rasterio.open(
    argv[2],
    "w",
    driver="PNG",
    height=rgb.shape[1],
    width=rgb.shape[2],
    count=3,
    photometric="RGB",
    dtype="uint8",
) as dst:
    dst.write(rgb8)
