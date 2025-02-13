#!/usr/bin/env python

"""
oklab-vectorscope.py input.image output.image
Draw a vectorscope of the input image in Oklab color space.

There are many different ways to do vectorscopes, for example
in original color v. monochrome v. at middle brightness. More
or less arbitrarily, and because it’s easy to draw, we use the 
following conventions:

1. Oklab color space oriented with +a right and +b down.
2. Pixels are drawn from the image in reading order and painted
   onto the output over previous pixels. There’s no way to tell
   for sure how much one color appears compared to another, and 
   if the pixel happens to be indistinguishable from black, it 
   won’t be visible in practice. It also looks messy. On the other
   hand, it’s conceptually simple, and all colors that appear in
   the output are colors that appeared in the input (with the
   possible exception of the background black, of course). I am
   certainly not arguing that this is The Right Way to do vector-
   scopes! It’s just worked pretty well for me over time.

Caveats:
- Only works with integer-typed inputs.
- Assumes sRGB input.
- Uses a constant 2.2 gamma instead of proper RGB conversion.
"""

import imageio.v3 as iio
import numpy as np
from sys import argv
from einops import einsum
import tqdm



rgbl_to_lms = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)

M1 = np.array(
    [
        [0.8189330101, 0.0329845436, 0.0482003018],
        [0.3618667424, 0.9293118715, 0.2643662691],
        [-0.1288597137, 0.0361456387, 0.6338517070],
    ]
)

M2 = np.array(
    [
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ]
)

dst_side = 1024


def scale_to_unit(x):
    return x.astype(np.float32) / np.iinfo(src.dtype).max


def unit_srgb_to_oklab(rgb):
    rgbl = rgb**2.2
    lms = einsum(rgbl_to_lms, rgbl, "l r, ... r -> ... l")
    lms = lms ** (1 / 3)
    oklab = einsum(M2, lms, "l o, ... l -> ... o")
    return oklab


src = iio.imread(argv[1])

dst = np.zeros((dst_side, dst_side, 3), dtype=src.dtype)

for row in tqdm.tqdm(range(src.shape[0]), unit=" row"):
    unit_rgb = scale_to_unit(src[row])
    oklab = unit_srgb_to_oklab(unit_rgb)
    for col in range(oklab.shape[0]):
        a = int(oklab[col, 1] * dst_side) + dst_side // 2
        b = int(oklab[col, 2] * dst_side) + dst_side // 2
        dst[a, b] = src[row, col]

iio.imwrite(argv[2], dst)
