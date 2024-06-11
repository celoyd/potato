'''
demo.py: pansharpen big ARD-style images on the CPU
python demo.py pan.tiff mul.tiff weights.pt output.tiff

This is demo-quality in that a lot of its choices are hardcoded
and not carefully considered. It's also very verbose, for testing.

Example workflow:

# We get the pan and mul parts of a big image (downtown Nairobi):

aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-pan.tif .

aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-ms.tif .

python demo.py 104001008E063C00-{pan,ms}.tif weights/space_heater-gen-95.pt test.tiff

# This will run for at least several minutes!
# You might then want to, for example, make it a cloud-optimized geotiff:

pip install -U rio-cogeo

rio cogeo create test.tiff nairobi.tiff

# You could then fetch the default pansharpening:

aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-visual.tif .

# And then open both in QGIS to compare:

qgis nairobi.tiff 104001008E063C00-visual.tif

TODO:
- Handle nodata better.
- Add option to use device (*PU).
- Relax ARD assumptions.
- Multiprocessing?
- Tonemapping?

'''

import rasterio
from rasterio import windows
from ripple.model import Ripple
from ripple.util import pile
from sys import argv
import torch
import colour
import numpy as np

from skimage import io

block = 1024
apron = 32
big_block = block + 2 * apron


def buffer_window(w, margin):
    return windows.Window(
        w.col_off - margin,
        w.row_off - margin,
        w.width + margin * 2,
        w.height + margin * 2,
    )


def clip_window(w, img):
    return windows.intersection(w, windows.Window(0, 0, img.width, img.height))


def quarter_window(w):
    return windows.Window(w.col_off // 4, w.row_off // 4, w.width // 4, w.height // 4)


model = Ripple()
model.load_state_dict(torch.load(argv[3], map_location="cpu"))
model.eval()

ctr = 0

pan, mul = (rasterio.open(path) for path in argv[1:3])

assert mul.width == pan.width // 4
assert mul.height == pan.height // 4

profile = pan.profile
profile.update(
    {
        "count": 3,
        "photometric": "RGB",
        "blocksize": 1024,
        "tiled": True,
        "blockxsize": 1024,
        "blockysize": 1024,
        "compress": "LZW",
        "interleave": "pixel"
    }
)

print(profile)

dst = rasterio.open(argv[4], "w", **profile)

for i, w in dst.block_windows():
    print()
    print(i)
    buffered_window = buffer_window(w, apron)
    reading_window = clip_window(buffered_window, pan)

    print(f"{w = }")
    print(f"{buffered_window = }")
    print(f"{reading_window = }")

    pan_pixels = pan.read(window=reading_window)
    mul_pixels = mul.read(window=quarter_window(reading_window))

    skip_chunk = False
    if np.all(pan_pixels == 0) and np.all(mul_pixels == 0):
        print("Skipping all-zero chunk.")
        skip_chunk = True

    pan_pixels_shape = pan_pixels.shape
    print(f"{pan_pixels_shape = }")
    mul_pixels_shape = mul_pixels.shape

    original_mul_pixels = torch.tensor(mul_pixels.astype("float32") / 10_000)
    original_pan_pixels = torch.tensor(pan_pixels.astype("float32") / 10_000)

    pan_pixels = torch.zeros((1, big_block, big_block))
    pan_pixels[:, : pan_pixels_shape[1], : pan_pixels_shape[2]] = original_pan_pixels

    mul_pixels = torch.zeros((8, big_block // 4, big_block // 4))
    mul_pixels[:, : mul_pixels_shape[1], : mul_pixels_shape[2]] = original_mul_pixels

    pack = torch.concat([pile(pan_pixels, factor=4), mul_pixels], dim=0).unsqueeze(0)

    # pack = pack.cuda() ###

    if not skip_chunk:
        _, _, sharp = model(pack)
        sharp = sharp.detach().numpy()
        sharp = np.array(sharp, order="C")[0]
        sharp = sharp[:, : pan_pixels_shape[1], : pan_pixels_shape[2]]
    else:
        sharp = np.zeros((pan_pixels.shape[-3]*3, pan_pixels.shape[-2], pan_pixels.shape[-1]))

    print(f"{sharp.shape = }")

    left_start = w.col_off - reading_window.col_off
    top_start = w.row_off - reading_window.row_off
    right_end = w.width + left_start
    bottom_end = w.height + top_start

    print(f"{left_start = }")
    print(f"{top_start = }")
    print(f"{right_end = }")
    print(f"{bottom_end = }")

    print(f"{sharp.shape = }")
    sharp = sharp[:, top_start:bottom_end, left_start:right_end]
    print(f"{sharp.shape = }")

    print(sharp.mean(), sharp.std())

    sharp = np.moveaxis(sharp, 0, 2)
    sharp = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(sharp))
    sharp = np.moveaxis(sharp, 2, 0)

    sharp = np.clip(sharp * 65_535, 0, 65_535).astype(np.uint16)

    print(f"{sharp.shape = }")

    print(sharp.mean(), sharp.std())

    dst.write(sharp, window=w)

    ctr += 1
