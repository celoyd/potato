"""
demo.py: pansharpen with potato
Usage: python demo.py pan.tiff mul.tiff -w weights.pt output.tiff
Also see the --help.

This code contains many hardcoded assmptions about data format and user 
preferences. It is truly a demo.

Example workflow:

We get the pan and mul parts of a big image (downtown Nairobi):

$ aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-pan.tif .
$ aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-ms.tif .

$ python demo.py -d cuda -w weights/space_heater-gen-95.pt 104001008E063C00-{pan,ms}.tif test.tiff

You might then want to, for example, make it a cloud-optimized geotiff:

$ pip install -U rio-cogeo
$ rio cogeo create test.tiff nairobi.tiff

You could then fetch the default pansharpening:

$ aws s3 cp s3://maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-visual.tif .

And then open both in QGIS to compare:

$ qgis nairobi.tiff 104001008E063C00-visual.tif

"""

from sys import argv, stderr
import click

import torch
import numpy as np

# import colour

from tqdm import tqdm

import rasterio
from rasterio import windows

from ripple.model import Ripple
from ripple.util import pile
from ripple.color import OklabTosRGB

block = 1024  # edge of square to pansharpen at a time
apron = 32  # extra space around each block
big_block = block + 2 * apron
# device = 'cpu'
device = "cuda:0"


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


@click.command()
@click.argument("panpath", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("mulpath", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("dstpath", nargs=1, type=click.Path(exists=False), required=True)
@click.option(
    "-w",
    "--weights",
    type=click.Path(exists=True),
    required=True,
    help="Checkpoint (weights) file",
)
@click.option(
    "-d", "--device", default="cuda", help="Torch device (e.g., cuda, cpu, mps)"
)
def pansharpen(panpath, mulpath, dstpath, weights, device):
    print(device)
    pan_file, mul_file = (rasterio.open(path) for path in (panpath, mulpath))

    assert mul_file.width == pan_file.width // 4
    assert mul_file.height == pan_file.height // 4

    profile = pan_file.profile
    profile.update(
        {
            "count": 3,
            "photometric": "RGB",
            "blocksize": block,
            "tiled": True,
            "blockxsize": block,
            "blockysize": block,
            "compress": None,  # "zstd"
            # "predictor": 2,
            "interleave": "pixel",
            "dtype": np.uint16,
            "nodata": 0,
        }
    )

    dst = rasterio.open(dstpath, "w", **profile)

    model = Ripple().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    sRGB = OklabTosRGB().to(device)

    with tqdm(list(dst.block_windows()), unit=" block") as progress:
        for i, w in progress:
            buffered_window = buffer_window(w, apron)
            reading_window = clip_window(buffered_window, pan_file)

            pan_pixels = pan_file.read(window=reading_window)
            mul_pixels = mul_file.read(window=quarter_window(reading_window))

            if np.all(pan_pixels == 0) and np.all(mul_pixels == 0):
                continue

            pan_pixels_shape = pan_pixels.shape
            mul_pixels_shape = mul_pixels.shape

            pan = torch.tensor(pan_pixels.astype("float32") / 10_000)
            mul = torch.tensor(mul_pixels.astype("float32") / 10_000)

            pack = torch.concat([pile(pan, factor=4), mul], dim=0).unsqueeze(0)

            pack = pack.to(device)

            _, _, sharp = model(pack)
            sharp = sRGB.forward(sharp)
            sharp = sharp.detach().cpu().numpy()[0]

            left_start = w.col_off - reading_window.col_off
            top_start = w.row_off - reading_window.row_off
            right_end = w.width + left_start
            bottom_end = w.height + top_start

            sharp = sharp[:, top_start:bottom_end, left_start:right_end]

            trimmed_pan = pan_pixels[:, top_start:bottom_end, left_start:right_end]
            pan_nulls = trimmed_pan == 0

            sharp = np.clip(sharp * 65_535, 1, 65_535).astype(np.uint16)
            sharp[:, pan_nulls[0]] = 0

            dst.write(sharp, window=w)


if __name__ == "__main__":
    pansharpen()
