"""
demo.py: pansharpen with potato
Usage: python demo.py pan.tiff mul.tiff -w weights.pt output.tiff
Also see the --help.

This code contains many hardcoded assmptions about data format and 
user preferences. It is truly a demo.

Example workflow:

First let’s break up the super long filenames by part for ease
(and to make my linter stop complaining about line length):

$ export nbo_demo_base=maxar-opendata/events/Kenya-Flooding-May24/ard/37/211111023311
$ export nbo_demo_file=2023-11-30/104001008E063C00

We get the pan and mul parts of a big image (downtown Nairobi):

$ aws s3 cp s3://${nbo_demo_base}/${nbo_demo_file}-pan.tif .
$ aws s3 cp s3://${nbo_demo_base}/${nbo_demo_file}-ms.tif .

$ python demo.py -d cuda -w sessions/bintje/23-gen.pt \
    104001008E063C00-{pan,ms}.tif test.tiff

You might then want to, for example, make it a cloud-optimized geotiff:

$ pip install -U rio-cogeo
$ rio cogeo create test.tiff nairobi.tiff

You could then fetch the default pansharpening:

$ aws s3 cp s3://${nbo_demo_base}/${nbo_demo_file}-visual.tif .

And then open both in QGIS to compare:

$ qgis nairobi.tiff 104001008E063C00-visual.tif

"""

import click

import torch
import numpy as np

from pathlib import Path

from tqdm import tqdm

import rasterio
from rasterio import windows

from potato.model import Potato
from potato.util import pile
from potato.color import OklabTosRGB

import concurrent.futures
import threading


class ImageDimensionError(Exception):
    pass


block = 1024  # edge of square to pansharpen at a time
apron = 32  # extra space around each block
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


@click.command()
@click.argument("panpath", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("mulpath", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("dstpath", nargs=1, type=click.Path(exists=False))
@click.option(
    "-w",
    "--weights",
    type=click.Path(exists=True),
    required=True,
    help="Checkpoint (weights) file",
)
@click.option(
    "-c",
    "--compile",
    is_flag=True,
    default=False,
    help="Compile model with JIT (only works on some devices)",
)
@click.option(
    "-d", "--device", default="cuda", help="Torch device (e.g., cuda, cpu, mps)"
)
@click.option(
    "-o", "--overwrite", is_flag=True, help="Destroy existing destination file"
)
def pansharpen(panpath, mulpath, dstpath, weights, compile, device, overwrite):
    with rasterio.open(panpath) as panfile, rasterio.open(mulpath) as mulfile:
        if mulfile.count != 8:
            raise ImageDimensionError(
                "Expected an 8-band multispectral image but got "
                f"{mulfile.count} bands. Check that this is a "
                "WV-2/3 image (CID should start 103 or 104)."
            )
        if (mulfile.width != panfile.width // 4) or (
            mulfile.height != panfile.height // 4
        ):
            raise ImageDimensionError(
                "Expected multispectral edge length exactly "
                "1/4 pan edge length but got "
                f"pan w×h {panfile.width}×{panfile.height} and "
                f"mul w×h {mulfile.width}×{mulfile.height}"
            )

        profile = panfile.profile
        profile.update(
            {
                "driver": "gtiff",  # in case some jokester wasts to try VRTs
                "bigtiff": "if_safer",
                "count": 3,
                "photometric": "RGB",
                "blocksize": block,
                "tiled": True,
                "blockxsize": block,
                "blockysize": block,
                "compress": "zstd",
                "predictor": 2,
                "interleave": "pixel",
                "dtype": np.uint16,
                "nodata": 0,
            }
        )

        if Path(dstpath).exists() and not overwrite:
            raise FileExistsError(
                f"{dstpath} already exists but you didn’t ask for --overwrite"
            )

        with rasterio.open(dstpath, "w", **profile) as dstfile:
            model = Potato().to(device)

            model.load_state_dict(
                torch.load(weights, map_location=device, weights_only=True)
            )
            model.eval()

            if compile:
                model = torch.compile(model)

            sRGB = OklabTosRGB().to(device)

            read_lock = threading.Lock()
            device_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(w):
                buffered_window = buffer_window(w, apron)
                reading_window = clip_window(buffered_window, panfile)

                with read_lock:
                    pan_pixels = panfile.read(window=reading_window)
                    mul_pixels = mulfile.read(window=quarter_window(reading_window))

                pan_pixels_shape = pan_pixels.shape

                if np.all(pan_pixels == 0) and np.all(mul_pixels == 0):
                    blank = torch.ones((3, pan_pixels_shape[-2], pan_pixels_shape[-1]))
                    dstfile.write(blank, window=w)
                    return

                pan = torch.tensor(pan_pixels.astype("float32") / 10_000)
                mul = torch.tensor(mul_pixels.astype("float32") / 10_000)

                pack = torch.concat([pile(pan, factor=4), mul], dim=0).unsqueeze(0)

                with device_lock:
                    pack = pack.to(device)
                    sharp = model(pack)
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

                with write_lock:
                    dstfile.write(sharp, window=w)

            windows = [w for i, w in dstfile.block_windows()]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                _ = list(
                    tqdm(
                        executor.map(process, windows),
                        total=len(windows),
                        unit=" block",
                    )
                )


if __name__ == "__main__":
    pansharpen()
