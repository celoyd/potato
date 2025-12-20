"""Visualize only the chroma plane of an image.

$ potato ab input.image output.tiff

Convert the sRGB input to oklab, set its L channel to 0.5, convert back to
sRGB, and write it out.

This is a simple operation but we made it complicated to handle different data
types, to check for common mistakes, to process concurrently, etc.
"""

import concurrent
import threading
import warnings
from pathlib import Path

import click
import numpy as np
import rasterio
import torch
from tqdm import tqdm

from potato.color import OklabTosRGB, sRGBToOklab

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

r_to_l = sRGBToOklab()
l_to_r = OklabTosRGB()


def work(img, src_dtype, l_level):
    """Do the actual thing."""
    top = np.iinfo(src_dtype).max
    img = img.astype(np.float32) / top  # squeezing to unit range
    img = torch.tensor(img)
    img = r_to_l(img)
    img[-3] = l_level  # <- this’s all we’re really here to do
    img = l_to_r(img)
    img = img.numpy()
    img = (img * top).clip(0, top).astype(src_dtype)
    return img


@click.command()
@click.option("--l-level", "-l", default=0.5, help="L level to set")
@click.option("--worker-count", "-w", default=4, help="Concurrency")
@click.argument(
    "srcpath",
    nargs=1,
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
)
@click.argument(
    "dstpath",
    nargs=1,
    type=click.Path(file_okay=True, dir_okay=False),
    # Why can’t you just do click.Path(..., exists=False) and have that work?
    # Great question. For the click team, not for me.
    callback=(
        lambda _, __, p: (
            p if not Path(p).exists() else click.BadParameter(f"{p} already exists")
        )
    ),
)
def cli(l_level, worker_count, srcpath, dstpath):
    """Copy an image’s a/b plane (color) to a new image."""
    with rasterio.open(srcpath, "r") as src:
        profile = src.profile
        if profile["count"] != 3:
            raise ValueError(f"Expected 3-band input but got {profile['count']} bands!")
        if profile["dtype"] not in ("uint8", "uint16"):
            raise ValueError(
                f"Expected uint8 or uint16 input but got {profile['dtype']}!"
            )
        profile.update({"driver": "gtiff", "photometric": "rgb"})
        read_lock = threading.Lock()
        write_lock = threading.Lock()
        with rasterio.open(dstpath, "w", **profile) as dst:

            def do_a_window(w):
                with read_lock:
                    chunk = src.read(window=w)
                chunk = work(chunk, profile["dtype"], l_level)
                with write_lock:
                    dst.write(chunk, window=w)

            ws = [w for i, w in dst.block_windows()]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as executor:
                futures = [executor.submit(do_a_window, w) for w in ws]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(ws),
                    unit=" block",
                ):
                    future.result()


if __name__ == "__main__":
    cli()
