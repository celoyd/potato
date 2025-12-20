"""
ab_only.py input.image output.tiff
Convert the sRGB input to oklab, set its L channel to 0.5, convert back to
sRGB, and write it out.

This is a simple operation but we made it complicated to handle different
data types, to check for common mistakes, to process concurrently, etc. This
is a useful script but not an elegant one.
"""

import warnings

import concurrent
import threading

import rasterio
import numpy as np
import torch
from pathlib import Path

from tqdm import tqdm
import click

from potato.color import sRGBToOklab, OklabTosRGB

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

r_to_l = sRGBToOklab()
l_to_r = OklabTosRGB()


def work(img, src_dtype, l_level):
    """The core task"""
    top = np.iinfo(src_dtype).max
    img = img.astype(np.float32) / top  # squeezing to unit range
    img = torch.tensor(img)
    img = r_to_l(img)
    img[-3] = l_level  # <- this all we’re really here to do
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
def wrap(l_level, worker_count, srcpath, dstpath):
    """All the i/o and iteration paperwork"""
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
    wrap()
