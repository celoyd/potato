"""Apply a simple contrast curve to a uint16 sRGB image.

This is based on Jed Smith’s Hill-Langmuir–style tonescale, proposed at
https://community.acescentral.com/t/output-transform-tone-scale/3498/177 and
https://www.desmos.com/calculator/yts27i1fio – a curve I happen to like, but
that is chosen mainly to be simple. We only use two parameters: exposure and
contrast. The idea here is not to provide a canonical display treatment for
Potato  output, only to show how one might think about adjustments.

The default parameters are meant to make the image roughly comparable in tone
to the “official” *-visual.tiff version, without actually fitting curves or
anything.

In anything approaching a production implementation of pansharpening for
display, this kind of thing would presumably be folded into the pansharpening
script, so the file only had to be written once.
"""

import concurrent.futures
import threading
from pathlib import Path

import click
import numpy as np
import rasterio
import torch
from tqdm import tqdm

from potato.color import linearize_srgb, safe_pow, unlinearize_lsrgb
from potato.errors import ImageDimensionError

block_side = 512


def hill_langmuir_smith(x, n, expo, red_corr, blue_corr):
    """Apply a Hill-Langmuir curve as outlined by Jed Smith."""
    x *= expo
    x[0] *= red_corr
    x[2] *= blue_corr

    # Naturally all these little things would more optimally be calculated
    # once instead of per-tile, but the cost is small.
    g0 = 0.5
    g1 = 0.5
    m0 = 1.5
    m1 = 1.0

    p0 = g0**n
    p1 = m0**n

    s1 = (m1 * g1 * (p0 - p1)) / (m1 * p0 - g1 * p1)
    s0 = (p0 * (s1 - g1)) / g1

    x2n = safe_pow(x.clip(0, None), n)
    return (s1 * x2n) / (x2n + s0)


@click.command()
@click.argument("src_path", nargs=1, type=click.Path())
@click.argument("dst_path", nargs=1, type=click.Path())
@click.option("-n", "--contrast", default=2.0, help="Amount of contrast.")
@click.option(
    "-e",
    "--exposure",
    default=2.4,
    help="Overall radiance multiplier",
)
@click.option("-r", "--red", default=0.9, help="Red exposure (relative to green).")
@click.option("-b", "--blue", default=1.1, help="Blue exposure (relative to green).")
@click.option(
    "-w",
    "--workers",
    required=False,
    type=int,
    default=4,
    help="Number of threads to use.",
)
@click.option("--overwrite", default=False, is_flag=True, help="Overwrite destination.")
def cli(src_path, dst_path, contrast, exposure, red, blue, workers, overwrite):
    """Add contrast to an image."""
    with rasterio.open(src_path) as src:
        profile = src.profile
        if (profile["dtype"] != "uint16") or (profile["count"] != 3):
            raise ImageDimensionError(
                "Expected 3-channel uint16 input, but got "
                f"{profile['count']}-channel {profile['dtype']}."
            )

        profile.update(
            {
                "driver": "gtiff",
                "photometric": "RGB",
                "tiled": True,
                "blockxsize": block_side,
                "blockysize": block_side,
            }
        )

        if Path(dst_path).exists() and not overwrite:
            raise FileExistsError(dst_path)

        with rasterio.open(dst_path, "w", **profile) as dst:
            reading_lock = threading.Lock()
            writing_lock = threading.Lock()

            @torch.inference_mode()
            def work(w):
                with reading_lock:
                    chunk = src.read(window=w)

                chunk = torch.tensor(chunk.astype(np.float32) / 65_535.0)
                chunk = linearize_srgb(chunk)

                chunk = hill_langmuir_smith(chunk, contrast, exposure, red, blue)

                chunk = unlinearize_lsrgb(chunk)
                chunk = (chunk * 65_535).clip(0, 65_535).numpy().astype(np.uint16)

                with writing_lock:
                    dst.write(chunk, window=w)

            windows = [w for _, w in dst.block_windows()]

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(work, w) for w in windows]
                for _ in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    unit=" block",
                    smoothing=0.005,
                ):
                    pass


if __name__ == "__main__":
    cli()
