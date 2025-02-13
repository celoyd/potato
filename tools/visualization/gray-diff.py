"""
gray_diff before.tiff after.tiff change.tiff

Given existing same-size sRGB before.tiff and after.tiff, write change.tiff.
This is a gray-centered image diff, so for example green in the output means
that the after image was more green there.

Assumptions and limits:
- We only handle (geo)TIFF and PNG.
- Output is the same type as the input.
- Nodata is not considered.

It’s a relatively elaborate script, in order to use correct color conversions
and to be memory-efficient. The basic operation is very similar to a simple
(b - a) + 1/2. See the work() function for details.
"""

import rasterio

import torch
import numpy

from tqdm import tqdm

import click

from potato.color import OklabTosRGB, sRGBToOklab

import concurrent.futures
import threading


class ImageDimensionError(Exception):
    pass


block_side = 512


@click.command()
@click.argument("src_a_path", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("src_b_path", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("dst_path", nargs=1, type=click.Path(exists=False))
@click.option(
    "-e",
    "--exaggeration",
    required=False,
    type=float,
    default=1.0,
    help="Factor by which to exaggerate differences",
)
@click.option(
    "-w",
    "--workers",
    required=False,
    type=int,
    default=4,
    help="Number of threads to use",
)
def diff(src_a_path, src_b_path, dst_path, exaggeration, workers):
    with rasterio.open(src_a_path) as src_a, rasterio.open(src_b_path) as src_b:
        try:
            assert src_a.shape == src_b.shape
            assert src_a.count == src_b.count == 3
            assert src_a.profile["dtype"] == src_b.profile["dtype"]
        except AssertionError:
            raise ImageDimensionError(
                "Expected inputs with 3 channels, the same dimensions, "
                "and the same data type, but found "
                f"{src_a.count, *src_a.shape} {src_a.profile['dtype']} v. "
                f"{src_b.count, *src_b.shape} {src_b.profile['dtype']}."
            )

        profile = src_a.profile

        if profile["driver"] not in ("TIFF", "GTiff", "PNG"):
            raise ValueError(
                f"Expected TIFF, geoTIFF, or PNG but got {profile['driver']}."
            )

        # used to scale unit range (for converions) <-> image range
        max_value = numpy.iinfo(profile["dtype"]).max

        profile.update(
            {
                "photometric": "RGB",
                "tiled": True,
                "blockxsize": block_side,
                "blockysize": block_side,
                "dtype": "uint16",
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            src_a_lock = threading.Lock()
            src_b_lock = threading.Lock()
            dst_lock = threading.Lock()

            def work(w):
                torch.set_num_threads(1)

                r_to_l = sRGBToOklab()
                l_to_r = OklabTosRGB()

                with src_a_lock:
                    a = src_a.read(window=w).astype("float32") / float(max_value)
                with src_b_lock:
                    b = src_b.read(window=w).astype("float32") / float(max_value)

                # The unsqueeze before the diff and the [0] after are to wrap the
                # image data in a pseudo-batch dimension for the conversions.
                oklab_a = r_to_l(torch.tensor(a).unsqueeze(0))
                oklab_b = r_to_l(torch.tensor(b).unsqueeze(0))

                diff = oklab_b - oklab_a
                diff *= exaggeration
                diff[0, 0] += 0.5

                srgb_diff = l_to_r(diff)[0]
                srgb_diff = (srgb_diff * max_value).clip(0, max_value)
                srgb_diff = srgb_diff.numpy().astype("uint16")

                with dst_lock:
                    dst.write(srgb_diff, window=w)

            windows = [w for i, w in dst.block_windows()]

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                list(
                    tqdm(
                        executor.map(work, windows),
                        total=len(windows),
                        unit="b",
                        smoothing=0.1,
                    )
                )


if __name__ == "__main__":
    diff()
