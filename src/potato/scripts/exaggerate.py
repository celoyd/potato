"""Exaggerate images.

potato exaggerate -e n a.tiff b.tiff a-exaggerated.tiff b-exaggerated.tiff

Given same-size sRGB a.tiff and b.tiff, write exaggerated versions. An -e of 1
means no exaggeration at all; -e 10 would mean to make exaggerate(A) 10 times
more different from (A + B)/2 than A already is.

Assumptions and limits:
- We only handle (geo)TIFF and PNG input, and TIFF output.
- Nodata is not considered.
"""

import concurrent.futures
import threading

import click
import numpy
import rasterio
import torch
from tqdm import tqdm

from potato.color import OklabTosRGB, sRGBToOklab
from potato.errors import ImageDimensionError

block_side = 512


@click.command()
@click.argument("src_a_path", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("src_b_path", nargs=1, type=click.Path(exists=True), required=True)
@click.argument("dst_a_path", nargs=1, type=click.Path())
@click.argument("dst_b_path", nargs=1, type=click.Path())
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
def cli(src_a_path, src_b_path, dst_a_path, dst_b_path, exaggeration, workers):
    """Exaggerate two images’ differences."""
    with rasterio.open(src_a_path) as src_a, rasterio.open(src_b_path) as src_b:
        try:
            assert src_a.shape == src_b.shape
            assert src_a.count == src_b.count == 3
            assert src_a.profile["dtype"] == src_b.profile["dtype"]
        except AssertionError as exc:
            raise ImageDimensionError(
                "Expected inputs with 3 channels, the same dimensions, "
                "and the same data type, but found "
                f"{src_a.count, *src_a.shape} {src_a.profile['dtype']} v. "
                f"{src_b.count, *src_b.shape} {src_b.profile['dtype']}."
            ) from exc

        profile = src_a.profile

        if profile["driver"] not in ("GTiff", "PNG"):
            raise ValueError(f"Expected (geo)TIFF or PNG but got {profile['driver']}.")

        # used to scale unit range (for converions) <-> image range
        max_value = float(numpy.iinfo(profile["dtype"]).max)

        profile.update(
            {
                "driver": "gtiff",
                "photometric": "RGB",
                "tiled": True,
                "blockxsize": block_side,
                "blockysize": block_side,
                "dtype": "uint16",
            }
        )

        with (
            rasterio.open(dst_a_path, "w", **profile) as dst_a,
            rasterio.open(dst_b_path, "w", **profile) as dst_b,
        ):
            src_a_lock = threading.Lock()
            src_b_lock = threading.Lock()
            dst_a_lock = threading.Lock()
            dst_b_lock = threading.Lock()

            r_to_l = sRGBToOklab()
            l_to_r = OklabTosRGB()

            def work(w):
                torch.set_num_threads(1)

                with src_a_lock:
                    a = src_a.read(window=w).astype("float32") / max_value
                with src_b_lock:
                    b = src_b.read(window=w).astype("float32") / max_value

                # The unsqueeze before the diff and the [0] after are to wrap
                # the image data in a pseudo-batch dimension for the
                # conversions.
                a_oklab = r_to_l(torch.tensor(a).unsqueeze(0))
                b_oklab = r_to_l(torch.tensor(b).unsqueeze(0))

                difference = (a_oklab - b_oklab) * exaggeration
                a_exag = a_oklab + difference
                b_exag = b_oklab - difference

                a_exag = l_to_r(a_exag)[0]
                b_exag = l_to_r(b_exag)[0]

                a_exag = (
                    (a_exag * max_value).clip(0, max_value).numpy().astype("uint16")
                )
                b_exag = (
                    (b_exag * max_value).clip(0, max_value).numpy().astype("uint16")
                )

                with dst_a_lock:
                    dst_a.write(a_exag, window=w)

                with dst_b_lock:
                    dst_b.write(b_exag, window=w)

            windows = [w for i, w in dst_a.block_windows()]

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
