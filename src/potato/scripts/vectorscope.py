"""Make a simple vectorscope.

potato vectorscope input.image output.tiff

There are many different ways to do vectorscopes, for example in original
color v. monochrome v. at middle brightness. More or less arbitrarily, and
because it’s easy to draw, we simply pull pixels from the source image in
reading order and put them in the destination image. Later-encountered pixels
cover previous ones. This is somewhat silly but it works for us.
"""

import warnings

import click
import numpy as np
import rasterio
import torch
import tqdm
from rasterio.windows import Window

from potato.color import OklabTosRGB, sRGBToOklab

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

r_to_l = sRGBToOklab()
l_to_r = OklabTosRGB()


@click.command()
@click.option(
    "-s",
    "--size",
    type=int,
    default=1024,
    help="Side length of vectorscope square",
)
@click.argument("source_path", required=True, type=click.Path(), nargs=1)
@click.argument("vectorscope_path", required=True, type=click.Path(), nargs=1)
def cli(size, source_path, vectorscope_path):
    """Make a simple vectorscope."""
    with rasterio.open(source_path) as src:
        if src.profile["count"] != 3:
            raise ValueError(
                f"Expected a 3-band sRGB image, but got {src.profile['count']} bands"
            )

        src_dtype = src.profile["dtype"]
        src_max = np.iinfo(src_dtype).max
        vs = np.zeros((3, size, size), dtype=src_dtype)

        for row in tqdm.tqdm(range(src.profile["height"]), unit=" row"):
            w = Window(0, row, src.profile["width"], 1)
            raw_pixels = src.read(window=w)
            pixels = torch.tensor(raw_pixels.astype(np.float32)).unsqueeze(0)

            oklab = r_to_l(pixels / src_max).numpy()

            a = ((oklab[0, 1] * size) + size / 2).astype(np.uint16)
            b = ((oklab[0, 2] * size) + size / 2).astype(np.uint16)

            a_valid = (a >= 0) & (a < size)
            b_valid = (b >= 0) & (b < size)

            valid = a_valid & b_valid

            raw_pixels = raw_pixels[:, 0, :]  # drop the height dimension

            vs[:, a[valid], b[valid]] = raw_pixels[:, valid[0]]

        with rasterio.open(
            vectorscope_path,
            "w",
            driver="gtiff",
            count=3,
            photometric="RGB",
            height=size,
            width=size,
            dtype=src_dtype,
        ) as dst:
            dst.write(vs)


if __name__ == "__main__":
    cli()
