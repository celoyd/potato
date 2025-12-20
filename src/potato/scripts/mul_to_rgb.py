"""Convert an ARD WV-2/3 multispectral image to (an) RGB image(s).

Choose whether you want using Potato’s fancy conversion and/or a more typical
RGB-alone method:

potato mul-to-rgb ...-ms.tif --potato p.tiff --standard s.tiff

N.b., this is implemented in a memory-wasteful way, basically assuming that
anything we ever want to convert will fit in memory at once. It’s fine for
me so far, but definitely wrong in principle.
"""

from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import rasterio
import torch
from tqdm import tqdm

from potato.color import BandsToOklab, OklabTosRGB

block_side = 1024

ref2lab = BandsToOklab()
lab2rgb = OklabTosRGB()


def to_unit(n):
    """Scale from uint16 reflectance * 10_000 to float in 0..1."""
    return torch.tensor(n.astype(np.float32) / 10_000)


def to_uint16(n):
    """Scale from unit to uint16."""
    return (n.numpy() * 65_535).clip(0, 65_535).astype(np.uint16)


def fancy(refl):
    """Do the cool Potato-style bands -> RGB conversion."""
    oklab = ref2lab(refl)
    srgb = lab2rgb(oklab)
    return srgb


def unfancy(refl):
    """Do normal RGB -> RGB conversion."""
    rgb = torch.stack([refl[4], refl[2], refl[1]])
    rgb = rgb ** (1 / 2.2)  # sRGB gamma
    return rgb


@click.command()
@click.argument("mul_path", nargs=1, type=click.Path(), required=True)
@click.option(
    "-p", "--potato", type=click.Path(), help="Write Potato’s sRGB version here"
)
@click.option(
    "-s",
    "--standard",
    type=click.Path(),
    help="Write a standard (RGB-only) sRGB conversion here",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=4,
    help="Number of workers",
)
def cli(mul_path, potato, standard, workers):
    """Convert a multispectral image to sRGB."""
    if potato is None and standard is None:
        raise ValueError("Very funny!")

    with rasterio.open(mul_path, "r") as src:
        profile = src.profile
        if profile["count"] != 8 or profile["dtype"] != "uint16":
            raise ValueError("This is for 8-band WV-2/3 ARD multispectral images only.")

        profile.update(
            {
                "count": 3,
                "photometric": "RGB",
                "tiled": True,
                "blockxsize": block_side,
                "blockysize": block_side,
                "compression": "zstd",
            }
        )

        dst_potato = rasterio.open(potato, "w", **profile) if potato else None
        dst_standard = rasterio.open(standard, "w", **profile) if standard else None

        # This is just to have something to iterate over the windows of:
        out = dst_potato if dst_potato else dst_standard
        window_count = len(list(out.block_windows()))

        potato_futures = []  # https://www.investing.com/commodities/potato
        standard_futures = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, w in tqdm(list(out.block_windows()), desc="Reading"):
                pixels = to_unit(src.read(window=w))

                if dst_potato:
                    potato_futures.append(executor.submit(fancy, pixels.clone()))
                if dst_standard:
                    standard_futures.append(executor.submit(unfancy, pixels.clone()))

        if dst_potato:
            for future, (_, window) in tqdm(
                zip(potato_futures, dst_potato.block_windows(), strict=True),
                desc="Writing Potato image",
                total=window_count,
            ):
                block = to_uint16(future.result())
                dst_potato.write(block, window=window)
            dst_potato.close()

        if dst_standard:
            for future, (_, window) in tqdm(
                zip(standard_futures, dst_standard.block_windows(), strict=True),
                desc="Writing standard image",
                total=window_count,
            ):
                block = to_uint16(future.result())
                dst_standard.write(block, window=window)
            dst_standard.close()


if __name__ == "__main__":
    cli()
