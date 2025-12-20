"""Get reflectances from corresponding ARD files.

Given matching WV-2/3 ARD pan and mul files in ARD, print the reflectances
at the given point (in pan image coordinates) in pan,deep-blue,blue,...
order, scaled by 10,000 per the ARD spec.
"""

from pathlib import Path

import click
import rasterio
from rasterio import windows

from potato.errors import ImageDimensionError, OldSatelliteError


@click.command()
@click.option(
    "-p",
    "--path-stub",
    type=str,
    required=True,
    help="Shared path up to -pan.tif/ms.tif",
)
@click.option(
    "-x", type=int, required=True, help="x coordinate of the pixel in the pan image"
)
@click.option("-y", type=int, required=True, help="y ditto")
def cli(path_stub, x, y):
    """Pull a reflectance from a pan and mul image pair."""
    pan_path = Path(path_stub + "-pan.tif")
    mul_path = Path(path_stub + "-ms.tif")

    if not pan_path.exists():
        raise FileNotFoundError(
            f"Expected a panchromatic file, but {pan_path} isn’t there"
        )

    if not mul_path.exists():
        raise FileNotFoundError(
            f"Expected a multispectral file, but {mul_path} isn’t there"
        )

    pan_window = windows.Window((x // 4) * 4, (y // 4) * 4, 4, 4)
    mul_window = windows.Window(x // 4, y // 4, 1, 1)

    with rasterio.open(pan_path) as pan:
        pan_patch = pan.read(window=pan_window)

    with rasterio.open(mul_path) as mul:
        mul_patch = mul.read(window=mul_window)

    if (pan_patch.size == 0) or (mul_patch.size == 0):
        raise ImageDimensionError(
            "Failed to find a pixel there. Check -x/-y against image dimensions."
        )

    if mul_patch.shape[0] != 8:
        raise OldSatelliteError(f"Expected 8 bands but got {mul_patch.shape[0]}.")

    reflectances = [pan_patch.mean().item(), *list(int(x) for x in mul_patch.flatten())]

    print(",".join(str(r) for r in reflectances))


if __name__ == "__main__":
    cli()
