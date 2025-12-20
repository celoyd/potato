"""Various things useful to do to a single chip."""

from pathlib import Path

import click
import torch

from potato.color import OklabTosRGB
from potato.losses import ΔEOK
from potato.model import Potato
from potato.util import unit_srgb_to_uint16_tiff

# def oklab_to_tiff(okl, converter, dst_path):
#     """Write an oklab to dst_path, through converter."""

#     unit_srgb_to_uint16_tiff(sRGB, dst_path)


def psnr_unit(reference, test):
    """Return standard PSNR. Assumes attitive color in 0..1."""
    mse = ((reference - test) ** 2).mean()
    return 20 - (10 * torch.log10(mse))


@torch.inference_mode()
def fuse(gen, pan, mul):
    """Give the pansharpened version and its loss against the target."""
    result = gen((pan.unsqueeze(0), mul.unsqueeze(0)))[0]
    return result


@click.command()
# @click.option("--device", default="cpu", help="Torch device to use")
@click.argument("chip_path", nargs=1)
@click.option(
    "--target",
    type=click.Path(),
    help="Path to make a PNG of the target image in this chip",
)
@click.option(
    "--potato",
    type=click.Path(),
    help="Path to make a PNG of Potato’s output from this chip",
)
@click.option(
    "--loss",
    type=bool,
    is_flag=True,
    help="Print the mean ΔEOK2 of Potato’s output (in JNDs)",
)
@click.option(
    "--psnr",
    type=bool,
    is_flag=True,
    help="Print the PSNR of Potato’s output (deprecated)",
)
@click.option(
    "--overwrite",
    type=bool,
    is_flag=True,
    help="Overwrite output file(s)",
)
@click.option(
    "-w",
    "--weights",
    type=click.Path(),
    default="sessions/latest.pt",
    help="Weights for Potato",
)
def cli(chip_path, target, potato, loss, psnr, overwrite, weights):
    """Display, or test Potato on, a chip."""
    chip_path = Path(chip_path)
    if not chip_path.exists():
        raise FileNotFoundError(chip_path)

    chip = torch.load(chip_path, weights_only=False)

    if potato or loss or psnr:
        # We’ll need the model
        gen = Potato(48)
        gen.load_state_dict(torch.load(weights, weights_only=True))

    if potato or target or psnr:
        # We’ll need color conversion
        o2r = OklabTosRGB()

    if target:
        unit_srgb_to_uint16_tiff(o2r(chip[1]), target, overwrite_ok=overwrite)

    if potato or loss or psnr:
        fused = fuse(gen, chip[0][0], chip[0][1])

        if potato or psnr:
            rgb_potato = o2r(fused)

        if potato:
            unit_srgb_to_uint16_tiff(rgb_potato, potato, overwrite_ok=overwrite)

        if loss:
            ΔEOK_loss = ΔEOK(fused, chip[1]).item()
            print(f"50 × ΔEOK2: {ΔEOK_loss:.3g}")

        if psnr:
            psnr_loss = psnr_unit(o2r(fused), o2r(chip[1])).item()
            print(
                f"PSNR: {psnr_loss:.3g}",
            )


if __name__ == "__main__":
    cli()
