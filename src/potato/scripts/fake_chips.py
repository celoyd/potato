"""Create synthetic chips.

potato fake-chips -i <c.csv> -c <n> -o <dir>

Make chips with difficult colors (from colors.csv) and reasonably
plausible spatial statistics.

Please see the accompanying documentation for an illustrated overview.
"""

from pathlib import Path

import click
import torch
from tqdm import tqdm

from potato.color import BandsToOklab
from potato.util import cheap_half, noisebox  # , Chip

b_to_l = BandsToOklab()

"""This is written in a fairly monolithic and unoptimized way for clarity and 
ease of editing. It assumes that the editor has taken the time to understand it 
visually and is not working from the source alone. Hint hint.

Definitions/conventions:
- A reflectance is a 9-tuple of reflectances (sampled from real imagery).
- A field is a (9, 512, 512) tensor of reflectances.
- An alpha is a (512, 512) tensor used as a weight to mix fields.
- A chip is a field with its mul part downsampled to (8, 128, 128); it
  is the format used as input for training.
"""


def make_simple_chip(reflectances, side):
    """Combine two reflectances."""
    rA, rB = two_different_reflectances(reflectances)
    A_field = reflectance_to_field(rA, side)
    B_field = reflectance_to_field(rB, side)

    alpha = noisebox(power=2, side=2 * side)

    C_field = alpha * A_field + (1 - alpha) * B_field
    return C_field


def make_complex_chip(reflectances, side):
    """Make a gnarly synthetic chip."""
    cut_ct = 16

    a = make_simple_chip(reflectances, side)
    b = make_simple_chip(reflectances, side)
    c = make_simple_chip(reflectances, side)
    edge_mask = noisebox(power=2, side=2 * side)
    gap_mask = noisebox(power=3, side=2 * side)

    A = torch.where(edge_mask > 0, a, b)
    B = torch.where(edge_mask > 0, b, a)

    chip = a.clone()

    cuts = torch.randn(cut_ct)
    cuts[0] = -torch.inf
    cuts[1] = torch.inf
    cuts, _idxs = torch.sort(cuts)

    for i in range(1, cut_ct):
        chip = torch.where(
            (gap_mask > cuts[i - 1]) & (gap_mask < cuts[i]), [A, B][i % 2], chip
        )
        chip = torch.where(
            (gap_mask - cuts[i]).abs() < 0.025, (c * torch.rand(1) * 0.1), chip
        )

    chip = cheap_half(torch.nan_to_num(chip)).clamp(1 / 10_000.0, 10_000)
    return chip


def field_to_chip(field):
    """Do the “observing” or “inverse pansharpening” operation."""
    oklab = b_to_l(field[1:])
    mul_small = cheap_half(cheap_half(field[1:]))
    x = (field[0].unsqueeze(0), mul_small)
    return x, oklab


def reflectance_to_field(reflectance, side):
    """Fill an ~image with a given reflectance."""
    return torch.tile(reflectance.view(9, 1, 1), (1, 2 * side, 2 * side))


def two_different_reflectances(reflectances):
    """Pull two reflectances without replacement."""
    i, j = torch.randperm(len(reflectances))[:2]
    return reflectances[i], reflectances[j]


@click.command()
@click.option(
    "-i",
    "--reflectance-file",
    required=True,
    help="Path to CSV of reflectances (try ancillary-data/fake-chips/reflectances.csv).",
)
@click.option(
    "-o",
    "--destination",
    required=True,
    type=Path,
    help="Directory to write chips into",
)
@click.option("-c", "--count", required=True, type=int, help="Number of chips to make")
@click.option("-r", "--random-seed", required=False, default=0, help="PRNG seed")
@click.option(
    "-s", "--side", required=False, default=512, help="Side length of squares"
)
def cli(reflectance_file, destination, count, random_seed, side):
    """Synthesize textured chips from reflectance profiles."""
    torch.manual_seed(random_seed)

    with open(reflectance_file, encoding="utf-8") as rf:
        reflectances = list(
            torch.tensor(list(float(r.strip()) for r in line.split(",")[:9])) / 10_000
            for line in rf
        )

    if len(reflectances) <= 6:
        raise ValueError("Need more than half a dozen reflectances")

    for c in tqdm(range(count)):
        x, y = field_to_chip(make_complex_chip(reflectances, side))
        torch.save((x, y), destination / f"{c}.pt")


if __name__ == "__main__":
    cli()
