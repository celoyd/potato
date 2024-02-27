import datetime
import json
import pathlib
from pathlib import Path

import sys

from ripple.color import BandsToOklab
from ripple.util import tile, pile, cheap_half

import rasterio as rio
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

# from typing import Sequence, Tuple
from torch.utils.data import DataLoader, Dataset

import click
import logging
from pyproj import Transformer

phi = (1 + 5**0.5) / 2


class OldSatellite(Exception):
    pass


class TooManyNulls(Exception):
    pass


class R2:
    # https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    def __init__(self, start=0):
        self.ρ = 1.3247179572447460259609088
        self.a1 = 1 / self.ρ
        self.a2 = 1 / (self.ρ**2)

        self.position = start

    def __iter__(self):
        return self

    def __next__(self):
        self.position += 1
        return self[self.position - 1]

    def __getitem__(self, idx):
        return (self.a1 * idx) % 1, (self.a2 * idx) % 1


class Tile(Dataset):
    """
    This represents one of the large (~5 km on a side) tiles in
    Maxar’s ARD format. It’s intended to be used within a Chipper
    object. For specifics of ARD, see the oficial documentation:
    https://ard.maxar.com/docs/about/
    """

    def __init__(self, basepath: Path, cid: str):
        self.root_dir = basepath
        self.cid = cid  # catalog ID

        self.json = json.load(open(basepath / f"{cid}.json"))

        self.metadata = self.json["properties"]

        if self.metadata["platform"] not in (
            "worldview-02",
            "worldview-03",
            "WV02",
            "WV03",
        ):
            raise OldSatellite

        self.pan = rio.open(self.root_dir / f"{self.cid}-pan.tif")
        self.mul = rio.open(self.root_dir / f"{self.cid}-ms.tif")

        self.mul_side = self.mul.profile["width"]

        self.to_latlon = Transformer.from_crs(self.mul.crs, 4326)
        # print(self.to_latlon)

    def coords(self, mul_window):
        center = (
            mul_window.col_off + mul_window.width / 2,
            mul_window.row_off + mul_window.height / 2,
        )
        xy = rio.transform.xy(self.mul.transform, cols=center[0], rows=center[1])
        lalo = self.to_latlon.transform(*xy)  # center[1], center[0])
        return {"lat": round(lalo[0], 5), "lon": round(lalo[1], 5)}

    def read(self, side, proportion):
        max_valid = self.mul_side - side

        left, top = (int(z * max_valid) % max_valid for z in proportion)

        mul_dims = (left, top, side, side)
        mul_window = rio.windows.Window(*mul_dims)
        pan_window = rio.windows.Window(*(4 * z for z in mul_dims))

        mul = torch.tensor(self.mul.read(window=mul_window).astype("float32"))
        pan = torch.tensor(self.pan.read(window=pan_window).astype("float32"))

        pan = pile(pan, factor=4)

        pack = torch.cat((pan, mul), 0)
        path = self.mul.name
        coords = self.coords(mul_window)

        return pack, path, coords


class Chipper(Dataset):
    """
    Represents a set of ARDs and makes it easy to read dispersed
    chips out of them. Covers operations like image degradation that
    are not specific to a file (those goes in the Tile object).
    """

    def __init__(self, ard_roots, length, starting_from):

        self.length = length
        self.starting_from = starting_from

        self.chips = []

        self.scattering = R2()

        self.id_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).float()

        self.oklab = BandsToOklab()

        for ard_root in ard_roots:
            for metadata in ard_root.glob("acquisition_collections/*.json"):
                with open(metadata) as md:
                    md = json.load(md)

                inner_json_path = [x["href"] for x in md["links"] if x["rel"] == "item"]
                inner_json_path = inner_json_path[0]

                chip_dir = (metadata.parent / Path(inner_json_path)).parent.absolute()
                chip_cid = Path(inner_json_path).stem

                try:
                    chip = Tile(chip_dir, chip_cid)
                    self.chips.append(chip)
                except OldSatellite:
                    logging.warning(
                        f"Skipping CID {chip_cid}: satellite is not WV-2 or -3"
                    )

    def validish(self, tensor):
        minimum = 0.75
        pixel_count = torch.prod(torch.tensor(tensor.shape))
        valid_count = torch.count_nonzero(tensor)
        return valid_count > (pixel_count * minimum)

    def m_noise(self, shape, scale):
        # Multiplicative noise centers on 1
        return torch.normal(1.0, scale, shape)

    def a_noise(self, shape, scale):
        # Additive noise centers on 0
        return torch.normal(0.0, scale, shape)

    def a_noisy_kernel(self, std=0.1):
        noise = self.a_noise(self.id_kernel.shape, std)
        noisy = self.id_kernel + noise
        noisy /= noisy.sum()
        noisy = noisy.expand(8, -1, -1, -1)
        return noisy

    def eight_noisy_kernels(self, std=0.1):
        identities = self.id_kernel.expand(8, -1, -1, -1)
        noise = self.a_noise(identities.shape, std)
        noisy = identities + noise
        sums = noisy.sum(dim=(-1, -2))
        noisy /= sums.view(8, 1, 1, 1)
        return noisy

    def worsen(self, x, all_std=0.05, each_std=0.1):
        all_k = self.a_noisy_kernel(all_std)
        x = F.conv2d(x, all_k, groups=8, padding="same")

        each_k = self.eight_noisy_kernels(each_std)
        x = F.conv2d(x, each_k, groups=8, padding="same")

        return x

    def pack_to_xy(self, pack):
        pack = pack.float() / 10_000

        # pan = cheap_half(shuf2(y[:16]))
        pan = cheap_half(tile(pack[:16], 4))
        mul = cheap_half(pack[16:])

        rots = int(torch.rand((1,)) * 4)
        pan = torch.rot90(pan, dims=(-1, -2), k=rots)
        mul = torch.rot90(mul, dims=(-1, -2), k=rots)

        pan_down = cheap_half(cheap_half(pan)).unsqueeze(0)
        # pan_down = unshuf2(pan_down)
        pan_down = pile(pan_down, 4)

        mul_down = cheap_half(cheap_half(mul)).unsqueeze(0)
        mul_down = self.worsen(mul_down)

        mul_down = mul_down * self.m_noise(
            mul_down.shape, scale=1 / 250
        ) + self.a_noise(mul_down.shape, scale=1 / 1_000)

        pan_down = pan_down * self.m_noise(
            pan_down.shape, scale=1 / 1_000
        ) + self.a_noise(pan_down.shape, scale=1 / 2_000)

        x = torch.squeeze(torch.cat([pan_down, mul_down], dim=1))

        y = self.oklab(mul)

        return x, y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        chip = index % len(self.chips)  # ?

        x_off, y_off = self.scattering[index + self.starting_from]

        pack, path, coords = self.chips[chip].read(1024, (x_off, y_off))

        if not self.validish(pack):
            raise TooManyNulls

        x, y = self.pack_to_xy(pack)

        return x, y, path, coords


@click.group()
@click.option(
    "--log",
    "-l",
    default="/dev/stdout",  # this seems wrong
    type=click.Path(exists=False, dir_okay=False, allow_dash=True, path_type=Path),
)
def cli(log):
    logging.basicConfig(
        filename=log,
        format="%(asctime)s (%(levelname)s): %(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )


@cli.command()
@click.option(
    "--ard-dir",
    "-a",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--chip-dir",
    "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--count",
    "-n",
    required=True,
    type=int,
)
@click.option("--starting-from", "-s", type=int, default=0)
def make_chips(ard_dir, chip_dir, count, starting_from):
    chipper = Chipper([ard_dir], count, starting_from)
    completed = 0
    failed = 0
    n = starting_from

    while completed < count:
        try:
            x, y, path, coords = chipper[n]
            dst = chip_dir / f"{completed}.pt"
            torch.save((x, y), dst)
            logging.info(f"{dst} at {coords} from {path}")
            completed += 1
        except TooManyNulls:
            failed += 1
        n += 1

    logging.info(f"Fishined {completed} chips. There were {failed} failures.")


@cli.command()
@click.argument(
    "srcs", type=click.Path(exists=True, file_okay=False, path_type=Path), nargs=-1
)
@click.argument(
    "dst", type=click.Path(exists=True, file_okay=False, path_type=Path), nargs=1
)
def link_chips(srcs, dst):
    all_paths = []
    for src in srcs:
        all_paths += list(src.glob("*.pt"))

    total = len(all_paths)
    logging.info(
        f"Preparing to link {total} files from {tuple(str(s) for s in srcs)} to {dst}."
    )

    for p in range(total):
        if p > 0 and p % 1000 == 0:
            logging.info(f"Completed {p} links.")

        n = int((p * phi)) % len(
            all_paths
        )  # arguably random would be better but I feel fancy
        pt = Path(all_paths[n])
        link = Path(dst / f"{p}.pt")

        try:
            pathlib.os.symlink(pt, link)
        except FileExistsError:
            logging.critical(f"{link} already exists. Will not delete or overwrite.")
            sys.exit()

        all_paths = all_paths[:n] + all_paths[n + 1 :]

    logging.info(f"Made {total} links in {dst}.")


if __name__ == "__main__":
    cli()

"""
scattering = R2()

ard, chip_count, dst_dir = argv[1:]
ard = Path(ard)
chip_count = int(chip_count)
dst_dir = Path(dst_dir)

dst_dir.mkdir(exist_ok=True)

ch = Chipper([ard], chip_count)

done = 0
failed = 0

while done < chip_count:
    # Todo: logging
    try:
        pack = ch[done + failed]
        torch.save(pack, dst_dir / f"{done}.pt")
        done += 1
    except TooManyNulls:
        failed += 1
"""
# if failed > done * 100:  # > (length / 2) and (tried / (done+1)) > 2:
#    raise ValueError(f"Too many bad ones in {ard}.")

# python chipper.py make-chips ard chipdir --log chiplog1.log
# python chipper.py link-chips chipdir/{foo,bar} chipdir/pool
