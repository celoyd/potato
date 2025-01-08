import datetime
import json
import pathlib
from pathlib import Path

import sys

from potato.color import BandsToOklab
from potato.util import tile, pile, cheap_half

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


class MissingImage(Exception):
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

        self.pan_path = self.root_dir / f"{self.cid}-pan.tif"
        self.mul_path = self.root_dir / f"{self.cid}-ms.tif"

        if not (self.pan_path.exists() and self.mul_path.exists()):
            raise MissingImage

        with rio.open(self.mul_path) as mul:
            self.mul_side = mul.profile["width"]
            self.mul_transform = mul.profile["transform"]
            self.mul_crs = mul.profile["crs"]

        self.to_latlon = Transformer.from_crs(self.mul_crs, 4326)

    def coords(self, mul_window):
        center = (
            mul_window.col_off + mul_window.width / 2,
            mul_window.row_off + mul_window.height / 2,
        )
        xy = rio.transform.xy(self.mul_transform, cols=center[0], rows=center[1])
        lalo = self.to_latlon.transform(*xy)  # center[1], center[0])
        return {"lat": round(lalo[0], 5), "lon": round(lalo[1], 5)}

    def read(self, side, proportion):
        with rio.open(self.pan_path) as pan, rio.open(self.mul_path) as mul:
            max_valid = self.mul_side - side

            left, top = (int(z * max_valid) % max_valid for z in proportion)

            mul_dims = (left, top, side, side)
            mul_window = rio.windows.Window(*mul_dims)
            pan_window = rio.windows.Window(*(4 * z for z in mul_dims))

            mul = torch.tensor(mul.read(window=mul_window).astype("float32"))
            pan = torch.tensor(pan.read(window=pan_window).astype("float32"))

            pan = pile(pan, factor=4)

            pack = torch.cat((pan, mul), 0)
            path = self.mul_path
            coords = self.coords(mul_window)

            return pack, path, coords


class Chipper(Dataset):
    """
    Represents a set of ARDs and makes it easy to read dispersed
    chips out of them. Covers operations that are not specific to
    a file (those goe in the Tile object).
    """

    def __init__(self, ard_roots, length, accept_cids, starting_from):

        self.length = length
        self.starting_from = starting_from

        self.accept_cids = accept_cids

        self.chips = []

        self.scattering = R2()

        self.oklab = BandsToOklab()

        for ard_root in ard_roots:
            for metadata in ard_root.glob("acquisition_collections/*.json"):
                with open(metadata) as md:
                    md = json.load(md)

                inner_json_paths = [
                    x["href"] for x in md["links"] if x["rel"] == "item"
                ]
                for inner_json_path in inner_json_paths:
                    chip_dir = (
                        metadata.parent / Path(inner_json_path)
                    ).parent.absolute()
                    chip_cid = Path(inner_json_path).stem
                    if (accept_cids != []) and (chip_cid not in accept_cids):
                        logging.warning(
                            f"Skipping CID {chip_cid}: not in the accept list"
                        )
                        continue

                    try:
                        chip = Tile(chip_dir, chip_cid)
                        self.chips.append(chip)
                    except OldSatellite:
                        logging.warning(
                            f"Skipping CID {chip_cid}: satellite is not WV-2 or -3"
                        )
                    except MissingImage:
                        logging.warning(
                            f"Skipping CID {chip_cid} in dir {chip_dir}: "
                            "one of its files is missing"
                        )
        if len(self.chips) < 1:
            logging.error("No (accept-listed) chips found in this ARD")
            raise ValueError("No (accept-listed) chips found in this ARD")

    def validish(self, tensor):
        minimum = 0.75
        pixel_count = torch.prod(torch.tensor(tensor.shape))
        valid_count = torch.count_nonzero(tensor)
        return valid_count > (pixel_count * minimum)

    def pack_to_xy(self, pack):
        pack = pack.float() / 10_000

        pan = cheap_half(tile(pack[:16], 4))
        mul = cheap_half(pack[16:])

        rots = int(torch.rand((1,)) * 4)
        pan = torch.rot90(pan, dims=(-1, -2), k=rots)
        mul = torch.rot90(mul, dims=(-1, -2), k=rots)

        pan_down = cheap_half(cheap_half(pan)).unsqueeze(0)
        pan_down = pile(pan_down, 4)

        mul_down = cheap_half(cheap_half(mul)).unsqueeze(0)

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
@click.option(
    "--accept-cids",
    "-a",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
@click.option("--starting-from", "-s", type=int, default=0)
def make_chips(ard_dir, chip_dir, count, accept_cids, starting_from):
    if accept_cids:
        accept_cids = [x.strip() for x in open(accept_cids)]
    else:
        accept_cids = []

    chipper = Chipper([ard_dir], count, accept_cids, starting_from)
    completed = 0
    failed = 0
    n = starting_from

    while completed < count:
        try:
            x, y, path, coords = chipper[n]
            dst = chip_dir / f"{starting_from + completed}.pt"
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

        n = int(p * len(all_paths) * phi) % len(all_paths)

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
