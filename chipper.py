import datetime
import json
from pathlib import Path
from sys import stdout
import numpy as np
import rasterio as rio
import torch
from einops import rearrange
from torch import nn
from typing import Sequence, Tuple
from torch.utils.data import DataLoader, Dataset
from sys import argv


# PixelShuffle operations by other names:
def pile(x, factor):
    return rearrange(
        x, "... c (h f0) (w f1) -> ... (c f0 f1) h w", f0=factor, f1=factor
    )


def tile(x, factor):
    return rearrange(
        x, "... (c f0 f1) h w -> ... c (h f0) (w f1)", f0=factor, f1=factor
    )


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


scattering = R2()


class OldSatellite(Exception):
    pass


class TooManyNulls(Exception):
    pass


class Chip(Dataset):
    def __init__(self, basepath: Path, cid: str):
        self.root_dir = basepath
        self.cid = cid

        self.json = json.load(open(basepath / f"{cid}.json"))

        self.metadata = self.json["properties"]

        if self.metadata["platform"] not in (
            "worldview-02",
            "worldview-03",
            "WV02",
            "WV03",
        ):
            raise OldSatellite

        # self.gsd_max = self.metadata["gsd_max"]
        self.pan = rio.open(self.root_dir / f"{self.cid}-pan.tif")
        self.mul = rio.open(self.root_dir / f"{self.cid}-ms.tif")

        self.mul_side = self.mul.profile["width"]

    def validish(self, tensor):
        minimum = 0.75
        pixel_count = torch.prod(torch.tensor(tensor.shape))
        valid_count = torch.count_nonzero(tensor)
        return valid_count > (pixel_count * minimum)

    def read(self, side: int, proportion: Tuple[float, float]):
        max_valid = self.mul_side - side

        left, top = (int(z * max_valid) % max_valid for z in proportion)

        mul_dims = (left, top, side, side)
        mul_window = rio.windows.Window(*mul_dims)
        pan_window = rio.windows.Window(*(4 * z for z in mul_dims))

        mul = torch.tensor(self.mul.read(window=mul_window).astype(np.int16))
        pan = torch.tensor(self.pan.read(window=pan_window).astype(np.int16))

        if not ((self.validish(pan) and self.validish(mul))):
            raise TooManyNulls

        pan = pile(pan, factor=4)

        pack = torch.cat((pan, mul), 0)

        return pack


class Chipper(Dataset):
    def __init__(self, ard_roots: Sequence[Path], length: int):
        self.root = ard_roots
        self.length = length

        self.chips = []

        self.scattering = R2()

        for ard_root in ard_roots:
            for metadata in ard_root.glob("acquisition_collections/*.json"):
                with open(metadata) as md:
                    # print(md)
                    md = json.load(md)

                inner_json_path = [x["href"] for x in md["links"] if x["rel"] == "item"]
                inner_json_path = inner_json_path[0]

                chip_dir = (metadata.parent / Path(inner_json_path)).parent.absolute()
                chip_cid = Path(inner_json_path).stem

                try:
                    chip = Chip(chip_dir, chip_cid)
                    self.chips.append(chip)
                except:
                    pass

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        chip = index % len(self.chips)  # ?

        x, y = self.scattering[index]

        pack = self.chips[chip].read(256, (x, y))
        return pack


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

    # if failed > done * 100:  # > (length / 2) and (tried / (done+1)) > 2:
    #    raise ValueError(f"Too many bad ones in {ard}.")
