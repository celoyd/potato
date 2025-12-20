"""Make chips, given an ARD directory.

This is how we create training data. Each sample is a pair of items:

  x: (pan band data, mul band data)
  y: (oklab image of mul data)

where the most important features are that:

- x’s pan band is the same resolution as y,
- x’s mul bands are 1/4 the (linear) resolution of y, and
- the output is stored in .pt (pytorch pickle) files.

For futher details, I trust that the reader would prefer to read the code than
to read a 300-line comment full of ascii diagrams and then have to check it
against the code.

Many parts of this depend in great detail on the ARD format. Find its docs here:
https://developers.maxar.com/docs/specifications/ard-user-guide/about-ard

But for anyone intending to jump in and make changes, a few design notes that
should help make sense of things:

- There is no explicit representation of the idea of an ARD per se; a Chipper
  transparently merges any number of them, and the next lower level of hierarchy
  is a Tile, which represents a pan and mul TIFF pair.

- The CLI is only set up to instantiate a Chipper with a single ARD, so *in
  practice* the previous point is wrong and a Chipper is really one ARD, but
  don’t think about it like that.

- Pixel data is immediately converted to torch, and numpy is never actually seen
  (although of course rasterio is using it).

- This process is slower than it needs to be because we open, read from, and
  close each Tile (pair of TIFFs) sequentially, wantonly spreading out i/o
  almost exactly as if we were trying not to benefit from caching. The crux of
  this is that unpredictable-ish null areas mean we don’t know up front how many
  chips would be read from a given tile in a fair sample. There are ways to
  tackle that, but they’d substantially increase SLOC, and an overarching
  assumption here is that chipping is a few-time cost where speedups are not
  vital.

- We log with f-strings. This is often considered bad style because it always
  builds the string, which can be wasteful, especially for large interpolants.
  Here we know that almost all constructed strings will typically be written,
  that we are dealing with nothing bulkier than some long paths, and that we
  have various actually expensive operations (i/o, image resizing, etc.) that
  massively outweigh string-building in the optimization stakes. So we don’t
  worry about it.
"""

import json
import logging
from pathlib import Path

import click
import rasterio
import torch
from pyproj import Transformer
from torch.utils.data import Dataset
from tqdm import tqdm

from potato.color import BandsToOklab
from potato.errors import MissingImageError, OldSatelliteError, TooManyNullsError
from potato.util import Chip

# We actually never want this on, but since we’re doing a lot of opens here,
# it’s worth bothering to turn it off.
rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True)


class R2:
    """Generate a 2D low-discrepancy sequence.

    See:
    https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    def __init__(self, start=0.5):
        """Set up constants.

        Please see the web link in the class docstring for details.
        """
        ρ = 1.324717957244746
        self.a1 = 1 / ρ
        self.a2 = 1 / (ρ**2)

        self.position = start

    def __iter__(self):
        """Be an iterator."""
        return self

    def __next__(self):
        """Step."""
        self.position += 1
        return self[self.position - 1]

    def __getitem__(self, idx):
        """Return (x: float, y: float)."""
        return (self.a1 * idx) % 1, (self.a2 * idx) % 1


class Tile(Dataset):
    """Represent an ARD tile.

    This is intended to be used within a Chipper object (see below).
    """

    def __init__(self, basepath: Path, cid: str):
        """Pull in tile metadata."""
        self.root_dir = basepath
        self.cid = cid  # catalog ID

        with open(basepath / f"{cid}.json", encoding="utf-8") as jsonf:
            # json.load doesn’t know what to do with a Path.
            self.json = json.load(jsonf)

        metadata = self.json["properties"]

        if metadata["platform"] not in (
            "worldview-02",
            "worldview-03",
            "WV02",
            "WV03",
        ):
            # Hello to me from the future who wondered why a Legion ARD didn’t
            # chip and remembered that this check wasn’t future- safe. (In my
            # defense, I don’t know what name will be used here for Legion
            # platforms, and having a deny list would be extremely risky.)
            raise OldSatelliteError

        self.pan_path = self.root_dir / f"{self.cid}-pan.tif"
        self.mul_path = self.root_dir / f"{self.cid}-ms.tif"

        if (not self.pan_path.exists()) or (not self.mul_path.exists()):
            raise MissingImageError

        with rasterio.open(self.mul_path) as mul:
            self.mul_side = mul.profile["width"]
            self.mul_transform = mul.profile["transform"]
            self.mul_crs = mul.profile["crs"]

        self.to_latlon = Transformer.from_crs(self.mul_crs, 4326)

    def coords(self, mul_window):
        """Given a window of a georeferenced image, give its lon/lat."""
        center = (
            mul_window.col_off + mul_window.width / 2,
            mul_window.row_off + mul_window.height / 2,
        )
        xy = rasterio.transform.xy(self.mul_transform, cols=center[0], rows=center[1])
        lalo = self.to_latlon.transform(*xy)
        return {"lat": round(lalo[0], 5), "lon": round(lalo[1], 5)}

    def read(self, side, proportion):
        """Return a chip.

        This does the physical reading, given a side length and a relative
        position (i.e., x and y both in 0..1) in the image. We open and close
        the files with each call, which is a bit unfortunate but unavoidable; a
        large ARD can easily contains more TIFFs than a sensible OS will let you
        have open file handles. (And we can’t simply extract what we need from
        each tile in turn, with a depth-first approach, because we don’t know
        what proportion of each tile is valid up front. See the last design note
        in the head comment.)
        """
        with (
            rasterio.open(self.pan_path) as pan_file,
            rasterio.open(self.mul_path) as mul_file,
        ):
            # A window has extent, so starting a window at position (1, 1) in
            # the image would make it reach outside the footprint, and output
            # chips must all be the same size. So the relative positions don’t
            # actually address the whole image space: they address the subset of
            # the image where the top left of a window of given side length that
            # is fully in the image can lie.

            # ARD images must have the same width and height, and our chips are
            # also square, so we can use the same arithmetic for x and y dims.
            # We call x or y “z”.
            safe_edge = self.mul_side - side

            left, top = (int(z * safe_edge) % safe_edge for z in proportion)

            mul_dims = (left, top, side, side)
            mul_window = rasterio.windows.Window(*mul_dims)
            pan_window = rasterio.windows.Window(*(4 * z for z in mul_dims))

            # Pixels are converted to tensor here, but scaling happens in Chip.
            pan = torch.tensor(pan_file.read(window=pan_window))
            mul = torch.tensor(mul_file.read(window=mul_window))

            chip = Chip(pan, mul)

            # Some notes for the log:
            path = self.mul_path
            coords = self.coords(mul_window)

            return chip, path, coords

    def __getitem__(self, _):
        """Technically satisfy the qualities of a Dataset."""
        pass


class Chipper(Dataset):
    """Represent a set of ARDs as a data source.

    An overview of our concepts here:

    - An ARD is (for our purposes) a set of CIDs and some metadata.
    - A CID is (f.o.p.) a set of image tiles.
    - A tile is (f.o.p.) a panchromatic image and a multispectral image. See the
      Tile object, above.
    - These pan/mul image pairs are our source of chips (see util.Chip), which
      are small sub-images that we pull out of them in a dispersed pattern
      implemented in R2, above.

    Chipper represents one or more ARDs for chipping purposes. It uses the ARD
    metadata and a given list of allowed CIDs to filter down to eligible CIDs,
    then wraps each of their tiles in a Tile. When its __getitem__ is called, it
    (deterministically) picks a Tile from its pool, calculates a good offset
    within it to pull a chip out of, calls the Tile’s read() to get the chip,
    and returns it with some

    We take a list of ARD roots and traverse them to collect all their image
    tiles that belong to CIDs in allow_list.

    All tiles in the ARD(s) and whose CIDS are listed in allow_cids are turned
    into Tile objects and put in a single pool. There is no concept of an
    individual ARD in the code except as something to traverse to find tiles.

    Operations specific to a file or matching files are above in Tile. This is
    only a collection type that knows how to diffuse reads across its members.
    """

    def __init__(self, ard_roots, length, allow_cids, starting_from, minimum_valid):
        """Scan the ARDs and assemble their tiles."""
        self.length = length
        self.starting_from = starting_from

        self.allow_cids = allow_cids

        # Not strictly necessary, but lets us skip dupplicate warnings.
        self.skipped_cids = []

        self.tiles = []

        self.minimum_valid = minimum_valid

        self.scattering = R2()

        self.oklab = BandsToOklab()

        for ard_root in ard_roots:
            # Step through every tile in the given ARDs and, for each
            # one that meets the conditions, wrap it in a Tile and add it
            # to self.tiles.

            for metadata in ard_root.glob("acquisition_collections/*.json"):
                with open(metadata, encoding="utf-8") as md:
                    md = json.load(md)

                inner_json_paths = [
                    x["href"] for x in md["links"] if x["rel"] == "item"
                ]
                for inner_json_path in inner_json_paths:
                    chip_dir = (
                        metadata.parent / Path(inner_json_path)
                    ).parent.resolve()

                    chip_cid = Path(inner_json_path).stem
                    if chip_cid not in self.skipped_cids:
                        if (self.allow_cids != []) and (
                            chip_cid not in self.allow_cids
                        ):
                            logging.warning(
                                f"Skipping CID {chip_cid}: not in the allow list"
                            )
                            self.skipped_cids.append(chip_cid)
                            continue

                        try:
                            tile = Tile(chip_dir, chip_cid)  # <- the actual work
                            self.tiles.append(tile)
                        except OldSatelliteError:
                            logging.warning(
                                f"Skipping CID {chip_cid}: satellite is not WV-2 or -3"
                            )
                            self.skipped_cids.append(chip_cid)
                        except MissingImageError:
                            logging.warning(
                                f"Skipping CID {chip_cid} in dir {chip_dir}: "
                                "one of its files is missing"
                            )
                            self.skipped_cids.append(chip_cid)

        if len(self.tiles) == 0:
            logging.error(f"No (allow-listed) tiles found in {ard_roots}")
            raise ValueError(f"No (allow-listed) tiles found in {ard_roots}")

    def __len__(self):
        """Return a length as demanded by Dataset."""
        return self.length

    def __getitem__(self, index):
        """Return the chip, its path, and its coodinates."""
        tile = (index + self.starting_from) % len(self.tiles)
        position = (index + self.starting_from) // len(self.tiles)

        x_off, y_off = self.scattering[position]

        chip, path, coords = self.tiles[tile].read(1024, (x_off, y_off))

        return chip, path, coords


@click.command()
@click.option(
    "--ard-dir",
    "-a",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source ARD directory",
)
@click.option(
    "--chip-dir",
    "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Destination for chips",
)
@click.option("--count", "-n", required=True, type=int, help="Number of chips to make")
@click.option(
    "--allow-cids",
    "-y",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    help="File with one CID per line; if given, only those CIDs will be used",
)
@click.option("--minimum-valid", "-m", type=float, default=0.75)
@click.option(
    "--starting-from",
    "-s",
    type=int,
    default=0,
    help="Counter number to start from. Depreceated: see docs.",
)
@click.option(
    "--log",
    "-l",
    default="/dev/stdout",
    type=click.Path(exists=False, dir_okay=False, allow_dash=True, path_type=Path),
)
def cli(ard_dir, chip_dir, count, allow_cids, starting_from, minimum_valid, log):
    """Extract training chips from ARD files."""
    logging.basicConfig(
        filename=log,
        format="%(asctime)s (%(levelname)s): %(message)s",
        encoding="utf-8",
        level=logging.INFO,
    )

    if allow_cids:
        with open(allow_cids, encoding="utf-8") as cids:
            allow_cids = [x.strip() for x in cids]
    else:
        allow_cids = []

    chipper = Chipper([ard_dir], count, allow_cids, starting_from, minimum_valid)
    completed = 0
    failed = 0
    n = starting_from

    pbar = tqdm(total=count)
    while completed < count:
        try:
            chip, path, coords = chipper[n]
            dst = chip_dir / f"{starting_from + completed}.pt"
            chip.write_pt(dst)
            logging.info(f"{dst} at {coords} from {path}")
            completed += 1
            pbar.update(1)
        except TooManyNullsError:
            failed += 1
        n += 1
    pbar.close()

    logging.info(f"Fishined {completed} chips. There were {failed} failures.")


if __name__ == "__main__":
    cli()
