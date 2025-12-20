"""Pansharpen with Potato.

potato fuse -w weights.pt pan.tiff mul.tiff output.tiff

For details, see the --help and the quickstart documentation.

As befits a demo, this code uses many assumptions and shortcuts. It’s here to
illustrate one way to do things, not the only correct way.
"""

import concurrent.futures
import threading
from pathlib import Path

import click
import numpy as np
import rasterio
import torch
from rasterio import windows
from tqdm import tqdm

from potato.color import OklabTosRGB
from potato.errors import ImageDimensionError
from potato.model import Potato

# Edge length of logical (as-written) blocks to pansharpen. The range of
# reasonable values will depend sensitively on your hardware, so you should
# experiment. For clarity, we do not batch; we pansharpen one block at a time,
# even though this brings in overheads.
# block = 1280 # 768

# Extra pixels to take around each edge of input for context, creating
# physical (“reading”) blocks. In practice this could probably be as low as
# about 8. As of press time, the radius of the receptive field is 28, but
# empirically, error at that distance falls well within quantization on even
# the gnarliest real-world data.
# apron = 24


def buffer_window(w, apron):
    """Widen a window."""
    return windows.Window(
        w.col_off - apron,
        w.row_off - apron,
        w.width + apron * 2,
        w.height + apron * 2,
    )


def clip_window(w, img):
    """Fit a window into an image."""
    return windows.intersection(w, windows.Window(0, 0, img.width, img.height))


def clip_pixels(px, rw, w):
    """Take pixels fitting window rw and trim them to fit inner window w."""
    if rw.intersection(w) != w:
        raise ValueError(f"Expected {rw} and {w} to intersect, but they do not.")
    left, top = w.col_off - rw.col_off, w.row_off - rw.row_off
    right, bottom = w.width + left, w.height + top
    return px[..., top:bottom, left:right]


def quarter_window(w):
    """Convert a pan window to the physically corresponding mul window."""
    return windows.Window(w.col_off // 4, w.row_off // 4, w.width // 4, w.height // 4)


@click.command()
# Strings instead of paths for inputs so that rasterio can accept any GDAL
# path-alike (s3, zip+https, etc.). Rasterio’s open()s are the first thing in
# the function, so bad paths will fail quickly even without click’s check.
@click.argument("panpath", nargs=1, type=click.STRING, required=True)
@click.argument("mulpath", nargs=1, type=click.STRING, required=True)
@click.argument("dstpath", nargs=1, type=click.STRING, required=True)
@click.option(
    "-w",
    "--weights",
    type=click.STRING,
    # type=click.Path(), #exists=True, dir_okay=False, resolve_path=True),
    default="sessions/latest.pt",
    required=True,
    help="Checkpoint (model weights) file",
)
@click.option(
    "-b", "--block-side", default=1024, help="Pixel size of square blocks to write"
)
@click.option("-a", "--apron", default=20, help="Pixel size of block margins")
@click.option(
    "-d", "--device", default="cuda", help="Torch device (e.g., cuda, cpu, mps)"
)
@click.option(
    "-o", "--overwrite", is_flag=True, help="Destroy existing destination file"
)
@click.option("--workers", type=int, default=4, help="Worker count")
def cli(
    panpath, mulpath, dstpath, weights, block_side, apron, device, overwrite, workers
):
    """Pansharpen 🫟."""
    with rasterio.open(panpath) as panfile, rasterio.open(mulpath) as mulfile:
        if mulfile.count != 8:
            raise ImageDimensionError(
                f"Expected {mulpath} to be 8-band multispectral but got "
                f"{mulfile.count} bands. Check that this is a multispectral "
                "ARD WV-2/3 image (name should look like 10{3,4}*-ms.tiff)."
            )
        if (mulfile.width != panfile.width // 4) or (
            mulfile.height != panfile.height // 4
        ):
            # If you’re experimenting with non-ARD data, you could merely warn
            # here, to get empty blocks on edges instead of this bail-out. You
            # would also want to knock out the error handling way down in the
            # as_completed.
            raise ImageDimensionError(
                "Expected multispectral edge length to be 1/4 pan’s, but got "
                f"pan {panfile.width}×{panfile.height} versus "
                f"mul {mulfile.width}×{mulfile.height}."
            )

        profile = panfile.profile
        profile.update(
            {
                "driver": "gtiff",  # in case some joker tries VRTs
                "bigtiff": "if_safer",
                "count": 3,
                "photometric": "RGB",
                "tiled": True,
                "blockxsize": block_side,
                "blockysize": block_side,
                "compress": "zstd",
                "predictor": 2,
                "interleave": "pixel",
                "dtype": "uint16",
                "nodata": 0,
            }
        )

        # We’ll use this to track progress.
        total_pixels = profile["width"] * profile["height"]

        if not overwrite and Path(dstpath).exists():
            raise FileExistsError(
                f"{dstpath} already exists (and you didn’t ask for --overwrite)."
            )

        with rasterio.open(dstpath, "w", **profile) as dstfile:
            # We open the destination early just to be sure we can; we don’t
            # want to have loaded up the model and maybe even run data through
            # it before noticing that we have nowhere to put output.

            model = Potato().to(device)

            # This can of course raise FileNotFound, but we would have nothing
            # to add if it did, so no need to catch.
            model.load_state_dict(
                torch.load(
                    Path(weights).resolve(), map_location=device, weights_only=True
                )
            )

            model.eval()

            sRGB = OklabTosRGB().to(device)

            # Rasterio/GDAL would actually let us get away with far less
            # locking, but you’d rather have it and not need it.
            read_lock = threading.Lock()
            device_lock = threading.Lock()
            write_lock = threading.Lock()

            @torch.inference_mode()
            def process(w):
                """Pansharpen, handle window i/o, and return pixel count."""
                # We juggle several kinds of windows:
                # w: the nominal window; does not overlap with any other w.
                #   This also serves as the window to write output into.
                # buffered_window: w expanded by the apron, to avoid edge
                #   artifacts in pansharpening. Overlaps with others.
                # reading_window: buffered_window clipped to fit in the image
                #   (so we don’t try to read off the image edge). Overlaps.
                buffered_window = buffer_window(w, apron)
                reading_window = clip_window(buffered_window, panfile)

                with read_lock:
                    pan = panfile.read(window=reading_window)
                    mul = mulfile.read(window=quarter_window(reading_window))

                # Nodata == 0 per ARD spec. We use that for a shortcut here: if
                # the tile is all-null, we know its output should be too, and we
                # know that the destination TIFF defaults to nulls, so we don’t
                # have to write anything at all. We only track pixel count for
                # the progress bar.
                if np.all(pan == 0) or np.all(mul == 0):
                    clipped = clip_pixels(pan, reading_window, w)
                    return np.prod(clipped.shape)

                # If the pan band contains both nulls and non-nulls, we’ll use
                # the nulls to mask the output. This prevents a class of
                # problems such as the model accidentally spilling faint halos
                # into what should be null areas. A nice minor optimization, if
                # you could do it clearly, would be to combine this with the
                # previous step by counting nulls once. If all nulls, skip (as
                # done above). If some nulls, mask (as done below). If no nulls,
                # skip masking.
                pan_nulls = clip_pixels(pan, reading_window, w) == 0

                # Scaling factor per ARD spec.
                pan = pan.astype(np.float32) / 10_000.0
                mul = mul.astype(np.float32) / 10_000.0

                # Our i/o and ETL is all numpy-based but our business
                # (pansharpening and color conversion) is all torch-based. This
                # is the numpy->torch boundary.
                #
                # The unsqueezes here and the squeeze immemmediately after the
                # device lock are to add and remove a placebo batch, since the
                # model expects data in BCHW layout.
                pan = torch.tensor(pan).unsqueeze(0)
                mul = torch.tensor(mul).unsqueeze(0)

                # We touch the device with ansynchronous transfers outside its
                # lock. The lock is really more about avoiding OOM errors than
                # about strictly bounding i/o. Optimization-wise, the fastest
                # order of operations here, whether .pin_memory() helps or
                # hurts, etc., all depends entirely on your setup. Experiment!
                pan = pan.to(device, non_blocking=True)
                mul = mul.to(device, non_blocking=True)

                with device_lock:
                    # Pansharpening happens here.
                    sharp = model((pan, mul)).detach()
                    sharp = sRGB(sharp)

                    # If you’re on CUDA, make this non-blocking to see some cool
                    # glitch art! (In the CUDA memory model, GPU->CPU generally
                    # must block or it can, and in my experience will, corrupt.)
                    sharp = sharp.to("cpu")

                # Back through the gate to the numpy side.
                sharp = sharp.squeeze(0).numpy()
                sharp = clip_pixels(sharp, reading_window, w)

                sharp = (65_535 * sharp).clip(1, 65_535).astype(np.uint16)

                # Carry the source nulls to the destination.
                sharp[np.tile(pan_nulls, (3, 1, 1))] = 0

                with write_lock:
                    dstfile.write(sharp, window=w)

                # All done. Report out the number of pixels written.
                return sharp.shape[-2] * sharp.shape[-1]

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as tpe:
                futures = [tpe.submit(process, w) for _, w in dstfile.block_windows()]

                with tqdm(
                    total=total_pixels,
                    # Because of some choices in tqdm, we can’t easily get a
                    # string properly formatted with a space between the
                    # number and its unit. With a space before the unit, we
                    # can have the next best thing: the SI prefix acting as a
                    # suffix on the number. So tqdm wants this ugly and
                    # technically incorrect thing:
                    #   7Mpx/s
                    # And we want the correct thing:
                    #   7 Mpx/s
                    # But here we trick it into giving us the tolerable:
                    #   7M px/s
                    # I like tqdm but this chafes. BIPM and NIST have
                    # formatting guidelines for a reason >:(
                    unit=" px",
                    # smoothing=1 / 1000,
                    unit_scale=True,
                    leave=True,
                ) as prog:
                    for future in concurrent.futures.as_completed(futures):
                        # If you want individual block failures not to stop the
                        # show, just comment out this if block:
                        if (problem := future.exception(timeout=0)) is not None:
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            tpe.shutdown(wait=False, cancel_futures=True)
                            raise problem

                        block_pixel_count = future.result()
                        prog.update(block_pixel_count)


if __name__ == "__main__":
    cli()
