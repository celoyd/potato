# Quickstart

This quickstart will take you on the shortest path to pansharpen imagery with Potato. It assumes that you’re comfortable installing and using a python package on the command line, including fixing minor snags and cleaning up outright failures as they arise. It requires downloading several gigabytes of libraries and test data, so don’t save this to do offline. The examples use Ubuntu 24 with [uv](https://github.com/astral-sh/uv) and `zsh`; if you do things in other ways (e.g., vanilla `pip`/`virtualenv` on macOS with `bash`, or `conda` on Alpine with `fish`), you are entrusted to translate the examples.

## Preliminaries

### Installation

_These instructions duplicate the ones in [the CLI reference](cli.md)_.

Here you will install a virtual environment with Potato the library, its dependencies, and `potato` the command-line tool.

These instructions use the [uv](https://docs.astral.sh/uv/) management tool, which I warmly recommend. However, all that’s important is that you end up in a virtual environment with the dependencies named in `pyproject.toml`.

With `uv`, do this in the top directory of this repo:

```bash
uv venv
uv pip install -e .
source .venv/bin/activate
```

The standard venv installation depends on how your system python, if any, is set up (_this is a problem `uv` can avoid!_), but something like this is likely to work:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Unless you’re in the kind of hurry where you can’t spare 5 seconds, run the tests. If you’re in the virtual environment (you see `(potato)` _with parentheses_ somewhere left of the prompt):

```bash
pytest
```

Otherwise:

```bash
uv run pytest
```

If there are failures, make a reasonable effort to work out why, and if it’s not your fault, file a bug.

### Selecting a device

Decide which hardware [backend](https://pytorch.org/docs/stable/backends.html) to run Potato on. This will be something you remember and use as a flag to scripts. (It’s not a global configuration option because it’s convenient to use different backends at different times. For example, you might use `cpu` to test things while training is using all the `cuda` RAM.)

The safest but slowest choice is `cpu`: the model will use the main processor. Everything will work correctly, but it will be relatively slow. If you have a GPU or GPU-equivalent, figure out its brand. If it’s AMD or Nvidia, use the `cuda` backend; if it’s Apple (with an M series chip), use `mps`. For anything else, you’re on your own. To select a specific device, like the first of two Nvidia GPUs, use syntax like `cuda:0`.

This quickstart will use `cuda` because that’s my best option. Replace it with yours wherever you see it.

## Pansharpening

Here you will download a panchromatic image and a multispectral image and combine them with Potato to make an RGB image.

### Download pansharpening inputs

To find sample input data, I went to [the Maxar Open Data Program landing page](https://registry.opendata.aws/maxar-open-data/), clicked the STAC Browser link, and navigated among many other good choices to [here](https://stacindex.org/catalogs/maxar-open-data-catalog-ard-format#/item/aCtvMLE92XskBWQbvt9J3vsA7EgRSdJ15SD3LJA6JS6f5SAxD/4dKymbGqAdScTepcQsBdACXvFvWipUToM2xs4gbaKtavizTRjBNSoaWKYELhidUbN2hF3DhyD1jwqeqhcZs1BuxpzChBDkqEB43meyRhi4D3YSy/5utsGWgkk8Rmyft4fhmmxhnoUPK96JjztCbzDmqpoMsS34t1fsuwh3R85msyGVfHd1fYvjV5yChWzjUf6mvpqnzhsFxT1Ws3iAcu?si=2#13/-1.288302/36.820695), which shows direct TIFF links in the Assets section. Get the panchromatic and multispectral images. (You could actually do everything straight over the net, with HTTPS URIs in place of filenames, but that would be showing off, and also slow.)

<details>
  <summary>Sidebar on input data</summary>

Potato expects input that looks like WorldView-2 or -3 data in [Maxar’s ARD format](https://ard.maxar.com/docs/). The images are physically pixel-aligned at a factor of 4 (not merely both georeferenced), the multispectral image has 8 bands as documented for the WV-{2,3} sensor, and the DNs are (modeled) reflectance, mapped from 0..1 to 1..10,000 in `uint16`. Anything with roughly those characteristics may roughly work. For example, I’ve sent TOA radiance through, scaled by an empirical constant, and it was, well, it was fine.
</details>

Download the TIFFs in the browser and move them to your working directory, or on the CLI like this:

```bash
curl -O "https://maxar-opendata.s3.amazonaws.com/events/Kenya-Flooding-May24/ard/37/211111023311/2023-11-30/104001008E063C00-{pan,ms}.tif"
```

Now you have the two TIFF files:

```console
user@host:~/potato$ du -h 104001008E063C00-*
295M  104001008E063C00-ms.tif
561M  104001008E063C00-pan.tif
```

### Pansharpening

Go for it (remembering to set your device):

```bash
potato fuse --device=cuda 104001008E063C00-{pan,ms}.tif 104001008E063C00-ps.tiff
```

Expect either some kind of reasonably helpful error or a progress bar. The M units in the progress bar are megapixels. On my 1070 (a GPU released in 2016), it takes a little under 30 seconds; on my CPU alone (`--device=cpu`), it takes 4–5 minutes. A fan may turn on. All being well, you now have a big, pansharpened output file:

```console
user@host:~/potato$ du -h 104001008E063C00-ps.tiff
1.4G  104001008E063C00-ps.tiff
```

This is a 16-bit lossless RGB TIFF, and should be readable by ordinary image libraries, photo editing software, and so on. (The most likely problem is that it uses the zstd compression mode, which is still “the new one”, but libtiff has supported it [since 2018](http://libtiff.maptools.org/v4.0.10.html). If it causes problems, search for `zstd` in `src/potato/scripts/fuse.py` and replace it with, say, `deflate` or `LZW`.)

Your first impression of the image will likely be that it’s low-contrast. This is intended behavior. Low contrast makes it easy to represent extremes – the huge range of brightness and color that can appear on Earth – without clipping. To see more contrast, turn the contrast up. An image editor’s “auto contrast”, “auto levels”, or “normalize” tool should work; a “curves” tool is also a good choice.

The image is a geoTIFF, meaning it’s in a defined projection. This gives it entry to the universe of geospatial operations. It can be reprojected, for example, or matched to other geoTIFFs (such as the default pansharpening, at the same web address but ending in `-visual.tif`), and mixed with other geographical data in tools like [QGIS](https://qgis.org/):

![The image overlaid on OSM](images/Wakulima/overlay.png)

_Pansharpened image translucently overlaid on [OSM](https://www.openstreetmap.org) (© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)) in QGIS. This is not good cartography, but it works; the image knows where it is at all times._

A cost of being a geoTIFF is that non-geospatial tools may warn about the geotags. For example, if you use [ImageMagick](https://imagemagick.org/index.php) to crop into Wakulima Market, the famous produce hub, you get warnings:

```console
user@host:~/potato$ magick 104001008E063C00-ps.tiff -crop 768x512+12000+8000 Wakulima-market.png
magick: Unknown field with tag 33550 (0x830e) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/928.
magick: Unknown field with tag 33922 (0x8482) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/928.
magick: Unknown field with tag 34735 (0x87af) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/928.
magick: Unknown field with tag 34737 (0x87b1) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/928.
magick: Unknown field with tag 42113 (0xa481) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/928.
```

These are safely ignored, and you get the image you should:

![Wakulima market, looking under-contrasty](images/Wakulima/Wakulima-market.jpeg)

<!-- Even if this were a TIFF, it would not be a geoTIFF. ImageMagick, like most tools, will correctly decline to copy tags that it doesn’t know how to carry. (If it had resized or rotated the image, for example, then the input geotags would be incorrect for the output, so always copying them would be wrong.) -->

You might add some contrast with `convert`, for example with a channelwise auto-level:

```bash
magick Wakulima-market.png -channel R,G,B -normalize +channel Wakulima-market-contrast.png
```

And get a nicer image, resolving even some of the colors of the fruits and vegetables themselves:

![Wakulima market, looking nicer](images/Wakulima/Wakulima-market-normed.jpeg)

And that’s the pansharpening demo.

To familiarize yourself with the process a little more, you might try it on other images from the Maxar Open Data Program, and you might try postprocessing the outputs a little further (with some gamma, for example). Enjoy!
