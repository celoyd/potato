# Quickstart

This quickstart contains instructions on setting up a python environment for Ripple and then:

1. pansharpening real imagery with Ripple or
2. preparing and running a training session with Ripple

with an emphasis on getting results as quickly as possible without skipping any crucial ideas. Each section has extra notes, so don’t be worried by this page’s length.

## Preliminaries: python environment setup and device selection

### Python environment setup

The following works on a recent Ubuntu system. Readers who prefer some other way of doing things (e.g., a different operating system or a different python environment system) are entrusted with making the appropriate translations for themselves.

[Install pip](https://pip.pypa.io/en/stable/installation/) and create a new virtual environment. This code is tested with python 3.12.

```bash

# create a new virtual environment
virtualenv ~/ripple -p python3.12

# enter it
~/ripple/bin/activate

# populate it
pip install -r requirements.txt
```

### Device selection

### Finding a device

You should decide up front which hardware [backend](https://pytorch.org/docs/stable/backends.html) (“device”) to use. The safest choice, because it’s available on any machine, is `cpu` – the model will run on the main processor. For hardware acceleration, figure out the brand of the best GPU on your machine. If it’s AMD or Nvidia, use the `cuda` backend. If it’s Apple, use `mps`. There are probably situations where this advice is misleading, but I can’t anticipate them, so you’re going to have to work it out on your own. The important points are that `cpu` is enough for testing (but not training, unless you are extremely patient), and that this documentation will use `cuda`, because it happens to be my best option, so you should replace that with yours.

The device selection is something you remember and use as a flag, not a global configuration option. Among other things, this makes it easy to mix devices. For example, I often train on the GPU with a physical batch size set to nearly max out its RAM; then if I want to test, instead of pausing training, I do it on the CPU.

## Pansharpening

Here we will pansharpen a panchromatic image and a multispectral image to make an RGB image.

### Download pansharpening inputs

To find sample input data, I went to [the Maxar Open Data Program landing page](https://registry.opendata.aws/maxar-open-data/), clicked the STAC Browser link, and navigated among may other good choices to [here](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900.json?.language=en), which shows direct TIFF links in the Assets section. We will get the panchromatic and multispectral images.

If you’re thinking of other inputs: Ripple expects data that looks like WorldView-2 or -3 bands in Maxar’s ARD format – in short, reflectance mapped from 0..1 to 0..10,000 in `uint16`. Anything with _approximately_ those spectral bands and where the values are _approximately_ linear reflectance in 0..10,000 is likely to _approximately_ work. But the design input is standard ARD data.

We use a {} expansion to make this slightly more legible with the long and very similar URIs:

```bash
curl -O "https://maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900-{pan,ms}.tif"
```

We now have the two TIFF files:

```console
user@host $ du -h 10300100D6740900-*
102M  10300100D6740900-ms.tif
199M  10300100D6740900-pan.tif
```

### Pansharpening

Let’s go for it:

```bash
python demo.py --device=cuda 10300100D6740900-{pan,ms}.tif weights/space_heater-gen-99.pt 10300100D6740900-ps.tiff 
```

You should see either some kind of reasonably helpful error or a progress bar. On my 1070 (a GPU about 5 years old), it takes exactly a minute. We now have a big output file:

```console
user@host $ du -h 10300100D6740900-ps.tiff
484M  10300100D6740900-ps.tiff
```

This is a reasonably ordinary RGB TIFF – other than its substantial size of nearly 100 megapixels – that should be readable by most image libraries, photo editing software, and so on. (It does use zstd compression, which is still considered “the new one”, but libtiff has supported it [since 2018](http://libtiff.maptools.org/v4.0.10.html) and it’s really good, so make your own choices.)

It’s also a geotiff, meaning it’s in a defined projection, which can make some cautious tools complain that it has unknown tags. This should be harmless. For example, if we use the [ImageMagick](https://imagemagick.org/index.php) tool `convert` to crop out a section, we get warnings:

```console
user@host $ convert 10300100D6740900-ps.tiff -crop 512x512+8200+9150 Yeşilvadi.png
convert-im6.q16: Unknown field with tag 33550 (0x830e) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 33922 (0x8482) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 34735 (0x87af) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 34737 (0x87b1) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
```

These are safely ignored; we get the image we should:

TK Yeşilvadi crop

If we add some contrast with `convert`, like this:

```console
$ convert 10300100D6740900-ps.tiff -crop 512x512+8200+9150 -sigmoidal-contrast 20x50% Yeşilvadi-pretty.png
```

We get a nicer image:

TK Yeşilvadi-pretty

And that’s the pansharpening demo. To familiarize yourself with the process a little more, you might try it on other images from the Maxar Open Data Program.

## Bonus geotiff tricks

Let’s do some spatial things for the readers who want to see how fun that can be. Suppose we want to compare Ripple’s output to Maxar’s default pansharpening. We could use [QGIS](https://www.qgis.org/), [`rio`](https://rasterio.readthedocs.io/en/stable/cli.html), or other tools, but for this example let’s try `gdalwarp` (from the GDAL package).

First we’ll draw a box around the corner of Yeşilvadi Park that’s visible in the image on [geojson.io](https://geojson.io). Copy and paste the JSON (in the sidebar on the right) into a file named `box.json`, or use mine:

```bash
cat << EOF > box.json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {},
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [37.418, 37.035],
        [37.422, 37.035],
        [37.422, 37.037],
        [37.418, 37.037],
        [37.418, 37.035]
      ]]
    }
  }]
}
EOF
```

I now regret using this region for an example because it’s near the curve where longitude = latitude, so we see an implausible number of 37s, but that’s how it goes sometimes. We have to check the resolution of Ripple’s output (which is also the resolution of the original pan image), because Maxar’s pansharpening gets upsampled for the ARD format, so we’ll want to downsample it for comparison. We can do that like so:

```bash
gdalinfo 10300100D6740900-ps.tiff
```

There’s a lot of output but the line we want is `Pixel Size = (0.549265922249793,-0.549265922249793)`.

Now we need the bit of knowledge about Maxar’s ARD format that the official pansharpened image will have the same name as the `-pan.tif` and `-ms.tif` but with `-visual.tif`. With this, we can use `gdalwarp` to punch out the shape of our box around the corner of Yeşilvadi Park. The tooling knows how to use the network, so we can give it an HTTPS URI instead of a filename, but we’ll need to tell it the resolution to resample to, and what resampling method to use (we’ll go with [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling), for sharpness – it’s debatable whether that’s the best choice but this is just a demo). This looks like:

```bash
gdalwarp -cutline box.json -crop_to_cutline -r Lanczos -tr 0.5493 0.5493 https://maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900-visual.tif Maxar-Yeşilvadi.tiff

# same except source and destination:
gdalwarp -cutline box.json -crop_to_cutline -r Lanczos -tr 0.5493 0.5493 10300100D6740900-ps.tif Ripple-Yeşilvadi.tiff
```

Now, although they started at different resolutions, we have pixel-aligned images. Maxar’s:

TK

And Ripple’s:

TK

Now we can zoom in and compare how they render the blue bike lanes along some of the park paths, for example. 

## Training quickstart

Training itself is done with `train.py`, but setting up the data for it to use is a relatively involved process, called chipping, which takes up most of this tutorial.

### Source data and directory setup

Here we will use [the Maxar Open Data Program’s imagery for the Emilia-Romagna flooding of 2023](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Emilia-Romagna-Italy-flooding-may23/collection.json?.language=en). This is a 51 gigabyte download (some of which we won’t end up using); if this is prohibitive, pick a subset of it or another data source. Welcome to remote sensing and its inconvenient data.

We’ll need two folders, each of which will hold a lot of data. I do i/o-heavy tasks like this on low-end external solid-state drives, and use symlinks to keep the paths convenient. You can skip all that and just do:


```bash
mkdir ards
mkdir chips
```

Or you take what I do and modify it to suit your path setup:

```bash
mkdir /media/ch/tuna/ards
ln -s /media/ch/tuna/ards .

mkdir /media/ch/uaru/chips
ln -s /media/ch/uaru/chips .

mkdir ards/italy23
mkdir chips/italy23
```

Now we can fetch the data:

```bash
aws s3 sync s3://maxar-opendata/events/Emilia-Romagna-Italy-flooding-may23/ard/ ards/italy23
```

## Chipping

A _chip_ is jargon for a small image, often a sample pulled out of a larger image.

The chipping process copies chips out of collections of images and resamples them into training pairs. The main arguments to the chipper are the path of an ARD (the tiled delivery package for an image) and the path of a directory in which to put training pairs of chips (as [`.pt` files](https://pytorch.org/docs/stable/generated/torch.save.html)).

We can chip our data like this:

```bash
python chipper.py --log italy23.log make-chips --ard-dir ards/italy23 --chip-dir chips/italy23 -n 1024
```

### CID allow-lists

You may want to limit which CIDs (catalog IDs, or individual images) you use to make training data. For example, as described in the documentation on band misalignment, I prefer to train on cloud-free CIDs with little surface water. To support this, I spent a day early this summer subjectively evaluating every image in the Maxar Open Data Program on axes of cloudiness, surface water coverage, and complexity of landcover. These are weighted into an overall quality index, and I selected only the top-scoring CIDs. There are plenty of other things you might want to do with an allow-list; for example, you might want to select individual scenes for train/test splits.

The `--allow-list` option expects a path that’s a plaintext file with one CID per line. There’s no extra parsing; it won’t recognize a regex, for example.

### Restarting

For some development and training strategies, it’s handy to chip a few hundred or a few thousand at a time. The chip selection within source images is diffused with [a deterministic low-discrepancy sequence](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) that (to simplify) does a reasonable job of maximizing distance between chip centerpoints for any given number of chips. It’s thus slightly better at information-per-chip efficiency than random selection would be. However, since it’s deterministic, if we ran the chipper twice with the same arguments, we’d get the same chips. So there’s an `-s` or `--starting-from` argument that picks up from some point. Chip numbering is 0-based, so you would run with, say, `--count 1024` and then pick up with `--starting-from 1024`.

### Linking (storage load balancing)

If you have more than one folder of chips, it’s nice to be able to mix them into a single folder. The `link-chips` chipper command does this by taking a set of source directories and making symlinks to their chips in the destination directory. The links are renamed (with the same _integer_.pt naming scheme) and deterministically shuffled so that a `DataLoader` gets a mixed sample even if it only reads the first _n_ chips. This is convenient to change the mix of source data for training, and to load-balance chip reads across multiple storage devices.

## Training

To start training from the pretrained weights shipped in the repo, we can use:

TK fix code + change training script

```bash
python train.py --load-epoch 63 --lr 1e-4 --chips chips7 --epoch-length 10240 --epochs 100 --test-chips chips7 --workers 5

python train.py --session spaceheater --chips chips --test-chips chips --epoch-length 1024 --epochs 10 --lr 1e-4
```

To train from scratch:

```bash
python train.py --session demo --chips chips --test-chips chips --epoch-length 1024 --epochs 10 --lr 1e-4
```

Using the same chips for training and testing is a bad idea for serious training sessions (because it makes the test loss uninformative); you will want to set up a separate testing chip directory.