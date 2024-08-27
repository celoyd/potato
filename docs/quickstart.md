# Quickstart

This quickstart contains instructions on setting up a python environment for Ripple and then:

1. pansharpening real imagery with Ripple or
2. preparing and running a training session with Ripple

with an emphasis on getting results as quickly as possible without skipping any crucial ideas. Each section has some extra notes, so don’t be worried by this page’s length.

## 0. Preliminaries: python environment setup and device selection

### Python environment setup

The following works on a recent Ubuntu system. Readers who prefer some other way of doing things (e.g., a different operating system or a different python environment system) are entrusted with making the appropriate translations for themselves.

[Install pip](https://pip.pypa.io/en/stable/installation/) and create a new virtual environment with a reasonable python version (3.12 is what I use):

```sh

# create a new virtual environment
$ virtualenv ~/ripple -p python3

# enter it
$ ~/ripple/bin/activate

# populate it
$ pip install -r requirements.txt
```

### Device selection

### Finding a device

You must decide what hardware backend (“device”) to use. The safest choice, because it’s available on any machine, is `cpu` – the model will run on the main processor. If you have an NVIDIA GPU, `cuda` is the accelerated backend. If you’re on recent Apple hardware, `mpu` should work. Because it’s completely dependent on your hardware setup, you’re going to have to figure this out for yourself; if it gets frustrating, fall back on `cpu`. Here I will use `cuda` simply because it’s what’s best for me.

The device selection is something you remember and use as a flag, not a global configuration option. Among other things, this makes it easy to mix things – for example, I’ve often done test pansharpening runs on the CPU while leaving all the GPU’s resources focused on training.


## 1. Pansharpening quickstart

### Downloading pansharpening inputs

To find sample input data, I went to [the Maxar Open Data Program landing page](https://registry.opendata.aws/maxar-open-data/), clicked the STAC Browser link, and navigated to [here](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900.json?.language=en), which shows direct TIFF links in the Assets section. We will get the panchromatic and multispectral images.

A sidebar if you’re thinking of other inputs: Ripple expects data that looks like WorldView-2 or -3 bands in Maxar’s ARD format – in short, reflectance mapped from 0..1 to 0..10,000 in `uint16`. Anything with _approximately_ those spectral bands and where the values are _approximately_ linear reflectance in 0..10,000 is likely to _approximately_ work. But the design input is standard ARD data.

We use a {} expansion to make this slightly more legible with the long and very similar URIs:

```sh
$ curl -O "https://maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900-{pan,ms}.tif"
```

We now have the two TIFF files:

```sh
$ du -h 10300100D6740900-*
102M  10300100D6740900-ms.tif
199M  10300100D6740900-pan.tif
```

### Pansharpening

Let’s go for it:

```sh
$ python demo.py 10300100D6740900-{pan,ms}.tif weights/space_heater-gen-99.pt 10300100D6740900-ps.tiff 
```

You should see either a reasonably helpful error of some kind or a progress bar. On my 1070 (a GPU about 5 years old), it takes exactly a minute. We now have a big output file:

```sh
$ du -h 10300100D6740900-ps.tiff
484M  10300100D6740900-ps.tiff
```

This is a reasonably ordinary RGB TIFF – other than its substantial size of nearly 100 megapixels – that should be readable by most image libraries, photo editing software, and so on. (It does use zstd compression, which is still considered “the new one”, but libtiff has supported it [since 2018](http://libtiff.maptools.org/v4.0.10.html) and it’s really good, so make your own choices.)

It is a geotiff, meaning it’s in a defined projection, which can make some cautious tools complain that it has unknown tags. This should be harmless. For example, if we use the [ImageMagick](https://imagemagick.org/index.php) tool `convert` to crop out a section, we get warnings:

```sh
$ convert 10300100D6740900-ps.tiff -crop 512x512+8200+9150 Yeşilvadi.png
convert-im6.q16: Unknown field with tag 33550 (0x830e) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 33922 (0x8482) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 34735 (0x87af) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
convert-im6.q16: Unknown field with tag 34737 (0x87b1) encountered. `TIFFReadDirectory' @ warning/tiff.c/TIFFWarnings/905.
```

These are safely ignored; we get the image we should:


## Bonus geotiff tricks

Let’s do some spatial things for the readers who don’t know what that means. Suppose we want to compare Ripple’s output to Maxar’s default pansharpening. We could use QGIS or `rio` or other tools, but for this example let’s try `gdalwarp` (from the GDAL package).

First we’ll draw a box around the corner of Yeşilvadi Park that’s visible in the image on [geojson.io](https://geojson.io). Copy and paste the JSON (in the sidebar on the right) into a file named `box.json`, or use mine:

```sh
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

I now regret using this region for an example because it’s near the curve where longitude = latitude, so we see an implausible number of 37s, but that’s how it goes sometimes. We have to check the resolution of Ripple’s output (which is also the resolution of the original pan image), because Maxar’s pansharpening gets upsampled, so we’ll want to downsample it for comparison. We can do that like so:

```sh
$ gdalinfo 10300100D6740900-ps.tiff
```

There’s a lot of output but the line we want is `Pixel Size = (0.549265922249793,-0.549265922249793)`.

Now we need the bit of knowledge about Maxar’s ARD format that the official pansharpened image will have the same name as the `-pan.tif` and `-ms.tif` but with `-visual.tif`. With this, we can use `gdalwarp` to punch out the shape of our box around the corner of Yeşilvadi Park. It knows how to use the network, so we can give it an HTTPS URI instead of a filename, but we’ll need to tell it the resolution to resample to, and what resampling method to use (we’ll go with [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling), for sharpness – it’s debatable whether that’s the best choice but this is just a demo). This looks like:

```sh
gdalwarp -cutline box.json -crop_to_cutline -tr 0.5492 0.5492 https://maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/ard/37/031133102033/2022-07-20/10300100D6740900-visual.tif Maxar-Yeşilvadi.tiff

# same except source and destination:
gdalwarp -cutline box.json -crop_to_cutline -tr 0.5492 0.5492 10300100D6740900-ps.tif Ripple-Yeşilvadi.tiff
```

Now, although they started at different resolutions, we have pixel-aligned images. Maxar’s:

And Ripple’s:

Now we can zoom in and compare how they render the blue bike lanes along some of the park paths, for example. 

## Training quickstart

Training itself can be done by running `train.py`, but setting up the data for it to use is a relatively involved process, called chipping, which takes up most of this tutorial.

### Source data and directory setup

Here we will use [the Maxar Open Data Program’s imagery for the Emilia-Romagna flooding of 2023](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Emilia-Romagna-Italy-flooding-may23/collection.json?.language=en). This is a 51 gigabyte download (some of which we won’t end up using); if this is prohibitive, pick a subset of it or another data source.

We’ll need two folders, each of which will hold a lot of data. I stick big chunks of i/o-heavy data like this on external USB drives, and use symlinks to keep the paths convenient. You can skip all that and do `mkdir ards; mkdir chips`, or you can modify what I do to suit your path setup:

```sh
mkdir /media/ch/tuna/ards
ln -s /media/ch/tuna/ards .

mkdir /media/ch/uaru/chips
ln -s /media/ch/uaru/chips .

mkdir chips/italy23
```

Now we can fetch the data:

TK TK

```sh
aws s3 sync s3://maxar-opendata/events/ ards/
```

## Chipping

A _chip_ is jargon for a small image, typically a sample pulled out of a larger image.

The chipping process copies chips out of collections of images, resamples them into training pairs, and injects sampling artifacts into the input samples. see the band misalignment section – or the code, of course – for more details. Overall, the input argument to the chipper is an ARD (a set of images) and the output is chips stored as individual .pt files.

<!-- Ripple is trained on 128×128 panchromatic chips and corresponding 8×32×32 multispectral chips to produce 3×128×128 oklab color chips. The pan band goes through a space-to-depth transform so that it’s _physically_ 16×32×32 and is concatenated with the multispectral data to make a 24×32×32 package. -->

TK

### CID allow-lists

You may want to limit which CIDs (catalog IDs, or individual images) you use to make training data. For example, as described in the documentation on band misalignment, I prefer to train on cloud-free CIDs with little surface water. To support this, I spent a day early this summer subjectively evaluating every image in the Maxar Open Data Program on axes of cloudiness, surface water coverage, and complexity of landcover. These are weighted into an overall quality index, and I selected only the top-scoring CIDs.

Do this with the `--allow-list` option, which expects a path with one CID per line.

### Linking (storage load balancing)


## Training

