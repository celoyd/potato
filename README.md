# Potato: a pansharpening model

:potato:

## About Potato

Potato is a research pansharpening model with a training setup and documentation. Its goals are to:

1. Raise interest in pansharpening.
2. Raise expectations of the quality per complexity of pansharpening methods.
3. Introduce some transferable techniques to improve pansharpening.

The model is a convolutional neural network that is small (~50k parameters) and of simple design. Potato’s training data is from the Maxar Open Data Program, and like that imagery, it is licensed CC BY-NC. It can be applied (noncommercially) in its present form – see [the quickstart](docs/quickstart.md) – but it is not tested, reviewed, or optimized for general-purpose use; it’s a research project. It aims to be a good influence on the next generation of applied pansharpening models, not to lead them.

## About pansharpening

_This explanation will skip many details, even some that specifically motivate this project. The only goal is to show the core idea of pansharpening to people seeing it for the first time._

A very high resolution optical satellite collects an image in two parts. It gets a grayscale image at full resolution:

![Monochrome image of a mountain village at a river confluence](docs/images/Chungthang/pan.png)

_The samples images here are of Chungthang, Sikkim, from Maxar collect 10300100CE8D0400 on 2022-03-07. Like all images shown in this repo except otherwise credited, they are [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en) from the [Maxar Open Data Program](https://registry.opendata.aws/maxar-open-data/)._

This is called the panchromatic or pan band, and it works like black and white film, or like applying a black and white filter to a photo. A band is a range of the spectrum, and pan-chromatic means all-color – this band collects the range of all visible light.

At the same time, the satellite collects a full-color image, with red, green, and blue bands, at lower resolution:

![Upsampled color image of the same scene](docs/images/Chungthang/rgb-upsampled.png)

So we have one image that is sharp, and one that is full-color. We want to merge them into a combined version that is both sharp _and_ full-color at the same time. We can think of this as using the pan image to sharpen the full-color image, so the process is called **pansharpening**.

Like compression or instant messaging, pansharpening isn’t any one algorithm or technique; it’s a category defined only by the problem being solved.

## Comparisons

### Comparison with Maxar’s pansharpening (Chungthang)

Two examples of pansharpening these pan and multispectral images of Chungthang. (For the connoisseurs, this is CID 10300100CE8D0400, 2022-03-07.)

First, Maxar’s pansharpening, provided with the raw data:

![Maxar output](docs/images/Chungthang/mx-nice.png)

Second, Potato’s output for the same data:

![Potato output](docs/images/Chungthang/potato.png)

I’ve adjusted the brightness and contrast of both images to make them roughly match. (Image-wide adjustments that you can make with a [curves tool](https://en.wikipedia.org/wiki/Curve_(tonality)) are independent of pansharpening – we’re looking at local details here, not overall tones.)

To see differences, try opening each in its own tab and flipping between them. There is no way to make a completely fair comparison, for example because the Maxar image has been lossily compressed and resampled back to nominal resolution. However, several of what I consider its shortcomings – the dark water edges, sensitivity to grainy noise, difficulty rendering deeper blue hues – are not plausibly artifacts of those processing steps. Google Earth [also uses this collect](https://earth.google.com/web/@27.60326593,88.64660971,1610.66494177a,1045.83623347d,35y,0h,0t,0r/data=ChYqEAgBEgoyMDIyLTAzLTA3GAFCAggBQgIIAEoNCP___________wEQAA), and although their color treatment is much darker than Maxar’s, the general look of the pansharpening is similar.

### Comparison with Google Earth’s pansharpening (Quito)

Here is a comparison of a different image chip with a different set of visualization choices. On the left we have a small part of Quito [as seen on Google Earth](https://earth.google.com/web/search/Quito,+Ecuador/@-0.08986503,-78.44810289,2628.27205598a,346.32203057d,35y,0h,0t,0r/data=Cj4iJgokCSF-pJOHnDtAEQdki5yumDtAGRLQGY0TKlZAIXMmJJ-4KFZAKhAIARIKMjAyMi0wMS0wOBgBQgIIAUICCABKDQj___________8BEAA) (from 10300100CC8F7900, 2022-01-08; image credit as given within the screenshot; n.b., the link is to the web version but the screenshot is from the desktop version). On the right is Potato’s output, resized and lightly color-adjusted to roughly match the screenshot.

![A side-by side comparison of images of a mixed neighborhood with a plaza park, a bus stop, and some large semi-industrial buildings in Quito](docs/images/Quito/both.png)

The rescaling is unflattering to the fine detail in both, but of note is the blocky artifacting around some of the [brick-through-grass](https://earth.google.com/web/search/Quito,+Ecuador/@-0.08986531,-78.44806574,2632.5715332a,0d,60y,270.41779775h,79.45498836t,0r/data=CiwiJgokCSF-pJOHnDtAEQdki5yumDtAGRLQGY0TKlZAIXMmJJ-4KFZAQgIIASIaChZIX0pETEI2d1VfcndHcS1fQVJTZklBEAJCAggASg0I____________ARAA) textures in the playground/plaza west of the bus stop. This is a relatively rare artifact but a revealing one: it shows a failure state of local adaptation–based pansharpening algorithms, which is confused by certain patterns. (The Maxar version of this image also fails in this area, but with blurriness instead of blockiness.) Potato is certainly not perfect, but it paints a picture that reasonably matches on-the-ground views.

### Comparison with ZS-Pan

Here we compare Ripple’s output to the output of [ZS-Pan](https://github.com/coder-qicao/ZS-Pan), a recent research algorithm which is [published](https://www.sciencedirect.com/science/article/abs/pii/S1566253523003172) and claims to exceed recent state-of-the-art methods. (The choice of ZS-Pan is not for any specifics of its design, but somewhat arbitrarily because (a) it makes a peer-reviewed claim to high quality and (b) it’s well enough implemented and documented that it’s easy to run.)

<details><summary>Process for generating ZS-Pan comparison</summary>

Other than the python libraries, this process will use the ImageMagic CLI tools and [uv](https://github.com/astral-sh/uv). For both there are many alternatives (Photoshop, numpy, …; pip, conda, …), and translations should be straightforward.

```bash
cd ~/Documents
git clone https://github.com/coder-qicao/ZS-Pan.git
cd ZS-Pan
uv venv
source .venv/bin/activate
uv pip install torch numpy scipy h5py torchsummary rasterio
```

Download [the WV-3 test data](https://drive.google.com/drive/folders/1x3b3ERBXKGXncTRL3gKcidV5BBdG2QjC) (having followed ZS-Pan’s link to [PanCollection](https://github.com/liangjiandeng/PanCollection).) Assuming it’s in `~/Downloads`:

```sh
mkdir dataset/wv3
ln -s ~/Downloads/test_wv3_OrigScale_multiExm1.h5 dataset/wv3/train.h5

python test.py --satellite wv3/ --name 19
```

The output is in linear radiance and is all-band, so we’ll extract it in a python session; then, still in the session, we’ll set the data up to be used by Potato. We use `rasterio` because it reliably produces multiband TIFFs, while something like `imageio` has a terser API but takes more work to produce correct behavior if the user happens to have certain plugins installed, &c. (This could also work as a script.)


```python
import scipy.io as sio
import rasterio as rio
import numpy as np
import h5py

result = sio.loadmat("result/wv3/19.mat")
r, g, b = (result["I_SR"].clip(0, 65535).astype('uint16')[..., b] for b in (4, 2, 1))

with rio.open("19-zs.tiff", "w", driver="gtiff", width=512, height=512, count=3, photometric="RGB", dtype="uint16") as zs_result:
    zs_result.write(np.stack([r, g, b]))

inputs = h5py.File('dataset/wv3/train.h5', 'r')

with rio.open("19-pan.tiff", "w", driver="gtiff", width=512, height=512, count=1, dtype="uint16") as pan:
    pan.write(np.stack((inputs['pan'][19] * 4).clip(0, 10_000).astype('uint16')))

with rio.open("19-mul.tiff", "w", driver="gtiff", width=128, height=128, count=8, dtype="uint16") as mul:
    mul.write(np.stack((inputs['lms'][19] * 4).clip(0, 10_000).astype('uint16')))
```

Notice the arbitrary `* 4` scaling, which _very_ roughly approximates turning radiance into reflectance. This shoddy conversion theoretically puts Potato at a disadvantage since the data statistics will be slightly out of its training distribution.

Now we can run Potato (remembering to adjust paths if necessary):

```sh
cd ~/Documents/potato

python demo.py -d cuda ~/Documents/ZS-Pan/19-{pan,mul}.tiff -w sessions/yukon-synths/147-gen.pt 19-potato.tiff
```

Now we have an RGB TIFF from each model, but they are scaled differently; ZS-Pan’s would look virtually all black if opened in an image viewer. (This is correct behavior for a radiance→radiance model, but it’s not what we want here.) We will bring it into a visible range with a channelwise `-normalize`, roughly equivalent to an auto-leveling operation in an image editor. We will also give it some modest gamma in order to be directly comparable to Potato’s output (which we will give the same adjustment, other than the gamma). This color mix is not particularly good-looking for either image but it is at least _fair_. Edit paths again as needed.

```sh
convert -channel R,G,B -normalize +channel -gamma 1.25 ~/Documents/ZS-Pan/19-zs.tiff zs-demo.jpeg

convert -channel R,G,B -normalize +channel 19-potato.tiff potato-demo.jpeg
```
</details>

ZS-Pan’s output, following its documentation (adjusted for display with the process above):

![ZS-Pan output](docs/images/ZS-Pan/zs-demo.jpeg)

_Image source data: Maxar via PanCollection._

And Potato’s, on the same data (also as explained above):

![Potato output](docs/images/ZS-Pan/potato-demo.jpeg)

_Image source data: Maxar via PanCollection._

### Comparison with Maxar’s pansharpening (Port NOLA)

Here is another 1:1 of Potato’s output with Maxar’s, this time with some more abstract comparison tools. We will be looking at the west bank of the Mississippi at river mile 100 – the West Riverside area of the Port of New Orleans (1040010073665D00, 2022-03-24), which we all know by its nickname: the [Coffee Gateway to the USA](https://stir-tea-coffee.com/features/coffee-gateway-to-the-usa/). Coffee Drive is in the top of the image, and the Silocaf building is the tall one in the northeast. The images are slightly shrunk for display in this readme; please click through for full size.

<details>
<summary>Processing details</summary>

To vary techniques in each comparison, this is in QGIS.

The images are from `s3://dg-opendata/louisiana-tornadoes22/post/15/120202312113/2022-03-24/10400100A0BAD600-*`. The Maxar image (being 8-bit) is presented without stretching; Potato’s output (16-bit) is auto-stretched by QGIS, and I’ve adjusted the parameters to make it basically resemble Maxar’s image: min, max for R, G, and B are: 12k, 46k; 10k, 44k; 10k, 42k. Both images’ sampling is cubic (4×4), and the scale is 1:2500; they are slightly undersampled (45 cm image pixels displayed at 51.5 cm per screen pixel). QGIS keeps them in UTM zone 15N.

I screenshotted them and cropped with `convert`. In an effort at some ill-defined notion of fairness, the Maxar image is left as a PNG, while the Potato image is compressed to a JPEG of roughly 2.1 bits per pixel, which I measured as a typical compression level for the Maxar JPEG-in-TIFF files (for the full image, excluding the pyramid), although I did not attempt to, for example, `jpegtran` and measure this specific image.

The only real loss for the compression is a lovely vivid yellow on the thin crane(?) seen diagonally among the blue cranes on the right; JPEG does tone it down.

The vectorscope below is from the original Potato screenshot and not its JPEG version because the JPEG conversion showed quantization, while the Maxar version’s (having been resized) is much subtler, and I didn’t want to go off on a tangent explaining why.

</details>

Maxar’s rendering, then Potato’s:

![Maxar image of the Port of New Orleans](docs/images/Nola/maxar.jpeg)
![Potato’s image of the Port of New Orleans](docs/images/Nola/potato-q85.png)

One way of comparing them is to create a difference image, where we subtract one from the other and then recenter the 0 level at medium gray. In this image, gray means no change, blue means Potato’s output is bluer than Maxar’s, dark gray means Potato is darker than Maxar, and so on:

![Difference image of the Port of New Orleans](docs/images/Nola/diff.jpeg)

Potato is better able to represent various bright hues like safety yellow and the hot fuchsia of the [ONE](https://www.one-line.com/en) brand. The shipping containers are a useful stress test because they are only slightly wider than a multispectral pixel. Both pansharpeners here clearly struggle with certain color adjacencies, for example. But Potato appears to struggle somewhat less overall. It also leds a more lifelike yellowness to living vegetation, instead of the cooler blue-greens that the standard method uses.

Many of the largest changes are to colors at the blue end of the spectrum. For example, Maxar renders the outside walls of the building in the lower left (Nashville Ave Wharf B) in periwinkle – a light violet-blue. Potato renders it with a more down-the-middle light sky blue, nearly the color of the cranes standing in front of it. Based on [videos of the port](https://youtu.be/xEJI3SkSH4A?t=229), [Google’s 3D aerial imagery](https://earth.google.com/web/search/New+Orleans/@29.91222285,-90.1157171,11.52396512a,163.67048562d,35y,0.00000001h,60.63998696t,0r/), etc., Potato is closer to reality. As a generalization, standard methods tends to avoid clear, medium blues; they tend to make them either darker or more violet than reality. In short this is because they do not use the sensor’s deep blue band; in long, see the [deeper documentation](https://github.com/celoyd/potato/blob/docs/docs/features.md#all-band-color-conversion).

We can also explore these color characteristics with a _vectorscope_, a visualization used in film grading to see pixels (or their counts) projected onto the chroma plane of a luma/chroma colorspace. Here is a vectorscope visualization of Maxar’s image: 

![Vectorscope of Maxar’s image](docs/images/Nola/mv.png)

And Potato’s:

![Vectorscope of Potato’s image](docs/images/Nola/pv.png)


### Motion artifact comparison

This is a bonus comparison, to inform and not to persuade. A stress test for any pansharpener is an airplane in flight, since its motion makes its image violate some usual assumptions of how bands behave. No pansharpening algorithm that I know of seriously attempts to fully merge images of planes in flight; in fact I think [it would be a mistake to try](https://github.com/celoyd/potato/issues/20). So here we are not trying to judge which image is _better_, only to learn about what the different algorithms are doing.

![both](https://github.com/user-attachments/assets/3d20886b-fbf9-4dbc-80dc-5de76d5ef132)

This is Maxar’s pansharpening on the left and Potato’s output on the right. Two of the differences we might draw out:

- Maxar only uses the RGB multispectral bands. (I don’t actually know that for certain; it’s a deduction from cases like this one.) A disadvantage is that there’s less information to work with, but an advantage is that it cuts band misalignment problems to 3/8. There’s only one rainbowed plane on the left, but two on the right, where Potato is trying to integrate two groups of 4 multispectral bands. (See the [notes on band misalignment] for more on this topic.)

- Maxar’s pansharpener is clearly getting most of the overall brightness directly from the multispectral bands. This manifests as its lead plane being much brighter than Potato’s. Potato seems more willing to discount multispectral bands that appear mistaken. Conversely, the center (panchromatic) plane only exists in fine details in Maxar’s version, while Potato appears very confident that there is in fact an object there. Potato adds very little of the rich green color even though its signal must be strong.

I underline again that in _this_ comparison I don’t think it’s useful to ask which model is doing better. A model that could handle this case convincingly would probably be overdesigned and inefficient for typical cases. This sort of outlier input, where pansharpeners must deal with incoherent data, makes their different strategies easier to understand. Here I only want to show some of what’s interesting in pansharpening, to delight and intrigue lay readers.

## Documentation

Please see [the docs directory](/docs) for a quickstart guide and lengthy discussion of the unusual techniques in Potato.

## License

Potato is licensed [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Contributing

Potato is basically a personal project, and is intended primarily as a one-and-done demonstration, not as a continually improving pansharpening package. It’s born in bug-fix–only mode. Feel free to file an issue to point out an error or ask a question, but ideas for bold new features are unlikely to interest me here; mine are going in other (as yet unreleased) projects. I warmly encourage forks that build on Potato in license-respecting ways. The whole point of this project is to get more people doing better work on pansharpening, and if Potato is a useful foundation or even a productive annoyance to you, I’m happy.
