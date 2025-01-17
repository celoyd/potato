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

## Examples

### Chungthang

Two examples of pansharpening these pan and multispectral images of Chungthang, Sikkim. (For the connoisseurs, this is CID 10300100CE8D0400, 2022-03-07.)

First, Maxar’s pansharpening that they provide with the data:

![Maxar output](docs/images/Chungthang/mx-nice.png)

Second, Potato’s output for the same data:

![Potato output](docs/images/Chungthang/potato.png)

I’ve adjusted the brightness and contrast of both images to make them roughly match. (Image-wide adjustments that you can make with a [curves tool](https://en.wikipedia.org/wiki/Curve_(tonality)) are independent of pansharpening – we’re looking at local details here, not overall tones.)

To see differences, try opening each in its own tab and flipping between them. There is no way to make a completely fair comparison, for example because the Maxar image has been lossily compressed and resampled back to nominal resolution. However, several of what I consider its shortcomings – the dark water edges, sensitivity to grainy noise, difficulty rendering deeper blue hues – are not plausibly artifacts of those processing steps. Google Earth [also uses this collect](https://earth.google.com/web/@27.60326593,88.64660971,1610.66494177a,1045.83623347d,35y,0h,0t,0r/data=ChYqEAgBEgoyMDIyLTAzLTA3GAFCAggBQgIIAEoNCP___________wEQAA), and although their color treatment is much darker than Maxar’s, the general look of the pansharpening is similar.

### Quito

Here is a comparison of a different image chip with a different set of visualization choices. On the left we have a small part of Quito [as seen on Google Earth](https://earth.google.com/web/search/Quito,+Ecuador/@-0.08986503,-78.44810289,2628.27205598a,346.32203057d,35y,0h,0t,0r/data=Cj4iJgokCSF-pJOHnDtAEQdki5yumDtAGRLQGY0TKlZAIXMmJJ-4KFZAKhAIARIKMjAyMi0wMS0wOBgBQgIIAUICCABKDQj___________8BEAA) (from 10300100CC8F7900, 2022-01-08; image credit as given within the screenshot; n.b., the link is to the web version but the screenshot is from the desktop version). On the right is Potato’s output, resized and lightly color-adjusted to roughly match the screenshot.

![A side-by side comparison of images of a mixed neighborhood with a plaza park, a bus stop, and some large semi-industrial buildings in Quito](docs/images/Quito/both.png)

The rescaling is unflattering to the fine detail in both, but of note is the blocky artifacting around some of the [brick-through-grass](https://earth.google.com/web/search/Quito,+Ecuador/@-0.08986531,-78.44806574,2632.5715332a,0d,60y,270.41779775h,79.45498836t,0r/data=CiwiJgokCSF-pJOHnDtAEQdki5yumDtAGRLQGY0TKlZAIXMmJJ-4KFZAQgIIASIaChZIX0pETEI2d1VfcndHcS1fQVJTZklBEAJCAggASg0I____________ARAA) textures in the playground/plaza west of the bus stop. This is a relatively rare artifact but a revealing one: it shows a failure state of local adaptation–based pansharpening algorithms, which is confused by certain patterns. (The Maxar version of this image also fails in this area, but with blurriness instead of blockiness.) Potato is certainly not perfect, but it paints a picture that reasonably matches on-the-ground views.

### San Fernando

Finally, a more abstract example. The subject is the area around [All American Asphalt](https://allamericanasphalt.com/) and [Moviemachines](https://moviemachines.net/collections/all) in San Fernando, California (from 10400100A0BAD600, 2025-01-14). To account for lossy compression favoring Potato in these comparisons, I’ve (1) resampled and (2) JPEG-compressed its output with ImageMagick to match Maxar’s JPEG-compressed pansharpened TIFF. (I only measured the size of the full-resolution subimage, extracted with `tiffsplit`, which was just over 80 megabytes. I used `convert -filter Catrom -resize 17408x -quality 87%` to make Potato’s output about 77.5 megabytes. Thus they both have slightly more than 2 bits per pixel.)

Here’s 

To clarify _color detail enhancement_, the core concept of pansharpening, we see these images in exaggerated chroma-only versions.

, I’ve used `tiffsplit` to extract the JPEG from the Maxar pansharpened TIFF (without recompression), then used ImageMagick with Potato’s rendering to (1) resize the scene, with Catmull-Rom interpolation, to the same dimensions, and (2) compress Potato’s rendering of the scene to a 

Here is Potato’s rendering of , lossily compressed with JPEG to TK kB (TK bpp).

Here is its ab plane in the Oklab color space, in other words just its chroma (hue and saturation) with luma set to 0.5, with a and b each normalized to a standard deviation of 0.25 for comparison.

Same for 

### On comparisons

Here I have tried to make comparisons that _I_ consider 


## Documentation

Please see [the docs directory](/docs) for a quickstart guide and lengthy discussion of the unusual techniques in Potato.

## License

Potato is licensed [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Contributing

Potato is basically a personal project, and is intended primarily as a one-and-done demonstration, not as a continually improving pansharpening package. It’s born in bug-fix–only mode. Feel free to file an issue to point out an error or ask a question, but ideas for bold new features are unlikely to interest me here; mine are going in other (as yet unreleased) projects. I warmly encourage forks that build on Potato in license-respecting ways. The whole point of this project is to get more people doing better work on pansharpening, and if Potato is a useful foundation or even a productive annoyance to you, I’m happy.
