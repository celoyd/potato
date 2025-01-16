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

_These images are of Chungthang, India, from Maxar collect 10300100CE8D0400 on 2022-03-07. Like all images shown in this repo except otherwise credited, it’s [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en) from the [Maxar Open Data Program](https://registry.opendata.aws/maxar-open-data/)._

This is called the panchromatic or pan band, and it works like black and white film, or like applying a black and white filter to a photo. A band is a range of the spectrum, and pan-chromatic means all-color – this band collects all visible light.

At the same time, the satellite collects a full-color image, with red, green, and blue bands, at lower resolution:

![Upsampled color image of the same scene](docs/images/Chungthang/rgb-upsampled.png)

So we have one image that is sharp, and one that is full-color. We want to merge them into a combined version that is both sharp _and_ full-color at the same time. We can think of this as using the pan image to sharpen the full-color image, so the process is called **pansharpening**.

Like compression or instant messaging, pansharpening isn’t any one algorithm or technique; it’s a category defined only by the problem being solved.

Two examples of pansharpening this pair of pan and multispectral images:

![Potato output](docs/images/Chungthang/mx-nice.png)

_Maxar’s pansharpening, adjusted to match the general brightness and contrast of…_

![Potato output](docs/images/Chungthang/potato.png)

_Potato’s pansharpening._

There is no way to make a completely fair comparison, for example because the Maxar image has been lossily compressed and resampled back to nominal resolution. However, several of what I consider the shortcomings of the Maxar-provided example here – the dark water edges, sensitivity to grainy noise, difficulty rendering deeper blue hues – are not plausibly artifacts of those processing steps.

## License

Potato is licensed [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
