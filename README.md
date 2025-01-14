# Potato: a pansharpening model

:potato:

## About Potato

Potato is a research pansharpening model with a training setup and documentation. Its goals are to:

1. Raise interest in pansharpening.
2. Raise expectations of the quality of pansharpening.
3. Introduce some transferable techniques to improve pansharpening.

The model is a convolutional neural network that is small (~50k parameters) and of simple design. Potato’s training data is from the Maxar Open Data Program, and like that imagery, it is licensed CC BY-NC. It can be applied (noncommercially) in its present form – see the quickstart – but it is not tested, reviewed, or optimized for general-purpose use; it’s a research project. It aims to be a good influence on the next generation of applied pansharpening models, not to lead them.

## About pansharpening

_This explanation will omit many details, even some that motivate this project. The goal here is to show lay readers the core idea of pansharpening._

A very high resolution optical satellite collects an image in two parts. It gets a grayscale image at full resolution:

TK pan band

This is called the panchromatic or pan band, and it works like black and white film, or like applying a black and white filter to a photo. A band is a range of the spectrum, and pan-chromatic means all-color – this band collects all visible light.

At the same time, the satellite collects a full-color image, with red, green, and blue bands, at lower resolution:

TK corresponding RGB

To make an image that is both sharp and full-color, we must merge the image that is sharp with the one that is full-color. There is no perfectly accurate way to do this, so there are many different approximate ways. All of these ways of sharpening the color image with the panchromatic image are different kinds of pansharpening. Like compression or instant messaging, pansharpening isn’t any one algorithm; it’s any process, conceptualized or implemented in any particular way, that solves a certain kind of problem. These are all ways of pansharpening our example images:

TK example pansharpenings

From left to right: nearest-neighbor interpolation of the RGB 


## License

[CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
