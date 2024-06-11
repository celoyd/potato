import torch
from torch import nn

from torchvision.transforms.functional import gaussian_blur, resize
from torchvision.transforms import InterpolationMode

import rasterio as rio

import numpy as np
from sys import argv, exit

import colour

from ripple.model import (
    Ripple,
    # shuf2,
    # unshuf2,
)

from ripple.tm import usm, aces_approx

import warnings

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def scale(n, a, b, x, y):
    return (n - a) / (b - a) * (y - x) + x


def pyramid_analysis(x, levels=5):
    # Laplacian pyramid
    pyramid = [x]
    for level in range(levels):
        sharp = pyramid[-1]
        # soft = gaussian_blur(sharp, 5, 1.0)
        small = resize(
            sharp.unsqueeze(0),
            (sharp.shape[-2] // 2, sharp.shape[-1] // 2),
            interpolation=InterpolationMode("bilinear"),
        )[0]
        soft = resize(
            small.unsqueeze(0),
            (sharp.shape[-2], sharp.shape[-1]),
            interpolation=InterpolationMode("bilinear"),
        )[0]
        diff = sharp - soft
        pyramid[-1] = diff
        pyramid.append(small)
    return pyramid


def pyramid_synthesis(pyramid):
    # base = None
    while len(pyramid) > 1:
        small = pyramid[-1]
        diff = pyramid[-2]
        soft = resize(
            small.unsqueeze(0),
            (diff.shape[-2], diff.shape[-1]),
            interpolation=InterpolationMode("bilinear"),
        )[0]
        rebuilt = soft + diff
        pyramid[-2] = rebuilt
        del pyramid[-1]

    return pyramid[0]


def tm(x):
    for b in []: #range(3):
        band = x[b]

        py = pyramid_analysis(band, levels=5)

        levels = len(py)
        for el in range(levels):
            layer = py[el]
            height = el/levels

            ls, lm = torch.std_mean(layer)

            layer -= lm
            scale = 1/((height*1)**2 + 1)
            print(el, height, scale)
            layer *= scale
            layer += lm

            py[el] = layer


        band = pyramid_synthesis(py)

        x[b] = band
        #x[b] = scale(x[b], np.quantile(x[b], 0.01), np.quantile(x[b], 0.99), 0, 1)

    # x = (np.tanh((x * 2 - 1)) + 1) / 2
    return x

def ltm(x):
    py = pyramid_analysis(x, levels=5)

    levels = len(py)
    for el in range(levels):
        layer = py[el]
        height = el/levels

        ls, lm = torch.std_mean(layer)

        layer -= lm
        scale = 1/((height*2)**2 + 1)
        print(el, height, scale)
        layer *= scale
        layer += lm

        py[el] = layer


    band = pyramid_synthesis(py)
    return band
    # x[b] = band

        #x[b] = scale(x[b], np.quantile(x[b], 0.01), np.quantile(x[b], 0.99), 0, 1)

    # x = (np.tanh((x * 2 - 1)) + 1) / 2
    # return x

def lrm(x, halvings=4, powers=[0.5, 2]):
    middle = resize(
        x.unsqueeze(0),
        (x.shape[-2]//(2**halvings), x.shape[-1]//(2**halvings)),
        interpolation=InterpolationMode("bilinear"),
    )[0]

    soft = resize(
        middle.unsqueeze(0),
        (x.shape[-2], x.shape[-1]),
        interpolation=InterpolationMode("bilinear"),
    )[0]

    upper = resize(
        x.unsqueeze(0)**powers[0],
        (x.shape[-2]//(2**halvings), x.shape[-1]//(2**halvings)),
        interpolation=InterpolationMode("bilinear"),
    )[0]

    lower = resize(
        x.unsqueeze(0)**powers[1],
        (x.shape[-2]//(2**halvings), x.shape[-1]//(2**halvings)),
        interpolation=InterpolationMode("bilinear"),
    )[0]

    diff = upper - lower
    
    diff = resize(
        diff.unsqueeze(0),
        (x.shape[-2], x.shape[-1]),
        interpolation=InterpolationMode("bilinear"),
    )[0]

    x = x + (4 * diff * (x - soft))

    return x

def rgb_sat(x):
    return torch.var(x, dim=-3, keepdims=True)

def rgb_tonemap(x):
    k = 15
    s = 7

    x = torch.tensor(x).clamp(1e-6, 1 - 1e-6) * 2
    x = (torch.tanh((x * 2 - 1)*2) + 1) / 2
    bright = x**0.5
    # bright = x**2
    dim = x**2
    x_sat = gaussian_blur(rgb_sat(bright).unsqueeze(0), k, s)[0]
    bright_sat = gaussian_blur(rgb_sat(bright).unsqueeze(0), k, s)[0]
    dim_sat = gaussian_blur(rgb_sat(dim).unsqueeze(0), k, s)[0]

    x = ((x*x_sat) + (bright*bright_sat) + (dim*dim_sat)) / (x_sat + bright_sat + dim_sat)
    x = (torch.tanh((x * 5 - 1.25)) + 1) / 2
    x = x.numpy()
    return x

def stm(x, r=8):
    # for b in len(x.shape[[-3]]):
    soft = gaussian_blur(x.unsqueeze(0), 31, r)[0]
    delta = soft - x
    x -= delta
    return x

def save_image(path, image, norm=True):
    image = torch.squeeze(image, dim=0).cpu().detach().numpy()

    image = torch.tensor(image)

    image = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(image.swapaxes(0, 2))).swapaxes(2, 0)

    # image = (rgb_tonemap(image) + image*3)/3

    image = np.clip(image * 65_535, 0, 65_535).astype(np.uint16)

    with rio.open(
        path,
        "w",
        width=image.shape[1],
        height=image.shape[2],
        dtype=image.dtype,
        driver="gtiff",
        count=image.shape[0],
        photometric="RGB",
        compress="LZW",
    ) as dst:
        for chan in range(image.shape[0]):
            dst.write(image[chan], chan + 1)


y = torch.load(argv[2], map_location="cpu")

if type(y) == tuple:
    y = y[0].float()
else:
    y = y.float() / 10_000

x = torch.unsqueeze(y, dim=0)

# x[:, 18, 0::2, 150] = 0.0
# x[:, 18, 1::2, 151] = 1.0

model = Ripple()

model.load_state_dict(torch.load(argv[1], map_location="cpu"))

model.eval()
with torch.no_grad():
    _, _, img = model(x)
    # _, img, _ = model(x)
    # img, _, _ = model(x)

# img[:, 0] = 0.5

# img[:, 0:1] = usm(img[:, 0:1])
# img[:, 0:1] = torch.clamp(img[:, 0:1] / torch.quantile(img[:, 0:1], 0.99), 0, 1)
# img[:, 0:1] = img[:, 0:1]**0.5
# img[:, 0:1] = aces_approx(img[:, 0:1])

save_image(argv[3], img)
