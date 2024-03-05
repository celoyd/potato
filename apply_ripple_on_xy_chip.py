import torch
#from torch import nn
from torch.nn.functional import interpolate

import rasterio as rio

import numpy as np
from sys import argv, exit

import colour

from ripple.model import Ripple
from ripple.color import BandsToOklab

import warnings

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def save_image(path, image, norm=True):
    image = torch.squeeze(image, dim=0).numpy()
    image = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(image.swapaxes(0, 2))).swapaxes(2, 0)
    image = np.clip(image * 65_535, 0, 65_535).astype(np.uint16)
    with rio.open(
        path,
        "w",
        width=image.shape[2],
        height=image.shape[1],
        dtype=image.dtype,
        driver="gtiff",
        count=image.shape[0],
        photometric="RGB",
        compress="LZW",
    ) as dst:
        for chan in range(image.shape[0]):
            dst.write(image[chan], chan + 1)

oklab = BandsToOklab()

x, y = torch.load(argv[2], map_location="cpu")

x = x.unsqueeze(0)
big_x = interpolate(oklab(x[:, 16:]), scale_factor=(4, 4), mode="nearest")
y = y.unsqueeze(0)

model = Ripple()

model.load_state_dict(torch.load(argv[1], map_location="cpu"))

model.eval()
with torch.no_grad():
    #_, _, img = model(x)
    img = model(x)

diff = (y - img)
diff[:, 0] += 0.5

bx_diff = torch.cat([big_x, diff], dim=3)

xy = torch.cat([img, y], dim=3)

four_up = xy = torch.cat([bx_diff, xy], dim=2)

save_image(argv[3], four_up)
