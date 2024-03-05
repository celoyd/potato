import torch
from torch import nn

import rasterio as rio

import numpy as np
from sys import argv, exit

import colour

from ripple.model import (
    Ripple,
    # shuf2,
    # unshuf2,
)

import warnings

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

def save_image(path, image, norm=True):
    image = torch.squeeze(image, dim=0).cpu().detach().numpy()
    image = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(image.swapaxes(0, 2))).swapaxes(2, 0)
    image = np.clip(image*65_535, 0, 65_535).astype(np.uint16)
    
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

x = torch.unsqueeze(y, dim=0)

x = x.float() / 10_000

model = Ripple()

model.load_state_dict(torch.load(argv[1], map_location="cpu"))

model.eval()
with torch.no_grad():
    #_, _, img = model(x)
    img = model(x)

save_image(argv[3], img)