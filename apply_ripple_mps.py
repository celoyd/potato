import torch
from torch import nn

#from torch.utils.data import Dataset
import rasterio as rio

import numpy as np
#import torch.nn.functional as F
#from skimage import io

#from typing import Tuple, Sequence

from sys import argv, exit

import colour

from model.ripple import (
    Ripple,
    #shuf,
    shuf2,
    #unshuf,
    unshuf2,
)
from model.ripple import Ripple as gen

import warnings

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


# python apply_ripple.py weights/gen-r5-${n}.pts /media/ch/Sablefish/lc3/${chip}.pt ok.tiff


def save_image(path, image, norm=True):
    image = torch.squeeze(image, dim=0).cpu().detach().numpy()

    image = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(image.swapaxes(0, 2))).swapaxes(2, 0)
    image = (np.clip(image * 10_000 * (65_553/10_000), 0, 65535)).astype(np.uint16)
    print(np.mean(image))

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

mps_device = torch.device("mps")

y = torch.load(argv[2], map_location=mps_device)

x = torch.unsqueeze(y, dim=0)

x = x.float() / 10_000
print(torch.mean(x))

model = Ripple().to(mps_device)

model.load_state_dict(torch.load(argv[1], map_location=mps_device))

model.eval()
# print(x)
with torch.no_grad():
    img = model(x)

print(torch.mean(img))

save_image(argv[3], img)