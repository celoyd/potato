import torch
import rasterio as rio
from potato.color import BandsToOklab, OklabTosRGB

import imageio.v3 as iio

def fast_half(x):
	return (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2]).mean()

def fast_quarter(x):
	return fast_half(fast_half(x))

b2l = BandsToOklab()
l2r = OklabTosRGB()

colors = {
    "classic blue roofs": (2172.3125, [1954, 2761, 1530, 1067, 1757, 4503, 4574, 4490]),
    "rust on blue roof": (1221.5, [539, 965, 959, 993, 1277, 2465, 2927, 3190]),
    "blue roof ridgeline": (2454.0, [1855, 2758, 1879, 1539, 2226, 4276, 4276, 4074]),
    "more blue roof": (2476.9375, [2565, 3456, 1734, 1149, 2038, 4600, 4720, 4648]),
    "red roof": (3392.3125, [756, 822, 1021, 4080, 5675, 5714, 5237, 4910]),
    "blue field": (405.25, [593, 911, 546, 349, 380, 459, 1408, 1661]),
    "giant flames": (8302.3125, [691, 801, 1005, 2072, 5296, 9995, 8654, 9999]),
    "dark tarp": (691.1875, [1482, 1782, 536, 390, 641, 831, 4494, 5367]),
    "medium tarp": (1383.875, [3102, 3974, 1283, 879, 970, 1210, 6200, 8096]),
    "bank roof": (938.5, [1381, 1895, 1067, 764, 1102, 1454, 1596, 1795]),
    "bright tarp": (2756.375, [3708, 4514, 2548, 2286, 2395, 2862, 6794, 8674]),
    "another blue roof": (681.125, [1537, 2063, 813, 523, 589, 713, 1784, 1862]),
    "off-white roof": (4003.0, [4177, 5185, 3932, 3233, 3728, 5568, 6284, 6470]),
    "rich red roof": (1427.625, [197, 503, 654, 1396, 1945, 2626, 2587, 2790]),
    "tarp roof": (658.8125, [1104, 1422, 739, 414, 555, 639, 2150, 2737]),
    "pool water": (779.5, [1592, 1997, 1958, 865, 624, 285, 143, 106]),
    "hotish orange roof": (
        2670.9375,
        [328, 437, 569, 3325, 4936, 5298, 5664, 6164],
    ),
    "dusty/faded blue": (
        2875.5625,
        [2843, 3381, 2811, 2694, 2809, 3839, 3813, 4282],
    ),
    "less dusty/faded": (2336.625, [3122, 3282, 2127, 1894, 2129, 3774, 4087, 4731]),
    "depot roof": (992.625, [1387, 1897, 1212, 1037, 1003, 1047, 1934, 2384]),
    "greenish roof": (822.1875, [622, 1214, 1185, 621, 645, 1189, 1522, 1617]),
    "ischool blue": (
        1001.0,
        [1359, 1732, 1091, 827, 1000, 1161, 2395, 2712],
    ),
    "bricky red roof": (1787.375, [309, 738, 907, 1821, 2574, 3452, 2998, 3362]),
}

pans = torch.tensor(list(c[0] for c in colors.values())) / 10_000.0
muls = torch.tensor(list(c[1] for c in colors.values())) / 10_000.0

pan = torch.zeros((1024, 1024))
mul = torch.zeros((8, 256, 256))

for n in range(len(colors.keys())):
	rgb = torch.zeros((3, 1024, 1024))
	m = muls[n]

	lab = b2l(m.unsqueeze(1).unsqueeze(1))
	true = l2r(lab)

	rgb += true
	rgb *= torch.normal(1, 0.05, rgb.shape)

	pngable = (rgb * 255).clamp(0, 255).byte().swapaxes(0, -1)
	iio.imwrite(f"difficult-colors/{n}.png", pngable)

#def make_gradient