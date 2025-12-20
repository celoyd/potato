"""
make-difficult-chips.py colors.csv chips/

Create training chips for hard-negative colors. Please see the readme.
"""

import torch
from potato.color import BandsToOklab, OklabTosRGB
from potato.util import pile  # not really necessary, but heck, why not
from sys import argv

full_edge = 512

b_to_l = BandsToOklab()
l_to_r = OklabTosRGB()


def make_chip(reflectances):
    p = reflectances[0]
    m = reflectances[1:]

    pan = torch.zeros((full_edge, full_edge)) + p
    mul = torch.zeros((8, full_edge // 4, full_edge // 4)) + m.unsqueeze(1).unsqueeze(1)

    y = b_to_l(m.unsqueeze(1).unsqueeze(1))
    y = torch.zeros((3, full_edge, full_edge)) + y

    pack = torch.cat([pile(pan.unsqueeze(0), 4), mul])
    xy = (pack, y)
    return xy


for i, line in enumerate(open(argv[1])):
    reflectances = torch.tensor(list(float(r) for r in line.split(",")[:9])) / 10_000
    chip = make_chip(reflectances)
    torch.save(chip, f"{argv[2]}/{i}.pt")
