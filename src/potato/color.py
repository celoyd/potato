import torch
from torch import nn

"""
mul_to_xyz is fitted as described in docs/features.md.

Other coefficents are from https://bottosson.github.io/posts/oklab/.

## Conventions:
- XYZ is XYZ[D65]
- Abbreviations in einsums:
 - XYZ as x
 - LMS as m
 - oklab as l
 - [l]sRGB as r
 - height and width as h and w
"""


def safe_pow(n, exp):
    return n.sign() * n.abs().pow(exp)


def unlinearize_lsrgb(lsrgb):
    return torch.where(
        lsrgb <= 0.0031308,
        lsrgb * 12.92,
        1.055 * safe_pow(lsrgb, 1 / 2.4) - 0.055,
    )


def linearize_srgb(srgb):
    return torch.where(
        srgb <= 0.04045, srgb / 12.92, safe_pow((srgb + 0.055) / (1.055), 2.4)
    )


class BandsToOklab(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mul_to_xyz",
            torch.tensor(
                [
                    [9.0677e-02, 0.00000e00, 4.0220e-01],
                    [1.1873e-01, 1.5157e-01, 7.5022e-01],
                    [2.0263e-01, 5.8714e-01, 0.00000e00],
                    [5.0440e-01, 2.9138e-01, 1.7287e-04],
                    [8.7978e-02, 2.7045e-02, 7.2419e-07],
                    [0.00000e00, 3.2608e-04, 0.00000e00],
                    [1.0887e-04, 1.7025e-04, 0.00000e00],
                    [8.0838e-05, 5.0579e-05, 0.00000e00],
                ]
            ),
            persistent=False,
        )

        self.register_buffer(
            "xyz_to_oklab_m1",
            torch.tensor(
                [
                    [0.8189330101, 0.3618667424, -0.1288597137],
                    [0.0329845436, 0.9293118715, 0.0361456387],
                    [0.0482003018, 0.2643662691, 0.6338517070],
                ]
            ),
            persistent=False,
        )

        self.register_buffer(
            "xyz_to_oklab_m2",
            torch.tensor(
                [
                    [0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660],
                ]
            ),
            persistent=False,
        )

    def mul_to_lab(self, mul):
        xyz = torch.einsum("...rhw, rx -> ...xhw", mul, self.mul_to_xyz)
        lms = torch.einsum("...xhw, mx -> ...mhw", xyz, self.xyz_to_oklab_m1)
        lms = safe_pow(lms, 1 / 3)
        oklab = torch.einsum("...mhw, lm -> ...lhw", lms, self.xyz_to_oklab_m2)
        return oklab

    def forward(self, x):
        return self.mul_to_lab(x)


class OklabTosRGB(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "oklab_to_lsrgb_m1",
            torch.tensor(
                [
                    [1, 0.3963377774, 0.2158037573],
                    [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480],
                ]
            ),
            persistent=False,
        )

        self.register_buffer(
            "oklab_to_lsrgb_m2",
            torch.tensor(
                [
                    [4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010],
                ]
            ),
            persistent=False,
        )

    def forward(self, lab):
        lms = torch.einsum("...lhw, ml -> ...mhw", lab, self.oklab_to_lsrgb_m1)
        lms = safe_pow(lms, 3)
        lsrgb = torch.einsum("...mhw, rm -> ...rhw", lms, self.oklab_to_lsrgb_m2)
        srgb = unlinearize_lsrgb(lsrgb)
        return srgb


class sRGBToOklab(nn.Module):
    """
    This is only used for analysis, not in pansharpening or training.
    """

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "lsrgb_to_xyz_m1",
            torch.tensor(
                [
                    [0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005],
                ]
            ),
        )

        self.register_buffer(
            "lsrgb_to_xyz_m2",
            torch.tensor(
                [
                    [0.2104542553, +0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, +0.4505937099],
                    [0.0259040371, +0.7827717662, -0.8086757660],
                ]
            ),
        )

    def forward(self, srgb):
        lsrgb = linearize_srgb(srgb)
        lms = torch.einsum("...rhw, mr -> ...mhw", lsrgb, self.lsrgb_to_xyz_m1)
        lms = safe_pow(lms, 1 / 3)
        oklab = torch.einsum("...mhw, lm -> ...lhw", lms, self.lsrgb_to_xyz_m2)
        return oklab
