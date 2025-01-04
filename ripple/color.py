import torch
from torch import nn

'''
All matrixes other than mul_to_xyz are from Ottosson’s documentation;
mul_to_xyz is fitted as described in docs/features.md.

As a convention, in einsum expressions we abbreviate:
 - XYZ as x
 - LMS as m
 - oklab as l
 - (s)RGB as r
 - height and width as h and w
'''

class BandsToOklab(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mul_to_xyz",
            torch.tensor(
                [
                    [9.0677e-02, 0.0000e00, 4.0220e-01],
                    [1.1873e-01, 1.5157e-01, 7.5022e-01],
                    [2.0263e-01, 5.8714e-01, 0.0000e00],
                    [5.0440e-01, 2.9138e-01, 1.7287e-04],
                    [8.7978e-02, 2.7045e-02, 7.2419e-07],
                    [0.0000e00, 3.2608e-04, 0.0000e00],
                    [1.0887e-04, 1.7025e-04, 0.0000e00],
                    [8.0838e-05, 5.0579e-05, 0.0000e00],
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

    def safe_pow(self, n, exp):
        return n.sign() * n.abs().pow(exp)

    def mul_to_lab(self, mul):
        xyz = torch.einsum("...rhw, rx -> ...xhw", mul, self.mul_to_xyz)
        lms = torch.einsum("...xhw, mx -> ...mhw", xyz, self.xyz_to_oklab_m1)
        lms = self.safe_pow(lms, 1 / 3)
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
            persistent=False
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
            persistent=False
        )

    def safe_pow(self, n, exp):
        return n.sign() * n.abs().pow(exp)

    def apply_gamma(self, lsrgb):
        # defined by sRGB itself
        return torch.where(
            lsrgb <= 0.0031308,
            lsrgb * 12.92,
            1.055 * self.safe_pow(lsrgb, 1 / 2.4) - 0.055,
        )

    def convert(self, lab):
        lms = torch.einsum("...lhw, ml -> ...mhw", lab, self.oklab_to_lsrgb_m1)
        lms = self.safe_pow(lms, 3)
        lrgb = torch.einsum("...mhw, rm -> ...rhw", lms, self.oklab_to_lsrgb_m2)
        srgb = self.apply_gamma(lrgb)
        return srgb
