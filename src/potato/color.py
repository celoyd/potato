"""Color conversions.

## Notes

- mul_to_xyz is fitted as described in docs/concepts.md.
- Other coefficents are from https://bottosson.github.io/posts/oklab/.

## Conventions

- For conversions, sRGB is always in unit range (not, e.g., uint8)
- XYZ means XYZ_D65
- In einsums, h and w are spatial height and width
"""

import torch
from einops import einsum
from torch import nn


def oklab_saturation(x):
    """Saturation in standard oklab space."""
    return (x[:, 1].square() + x[:, 2].square()).sqrt()


def safe_pow(n, exp):
    """Give signed magnitude instead of NaN for negative**fractional."""
    return n.sign() * n.abs().pow(exp)


def unlinearize_lsrgb(lsrgb):
    """Apply (unit) sRGB gamma."""
    return torch.where(
        lsrgb <= 0.0031308,
        lsrgb * 12.92,
        1.055 * safe_pow(lsrgb, 1 / 2.4) - 0.055,
    )


def linearize_srgb(srgb):
    """Remove (unit) sRGB gamma."""
    return torch.where(
        srgb <= 0.04045, srgb / 12.92, safe_pow((srgb + 0.055) / 1.055, 2.4)
    )


class BandsToOklab(nn.Module):
    """Convert from WV-2/3 band reflectance to oklab.

    We imply D65 illumination in mul_to_xyz. Then it’s just XYZ to oklab.
    """

    def __init__(self):
        """Keep the conversion tensors on the device."""
        super().__init__()

        self.register_buffer(
            "mul_to_xyz",
            torch.tensor(
                [
                    [0.117, 0.000, 0.536],
                    [0.045, 0.098, 0.558],
                    [0.262, 0.649, 0.000],
                    [0.435, 0.221, 0.000],
                    [0.091, 0.032, 0.000],
                    [0.000, 0.000, 0.000],
                    [0.000, 0.000, 0.000],
                    [0.000, 0.000, 0.000],
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

    def forward(self, mul):
        """Convert WV-2/3 multispectral bands to oklab.

        This is implemented as the definitional XYZ to oklab conversion but
        with the band convertion merged into the first matmul.
        """
        with torch.no_grad():
            lms = einsum(
                mul,
                self.mul_to_xyz,
                self.xyz_to_oklab_m1,
                "... rho h w, rho xyz, lms xyz -> ... lms h w",
            )
            lms = safe_pow(lms, 1 / 3)
            oklab = einsum(
                lms, self.xyz_to_oklab_m2, "... lms h w, oklab lms -> ... oklab h w"
            )
            return oklab


class OklabTosRGB(nn.Module):
    """Turn oklab color (in ..., c, h, w) into unit sRGB."""

    def __init__(self):
        """Keep the conversion tensors on the device."""
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
        """Convert oklab to unit sRGB.

        This should be (and is tested as) the exact inverse, up to reasonable
        float precision constraints, of sRGBToOklab.forward().
        """
        with torch.no_grad():
            lms = einsum(
                lab, self.oklab_to_lsrgb_m1, "... oklab h w, lms oklab -> ... lms h w"
            )
            lms = safe_pow(lms, 3)
            lsrgb = einsum(
                lms, self.oklab_to_lsrgb_m2, "... lms h w, lsrgb lms -> ... lsrgb h w"
            )
            srgb = unlinearize_lsrgb(lsrgb)
            return srgb


class sRGBToOklab(nn.Module):
    """Turn unit sRGB color (in ..., c, h, w) into oklab."""

    def __init__(self):
        """Keep the conversion tensors on the device."""
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
        """Convert unit sRGB to oklab.

        This should be (and is tested as) the exact inverse, up to reasonable
        float precision constraints, of OklabTosRGB.forward().
        """
        with torch.no_grad():
            lsrgb = linearize_srgb(srgb)
            lms = einsum(
                lsrgb, self.lsrgb_to_xyz_m1, "... lsrgb h w, lms lsrgb -> ... lms h w"
            )
            lms = safe_pow(lms, 1 / 3)
            oklab = einsum(
                lms, self.lsrgb_to_xyz_m2, "... lms h w, oklab lms -> ... oklab h w"
            )
            return oklab
