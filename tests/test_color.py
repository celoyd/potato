"""Test color functions."""

import pytest
import torch

from potato import color

# These are mostly just Chris Lilley’s pairs from
# https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md
srgb_oklab_pairs = (
    ((1, 1, 1), (1.0, 0.0, 0.0)),
    ((1, 0, 0), (0.627955, 0.224863, 0.125846)),
    ((0, 1, 0), (0.86644, -0.23388, 0.179498)),
    ((0, 0, 1), (0.45201, -0.03245, -0.311528)),
    ((0, 1, 1), (0.90539, -0.14944, -0.039398)),
    ((1, 0, 1), (0.70167, 0.27456, -0.169156)),
    ((1, 1, 0), (0.96798, -0.07136, 0.198570)),
    ((0, 0, 0), (0.0, 0.0, 0.0)),
    ((0.5, 0.5, 0.5), (0.598182, 0.0, 0.0)),
)

# Test conversions


@pytest.mark.parametrize("sRGB,oklab", srgb_oklab_pairs)
def test_srgb_to_oklab(sRGB, oklab):
    """SRGB -> oklab should fit others’ test pairs."""
    s_to_l = color.sRGBToOklab()

    # These reshapes turn the pixel value into a 1×1 image.
    sRGB = torch.tensor(sRGB).reshape(-1, 1, 1).float()
    oklab = torch.tensor(oklab).reshape(-1, 1, 1)

    test = s_to_l(sRGB)

    # Tolerance is 1/10 of an oklab JND
    assert torch.allclose(oklab, test, atol=0.02 * 0.1)


@pytest.mark.parametrize("sRGB,oklab", srgb_oklab_pairs)
def test_oklab_to_srgb(sRGB, oklab):
    """Oklab -> sRGB should fit others’ test pairs."""
    l_to_s = color.OklabTosRGB()
    sRGB = torch.tensor(sRGB).reshape(-1, 1, 1).float()
    oklab = torch.tensor(oklab).reshape(-1, 1, 1)

    test = l_to_s(oklab)

    # Tolerance is the equivalent of an 8-bit step.
    assert torch.allclose(sRGB, test, atol=1 / 256)


def test_32_srgb_oklab_roundtrips(random_srgbish_image):
    """Bouncing between sRGB and oklab many times should be stable."""
    s_to_l = color.sRGBToOklab()
    l_to_s = color.OklabTosRGB()

    for _ in range(32):
        oklab = s_to_l(random_srgbish_image)
        srgb = l_to_s(oklab)

    assert torch.allclose(srgb, random_srgbish_image, atol=1 / 256)


# Test saturation

# We don’t try to get fancy here; we’re looking for “accidentally deleted the
# wrong line”–type regressions and not whether this definition of saturation
# exactly matches CIECAM16 or whatever.
#
# In particular, test colors may be invalid/out-of-gamut.


def test_mean_saturation_of_noise():
    """The mean saturation of many random pixels should be large."""
    random_oklabish_image = torch.randn((1, 3, 256, 256))
    saturation_field = color.oklab_saturation(random_oklabish_image)
    mean_saturation = saturation_field.mean()
    assert mean_saturation > 1.0


def test_saturation_of_gray():
    """A gray pixel should have no saturation."""
    gray = torch.zeros((1, 3, 1, 1))
    saturation = color.oklab_saturation(gray)
    assert saturation == 0.0


def test_saturation_of_red():
    """A red pixel manually placed at distance 1 should have saturation 1."""
    red = torch.tensor([[0.5, 2**-0.5, 2**-0.5]])
    saturation = color.oklab_saturation(red)
    assert torch.allclose(saturation, torch.tensor(1.0))


def test_relative_saturation():
    """Saturated green should be more saturated than unsaturated green."""
    mild_green = torch.tensor([[0.5, -0.1, 0.1]])
    strong_green = torch.tensor([[0.5, -0.5, 0.5]])
    assert color.oklab_saturation(mild_green) < color.oklab_saturation(strong_green)


def test_saturation_invariance_over_L():
    """Otherwise equal pixels of different L should have same saturation."""
    dark_blue = torch.tensor([[0.1, 0.0, -0.3]])
    light_blue = torch.tensor([[0.9, 0.0, -0.3]])
    assert color.oklab_saturation(dark_blue) == color.oklab_saturation(light_blue)


def test_equal_saturation_of_opposites():
    """Medium green and medium purple should have the same saturation."""
    green = torch.tensor([[0.5, -(2**-0.5), 2**-0.5]])
    purple = torch.tensor([[0.5, 2**-0.5, -(2**-0.5)]])
    assert color.oklab_saturation(green) == color.oklab_saturation(purple)


def test_averaging_desaturates():
    """Mixing images should give net lower total saturation than either."""
    a = torch.randn((1, 3, 256, 256))
    b = torch.randn((1, 3, 256, 256))
    assert (color.oklab_saturation(a) + color.oklab_saturation(b)).mean() > (
        color.oklab_saturation(a + b)
    ).mean()


# Test band conversions


def test_null_reflectance_is_black():
    """Zero reflectance should induce oklab black."""
    b_to_l = color.BandsToOklab()
    dark = torch.zeros((8, 1, 1))
    test_article = b_to_l(dark)[:, 0, 0]
    assert torch.allclose(test_article, torch.tensor([0.0, 0.0, 0.0]))


def test_perfect_reflectance_is_white_ish():
    """Full reflectance should induce oklab white."""
    b_to_l = color.BandsToOklab()
    bright = torch.ones((8, 1, 1))
    test_article = b_to_l(bright)[:, 0, 0]

    # We apply some tolerance here because, given band geometry and the D65
    # illumination, it’s only approximately the case that 1.0 albedo as
    # measured should map to oklab(1, 0, 0). As of this writing, the
    # difference is about about a JND (0.02) off: L is slightly too bright.
    # Roughly 2–3 times that feels like about where it makes sense to suspect
    # something might be wrong. As with other such test thresholds, please
    # don’t hesitate to adjust it to your needs.
    assert torch.allclose(test_article, torch.tensor([1.0, 0.0, 0.0]), atol=0.05)
