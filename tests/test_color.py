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


@pytest.mark.parametrize("sRGB,oklab", srgb_oklab_pairs)
def test_srgb_to_oklab(sRGB, oklab):
    s_to_l = color.sRGBToOklab()

    # These reshapes turn the pixel value into a 1×1 image
    sRGB = torch.tensor(sRGB).reshape(-1, 1, 1).float()
    oklab = torch.tensor(oklab).reshape(-1, 1, 1)

    test = s_to_l(sRGB)

    # Tolerance is 1/10 of an oklab JND
    assert torch.allclose(oklab, test, atol=(0.02 * 0.1))


@pytest.mark.parametrize("sRGB,oklab", srgb_oklab_pairs)
def test_oklab_to_srgb(sRGB, oklab):
    l_to_s = color.OklabTosRGB()
    sRGB = torch.tensor(sRGB).reshape(-1, 1, 1).float()
    oklab = torch.tensor(oklab).reshape(-1, 1, 1)

    test = l_to_s(oklab)

    # Tolerance is the equivalent of an 8-bit step
    assert torch.allclose(sRGB, test, atol=(1 / 256))


@pytest.fixture
def random_srgbish_image():
    return torch.rand((3, 256, 256))


def test_32_srgb_oklab_roundtrips(random_srgbish_image):
    s_to_l = color.sRGBToOklab()
    l_to_s = color.OklabTosRGB()

    for _ in range(32):
        oklab = s_to_l(random_srgbish_image)
        srgb = l_to_s(oklab)

    assert torch.allclose(srgb, random_srgbish_image, atol=(1 / 256))
