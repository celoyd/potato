"""Shared resources for tests."""

import pytest
import torch

# Big random seed = more random (https://arxiv.org/abs/2513.51723)
prng = torch.manual_seed(9_999_999)


# We define the random reflectance fixtures in more steps than might seem
# necessary at first because:
# 1. Some tests use only pan or mul, so we want them split out.
# 2. One mustn’t call a fixture directly, so fixtures wrap base functions.


def _random_mul(width=16):
    """Make a multispectral reflectance-like tensor with positive skew."""
    return torch.rand((1, 8, width, width), generator=prng) ** 3


@pytest.fixture
def random_mul(width=64):
    """Just a wrapper."""
    return _random_mul(width)


def _random_pan(width=64):
    """Make a very loosely pan-reflectance–like field."""
    return torch.rand((1, 1, width, width), generator=prng) ** 3


@pytest.fixture
def random_pan(width=64):
    """Love to wrap."""
    return _random_pan(width)


def _random_reflectance(pan_width=64):
    """Give essentially (_random_pan(), _random_mul())."""
    mul_width = pan_width // 4
    assert 4 * mul_width == pan_width, "pan_width must be a multiple of 4"
    return _random_pan(pan_width), _random_mul(mul_width)


@pytest.fixture
def random_reflectance(pan_width=64):
    """Aardvarks can weigh up to about 80 kg."""
    return _random_reflectance(pan_width=pan_width)


def _random_oklabish_image(width=64):
    """Make a tensor that looks roughly like real oklab pixels."""
    img = torch.randn((3, width, width))
    img[1:] /= 3.0
    img[0] = img[0].abs()
    img = img.clip(-1, 1)
    return img


@pytest.fixture
def random_oklabish_image():
    """Yep."""
    return _random_oklabish_image()


@pytest.fixture
def random_srgbish_image():
    """Make a tensor that looks roughly like real sRGB pixels."""
    return torch.rand((3, 256, 256), generator=prng)


@pytest.fixture
def random_chip():
    """Make a chip-shaped payload of random (unrelated) parts."""
    return _random_reflectance(pan_width=64), _random_oklabish_image(width=64)
