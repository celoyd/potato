import pytest
import torch
from potato import util


# A random, independent, reflectance-like tensor
@pytest.fixture
def rir_1x8x32x32():
    return torch.rand((1, 8, 32, 32)) ** 3


def test_cheap_half(rir_1x8x32x32):
    assert util.cheap_half(rir_1x8x32x32).mean() == pytest.approx(rir_1x8x32x32.mean())


def test_pile2_of_tile2_roundtrip(rir_1x8x32x32):
    assert torch.equal(util.pile(util.tile(rir_1x8x32x32, 2), 2), rir_1x8x32x32)


def test_tile2_of_pile2_roundtrip(rir_1x8x32x32):
    assert torch.equal(util.tile(util.pile(rir_1x8x32x32, 2), 2), rir_1x8x32x32)


def test_tile2_tile2_tile2_of_pile8_roundtrip(rir_1x8x32x32):
    assert torch.equal(
        util.tile(util.tile(util.tile(util.pile(rir_1x8x32x32, 8), 2), 2), 2),
        rir_1x8x32x32,
    )


def test_pile_preserves_locality_horizontal(rir_1x8x32x32):
    """
    It’s very important for our purposes that pile is a local operation
    in the sense that a pixel does not change relative x/y position in an
    image thru pile/tile operations, except for the rounding intrinsic to
    the method.

    We test this by making sure that the same pixels are present in the
    left 2 columns of the normal version as in the leftmost col of the tiled
    version. You could craft a nonlocal algorithm that could pass this
    test, but it would be hard to do by accident at the same time as its
    rotated counterpart (see below).
    """
    piled = util.pile(rir_1x8x32x32, 2)
    piled_sorted, _ = torch.sort(piled[..., 0].flatten())
    orig_sorted, _ = torch.sort(rir_1x8x32x32[..., :2].flatten())
    assert torch.equal(piled_sorted, orig_sorted)


def test_pile_preserves_locality_vertical(rir_1x8x32x32):
    # See notes on horizontal version.
    piled = util.pile(rir_1x8x32x32, 2)
    piled_sorted, _ = torch.sort(piled[..., 0, :].flatten())
    orig_sorted, _ = torch.sort(rir_1x8x32x32[..., :2, :].flatten())
    assert torch.equal(piled_sorted, orig_sorted)
