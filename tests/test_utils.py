"""Test functions from the util module.

We skip the Chip class because it’s data-dependent.
"""

import pytest
import torch

from potato import util


def test_cheap_half(random_mul):
    """Downsampling should not change mean values."""
    assert util.cheap_half(random_mul).mean() == pytest.approx(random_mul.mean())


# Check tile and pile for self-consistency


def test_tile2_tile2_is_tile4():
    """Multiple small tile ops should give the same as one big one.

    We don’t use a proper fixture because a depth of 8 channels isn’t enough
    to test double tiling.
    """
    noise = torch.rand((1, 16, 24, 24))
    assert torch.equal(
        util.tile(util.tile(noise, 2), 2),
        util.tile(noise, 4),
    )


def test_pile2_pile2_is_pile4(random_mul):
    """Multiple small pile ops should give the same as one big one."""
    assert torch.equal(
        util.pile(util.pile(random_mul, 2), 2),
        util.pile(random_mul, 4),
    )


# Round-trip some tile/pile operations to check cross-consistency (in other
# words, tile undoes pile and vice versa).


def test_pile2_of_tile2_roundtrip(random_mul):
    """Piling something after tiling it should be a no-op."""
    assert torch.equal(util.pile(util.tile(random_mul, 2), 2), random_mul)


def test_tile2_of_pile2_roundtrip(random_mul):
    """Tiling something after piling it should be a no-op."""
    assert torch.equal(util.tile(util.pile(random_mul, 2), 2), random_mul)


def test_tile2_tile2_tile2_of_pile8_roundtrip(random_mul):
    """Multiple separate tiles should reverse one big tile."""
    assert torch.equal(
        util.tile(util.tile(util.tile(util.pile(random_mul, 8), 2), 2), 2),
        random_mul,
    )


# Locality tests
#
# It’s very important for our purposes that pile/tile are local operations in
# the sense that a pixel does not change relative (u,v) position in an image
# thru pile/tile operations, except for the quantizing intrinsic to the
# method.
#
# That is, if a pixel is at position x, y, then when its tensor is trans-
# formed with pile(..., n), the pixel should be at some dpeth in position
# x//n, y//n. Going the other way, if we start at x, y and tile(..., n) is
# applied to our tensor, we must end up somewhere in the interval (nx, nx+n],
# (ny, ny+n], where (a, b] means >= a, < b.
#
# We test this by making sure that the same pixels are present in the left 2
# columns of the raw version as in the leftmost col of the tiled version. An
# adversary could certainly craft a nonlocal algorithm that could pass this
# test, but it would be hard to do by accident, and especially not at the same
# time as its rotated counterpart (see below).


def test_pile_preserves_locality_horizontal(random_mul):
    """Pile should leave things near their relative horizontal positions."""
    piled = util.pile(random_mul, 2)
    piled_sorted, _ = torch.sort(piled[..., 0, :].flatten())
    orig_sorted, _ = torch.sort(random_mul[..., :2, :].flatten())
    assert torch.equal(piled_sorted, orig_sorted)


def test_pile_preserves_locality_vertical(random_mul):
    """Pile should leave things near their relative vertical positions."""
    piled = util.pile(random_mul, 2)
    piled_sorted, _ = torch.sort(piled[..., 0].flatten())
    orig_sorted, _ = torch.sort(random_mul[..., :2].flatten())
    assert torch.equal(piled_sorted, orig_sorted)


# Valid fraction tests


def test_valid_fraction_all_valid(random_mul):
    """All nonzero pixels should have validity 1.0."""
    test_article = (random_mul * 10_000).clip(1, None).to(torch.uint16)
    assert util.valid_fraction(test_article) == 1.0


def test_valid_fraction_all_invalid():
    """All zero pixels should have validity 0.0."""
    assert util.valid_fraction(torch.zeros((1, 8, 256, 256), dtype=torch.uint16)) == 0.0


def test_valid_fraction_spatial_half_valid(random_mul):
    """Half zero pixels (striped on the width axis) should give 0.5."""
    test_article = (random_mul * 10_000).clip(1, None).to(torch.uint16)
    test_article[:, :, :, ::2] = 0
    assert util.valid_fraction(test_article) == 0.5


def test_valid_fraction_spatial_half_valid_pan(random_pan):
    """Make sure we’re good on single-band images."""
    test_article = (random_pan * 10_000).clip(1, None).to(torch.uint16)
    test_article[:, :, :, ::2] = 0
    assert util.valid_fraction(test_article) == 0.5


def test_valid_fraction_channelwise_half_valid(random_mul):
    """Half zero pixels (striped on the channel axis) should give 0.5."""
    test_article = (random_mul * 10_000).clip(1, None).to(torch.uint16)
    test_article[:, ::2, :, :] = 0
    assert util.valid_fraction(test_article) == 0.5
