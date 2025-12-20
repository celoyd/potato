"""Test the methods from potato.augmentations."""

import pytest

from potato.augmentations import HaloMaker, RandomD4, WV23Misaligner

pan_halo = HaloMaker(1)
mul_halo = HaloMaker(8)
misalign = WV23Misaligner(side_length=64)  # matching fixture in conftest.py
flip_n_spin = RandomD4()


## 1. Haloing


def test_pan_null_halo_is_identity(random_pan):
    """No halo should mean no change on a pan band."""
    test_article = pan_halo(random_pan, mean_sharpening=0.0, std=0.0)
    assert 0.0 == pytest.approx((test_article - random_pan).abs().mean())


def test_mul_null_halo_is_identity(random_mul):
    """No halo should mean no change on multispectral bands."""
    test_article = mul_halo(random_mul, mean_sharpening=0.0, std=0.0)
    assert 0.0 == pytest.approx((test_article - random_mul).abs().mean())


def test_pan_big_halo_is_big(random_pan):
    """A halo should mean a change on a pan band."""
    test_article = pan_halo(random_pan, mean_sharpening=2.0, std=0.0)
    assert (test_article - random_pan).abs().mean() > 0.05


def test_mul_big_halo_is_big(random_mul):
    """A halo should mean a change on multispectral bands."""
    test_article = mul_halo(random_mul, mean_sharpening=2.0, std=0.0)
    assert (test_article - random_mul).abs().mean() > 0.05


## 2. Misalignment


def test_null_misalignment_is_identity(random_mul):
    """No misalignment should mean no change."""
    test_article = misalign(random_mul, amount=0, spikiness=1.0)
    assert 0.0 == pytest.approx((test_article - random_mul).abs().mean(), abs=0.1)


def test_mul_misalignment_but_big_spikiness_is_identity(random_mul):
    """No misalignment should mean no change even with a big exponent."""
    test_article = misalign(random_mul, amount=0, spikiness=10.0)
    assert 0.0 == pytest.approx((test_article - random_mul).abs().mean(), abs=0.1)


def test_big_misalignment_is_big(random_mul):
    """Heavy misalignment should mean a large change."""
    test_article = misalign(random_mul, amount=500.0, spikiness=5.0)
    assert (test_article - random_mul).abs().mean() > 0.001


## 3. Rotation/flips


def test_d4_changes_pan(random_chip):
    """D4 actions should make changes."""
    diffs = 0.0
    for _ in range(10):
        augmented = flip_n_spin(random_chip)
        diffs += (random_chip[0][0] - augmented[0][0]).abs().mean()
    assert diffs > 0.0


def test_d4_changes_mul(random_chip):
    """D4 actions should make changes."""
    diffs = 0.0
    for _ in range(10):
        augmented = flip_n_spin(random_chip)
        diffs += (random_chip[0][1] - augmented[0][1]).abs().mean()
    assert diffs > 0.0


def test_d4_changes_oklab(random_chip):
    """D4 actions should make changes."""
    diffs = 0.0
    for _ in range(10):
        augmented = flip_n_spin(random_chip)
        diffs += (random_chip[1] - augmented[1]).abs().mean()
    assert diffs > 0.0
