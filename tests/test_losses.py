"""Test losses."""

import torch

from potato import losses

epsilon = 1e-3


def test_ΔEOK_of_identity_is_zero(random_oklabish_image):
    """loss(x, x) should be 0."""
    assert losses.ΔEOK(random_oklabish_image, random_oklabish_image) < epsilon


def test_ΔEOK_of_small_change_is_small(random_oklabish_image):
    """loss(x, c(x)), where c is s a small change, should be small."""
    a_little_different = random_oklabish_image + (
        0.002 * torch.randn(random_oklabish_image.shape)
    )

    delta = losses.ΔEOK(random_oklabish_image, a_little_different).item()

    # Separate tests make for very slightly more useful errors.
    assert delta > epsilon
    assert delta < 1


def test_ΔEOK_of_large_change_is_large(random_oklabish_image):
    """loss(x, c(x)), where c is s a big change, should be big."""
    very_different = torch.rot90(random_oklabish_image.clone(), dims=(-1, -2))

    delta = losses.ΔEOK(random_oklabish_image, very_different).item()
    assert delta > 10


def test_ΔEOK_catches_L_alone(random_oklabish_image):
    """Tweak only the L channel."""
    weird = random_oklabish_image.clone()
    weird[0] = torch.rand(weird[0].shape) ** (1 / 3)

    delta = losses.ΔEOK(random_oklabish_image, weird).item()
    assert delta > 1


def test_ΔEOK_catches_a_alone(random_oklabish_image):
    """Tweak only the a channel."""
    weird = random_oklabish_image.clone()
    weird[1] = torch.randn(weird[1].shape)

    delta = losses.ΔEOK(random_oklabish_image, weird).item()
    assert delta > 1


def test_ΔEOK_catches_b_alone(random_oklabish_image):
    """Tweak only the b channel."""
    weird = random_oklabish_image.clone()
    weird[2] = torch.randn(weird[2].shape)

    delta = losses.ΔEOK(random_oklabish_image, weird).item()
    assert delta > 1
