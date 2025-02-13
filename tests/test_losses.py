import pytest
import torch
from potato import losses

epsilon = 1e-6


@pytest.fixture
def random_oklabish_image():
    img = torch.rand((3, 256, 256))
    img[0] = img[0].abs()
    img[1:] /= 3.0
    return img


def test_ΔEOK_of_identity_is_zero(random_oklabish_image):
    assert losses.ΔEOK(random_oklabish_image, random_oklabish_image) < epsilon


def test_ΔEOK_of_small_change_is_small(random_oklabish_image):
    a_little_different = random_oklabish_image.clone()
    a_little_different[:, :, 0] = torch.roll(a_little_different[:, :, 0], 1)
    delta = losses.ΔEOK(random_oklabish_image, a_little_different).item()

    assert (delta > epsilon) and (delta < 1)


def test_ΔEOK_of_large_change_is_large(random_oklabish_image):
    very_different = torch.rot90(random_oklabish_image.clone(), dims=(-1, -2))
    delta = losses.ΔEOK(random_oklabish_image, very_different).item()

    assert delta > 1.0


def test_rfft_texture_loss_of_identity_is_zero(random_oklabish_image):
    assert (
        losses.rfft_texture_loss(random_oklabish_image, random_oklabish_image) < epsilon
    )


def test_rfft_texture_loss_of_small_change_is_small(random_oklabish_image):
    a_little_different = random_oklabish_image.clone()
    a_little_different[:, :, 0] = torch.roll(a_little_different[:, :, 0], 1)
    delta = losses.rfft_texture_loss(random_oklabish_image, a_little_different).item()

    assert (delta > epsilon) and (delta < 3)


def test_rfft_texture_lossof_large_change_is_large(random_oklabish_image):
    very_different = torch.rot90(random_oklabish_image.clone(), dims=(-1, -2))
    delta = losses.rfft_texture_loss(random_oklabish_image, very_different).item()

    assert delta > 1.0
