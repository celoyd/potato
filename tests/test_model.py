"""Test the model on artificial inputs and on the known-difficult colors.

We do not test on real data because the repo does not ship with any.
"""

import pytest
import torch

from potato import color, model
from potato.losses import ΔEOK

# Oklab color axis names
L, a, b = (0, 1, 2)

b_to_l = color.BandsToOklab()

gen = model.Potato(48)
gen.eval()
gen.load_state_dict(torch.load("sessions/latest.pt", weights_only=True, map_location="cpu"))


def difficult_colors():
    """Parse difficult reflectances from list."""
    with open("ancillary-data/fake-chips/reflectances.csv", encoding="utf-8") as src:
        return list(
            list(float(i) / 10_000 for i in line.split(",")[:-1]) for line in src
        )


# 1. Is the model broadly, directionally correct on some skewed inputs?


def test_dark_input_makes_dark_output(random_reflectance):
    """Test the low end of the dynamic range."""
    pan, mul = random_reflectance
    pan = pan.clamp(0, 0.01)
    mul = mul.clamp(0, 0.01)
    picture = gen((pan, mul))
    assert picture[0, L].mean().item() < 0.25


def test_bright_input_makes_bright_output(random_reflectance):
    """Test the high end of the dynamic range."""
    pan, mul = random_reflectance
    pan = pan.clamp(0.5, 1.0)
    mul = mul.clamp(0.5, 1.0)
    picture = gen((pan, mul))
    assert picture[0, L].mean().item() > 0.75


# 2. Does the model introduce strong color bias?


def test_random_input_makes_net_grayish_output(random_reflectance):
    """Test that white noise looks white.

    This would have an expected value of 0 if we used the E illuminant, but we
    do not.
    """
    picture = gen(random_reflectance)
    picture_mean = picture.mean(dim=(-1, -2), keepdim=True)
    saturation = (picture_mean[0, a].square() + picture_mean[0, b].square()).sqrt()
    assert saturation.item() < 0.05


# 3. Does the model render specifically hard colors well?


@pytest.mark.parametrize("c", difficult_colors())
def test_difficult_colors(c):
    """Test the model on known-difficult reflectances.

    It’s a design limitation that we’re testing on training data here. The
    justification is: if we can make data we want the model to do well on,
    it’s more important to actually train it on that data than to test it
    rigorously. Feel free to, e.g., hold out some reflectances for testing.
    Or generate random ones!
    """
    # Mul side length for the tiny images. With my setup right now, this is
    # the dominant factor in the runtime of the whole test suite. Batching
    # would likely make everything much faster but might be a pain to set up.
    msl = 32

    pan_refl = torch.tensor(c[0])
    mul_refl = torch.tensor(c[1:])

    # Make the tensors big enough for the model to run on.
    tiny_pan_refl = pan_refl.view(1, 1, 1, 1).expand(-1, -1, 4 * msl, 4 * msl)
    tiny_mul_refl = mul_refl.view(1, 8, 1, 1).expand(-1, -1, 1 * msl, 1 * msl)

    output = gen((tiny_pan_refl, tiny_mul_refl)).detach()

    # Make the single-pixel ground truth color as big as the generated patch
    target = b_to_l(mul_refl.view(1, 8, 1, 1)).expand_as(output)

    diff = ΔEOK(target, output)

    # Here we must set a threshold past which output should fail. The loss
    # function is scaled so that 1.0 is a JND. My current best checkpoints
    # clear 1.0 for all the test colors, and clear 0.5 for 63% of test colors.
    # A reasonable person might consider the JND a natural threshold. However,
    # another reasonable person might say: “I want to know if there’s a logic,
    # config, or data bug that’s leading to a failure to converge, not whether
    # the pictures look nice 🥹🫴🖼️ in some pseudo-objective sense. I want to
    # see a test fail when and only when something is actually wrong, like
    # output is white for all inputs or something, at many JNDs.”
    #
    # So I’ve set this threshold to a sort of intermediate value that will
    # probably be much too loose for some and much too tight for others. If
    # you are reading this after unexpected test behavior and you need me to
    # say it: I give you permission to adjust this value as useful to you.

    assert diff < 2.5
