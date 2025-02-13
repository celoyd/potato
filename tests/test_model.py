import pytest
import torch
from potato import model

# Oklab color axes, for clarity later:
L = 0
a = 1
b = 2

# Color names for band indexes, for clarity later:
deep_blue, blue, green, yellow, red, deep_red = (16 + n for n in range(6))

gen = model.Potato(48)
gen.load_state_dict(torch.load("sessions/sprout/160-gen.pt", weights_only=True))


@pytest.fixture
def rir_1x24x32x32():
    return torch.rand((1, 24, 32, 32)) ** 3


def test_dark_input_makes_dark_output(rir_1x24x32x32):
    reflectance = rir_1x24x32x32.clamp(0, 0.01)
    picture = gen(reflectance)
    assert picture[0, L].mean().item() < 0.25


def test_bright_input_makes_bright_output(rir_1x24x32x32):
    reflectance = rir_1x24x32x32.clamp(0.5, 1.0)
    picture = gen(reflectance)
    assert picture[0, L].mean().item() > 0.75


def test_random_input_makes_net_grayish_output(rir_1x24x32x32):
    """
    This would only have an expected value of 0 if we used the E illuminant,
    and we don’t. In practice it seems to be around 0.07.
    """
    picture = gen(rir_1x24x32x32)
    saturation = torch.sqrt(torch.square(picture[0, a]) + torch.square(picture[0, b]))
    assert saturation.mean().item() < 0.1


"""
For these next tests we make up some reflectance functions that aren’t very 
plausible, and maybe not even physically possible. The validity of this is 
certainly debatable, because it means we’re using out-of-distribution inputs. 

Also, just to be explicit, the oklab hue orientation is:
-a = green; +a = red
-b = purple; +b = yellow
"""


def test_red_input_makes_red_output():
    red_reflectance = torch.zeros((1, 24, 32, 32))
    red_reflectance[0, :16] = 0.15  # pan bands
    red_reflectance[0, red] = 1.0
    picture = gen(red_reflectance)
    assert picture[0, a].mean().item() > 0.2


def test_green_input_makes_green_output():
    green_reflectance = torch.zeros((1, 24, 32, 32))
    green_reflectance[0, :16] = 0.15  # pan bands
    green_reflectance[0, green] = 1.0
    picture = gen(green_reflectance)
    assert picture[0, a].mean().item() < -0.2


def test_yellow_input_makes_yellow_output():
    yellow_reflectance = torch.zeros((1, 24, 32, 32))
    yellow_reflectance[0, :16] = 0.15  # pan bands
    yellow_reflectance[0, yellow] = 1.0
    picture = gen(yellow_reflectance)
    assert picture[0, b].mean().item() > 0.2


def test_purple_input_makes_purple_output():
    purple_reflectance = torch.zeros((1, 24, 32, 32))
    purple_reflectance[0, :16] = 0.15  # pan bands
    purple_reflectance[0, deep_blue] = 1.0
    purple_reflectance[0, deep_red] = 1.0
    picture = gen(purple_reflectance)
    assert picture[0, b].mean().item() < -0.2
