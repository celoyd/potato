import pytest
import torch
from potato import model, color

# Oklab color axes, for clarity later:
L = 0
a = 1
b = 2

# Color names for band indexes, for clarity later:
deep_blue, blue, green, yellow, red, deep_red = (16 + n for n in range(6))

gen = model.Potato(48)
gen.load_state_dict(torch.load("sessions/sprout/160-gen.pt", weights_only=True))

b_to_l = color.BandsToOklab()


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
Let’s see if we’re rendering some specific reflectances->colors well.
"""


def difficult_colors():
    with open("tools/difficult-colors/colors.csv") as src:
        return list(list(float(i) for i in line.split(",")[:-1]) for line in src)


@pytest.mark.parametrize("color", difficult_colors())
def test_difficult_colors(color):
    '''
    Basically we fetch the known difficult reflectances (hard negatives) and
    test whether the model produces a color field that’s similar to the
    canonical oklab version. There’s some song and dance to make the tensors
    big enough for the model to run on.
    '''
    pan_refl = torch.tensor(color[0]) / 10_000
    mul_refl = torch.tensor(color[1:]) / 10_000

    tiny_pan_refl_x16 = pan_refl.repeat(16).reshape(1, 16, 1, 1)
    tiny_mul_refl = mul_refl.reshape((1, 8, 1, 1))

    packed_refl = torch.concat([tiny_pan_refl_x16, tiny_mul_refl], dim=1)
    packed_refl = packed_refl.repeat(1, 1, 4, 4)

    tiny_oklab_image = b_to_l(tiny_mul_refl)
    output = gen(packed_refl).detach()

    output = output.mean(dim=(-2, -1))

    # Double weighting for chroma
    diff = (
        (tiny_oklab_image[0, 0] - output[0, 0]) ** 2
        + ((tiny_oklab_image[0, 1] - output[0, 1])*2) ** 2
        + ((tiny_oklab_image[0, 2] - output[0, 2])*2) ** 2
    ) ** 0.5

    # This is roughly 2 JNDs
    assert diff.item() < 0.05

"""

# The following are obsolete but may interest the reader

For these next tests we make up some reflectance functions that aren’t very 
plausible, and maybe not even physically possible. The validity of this is 
certainly debatable, because it means we’re using out-of-distribution inputs. 

Also, just to be explicit, the oklab hue orientation is:
-a = green; +a = red
-b = purple; +b = yellow

def test_red_input_makes_red_output():
    red_reflectance = torch.zeros((1, 24, 32, 32))
    red_reflectance[0, :16] = 0.15  # pan bands
    red_reflectance[0, red] = 1.0
    picture = gen(red_reflectance)
    assert picture[0, a].mean().item() > 0.15


def test_green_input_makes_green_output():
    green_reflectance = torch.zeros((1, 24, 32, 32))
    green_reflectance[0, :16] = 0.15  # pan bands
    green_reflectance[0, green] = 1.0
    picture = gen(green_reflectance)
    assert picture[0, a].mean().item() < -0.15


def test_yellow_input_makes_yellow_output():
    yellow_reflectance = torch.zeros((1, 24, 32, 32))
    yellow_reflectance[0, :16] = 0.15  # pan bands
    yellow_reflectance[0, yellow] = 1.0
    picture = gen(yellow_reflectance)
    assert picture[0, b].mean().item() > 0.15


def test_purple_input_makes_purple_output():
    purple_reflectance = torch.zeros((1, 24, 32, 32))
    purple_reflectance[0, :16] = 0.15  # pan bands
    purple_reflectance[0, deep_blue] = 1.0
    purple_reflectance[0, deep_red] = 1.0
    picture = gen(purple_reflectance)
    assert picture[0, b].mean().item() < -0.15
"""