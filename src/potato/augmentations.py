import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


class WV23Misaligner(Module):
    """
    Simulate multispectral band misalignments, including motion
    artifacts (a.k.a. rainbowing) in WorldView-2/3 data.

    See __init__() and forward() for technical notes.

    ## General todos

    1. Reduce unnecessary memory hunger and traffic, keeping reasonable
       clarity. For example, small_noise could be a reused instance variable
       instead of creating a new tensor every forward().
    2. Specify amount in pixels, so changining chip size doesn’t break things.
    3. Make things less hard-coded and more configurable.

    ## What this is all about

    The artifact that we’re trying to emulate here is band misalignment that
    stems from half the multispectral bands seeing the grounf just before the
    pan band does, and the other half just after. This means that image of
    things in motion or not at the modeled ground surface tend to split
    symmetrically away from the pan band in the multispectral band groups.

    (An excellent paper to build some intutions about all of this is
    “Exploiting Satellite Focal Plane Geometry for Automatic Extraction
    of Traffic Flow from Single Optical Satellite Imagery” by T. Krauß
    (http://dx.doi.org/10.5194/isprsarchives-XL-1-179-2014). It’s open
    access and has the best illustrations I’ve seen on this topic.)

    To emulate this, we generate an offset field that adds random smooth
    distortions to the image, then (a) roll it off at the image edges so we
    don’t have to think about pulling in outside pixels and (b) apply it to
    half the channels in the positive direction and the other half in the
    negtive direction.

    An ealier version of this code also did a “joint” offset, (i.e., a difference
    of all the multispectral bands at once v. the panchromatic band), but this
    is a rare enough problem that training with it as an augmentation seemed
    to attract more artifacts than it cleared away. There may still be a few
    traces of it.
    """

    def __init__(self, side_length, device, weight_power=2.0):
        super().__init__()
        self.device = device

        # number of channels
        self.C = 8

        # non-square input will raise in forward()
        self.side_length = side_length

        # layout of the WV-2 and -3 sensors
        self.upper_bands = [6, 4, 2, 1]
        self.lower_bands = [5, 3, 0, 7]

        # grid_warp() expects what are sometimes called “normalized
        # coordinates”, which are like u,v coordinates except they
        # go from -1 to 1. So we build a big coordinate tensor.

        ramp = torch.linspace(
            -1, 1, self.side_length, device=self.device, requires_grad=False
        )
        self.grid = torch.stack(torch.meshgrid(ramp, ramp, indexing="xy"), dim=-1)

        v = self.grid[..., 0]
        u = self.grid[..., 1]
        self.center_weight = ((1 - torch.sqrt(u**2 + v**2)) ** weight_power).clamp(0, 1)
        self.small_noise_shape = (2, side_length // 8, side_length // 8)

    def forward(self, x, amt):
        self.check_shape(x.shape)

        res = torch.zeros_like(x)

        N = x.shape[0]

        # FIXME: make this in terms of pixels (absolute, not relative to side)
        amt = amt / self.side_length

        small_noise = torch.normal(0, amt, self.small_noise_shape, device=self.device)

        noise = resize(
            small_noise,
            (self.side_length, self.side_length),
            InterpolationMode("bicubic"),
        )

        # from image layout to warp-field layout – clean this up
        noise = noise.swapaxes(1, -1)
        upper_offset = (noise * self.center_weight).swapaxes(-1, 0)
        lower_offset = (-1 * noise * self.center_weight).swapaxes(-1, 0)

        res[:, self.upper_bands] = grid_sample(
            x[:, self.upper_bands],
            self.grid.repeat(N, 1, 1, 1) + upper_offset,
            "bicubic",
            padding_mode="reflection",
            align_corners=False,
        )

        res[:, self.lower_bands] = grid_sample(
            x[:, self.lower_bands],
            self.grid.repeat(N, 1, 1, 1) + lower_offset,
            "bicubic",
            padding_mode="reflection",
            align_corners=False,
        )

        return res

    def check_shape(self, s):
        if s[1] != self.C:
            raise ValueError(f"Wrong band count. Expected {self.C} but got {s[1]}.")
        if s[2] != s[3]:
            raise ValueError(f"Non-square image: height {s[2]} is not width {s[3]}.")
        if s[2] != self.side_length:
            raise ValueError(
                (
                    "Wrong pixel dimension. Expected side length "
                    f"{self.side_length} but got {s[2]}."
                )
            )


class HaloMaker(Module):
    """
    Emulate a ringing point spread function for resampling.

    We create a binomial kernel (a small blur), use it to make a blurry
    version of the input, make a smooth random mask, and, modulated by the
    smooth random mask, subtract out some amount of the blurred version
    from the original version. This gives us an image that is oversharp
    by random amounts in different parts of the image. (It can also be a
    bit undersharp.)

    ## Todos
    1. Better docs, variable names, and general clarity.
    """

    def __init__(self, depth, device):
        super().__init__()
        self.device = device
        cCc = torch.tensor([1.0, 2.0, 1.0], device=self.device, requires_grad=False)
        kernel = torch.outer(cCc, cCc)
        kernel = kernel / kernel.sum()

        self.blur = Conv2d(
            depth,
            depth,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=False,
            groups=depth,
        )

        self.blur.weight.data = kernel.view(1, 1, 3, 3).repeat(depth, 1, 1, 1)

    def forward(self, x, mean=1.0, std=1.0):
        B = x.shape[0]
        r = torch.normal(mean, std, (B,), device=self.device).view(B, 1, 1, 1)
        x_blurry = self.blur(x)
        the_blur = x_blurry - x
        return x + (-r * the_blur)
