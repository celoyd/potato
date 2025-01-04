import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import grid_sample, conv2d
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


class WV23Misaligner(Module):
    """
        Simulate multispectral band misalignments, including motion
        artifacts (a.k.a. rainbowing) in WorldView-2/3 data

        See __init__() and forward() for technical notes.

        TODOs:
        1. This is stupendously memory-hungry right now.
        - square -> circle offset variation

    think about what makes sense
        to set at init v. at call.
        1. Right now we err on the side of too few options and too
           much hardcoding of parameters at non–rigorously-determined
           constant values.

        We want to accomplish two basic things here:
        1. Apply some random warps (like torchvison’s ElasticTransform),
           but rolled off toward the image sides to avoid edge effects.
           This simulates the multispectral bands not perfectly aligning
           with the panchromatic band. Call this joint warp.
        2. Much the same but with two sets of bands separately – the
           upper and lower bands, named for their placement to either
           side of the panchromatic band on the satellite’s sensor.
           These tend to symmetrically diverge from the pan band where
           there is motion (e.g., cars, planes) or departure from the
           modeled gound surface (e.g., clouds, skyscrapers, planes).
           Call this split warp.

        (An excellent paper to build some intutions about split warp is
        “Exploiting Satellite Focal Plane Geometry for Automatic Extraction
        of Traffic Flow from Single Optical Satellite Imagery” by T. Krauß
        (http://dx.doi.org/10.5194/isprsarchives-XL-1-179-2014). It’s open
        access and has the best illustrations I’ve seen on this topic.)

        The implementation is best understood line by line, because it
        involves a bunch of inter-related steps, pytorch quirks,
        optimizations, and so on. But basically we generate a weight
        field that peaks in the center, then multiply that with
        smoothed noise to generate an offset field for grid_sample.
    """

    def __init__(self, side_length, device, weight_power=2.0):
        super().__init__()
        self.device = device

        self.C = 8

        # Non-square input will raise in forward().
        self.side_length = side_length

        self.upper_bands = [6, 4, 2, 1]
        self.lower_bands = [5, 3, 0, 7]

        # grid_warp expects what are sometimes called “normalized
        # coordinates”, which are like u,v coordinates except they
        # go from -1 to 1. So we build a big coordinate tensor.

        ramp = torch.linspace(
            -1, 1, self.side_length, device=self.device, requires_grad=False
        )
        self.grid = torch.stack(torch.meshgrid(ramp, ramp, indexing="xy"), dim=-1)

        v = self.grid[..., 0]
        u = self.grid[..., 1]
        self.center_weight = (
            ((1 - torch.sqrt(u**2 + v**2)) ** weight_power).clamp(0, 1)
            # .unsqueeze(0)
        )
        self.small_noise_shape = (2, side_length // 8, side_length // 8)
        # print(f"{self.center_weight.shape = }")

        # self.small_noise = torch.empty(
        # (2, side_length // 8, side_length // 8), device=self.device
        # )
        # self.noise = torch.empty(
        # (2, side_length, side_length), device=self.device
        # )

        # We could save some space by juggling some of this data
        # between fewer tensors, but it would add operations and
        # make people trying to understand this justifiably angry.
        # self.upper_offsets = torch.empty((side_length, side_length, 2), device=self.device)
        # self.lower_offsets = torch.empty((side_length, side_length, 2), device=self.device)

    def forward(self, x, amt, joint_amt):
        self.check_shape(x.shape)

        res = torch.zeros_like(x)

        # N, _, H, W = x.shape
        N = x.shape[0]
        amt = amt / self.side_length

        small_noise = torch.normal(0, amt, self.small_noise_shape, device=self.device)

        noise = resize(
            small_noise,
            (self.side_length, self.side_length),
            InterpolationMode("bicubic"),
        )

        # From image layout to warp-field layout
        noise = noise.swapaxes(1, -1)

        # joint_offset = noise * joint_amt

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
