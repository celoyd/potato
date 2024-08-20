import torch
from torch.nn import Conv2d, Module
from torch.nn.functional import grid_sample, conv2d
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

upper_bands = [6, 4, 2, 1]
lower_bands = [5, 3, 0, 7]


def motion_warp(x, amt):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    b, c, h, w = x.shape
    if h != w:
        raise ValueError("Non-square image")

    idx = torch.linspace(-1, 1, h, device=x.device)
    grid = torch.stack(torch.meshgrid(idx, idx.t(), indexing="xy"), dim=-1).repeat(
        b, 1, 1, 1
    )

    amt /= w  # (h * w)**0.5

    noise = torch.normal(0, amt, (b, 2, h // 8, w // 8), device=x.device) ** 2
    noise = resize(noise, (h, w), InterpolationMode("bicubic"))

    hdist = grid[..., 0] ** 2
    wdist = grid[..., 1] ** 2
    mask = torch.sqrt(hdist + wdist).unsqueeze(1)

    mask = 1 - torch.clamp(mask, 0.0, 1.0)

    offset_field = (noise * mask).moveaxis(1, -1)  # .unsqueeze(0)

    offset_rand = torch.rand((1), device=x.device) + 0.5

    x[:, upper_bands] = grid_sample(
        x[:, upper_bands], grid + offset_field, "bicubic", align_corners=True
    )
    x[:, lower_bands] = grid_sample(
        x[:, lower_bands],
        grid - offset_field * offset_rand,
        "bicubic",
        align_corners=True,
    )

    return x


# class HaloMaker(Module):
#     def __init__(self, depth):
#         super().__init__()
#         cCc = torch.tensor([1.0, 2.0, 1.0])
#         kernel = torch.outer(cCc, cCc)
#         kernel /= kernel.sum()

#         self.blur = Conv2d(
#             depth, depth, kernel_size=3, padding=1, padding_mode="reflect", bias=False, groups=depth
#         )

#         self.blur.weight.data = kernel.view(1, 1, 3, 3).repeat(depth, 1, 1, 1)

#     def forward(self, x, mean=1.0, std=1.0):
#         self.blur = self.blur.to(x.device)
#         if len(x.shape) < 4:
#             x = x.unsqueeze(0)
#         B = x.shape[0]

#         r = torch.normal(mean, std, (B,)).view(B, 1, 1, 1).to(x.device)
#         x_blurry = self.blur(x)
#         the_blur = x_blurry - x

#         return x + (-r * the_blur)


def halo(x, mean=1.0, std=1.0):
    if len(x.shape) < 4:
        x = x.unsqueeze(0)
    B = x.shape[0]
    r = torch.normal(mean, std, (B,)).view(B, 1, 1, 1).to(x.device)

    cCc = torch.tensor([1.0, 2.0, 1.0], device=x.device)
    kernel = torch.outer(cCc, cCc)
    kernel /= kernel.sum()

    blurred = conv2d(
        x,
        kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1),
        # bias=False,
        # stride=(1, 1),
        # padding="reflect",
        padding=(1, 1),
        dilation=1,
        # padding_mode="reflect",
        groups=x.shape[1],
    )

    r = torch.normal(mean, std, (B,)).view(B, 1, 1, 1).to(x.device)
    the_blur = blurred - x

    return x + (-r * the_blur)


class HaloMaker(Module):
    def __init__(self, depth):
        super().__init__()
        cCc = torch.tensor([1.0, 2.0, 1.0])
        kernel = torch.outer(cCc, cCc)
        kernel /= kernel.sum()

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
        self.blur = self.blur.to(x.device)
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        B = x.shape[0]

        r = torch.normal(mean, std, (B,)).view(B, 1, 1, 1).to(x.device)
        x_blurry = self.blur(x)
        the_blur = x_blurry - x

        return x + (-r * the_blur)
