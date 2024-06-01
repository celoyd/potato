import torch
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

upper_bands = [6, 4, 2, 1]
lower_bands = [5, 3, 0, 7]

def motion_warp(x, amt):
    b, c, h, w = x.shape
    if h != w:
        raise ValueError("Non-square image")

    idx = torch.linspace(-1, 1, h, device=x.device)
    grid = torch.stack(torch.meshgrid(idx, idx.t(), indexing="xy"), dim=-1).repeat(b, 1, 1, 1)

    noise = torch.normal(0, amt, (b, 2, h // 8, w // 8), device=x.device)
    noise = resize(noise, (h, w), InterpolationMode("bicubic"))
    # noise = resize(noise, (h, w), "bicubic")

    hdist = grid[..., 0] ** 2
    wdist = grid[..., 1] ** 2
    mask = torch.sqrt(hdist + wdist).unsqueeze(1)

    mask = 1 - torch.clamp(mask, 0.0, 1.0)

    offset_field = (noise * mask).moveaxis(1, -1) #.unsqueeze(0)

    offset_rand = torch.rand((1), device=x.device) + 0.5

    x[:, upper_bands] = grid_sample(
        x[:, upper_bands], grid + offset_field, "bicubic", align_corners=True
    )
    x[:, lower_bands] = grid_sample(
        x[:, lower_bands], grid - offset_field * offset_rand, "bicubic", align_corners=True
    )

    return x