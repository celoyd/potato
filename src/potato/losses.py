import torch
from potato.util import cheap_half

def remap(t, a, b, x, y):
    return x + (t - a) * (y - x) / (b - a)

def oklab_saturation(x):
    return torch.sqrt(torch.square(x[:, 1]) + torch.square(x[:, 2]))

def sat_loss(y, ŷ):
    return (oklab_saturation(y) - oklab_saturation(ŷ)).abs().mean()

def ΔEOK(y, ŷ, ab_weight=2.0):
    """
    Ordinary 3D distance in the oklab space.

    Oklab theoretically has a JND (just-noticeable difference) of ~0.02.
    As a loss function, this means that if each pixel is _individually_
    just noticeably wrong before any scaling or tonemapping, the loss
    function could be scaled up by 50× to return 1.

    ab_weight gives more (or less) weight to the chroma plane. This lets
    us implement what’s sometimes called “deltaEOK2” as the default.
    """

    return (
        torch.square(y[:, 0] - ŷ[:, 0])
        + ab_weight
        * (torch.square(y[:, 1] - ŷ[:, 1]) + torch.square(y[:, 2] - ŷ[:, 2]))
    ).sqrt().mean() * 50

def detail_loss(y, ŷ):
    dev = y.device

    fty = torch.abs(torch.fft.rfft2(y))
    ftŷ = torch.abs(torch.fft.rfft2(ŷ))

    h, w = fty.shape[-2:]

    fty = fty.roll(h//2, 0)
    ftŷ = ftŷ.roll(h//2, 0)

    u, v = torch.meshgrid(
        torch.arange(h, device=dev) / h,
        (torch.arange(w, device=dev) / w - 0.5) * 2,
        indexing="ij",
    )

    dist = torch.sqrt(torch.square(u) + torch.square(v))
    mask = remap(dist, 6/16, 7/16, 0, 1).clip(0, 1)

    diff = (mask*fty - mask*ftŷ)
    return diff.abs().mean()

def sat_detail_loss(y, ŷ):
    return detail_loss(oklab_saturation(y), oklab_saturation(ŷ))
