import torch


def oklab_saturation(x):
    return torch.sqrt(torch.square(x[:, 1]) + torch.square(x[:, 2]))


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


def rfft_loss(y, ŷ):
    fy = torch.fft.rfft(y)
    fŷ = torch.fft.rfft(ŷ)
    return torch.mean(torch.abs(fy - fŷ))


def rfft_texture_loss(y, ŷ):
    dev = y.device
    yfft = torch.abs(torch.fft.rfft2(y))
    ŷfft = torch.abs(torch.fft.rfft2(ŷ))

    h, w = yfft.shape[-2:]

    vertical = torch.arange(h, device=dev) / h
    horizontal = torch.arange(w, device=dev) / (w * 2)

    coords = torch.cartesian_prod(vertical, horizontal)
    pt = torch.tensor([0.5, 0], device=dev).unsqueeze(0)
    distances = torch.cdist(pt, coords)
    distances = distances.reshape(h, w) * 2**0.5

    mask = distances**0.5

    diff = torch.mean(torch.abs(yfft - ŷfft) * mask)

    return diff


def rfft_saturation_loss(y, ŷ):
    y_sat = oklab_saturation(y)
    ŷ_sat = oklab_saturation(ŷ)

    return rfft_texture_loss(y_sat, ŷ_sat)
