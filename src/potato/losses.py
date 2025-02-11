import torch

# Not tested with float16
epsilon = 1e-9


def oklab_saturation(x):
    return torch.sqrt(x[:, 1] * x[:, 1] + x[:, 2] * x[:, 2])


def sat_diff(y, ŷ):
    y_sat = oklab_saturation(y)
    ŷ_sat = oklab_saturation(ŷ)
    return torch.mean(torch.abs(y_sat - ŷ_sat))


def ΔEuOK(y, ŷ, a_weight=1.0, b_weight=1.0):
    """
    Ordinary 3D distance in the oklab space.

    Oklab theoretically has a JND (just-noticeable difference) of 0.02.
    As a loss function, this means that if each pixel is _individually_
    just noticeably wrong before any scaling or tonemapping, the loss
    function could be scaled up by 50× to return 1.
    """

    return (
        torch.sqrt(
            torch.square(y[:, 0] - ŷ[:, 0])
            + torch.square(y[:, 1] - ŷ[:, 1]) * a_weight
            + torch.square(y[:, 2] - ŷ[:, 2]) * b_weight
        )
    ).mean() * 50


def ΔEOK(y, ŷ, c_weight=1.0, h_weight=1.0):
    # https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md#color-difference-metric
    # Variables named to follow that reference.

    L1, a1, b1 = y[:, 0], y[:, 1], y[:, 2]
    L2, a2, b2 = ŷ[:, 0], ŷ[:, 1], ŷ[:, 2]

    ΔL = L1 - L2

    C1 = torch.nan_to_num(torch.sqrt(torch.square(a1) + torch.square(b1)))
    C2 = torch.nan_to_num(torch.sqrt(torch.square(a2) + torch.square(b2)))

    ΔC = C1 - C2

    Δa = a1 - a2
    Δb = b1 - b2

    ΔH = torch.nan_to_num(
        torch.sqrt(torch.square(Δa) + torch.square(Δb) - torch.square(ΔC))
    )

    return (
        torch.nan_to_num(
            torch.sqrt(
                torch.square(ΔL)
                + torch.square(ΔC * c_weight)
                + torch.square(ΔH * h_weight)
            )
        ).mean()
        * 50.0
    )  # 50 is a JND – see ΔEuOK’s comment


def proportional_loss(y, ŷ):
    return torch.mean(torch.abs((ŷ / (y + 1e-8)) - 1))


def normalized_loss(y, ŷ):
    sl, ml = torch.std_mean(y[:, 0])
    dl = torch.mean(torch.abs(((y[:, 0] - ml) / sl) - ((ŷ[:, 0] - ml) / sl)))

    sa, ma = torch.std_mean(y[:, 1])
    da = torch.mean(torch.abs(((y[:, 1] - ma) / sa) - ((ŷ[:, 1] - ma) / sa)))

    sb, mb = torch.std_mean(y[:, 2])
    db = torch.mean(torch.abs(((y[:, 2] - mb) / sb) - ((ŷ[:, 2] - mb) / sb)))

    return (dl * dl + da * da + db * db).sqrt()


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
