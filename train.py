import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur

from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode

import numpy as np

from tensorboardX import SummaryWriter

from einops import rearrange

import click
from tqdm import tqdm

from ripple.model import Ripple, concat
from ripple.util import tile, pile, cheap_half, cheap_multilayer_db1_dwt
from ripple.color import BandsToOklab
from ripple.augmentations import motion_warp
from ripple.mma_loss import get_mma_loss

from pathlib import Path

from GetDevice import getDevice

device = getDevice()

### The loss part

l2_criterion = nn.MSELoss(reduction="mean")
l1_criterion = nn.L1Loss(reduction="mean")

class ChipReader(Dataset):
    def __init__(self, chip_dir, length, offset=0):
        self.length = length
        self.offset = offset
        self.src = Path(chip_dir)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        x, y = torch.load(self.src / f"{index}.pt")
        return x, y

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


# def ΔEOK(y, ŷ, c_weight=1.0, h_weight=1.0):
#     # https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md#color-difference-metric
#     # For oklab notes please see ΔEuOK.
#     ΔL = y[:, 0] - ŷ[:, 0]
#     C1 = torch.sqrt(torch.square(y[:, 1]) + torch.square(ŷ[:, 1]) + 1e-6)
#     C2 = torch.sqrt(torch.square(y[:, 2]) + torch.square(ŷ[:, 2]) + 1e-6)
#     ΔC = C1 - C2
#     Δa = y[:, 1] - ŷ[:, 1]
#     Δb = y[:, 2] - ŷ[:, 2]
#     ΔH = torch.sqrt(torch.square(Δa) + torch.square(Δb) + torch.square(ΔC) + 1e-6)
#     return torch.sqrt(
#         torch.square(ΔL)
#         + torch.square(ΔC * c_weight)
#         + torch.square(ΔH * h_weight)
#         + 1e-6
#     ).mean()


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


# def normalized_rfft_loss(y, ŷ):
#     sl, ml = torch.std_mean(y, axis=[-1, -2], keepdim=True)

#     yfft = torch.pow(
#         torch.clamp(torch.fft.rfft2((y - ml) / sl).real, 1e-8, None), 1 / 8
#     )
#     ŷfft = torch.pow(
#         torch.clamp(torch.fft.rfft2((ŷ - ml) / sl).real, 1e-8, None), 1 / 8
#     )

#     diff = torch.mean(torch.abs(yfft - ŷfft))  # * weight

#     return diff


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


def sampling_equivariant_loss(gen, x, y):
    j_pan = tile(x[:, :16], factor=4)
    j_mul = x[:, 16:]

    pan_jitter = (torch.rand(1, device=x.device) * 2 - 1).clamp(-1, 3)
    mul_jitter = (torch.rand(1, device=x.device) * 4 - 1).clamp(-2, 6)

    j_pan = jittery_quarter(j_pan, pan_jitter)
    j_mul = jittery_quarter(j_mul, mul_jitter)

    _, _, _, jittery = gen(concat(pile(j_pan, factor=4), j_mul))

    regular = cheap_half(cheap_half(y))

    sr, mr = torch.std_mean(regular, dim=[-1, -2], keepdim=True)
    regular = (regular - mr) / sr
    jittery = (jittery - mr) / sr

    return torch.mean(torch.abs(regular - jittery))


def edgenoise_equivariant_loss(gen, x, y):

    quarter_area = nn.Upsample(scale_factor=1 / 4, mode="area")
    quarter_bicu = nn.Upsample(scale_factor=1 / 4, mode="bicubic")

    pan = tile(x[:, :16], factor=4)
    mul = x[:, 16:]

    area_mul = quarter_area(mul)
    bicu_mul = quarter_bicu(mul)
    diff_mul = torch.mean(torch.abs(area_mul - bicu_mul), dim=-3) ** 2
    diff_mul = diff_mul.unsqueeze(1)
    del bicu_mul

    noise = torch.normal(1, 0.1, area_mul.shape, device=y.device)

    noised_mul = area_mul * noise * diff_mul

    _, _, _, ŷ = gen(concat(pile(quarter_area(pan), factor=4), noised_mul))

    y = quarter_area(y)

    return torch.mean(torch.abs(y - ŷ))

    # sr, mr = torch.std_mean(regular, dim=[-1, -2], keepdim=True)
    # regular = (regular - mr) / sr
    # jittery = (jittery - mr) / sr


def jittery_quarter(x, n):
    half_area = nn.Upsample(scale_factor=1 / 2, mode="area")
    half_bicu = nn.Upsample(scale_factor=1 / 2, mode="bicubic")

    area = half_area(half_area(x))
    bicu = half_bicu(half_bicu(x))

    diff = bicu - area

    # return torch.clamp(area + diff * n.view(-1, 1, 1, 1), 0, 1)
    return area + diff * n.view(-1, 1, 1, 1)


def output_losses(y, qŷ, hŷ, d, ŷ):

    d_loss = torch.mean(torch.abs(d)) * 0.0

    t_loss = rfft_texture_loss(y, ŷ) * 2.0

    s_loss = rfft_saturation_loss(y, ŷ) * 5.0

    ok_loss = ΔEuOK(y, ŷ) * 1.0

    hy = cheap_half(y)
    qy = cheap_half(hy)

    h_loss = ΔEuOK(hy, hŷ) * 1.0
    q_loss = ΔEuOK(qy, qŷ) * 1.0

    # extremes_loss = (
    #     proportional_loss(y[:, 0], ŷ[:, 0])
    #     + proportional_loss(1 - y[:, 0], 1 - ŷ[:, 0])
    # ) * 5e-5

    return d_loss + t_loss + ok_loss + h_loss + q_loss + s_loss #+ extremes_loss


### The training part


@click.command()
@click.option("--session", default="space_heater", help="Name of training session")
@click.option("--load-epoch", default=0, help="Completed epoch to start from")
@click.option("--lr", default=5e-4, help="Learning rate")
@click.option("--epochs", default=320, help="Epochs to train for")
@click.option("--chips", default="chips", help="Chip source")
@click.option("--test-chips", default="chips", help="Test chip source")  # fixme
@click.option("--epoch-length", default=4 * 1024, help="Number of chips per epoch")
@click.option("--test-length", default=64, help="Number of chips to test on")
@click.option("--workers", default=0, help="Chip-loading processes")
def train(
    session,
    load_epoch,
    lr,
    epochs,
    chips,
    test_chips,
    epoch_length,
    test_length,
    workers,
):

    physical_batch_size = 8  # parameterize
    logical_batch_size = 64  # parameterize

    loader_params = {
        "batch_size": physical_batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": True,
    }

    trainlen = epoch_length
    testlen = test_length

    Train = ChipReader(
        chips,
        trainlen,
        0,
    )

    Test = ChipReader(test_chips, testlen, trainlen)

    trainloader = DataLoader(Train, **loader_params)
    testloader = DataLoader(Test, **loader_params)

    epoch_counter = 0

    gen = Ripple().to(device)

    try:
        torch.compile(gen)
    except:
        print("Not compiling.")

    opt = torch.optim.AdamW(gen.parameters(), lr)

    weight_path = f"weights/{session}-gen-{load_epoch}.pt"
    opt_path = f"weights/{session}-opt-{load_epoch}.pt"

    try:
        gen.load_state_dict(torch.load(weight_path))
        opt.load_state_dict(torch.load(opt_path))
    except:
        print("Starting training from scratch.")

    batch_counter = 0

    for epoch in range(epochs):
        losses = []

        with tqdm(trainloader, unit="b", mininterval=2) as progress:
            for x, y in progress:
                progress.set_description(f"Ep {epoch_counter}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x[:, 16:] = motion_warp(
                    x[:, 16:], 1 / 96
                )  # change to a function of shape

                # x[:, 16:] += torch.normal(0, 0.002, x[:, 16:].shape, device=x.device)
                # x[:, 16:] *= torch.normal(1, 0.001, x[:, 16:].shape, device=x.device)
                # x[:, :16] += torch.normal(0, 0.002, x[:, :16].shape, device=x.device)
                # x[:, :16] *= torch.normal(1, 0.001, x[:, :16].shape, device=x.device)

                x[:, 16:] += torch.normal(0, 0.005, x[:, 16:].shape, device=x.device)
                x[:, 16:] *= torch.normal(1, 0.01, x[:, 16:].shape, device=x.device)

                # x[:, :16] += torch.normal(0, 0.005, x[:, :16].shape, device=x.device)
                # x[:, :16] *= torch.normal(1, 0.001, x[:, :16].shape, device=x.device)


                qŷ, hŷ, d, ŷ = gen(x)
                main_losses = output_losses(y, qŷ, hŷ, d, ŷ) 

                # e_loss = edgenoise_equivariant_loss(gen, x, y) * 5

                damaged = jittery_quarter(
                    x, torch.normal(1.0, 1.0, (x.shape[0],), device=x.device)
                )

                qŷ, hŷ, d, ŷ = gen(damaged)
                damaged_losses = output_losses(cheap_half(cheap_half(y)), qŷ, hŷ, d, ŷ)

                mma_loss = 0.0
                for name, m in gen.named_modules():
                    if hasattr(m, "mma_norm"):
                        mma_loss += get_mma_loss(m.weight)
                mma_loss *= 0.005

                sampeq = sampling_equivariant_loss(gen, x, y) * 50

                loss = main_losses + damaged_losses + mma_loss + sampeq #+ e_loss

                loss.backward()
                losses.append(float(loss.item()))

                batch_counter += 1

                progress.set_postfix(
                    avg=f"{float(np.mean(losses)):.3f}",
                )

                if batch_counter >= (logical_batch_size / physical_batch_size):
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    batch_counter = 0

            log.add_scalars("loss", {"train": np.mean(np.array(losses))}, epoch_counter)
            log.flush()

            if epoch % 1 == 0:
                testlosses = []
                with torch.no_grad():
                    if True:
                        torch.save(
                            gen.state_dict(), f"weights/{session}-gen-{epoch}.pt"
                        )
                        torch.save(
                            opt.state_dict(),
                            f"weights/{session}-opt-{epoch}.pt",
                        )

                    for x, y in testloader:
                        progress.set_description(f"Ep {epoch_counter}")
                        x = x.to(device, non_blocking=True)
                        # mul = cheap_half(x[:, 16:])
                        # pan = pile(cheap_half(tile(x[:, :16], factor=4)), factor=4)
                        # mul = x[:, 16:]
                        # pan = tile(x[:, :16], factor=4)
                        # x = concat(pan, mul)
                        y = y.to(device, non_blocking=True)

                        gen.eval()
                        _, _, _, ŷ = gen(x)
                        # ŷ = gen(x)

                        ok_test_loss = ΔEuOK(y, ŷ) * 100
                        # wave_test_loss = big_pyramid_loss(y, ŷ) * 100
                        test_loss = ok_test_loss  # wave_test_loss

                        testlosses.append(float(test_loss.item()))
                        gen.train()

                log.add_scalars(
                    "loss", {"test": np.mean(np.array(testlosses))}, epoch_counter
                )
                log.flush()

            epoch_counter += 1


log = SummaryWriter()

if __name__ == "__main__":
    train()
