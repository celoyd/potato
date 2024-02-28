import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import GaussianBlur

import numpy as np

import ptwt
from pytorch_wavelets import DTCWTForward

from tensorboardX import SummaryWriter

from einops import rearrange

import click
from tqdm import tqdm

from ripple.model import Ripple
from ripple.util import tile, pile

from pathlib import Path

from GetDevice import getDevice

device = getDevice()

### The loss part

l2_criterion = nn.MSELoss(reduction="mean")
l1_criterion = nn.L1Loss(reduction="mean")

dtcwt = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").to(device)


def ΔEuOK(y, ŷ, a_weight=1.0, b_weight=1.0):
    """
    Ordinary 3D distance in the oklab space.

    Oklab theoretically has a JND (just-noticeable difference) of 0.02.
    As a loss function, this means that if each pixel is _individually_
    just noticeably wrong before any scaling or tonemapping, the loss
    function could be scaled up by 50× to return 1.
    """

    return torch.sqrt(
        torch.square(y[:, 0] - ŷ[:, 0])
        + torch.square(y[:, 1] - ŷ[:, 1]) * a_weight
        + torch.square(y[:, 2] - ŷ[:, 2]) * b_weight
    ).mean()


def ΔEOK(y, ŷ, c_weight=1.0, h_weight=1.0):
    # https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md#color-difference-metric
    # For oklab notes please see ΔEuOK.
    ΔL = y[:, 0] - ŷ[:, 0]
    C1 = torch.sqrt(torch.square(y[:, 1]) + torch.square(ŷ[:, 1]) + 1e-6)
    C2 = torch.sqrt(torch.square(y[:, 2]) + torch.square(ŷ[:, 2]) + 1e-6)
    ΔC = C1 - C2
    Δa = y[:, 1] - ŷ[:, 1]
    Δb = y[:, 2] - ŷ[:, 2]
    ΔH = torch.sqrt(torch.square(Δa) + torch.square(Δb) + torch.square(ΔC) + 1e-6)
    return torch.sqrt(
        torch.square(ΔL)
        + torch.square(ΔC * c_weight)
        + torch.square(ΔH * h_weight)
        + 1e-6
    ).mean()


def big_pyramid_loss(y, ŷ, wt="db8"):
    sum = torch.tensor(0.0, device=y.device)

    _, yh = dtcwt(y)
    _, ŷh = dtcwt(ŷ)

    for lev in range(len(yh)):
        sum += torch.sum(torch.abs(yh[lev] - ŷh[lev]))

    return sum / torch.prod(torch.tensor(y.shape))


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

    physical_batch_size = 8 # parameterize
    logical_batch_size = 32 # parameterize

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

    opt = torch.optim.AdamW(gen.parameters(), lr)

    weight_path = f"weights/{session}-gen-{load_epoch}.pt"
    opt_path = f"weights/{session}-opt-{load_epoch}.pt"

    try:
        gen.load_state_dict(torch.load(weight_path))
        opt.load_state_dict(torch.load(opt_path))
    except:
        pass

    batch_counter = 0

    for epoch in range(epochs):
        losses = []

        with tqdm(trainloader, unit="b", mininterval=2) as progress:
            for x, y in progress:
                progress.set_description(f"Ep {epoch_counter}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                gen.train()
                ŷ = gen(x)

                ok_loss = ΔEOK(y, ŷ, c_weight=2.5, h_weight=2.5) * 25
                wave_loss = big_pyramid_loss(y, ŷ) * 250
                loss = wave_loss + ok_loss

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
                        y = y.to(device, non_blocking=True)

                        gen.eval()
                        ŷ = gen(x)

                        ok_test_loss = ΔEOK(y, ŷ, c_weight=4, h_weight=4) * 100
                        wave_test_loss = big_pyramid_loss(y, ŷ) * 100
                        test_loss = wave_test_loss + ok_test_loss

                        testlosses.append(float(test_loss.item()))

                log.add_scalars("loss", {"test": np.mean(np.array(testlosses))}, epoch_counter)
                log.flush()

            epoch_counter += 1


log = SummaryWriter()

if __name__ == "__main__":
    train()
