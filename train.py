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

from model.ripple import Ripple
from model.ripple import shuf2, unshuf2

from pathlib import Path


### The loss part

l2_criterion = nn.MSELoss(reduction="mean")
l1_criterion = nn.L1Loss(reduction="mean")

dtcwt = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").cuda()


def saturation_loss(y, ŷ):
    ysat = torch.sqrt(torch.square(y[:, 0]) + torch.square(y[:, 1]))
    ŷsat = torch.sqrt(torch.square(ŷ[:, 0]) + torch.square(ŷ[:, 1]))
    return torch.mean(torch.abs(ysat - ŷsat))

# def oklab_Δ_Euclidean(y, ŷ):
#     diff = y - ŷ
#     diff = torch.square(diff)
#     diff = torch.sum()

def oklab_ΔEOK(y, ŷ):
    # https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md#color-difference-metric
    ΔL = y[:, 0] - ŷ[:, 0]
    C1 = torch.sqrt(torch.square(y[:, 1]) + torch.square(y[:, 2]))
    C2 = torch.sqrt(torch.square(ŷ[:, 1]) + torch.square(ŷ[:, 2]))
    ΔC = C1 - C2
    Δa = y[:, 1] - ŷ[:, 1]
    Δb = y[:, 2] - ŷ[:, 2]
    ΔH = torch.sqrt(torch.square(Δa) + torch.square(Δb) + torch.square(ΔC))
    return torch.sqrt(torch.square(ΔL) + torch.square(ΔC) + torch.square(ΔH)).mean()

def big_pyramid_loss(y, ŷ, wt="db8", chroma_weight=8, highres_weight=2):
    sum = torch.tensor(0.0, device=y.device)

    # y[:, 1:] *= chroma_weight
    # ŷ[:, 1:] *= chroma_weight

    yl, yh = dtcwt(y)
    ŷl, ŷh = dtcwt(ŷ)

    sum = torch.sum(torch.abs(yl - ŷl))

    for lev in range(len(yh)):
        sum += torch.sum(torch.abs(yh[lev] - ŷh[lev])) * highres_weight

    return sum / torch.prod(torch.tensor(y.shape))


### The chip loading part

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


physical_batch_size = 8
logical_batch_size = 32

loader_params = {
    "batch_size": physical_batch_size,
    "shuffle": True,
    "num_workers": 4,
    "pin_memory": True,
}

trainlen = 3 * 1024
testlen = 64

Train = ChipReader(
    "chips2",
    trainlen,
    0,
)

Test = ChipReader("chips2", testlen, trainlen)

trainloader = DataLoader(Train, **loader_params)
testloader = DataLoader(Test, **loader_params)


### The training part

@click.command()
@click.option("--session", default="space_heater", help="Name of training session")
@click.option("--load-epoch", default=0, help="Completed epoch to start from.")
@click.option("--lr", default=5e-4, help="Learning rate.")
@click.option("--epochs", default=320, help="Epochs to train for.")
def train(session, load_epoch, lr, epochs):

    device = torch.device("cuda:0")  # FIXME
    te = 0

    gen = Ripple().cuda()

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

        ct = 0

        with tqdm(trainloader, unit=" b", mininterval=2) as progress:
            for x, y in progress:
                progress.set_description(f"Ep {te}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                gen.train()
                ŷ = gen(x)

                #simple_loss = l2_criterion(y, ŷ) * 500
                ok_loss = oklab_ΔEOK(y, ŷ) * 100
                wave_loss = big_pyramid_loss(y, ŷ) * 100
                sat_loss = saturation_loss(y, ŷ) * 50
                loss = wave_loss + sat_loss + ok_loss # simple_loss

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

            log.add_scalars("loss", {"train": np.mean(np.array(losses))}, te)
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
                        progress.set_description(f"Ep {te}")
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        gen.eval()
                        ŷ = gen(x)

                        ok_test_loss = oklab_ΔEOK(y, ŷ) * 100
                        #simple_test_loss = l2_criterion(y, ŷ) * 500
                        wave_test_loss = big_pyramid_loss(y, ŷ) * 100
                        sat_test_loss = saturation_loss(y, ŷ) * 50
                        test_loss = wave_test_loss + sat_test_loss + ok_test_loss # simple_test_loss

                        testlosses.append(float(test_loss.item()))

                log.add_scalars("loss", {"test": np.mean(np.array(testlosses))}, te)
                log.flush()

            te += 1

log = SummaryWriter()

if __name__ == "__main__":
    train()
