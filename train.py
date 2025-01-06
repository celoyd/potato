import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np

from einops import rearrange

from potato.model import Potato, concat
from potato.util import tile, pile, cheap_half
from potato.color import BandsToOklab
from potato.augmentations import HaloMaker, WV23Misaligner
from potato.losses import rfft_texture_loss, rfft_saturation_loss, ΔEuOK, ΔEOK

from tensorboardX import SummaryWriter
import click
from tqdm import tqdm


class ChipReader(Dataset):
    def __init__(self, chip_dir, length, offset=0):
        self.length = length
        self.offset = offset
        self.src = Path(chip_dir)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        x, y = torch.load(self.src / f"{index}.pt", weights_only=False)
        return x, y


def net_loss(y, ŷ):

    t_loss = rfft_texture_loss(y, ŷ)
    s_loss = rfft_saturation_loss(y, ŷ)
    ok_loss = ΔEOK(y, ŷ)

    return t_loss + ok_loss + s_loss


@click.command()
@click.option("--session", default="space_heater", help="Name of training session")
@click.option("--load-epoch", default=0, help="Completed epoch to start from")
@click.option("--lr", default=5e-4, help="Learning rate")
@click.option("--epochs", default=320, help="Epochs to train for")
@click.option("--chips", default="chips", help="Chip source")
@click.option("--test-chips", default="chips", help="Test chip source")  # fixme
@click.option("--epoch-length", default=4096, help="Number of chips per epoch")
@click.option("--test-length", default=64, help="Number of chips to test on")
@click.option("--workers", default=0, help="Chip-loading processes")
@click.option("--device", default="cuda", help="Torch device to run on")
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
    device,
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

    gen = Potato(48).to(device)

    try:
        torch.compile(gen)
    except:
        print("Not compiling.")

    opt = torch.optim.AdamW(gen.parameters(), lr)

    weight_path = f"weights/{session}-gen-{load_epoch}.pt"
    opt_path = f"weights/{session}-opt-{load_epoch}.pt"

    try:
        gen.load_state_dict(torch.load(weight_path, weights_only=True))
        opt.load_state_dict(torch.load(opt_path, weights_only=True))
    except:
        print("Starting training from scratch.")

    batch_counter = 0

    for epoch in range(epochs):
        losses = []

        with tqdm(trainloader, unit="b", mininterval=2) as progress:
            pan_halo = HaloMaker(1, device=device)
            mul_halo = HaloMaker(8, device=device)
            misalignment = WV23Misaligner(
                side_length=128, device=device, weight_power=2.0
            )

            for x, y in progress:
                progress.set_description(f"Ep {epoch_counter}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pan = x[:, :16]
                mul = x[:, 16:]

                # pan = pan + torch.normal(0, 0.0005, pan.shape, device=x.device)
                # pan = pan * torch.normal(1, 0.001, pan.shape, device=x.device)
                # mul = mul + torch.normal(0, 0.001, mul.shape, device=x.device)
                # mul = mul * torch.normal(1, 0.0025, mul.shape, device=x.device)

                pan = pile(pan_halo(tile(pan, 4), mean=0.25, std=0.1), 4)

                mul = misalignment(mul, amt=1.0, joint_amt=0.5)

                mul = mul_halo(mul, mean=1.0, std=0.5)

                x[:, :16] = pan
                x[:, 16:] = mul

                ŷ = gen(x)

                loss = net_loss(y, ŷ)

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
                            gen.state_dict(),
                            f"weights/{session}-gen-{epoch}.pt",
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

                        ok_test_loss = ΔEuOK(y, ŷ) * 50
                        test_loss = ok_test_loss

                        testlosses.append(float(test_loss.item()))
                        gen.train()

                log.add_scalars(
                    "loss",
                    {"test": np.mean(np.array(testlosses))},
                    epoch_counter,
                )
                log.flush()

            epoch_counter += 1


log = SummaryWriter()

if __name__ == "__main__":
    train()
