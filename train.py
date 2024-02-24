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


### The color part


mul_to_xyz_matrix = torch.tensor(
    [
        [9.0677e-02, 0.0000e00, 4.0220e-01],
        [1.1873e-01, 1.5157e-01, 7.5022e-01],
        [2.0263e-01, 5.8714e-01, 0.0000e00],
        [5.0440e-01, 2.9138e-01, 1.7287e-04],
        [8.7978e-02, 2.7045e-02, 7.2419e-07],
        [0.0000e00, 3.2608e-04, 0.0000e00],
        [1.0887e-04, 1.7025e-04, 0.0000e00],
        [8.0838e-05, 5.0579e-05, 0.0000e00],
    ]
)


xyz_to_oklab_m1 = torch.tensor(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ]
)

xyz_to_oklab_m2 = torch.tensor(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def safe_pow(n, exp):
    return n.sign() * n.abs().pow(exp)


def xyz_to_lab(xyz):
    lms = xyz_to_oklab_m1 @ xyz
    lamasa = safe_pow(lms, 1 / 3)
    Lab = xyz_to_oklab_m2 @ lamasa
    return Lab


def mul_to_xyz(mul):
    return mul @ mul_to_xyz_matrix


def mul_to_lab(mul):
    xyz = torch.einsum("rhw, rx -> xhw", mul, mul_to_xyz_matrix)
    lms = torch.einsum("xhw, mx -> mhw", xyz, xyz_to_oklab_m1)
    lms = safe_pow(lms, 1 / 3)
    oklab = torch.einsum("mhw, lm -> lhw", lms, xyz_to_oklab_m2)
    return oklab


### The loss part

l2_criterion = nn.MSELoss(reduction="mean")
l1_criterion = nn.L1Loss(reduction="mean")

dtcwt = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").cuda()


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


### The chip loading (and noising) part
# This all happens on the CPU so multicore workers can do it
# while the GPU focuses on GPU stuff. Todo: numpy->torch even here?


def m_noise(shape, scale):
    # Multiplicative noise centers on 1
    return torch.normal(1.0, scale, shape, device="cpu")


def a_noise(shape, scale):
    # Additive noise centers on 0
    return torch.normal(0.0, scale, shape, device="cpu")


def cheap_half(x):
    # Fast 2× downsample
    return (
        x[..., 0::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 0::2]
        + x[..., 1::2, 1::2]
    ) / 4.0


class Chipper(Dataset):
    def __init__(self, length: int, offset: int = 0):
        self.length = length
        self.offset = offset

        self.id_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).float()

    def a_noisy_kernel(self, std=0.1):
        noise = a_noise(self.id_kernel.shape, std)
        noisy = self.id_kernel + noise
        noisy /= noisy.sum()
        noisy = noisy.expand(8, -1, -1, -1)
        return noisy

    def eight_noisy_kernels(self, std=0.1):
        identities = self.id_kernel.expand(8, -1, -1, -1)
        noise = a_noise(identities.shape, std)
        noisy = identities + noise
        sums = noisy.sum(dim=(-1, -2))
        noisy /= sums.view(8, 1, 1, 1)
        return noisy

    def blur(self, x, all_std=0.1, each_std=0.1):
        all_k = self.a_noisy_kernel(all_std)
        x = F.conv2d(x, all_k, groups=8, padding="same")

        each_k = self.eight_noisy_kernels(each_std)
        x = F.conv2d(x, each_k, groups=8, padding="same")

        return x


    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        y = torch.load(f"chips/pool/{index}.pt")

        y = y.float() / 10_000

        pan = cheap_half(shuf2(y[:16]))
        mul = cheap_half(y[16:])

        rots = int(torch.rand((1,)) * 4)

        pan = torch.rot90(pan, dims=(-1, -2), k=rots)
        mul = torch.rot90(mul, dims=(-1, -2), k=rots)


        pan_down = cheap_half(cheap_half(pan)).unsqueeze(0)
        pan_down = unshuf2(pan_down)

        mul_down = cheap_half(cheap_half(mul)).unsqueeze(0)
        mul_down = self.blur(mul_down)


        mul_down = mul_down * m_noise(mul_down.shape, scale=1 / 500) + a_noise(
            mul_down.shape, scale=1 / 1_000
        )
        pan_down = pan_down * m_noise(pan_down.shape, scale=1 / 2_500) + a_noise(
            pan_down.shape, scale=1 / 10_000
        )

        x = torch.squeeze(torch.cat([pan_down, mul_down], dim=1))

        y = mul_to_lab(mul)

        return x, y


physical_batch_size = 32
logical_batch_size = 64

loader_params = {
    "batch_size": physical_batch_size,
    "shuffle": True,
    "num_workers": 0,
    "pin_memory": True,
}

trainlen = 24 * 1024
testlen = 64

Train = Chipper(
    trainlen,
    0,
)

Test = Chipper(testlen, trainlen)

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

                simple_loss = l2_criterion(y, ŷ) * 100
                wave_loss = big_pyramid_loss(y, ŷ) * 100
                loss = wave_loss + simple_loss

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

                        simple_test_loss = l2_criterion(y, ŷ) * 100
                        wave_test_loss = big_pyramid_loss(y, ŷ) * 100

                        test_loss = wave_test_loss + simple_test_loss

                        testlosses.append(float(test_loss.item()))

                log.add_scalars("loss", {"test": np.mean(np.array(testlosses))}, te)
                log.flush()

            te += 1


# device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

log = SummaryWriter()

if __name__ == "__main__":
    train()
