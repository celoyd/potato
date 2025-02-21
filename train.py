from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from potato.model import Potato
from potato.util import tile, pile
from potato.augmentations import HaloMaker, WV23Misaligner
from potato.losses import ΔEOK, detail_loss, sat_detail_loss

from tensorboardX import SummaryWriter

import click

from tqdm import tqdm

import datetime


class ChipReader(Dataset):
    def __init__(self, chip_dir, length, offset=0):
        self.length = length
        self.offset = offset
        self.dir = Path(chip_dir)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x, y = torch.load((self.dir / f"{index}.pt").absolute(), weights_only=False)
        return x, y


class Session(object):
    def __init__(self, name, load_from, dir="sessions", logical_epoch=0):
        self.name = name

        # self.dir is the base-base directory; we use
        # self.dir / self.name to get the actual seshdir
        self.dir = Path(dir)
        self.dir.mkdir(exist_ok=True)

        self.physical_epoch = 0
        self.logical_epoch = 0

        if self.check_session(self.name):  # this is an existing sesh
            latest = self.get_latest_epoch(self.name)
            if latest >= 0:
                self.starters = self.get_paths(self.name, latest, should_exist=True)
                self.logical_epoch = latest + 1
        else:  # new sesh
            (self.dir / self.name).mkdir(exist_ok=False)

        if load_from:
            load_session, load_epoch = self.parse_load_from(load_from)
            self.starters = self.get_paths(load_session, load_epoch, should_exist=True)

            if load_session == self.name:  # this is a continuation
                self.logical_epoch = load_epoch + 1  # debatable

    def has_starters(self):
        return hasattr(self, "starters")

    def get_starters(self):
        return (torch.load(w, weights_only=True) for w in self.starters)

    def parse_load_from(self, load_from):
        if "/" in load_from:
            try:
                load_session, load_epoch = load_from.split("/")
                load_epoch = int(load_epoch)
            except ValueError:
                raise ValueError(
                    "Expected --load-from to look like sesh or sesh/1 "
                    f"(name[/number]) but got {load_from}."
                )
        else:
            load_session = load_from
            load_epoch = self.get_latest_epoch(load_session)
        return load_session, load_epoch

    def get_latest_epoch(self, sesh):
        ckpts = list((self.dir / sesh).glob("*.pt"))

        if len(ckpts) == 0:
            return -1

        numbers = [int(p.name.split("-")[0]) for p in ckpts]
        last = sorted(numbers)[-1]

        # Abusing this as a check :/
        _ = self.get_paths(sesh, last, should_exist=True)

        return last

    def check_session(self, sesh):
        return (self.dir / sesh).is_dir()

    def get_paths(self, sesh, n, should_exist):
        gen_path, opt_path = (
            Path(self.dir / sesh / f"{n}-{k}.pt") for k in ("gen", "opt")
        )

        gen_exists, opt_exists = (path.exists() for path in (gen_path, opt_path))

        if gen_exists != opt_exists:
            raise FileNotFoundError(f"Only one of {gen_path} or {opt_path} exists!")

        epoch_exists = gen_exists
        if epoch_exists == should_exist:
            return gen_path, opt_path
        elif epoch_exists and not should_exist:
            raise FileExistsError(f"{sesh}/{n} already exists.")
        elif should_exist and not epoch_exists:
            raise FileNotFoundError(f"{sesh}/{n} does not exist.")

    def get_next_paths(self):
        return self.get_paths(self.name, self.logical_epoch, should_exist=False)

    def finish_epoch(self):
        self.logical_epoch += 1


@click.command()
@click.option("--session", required=True, help="Name of training session")
@click.option("--load-from", help="Checkpoint to start off: <session>/<number>")
@click.option("--lr", "--learning-rate", default=5e-4, help="Optimizer learning rate")
@click.option(
    "--physical-batch-size", "--pbs", default=8, help="Number of tiles to load at once"
)
@click.option(
    "--logical-batch-size", "--lbs", default=64, help="Backprop every this many tiles"
)
@click.option("--epochs", type=int, help="Epochs to train for")
@click.option(
    "--up-to-epoch",
    type=int,
    help="Stop just before this logical epoch (conflicts with --epochs)",
)
@click.option(
    "--chips", default="chips", type=click.Path(exists=True), help="Chip source"
)
@click.option(
    "--test-chips",
    default="test-chips",
    type=click.Path(exists=True),
    help="Test chip source",
)
@click.option("--train-length", default=4096, help="Number of chips per epoch")
@click.option("--test-length", default=64, help="Number of chips to test on")
@click.option(
    "--checkpoints", default="sessions", help="Where to save session checkpoints"
)
@click.option("--workers", default=0, help="Chip-loading workers")
@click.option("--logs", default="logs", help="TensorboardX log directory")
@click.option("--device", default="cuda", help="Torch device to run on")
@click.option(
    "--compile",
    is_flag=True,
    default=False,
    help="Compile model with JIT (only works on some devices)",
)
@click.option(
    "--agenda",
    is_flag=True,
    default=False,
    help="Print session parameters before starting",
)
def train(
    session,
    load_from,
    lr,
    physical_batch_size,
    logical_batch_size,
    epochs,
    up_to_epoch,
    chips,
    test_chips,
    train_length,
    test_length,
    checkpoints,
    workers,
    logs,
    device,
    compile,
    agenda,
):
    """
    Potato trainer script.

    Typical usage:
    
    \b
    $ python train.py --session yukon-silver --load-from yukon-gold/49 \
--chips chips --test-chips /media/ch/uaru/lc7/chido24 --lr 1e-4 \
--train-length 5120 --epochs 24 --agenda
    """

    # Set up the chip loaders.
    loader_params = {
        "batch_size": physical_batch_size,
        "num_workers": workers,
        "pin_memory": True,
    }

    Train = ChipReader(chips, train_length)
    Test = ChipReader(test_chips, test_length)

    trainloader = DataLoader(Train, **loader_params, shuffle=True)
    testloader = DataLoader(Test, **loader_params, shuffle=False)

    # Set up the model and optimizer so we can load their weights.
    gen = Potato(48).to(device)
    opt = torch.optim.AdamW(gen.parameters(), lr, weight_decay=0.025)

    sesh = Session(session, load_from, dir=checkpoints)

    if sesh.has_starters():
        gen_weights, opt_weights = sesh.get_starters()
        gen.load_state_dict(gen_weights)
        opt.load_state_dict(opt_weights)

    if up_to_epoch:
        if epochs:
            raise ValueError("Can’t have both --epochs and --up-to-epoch.")
        epochs = up_to_epoch - sesh.logical_epoch

    # Now we know what to name the log.
    log = SummaryWriter(f"{logs}/{sesh.name}")

    # Set up our image-damagers.
    pan_halo = HaloMaker(1, device=device)
    mul_halo = HaloMaker(8, device=device)
    misalignment = WV23Misaligner(side_length=128, device=device, weight_power=2.0)

    physical_per_logical = logical_batch_size // physical_batch_size
    if physical_per_logical < 1:
        raise ValueError(
            "Physical batch size cannot be smaller than logical batch size."
        )

    def net_loss(y, ŷ):
        ok_loss = ΔEOK(y, ŷ) * 1
        deet_loss = detail_loss(y, ŷ)
        sat_loss = sat_detail_loss(y, ŷ)
        return (
            ok_loss
            + deet_loss
            + sat_loss
        )

    if compile:
        net_loss = torch.compile(net_loss)
        gen = torch.compile(gen)

    if agenda:
        print(f"Training time! Running on {device}.")

        if sesh.has_starters():
            ckpt = sesh.starters[0]
            origin_string = f"Loaded {ckpt} and corresponding optimizer."
        else:
            origin_string = "Starting training from scratch."
        print(origin_string)
        print(f"The plan is {epochs} epochs of:")
        print(
            f"  Batch size: {logical_batch_size}\n"
            f"  Physical batch size: {physical_batch_size}\n"
            f"  Learning rate: {lr}\n"
            f"  Chip source: {chips}\n"
            f"  Test chip source: {test_chips}\n"
        )

        print(f"The next checkpoint will go in {sesh.get_next_paths()[0]}, and so on.")
        print("Training starts at " + datetime.datetime.now().isoformat() + ".\n")

    for epoch in range(epochs):
        batch_counter = 0

        with tqdm(trainloader, unit="b", mininterval=2) as progress:
            # Training part of epoch
            loss = 0.0
            loss_history = []
            progress.set_postfix(l="...")

            for x, y in progress:
                progress.set_description(f"Ep {sesh.logical_epoch}")

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pan = x[:, :16]
                mul = x[:, 16:]

                pan = pile(pan_halo(tile(pan, 4), mean=0.25, std=0.1), 4)
                mul = misalignment(mul, amt=1.0)
                mul = mul_halo(mul, mean=1.0, std=0.75)

                x[:, :16] = pan
                x[:, 16:] = mul

                ŷ = gen(x)

                loss = net_loss(y, ŷ) / physical_per_logical
                loss_history.append(float(loss.item()))
                loss.backward()

                batch_counter += 1
                if batch_counter >= physical_per_logical:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    batch_counter = 0

                    if len(loss_history) > 1:
                        loss_std, loss_mean = torch.std_mean(
                            torch.tensor(loss_history) * physical_per_logical
                        )
                        progress.set_postfix(l=f"{loss_mean:.3f}±{loss_std:.3f}")

            log.add_scalars(
                "loss",
                {"train": torch.mean(torch.tensor(loss_history))},
                sesh.logical_epoch,
            )

        # Write checkpoints in case user has been waiting for them.
        gen_path, opt_path = sesh.get_next_paths()
        torch.save(gen.state_dict(), gen_path)
        torch.save(opt.state_dict(), opt_path)

        # Testing part of epoch
        gen.eval()
        with torch.no_grad():
            test_losses = []

            for x, y in testloader:
                # progress.set_description(f"(Testing)")
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                ŷ = gen(x)

                test_loss = net_loss(y, ŷ)
                test_losses.append(test_loss.item())
        gen.train()

        log.add_scalars(
            "loss",
            {"test": torch.mean(torch.tensor(test_losses))},
            sesh.logical_epoch,
        )

        log.flush()
        sesh.finish_epoch()  # increments the logical_epoch


if __name__ == "__main__":
    train()
