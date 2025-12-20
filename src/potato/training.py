"""Helpers for training.

- Session, a training session manager
- ChipReader, a Dataset for DataLoader
"""

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from potato.augmentations import RandomD4


class ChipReader(Dataset):
    """Wrap a directory of chips named 0.pt, 1.pt…."""

    def __init__(self, chip_dir, length, offset=0, augment=True):
        """Set up the usual kind of data.Dataset stuff."""
        self.length = length
        self.offset = offset
        self.dir = Path(chip_dir)
        self.augment = augment

        if self.augment:
            self.d4 = RandomD4()

        self.check_directory()

    def check_directory(self):
        """Make sure the files that should be there are there.

        We do not check that they’re valid in any way, only present.

        Speed will depend on your filesystem of course, but these are all going
        to be read soon anyway, so think of this as a cache warmup :)
        """
        for index in range(self.offset, self.offset + self.length):
            if not (self.dir / f"{index}.pt").exists():
                raise FileNotFoundError(f"Missing {index}.pt")

    def __len__(self):
        """I feel like you can figure this one out on your own."""
        return self.length

    def __getitem__(self, index):
        """Read, and if requested augment, a chip."""
        pt_path = (self.dir / f"{index}.pt").absolute()
        chip = torch.load(pt_path, weights_only=False)

        if self.augment:
            chip = self.d4(chip)

        return chip


class LossAccumulator:
    """Flexibly collect losses and log them at epoch-end."""

    def __init__(self, run, subset, start_epoch, logical_per_physical):
        """Track and log multiple losses over multiple subsets.

        Args:
          run: name of run
          subset: subset (e.g., intermediate losses)
          start_epoch: where we’re starting from
          logical_per_physical: logical batch length / physical batch length

        """
        self.run = run
        self.subset = subset
        self.epoch = start_epoch
        self.lpp = logical_per_physical

        self._losses = defaultdict(list)

    def __enter__(self):
        """We need no special extra context."""
        return self

    def __exit__(self, *args):
        """Clean up."""
        self.finish_epoch()

    def log(self, k, v):
        """Return a loss float."""
        self._losses[k].append(v.detach().cpu().item())

    def __call__(self, name):
        """Give a wrapper that collects a loss."""

        def wrapper(value):
            self._losses.setdefault(name, []).append(value.detach().cpu())
            return value

        return wrapper

    def finish_epoch(self):
        """Track everything recorded over the epoch."""
        avgs = {k: (sum(v) / len(v)) / self.lpp for k, v in self._losses.items()}
        for k, v in avgs.items():
            self.run.track(v, name=k, epoch=self.epoch, context={"subset": self.subset})
        self.epoch += 1
        self._losses.clear()


class Session:
    """Represent the idea of a named training session.

    We do this by wrapping a folder of paired model and optimizer weights
    (organized by epoch number), e.g., training_again/0-{gen,opt}.pt
    """

    def __init__(self, name, load_from=None, root_dir="sessions"):
        """Build a session.

        Args:
          name: Session name
          load_from: Optional point, in format session/epoch, to load
          root_dir: Parent directory for all sessions (default: “sessions”)
          # logical_epoch: Starting epoch counter (default: 0)

        """
        self.name = name

        # self.dir is the base-base directory; we use
        # self.dir / self.name to get the actual seshdir
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

        self.physical_epoch = 0
        self.logical_epoch = 0

        if self.check_session(self.name):  # this is an existing sesh
            latest = self.get_latest_epoch(self.name)
            if latest >= 0:
                self.starters = self.get_paths(self.name, latest, should_exist=True)
                self.logical_epoch = latest + 1
        else:  # new sesh
            (self.root_dir / self.name).mkdir(exist_ok=False)

        if load_from:
            load_session, load_epoch = self.parse_load_from(load_from)
            self.starters = self.get_paths(load_session, load_epoch, should_exist=True)

            if load_session == self.name:  # this is a continuation
                self.logical_epoch = load_epoch + 1  # debatable

    def has_starters(self):
        """Check whether we have a starting point."""
        return hasattr(self, "starters")

    def get_starters(self):
        """Return the contents of the gen and opt we’re starting from."""
        return (torch.load(w, weights_only=True) for w in self.starters)

    def parse_load_from(self, load_from):
        """Parse a load-from string, which may specify an epoch number."""
        if "/" in load_from:
            try:
                load_session, load_epoch = load_from.split("/")
                load_epoch = int(load_epoch)
            except ValueError as exc:
                raise ValueError(
                    "Expected --load-from to look like sesh or sesh/1 "
                    f"(name[/number]) but got {load_from}."
                ) from exc
        else:
            load_session = load_from
            load_epoch = self.get_latest_epoch(load_session)
        return load_session, load_epoch

    def get_latest_epoch(self, sesh):
        """Return the most recent epoch in the named session."""
        ckpts = list((self.root_dir / sesh).glob("*.pt"))

        if len(ckpts) == 0:
            return -1  # I make fun of C programmers for doing this sort of thing

        numbers = [int(p.name.split("-")[0]) for p in ckpts]
        last = sorted(numbers)[-1]

        # Abusing this as a check :/
        _ = self.get_paths(sesh, last, should_exist=True)

        return last

    def check_session(self, sesh):
        """See if the named session exists."""
        return (self.root_dir / sesh).is_dir()

    def get_paths(self, sesh, n, should_exist):
        """Return paths for the given session and epoch."""
        gen_path, opt_path = (
            Path(self.root_dir / sesh / f"{n}-{k}.pt") for k in ("gen", "opt")
        )

        gen_exists, opt_exists = (path.exists() for path in (gen_path, opt_path))

        if gen_exists != opt_exists:
            raise FileNotFoundError(f"Only one of {gen_path} or {opt_path} exists!")

        epoch_exists = gen_exists
        if epoch_exists == should_exist:
            return gen_path, opt_path
        if epoch_exists and not should_exist:
            raise FileExistsError(f"{sesh}/{n} already exists.")
        if should_exist and not epoch_exists:
            raise FileNotFoundError(f"{sesh}/{n} does not exist.")

    def get_next_paths(self):
        """Return paths for new writes."""
        return self.get_paths(self.name, self.logical_epoch, should_exist=False)

    def finish_epoch(self):
        """At end of epoch, all we have to do is bump the counter."""
        self.logical_epoch += 1
