"""Train Potato.

Please see the documentation (in docs/) for general directions on using this
script, and in particular for chipping (preparing training data, which must be
done before training).

This script is written in a very basic, “big list of operations” style. The
rationale is that it’s meant to demonstrate many details of how one might want
to train Potato. Specifically, the voice of reason
might suggest that many hardcoded values and orders of operation in this
script be moved to, say, a training.toml. Instead, everything is crammed in
here. This is because a training.toml capable of containing everything a user
might want to adjust would be equivalent to a script, and this file would
become an interpreter for it. It seems simpler to leave this as the script.

There are also comments suggesting likely changes.
"""

import warnings
from pathlib import Path

import click
import torch
from aim import Run
from torch.utils.data import DataLoader
from tqdm import tqdm

from potato.augmentations import HaloMaker, WV23Misaligner
from potato.losses import ΔEOK
from potato.model import Potato
from potato.training import ChipReader, LossAccumulator, Session
from potato.util import cheap_half, timestamp


@click.command()
@click.option("--session", required=True, help="Name of training session")
@click.option("--load-from", help="Checkpoint to start off: <session>[/<number>]")
@click.option("--lr", "--learning-rate", default=5e-4, help="Optimizer learning rate")
@click.option(
    "--physical-batch-size", "--pbs", default=8, help="Load this many chips at once"
)
@click.option(
    "--logical-batch-size", "--lbs", default=64, help="Backprop every this many chips"
)
@click.option("--epochs", type=int, help="Number of epochs to train for")
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
@click.option("--test-length", default=256, help="Number of chips to test on")
@click.option(
    "--checkpoints", default="sessions", help="Where to save session checkpoints"
)
@click.option("--workers", default=3, help="Number of chip-loading workers")
@click.option("--device", default="cuda", help="Torch device to run on")
# @click.option(
#     "--compile",
#     "torch_compile",  # compile() is a builtin; let’s not shadow it
#     is_flag=True,
#     default=False,
#     help="Compile model with JIT",
# )
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=True,
    help="Print session parameters and (starting and ending) timestamps",
)
@click.option(
    "--weight-link",
    type=click.Path(),
    help="Link latest generator weights here",
    default="sessions/latest.pt",
    required=False,
)
def cli(
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
    device,
    # torch_compile,
    verbose,
    weight_link,
):
    """Train Potato."""
    # Set up the chip loaders.
    loader_params = {
        "batch_size": physical_batch_size,
        "num_workers": workers,
        "pin_memory": True,
    }

    train_reader = ChipReader(chips, train_length)
    test_reader = ChipReader(test_chips, test_length)

    train_loader = DataLoader(train_reader, **loader_params, shuffle=True)

    # For lengthy sessions you may prefer to shuffle here too:
    test_loader = DataLoader(test_reader, **loader_params, shuffle=False)

    # Set up the model and optimizer so we can load their weights.
    gen = Potato(48).to(device)

    opt = torch.optim.AdamW(gen.parameters(), lr, weight_decay=2e-3)

    sesh = Session(session, load_from, root_dir=checkpoints)

    if sesh.has_starters():
        gen_weights, opt_weights = sesh.get_starters()
        gen.load_state_dict(gen_weights)
        opt.load_state_dict(opt_weights)

    if up_to_epoch:
        if epochs:
            raise ValueError("Expected only one of --epochs and --up-to-epoch.")
        epochs = up_to_epoch - sesh.logical_epoch

    # This name has tripped me up. Keep in mind it’s in terms of batch size.
    # Perhaps worth renaming to accumulation_ratio or something.
    logical_per_physical = logical_batch_size // physical_batch_size
    if (logical_batch_size % physical_batch_size > 0) or (logical_per_physical < 1):
        raise ValueError("Physical batch size must evenly divide logical batch size.")

    # V1 – we’re taking off.
    run = Run()

    run["hparams"] = {
        "session": sesh.name,
        "starting_from": sesh.logical_epoch,
        "learning_rate": lr,
        "batch_size": logical_batch_size,
    }

    # Set up our image-damagers.
    misalign = WV23Misaligner(side_length=128).to(device)
    pan_halo = HaloMaker(1).to(device)
    mul_halo = HaloMaker(8).to(device)

    # If you have a class-based loss, instantiate it here.

    # full_, half_, and quarter_ in these names refer to image scale,
    # so half_loss and quarter_loss are the intermediate losses.
    def full_loss(y, ŷ, logger):
        µΔE = ΔEOK(y, ŷ)
        logger.log("µΔE_f", µΔE)

        # Your losses here!

        total = µΔE  # + x_weight * x_loss ...
        logger.log("total f", total)

        return total

    def half_loss(y_h, ŷ_h, logger):
        µΔE = ΔEOK(y_h, ŷ_h)
        logger.log("µΔE_h", µΔE)
        total = µΔE
        return total

    def quarter_loss(y_q, ŷ_q, logger):
        µΔE = ΔEOK(y_q, ŷ_q)
        logger.log("µΔE_q", µΔE)
        total = µΔE
        return total

    # Collect any regularization losses, about the model per se.
    # def model_losses(model, logger):
    #     Gramian = Gramian_weight_loss(model)
    #     logger.log("Gramian", Gramian)
    #     total = 0.01 * Gramian
    #     return total

    # Collect the ordinary losses, on how the model is doing the task.
    def task_losses(y, ŷ, intermediates, logger):
        task_loss = full_loss(y, ŷ, logger)

        y_h = cheap_half(y)  # image resize
        task_loss += half_loss(y_h, intermediates["h"], logger)

        y_q = cheap_half(y_h)
        task_loss += quarter_loss(y_q, intermediates["q"], logger)

        # Scale things up if you like big numbers:
        # task_loss = 64.0 * task_loss

        logger.log("task_losses", task_loss)

        return task_loss

    # if torch_compile:
    #     task_losses = torch.compile(task_losses)
    #     # model_losses = torch.compile(model_losses)
    #     gen = torch.compile(gen)

    if verbose:
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
        if weight_link:
            print(f"{weight_link} will stay pointed at the most recent checkpoint.")
        print(f"Training starts at {timestamp()}.")

    train_log = LossAccumulator(run, "train", sesh.logical_epoch, logical_per_physical)
    test_log = LossAccumulator(run, "test", sesh.logical_epoch, logical_per_physical)

    # Now we begin actual training and testing.
    for _ in range(epochs):
        physical_batch_counter = 0

        # Loss as displayed in the tqdm progress bar is always the running
        # mean of the epoch’s losses so far. This is where we store each
        # physical batch’s loss to calculate that:
        loss_history = []

        # The unit for reporting progress and rate is chip (not batch).
        with tqdm(
            total=train_length,
            unit=" c",
            bar_format=(
                "{desc} {bar} {n_fmt}/{total_fmt} ({percentage:.1f}%), "
                "{rate_fmt}, {elapsed}+{remaining}{postfix}"
            ),
        ) as prog:
            gen.train()
            prog.set_description(f"E {sesh.logical_epoch}")
            prog.set_postfix(l="...")  # until we have a loss

            for x, y in train_loader:
                # This is the physical batch loop. Here, we will sometimes use
                # regression-style notation, where x is input, y is truth
                # (label or target), and ŷ is model output, or y + ε (error).

                # First we move everything to the device.
                pan, mul = x
                pan = pan.to(device, non_blocking=True)
                mul = mul.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Now we augment (damage) our inputs.
                pan = pan_halo(pan, mean_sharpening=0.1, std=0.1)
                mul = misalign(mul, amount=1.0, spikiness=2.0)
                mul = mul_halo(mul, mean_sharpening=0.75, std=0.5)

                # Add other device-based augmentations here, e.g., uncomment:
                # pan = (
                #   pan + torch.randn(pan.shape, device=device) * 0.001
                # ).clamp_min(1/10_000)

                # Actually do the thing:
                ŷ, intermediates = gen((pan, mul), intermediates=True)

                loss = task_losses(y, ŷ, intermediates, train_log)
                # loss += model_losses(gen, train_log)

                pbatch_loss = loss
                (pbatch_loss / logical_per_physical).backward()

                loss_history.append(pbatch_loss.item())
                prog.update(physical_batch_size)  # remember, units are chips

                physical_batch_counter += 1
                if physical_batch_counter >= logical_per_physical:
                    # This block functions as the logical batch loop.
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    loss_mean = sum(loss_history) / len(loss_history)
                    prog.set_postfix(l=f"{loss_mean:.3f}")

                    physical_batch_counter = 0

        # Write checkpoints as soon as possible, for impatient users.
        gen_path, opt_path = sesh.get_next_paths()
        torch.save(gen.state_dict(), gen_path)
        torch.save(opt.state_dict(), opt_path)

        if weight_link:
            # We won’t delete anything other than an existing symlink, so this
            # is all reasonably safe, but it may fail to do the obvious thing in
            # various unusual situations (e.g., a link across drives under
            # Windows) that I have not tested, so don’t hesitate to edit it.

            link = Path(weight_link)
            if link.is_symlink():
                link.unlink()

            if not link.exists():
                # It was a symlink, or not there at all, so we’re safe to write.
                try:
                    link.symlink_to(gen_path.relative_to(link.parent))
                except Exception as e:  # bite me
                    warnings.warn(
                        f"Failed to link {link} to {gen_path}:"
                        f"\n\n{e}\n\n"
                        "Ignoring for this run; only writing numbered files.",
                        stacklevel=0,
                    )
                    weight_link = False
            else:
                # It exists, so it was not a symlink
                warnings.warn(
                    f"{weight_link} is not (already) a symlink, so I won’t "
                    "overwrite it. Ignoring for this run and only writing the "
                    "numbered weight files.",
                    stacklevel=0,
                )
                weight_link = False

        # Testing time!
        with torch.no_grad():
            gen.eval()
            test_losses = []

            for x, y in test_loader:
                pan, mul = x
                pan = pan.to(device, non_blocking=True)
                mul = mul.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # We don’t augment or look at anything but the primary, full-
                # scale loss here. This is debatable but I have no regrets yet.

                ŷ, intermediates = gen((pan, mul), intermediates=True)

                test_loss = task_losses(y, ŷ, intermediates, test_log)
                test_losses.append(test_loss.item())

        test_log.finish_epoch()
        train_log.finish_epoch()
        sesh.finish_epoch()  # increments logical_epoch

    if verbose:
        print(f"Training ends at {timestamp()}.")


if __name__ == "__main__":
    cli()
