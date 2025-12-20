"""Link chips from destination directory.

Given some chip-containing directories, and a target directory, fill the target
directory with links to the source directories, in dispersed order.
"""

import logging
import pathlib

import click

φ = 1.618033988749895  # the golden ratio, φ - 1 = φ^-1


@click.command()
@click.argument(
    "srcs",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    nargs=-1,
)
@click.argument(
    "dst",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    nargs=1,
)
def cli(srcs, dst):
    """Symlink chips to mix sources."""
    all_paths = []
    for src in srcs:
        all_paths += list(src.glob("*.pt"))

    total = len(all_paths)
    logging.info(
        f"Preparing to link {total} files from {tuple(str(s) for s in srcs)} to {dst}."
    )

    for p in range(total):
        if p > 0 and p % 1000 == 0:
            logging.info(f"Completed {p} links.")

        n = int(p * len(all_paths) * φ) % len(all_paths)

        pt = pathlib.Path(all_paths[n]).resolve()
        link = pathlib.Path(dst / f"{p}.pt")

        try:
            pathlib.os.symlink(pt, link)
        except FileExistsError:
            logging.critical(
                f"{link} already exists. I am forbidden to delete or overwrite."
            )
            raise

        all_paths = all_paths[:n] + all_paths[n + 1 :]

    logging.info(f"Made {total} links in {dst}.")


if __name__ == "__main__":
    cli()
