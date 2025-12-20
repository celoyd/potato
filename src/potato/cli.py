"""The potato CLI runnable."""

import importlib
import pkgutil

import click

from . import scripts


@click.group()
def cli():
    """Potato’s command-line interface."""
    pass


for _importer, name, _ispkg in pkgutil.iter_modules(scripts.__path__):
    mod = importlib.import_module(f"{scripts.__name__}.{name}")
    cmd = getattr(mod, "cli", None)
    if isinstance(cmd, click.Command):
        cli.add_command(cmd, name=name.replace("_", "-"))
