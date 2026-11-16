"""
Staged, config-driven replacement for apply_oe.

A retrieval is described by a single YAML config (see
``templates/retrieval.yml`` for every option and its default) and executed as
a sequence of stages (see :mod:`isofit.utils.retrieval.pipeline`), each of
which may also be run on its own from the CLI:

    isofit retrieval run config.yml --n_cores 8
    isofit retrieval presolve config.yml
"""

import logging
from pathlib import Path
from typing import Optional, Union

import click
from rich.console import Console as RichConsole
from rich.logging import RichHandler

# from isofit.configs.loader import expand_paths
# from isofit.configs.utils import interpolate
# from isofit.data import env
from isofit.configs import list_bases, load, make_config
from isofit.configs.sections.pipeline import LogConfig

Console = RichConsole(record=True, force_terminal=True, force_interactive=True)
Logger = logging.getLogger(__name__)
# Template = Path(__file__).resolve().parent / "templates" / "retrieval.yml"


def init(config: str, patch: Optional[tuple[str, ...]], ctx: click.Context) -> ...:
    """
    Load the retrieval config and set up logging for a CLI invocation.

    Parameters
    ----------
    config : str
        A base config name, a config file, or ``file:section``.
    patch : tuple of str
        Config references, in the same forms as ``config``, layered on top of it
        in order. Only each patch's own keys are applied, not its ``^^`` parents.
    ctx : click.Context
        Click context whose extra args supply the dotted ``--key value`` overrides.

    Returns
    -------
    RetrievalConfig
        The fully resolved configuration.
    """
    if not config:
        Logger.error(
            "-c/--config must be set to either a configuration file or a usecase base. Use '[file.yaml]:<section>' to select a section instead of the whole file."
        )
        Logger.error(f"Available usecase bases: {list_bases().keys()}")
        return

    c = load(config, overrides=ctx, patch=patch)

    handlers = [
        RichHandler(
            console=Console,
            rich_tracebacks=True,
            tracebacks_suppress=[click],
        )
    ]

    level = None
    if "log" in c:
        l = LogConfig(**c["log"])

        if l.file:
            l.file.parent.mkdir(parents=True, exist_ok=True)

            fh = logging.FileHandler(file, mode="a" if l.append else "w")
            fh.setLevel(logging.DEBUG)

            fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            fmt = logging.Formatter(fmt)
            fh.setFormatter(fmt)

            handlers.append(fh)

        level = getattr(logging, l.level)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s",
        datefmt="[%X]",
    )

    # if c["log"]["config"]:
    #     output = Path(c["log"]["config"])
    #     output.parent.mkdir(exist_ok=True, parents=True)
    #     output.write_text(c.to_yaml())

    return make_config(c)
