import logging
from typing import Any, Optional

import click
from rich.logging import RichHandler

from isofit.configs import list_bases, list_mixins, load, make_config
from isofit.configs.sections.pipeline import LogConfig
from isofit.retrieval import Console
from isofit.retrieval import pipeline as pl
from isofit.retrieval.stages import Stages

Logger = logging.getLogger(__name__)
CS = dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
)
Config = click.option(
    "-c",
    "--config",
    help="Either a configuration file or a usecase base. Use '[file.yaml]:<section>' to select a section instead of the whole file.",
)
Mixin = click.option(
    "-m", "--mixin", multiple=True, help="Layer mixins on top of the base, in order"
)


def init(config: str, mixin: Optional[tuple[str, ...]], ctx: click.Context) -> ...:
    """
    Load the retrieval config and set up logging for a CLI invocation.

    Parameters
    ----------
    config : str
        A base config name, a config file, or ``file:section``.
    mixin : tuple of str
        Config references, in the same forms as ``config``, layered on top of it
        in order. Only each mixin's own keys are applied, not its ``^^`` parents.
    ctx : click.Context
        Click context whose extra args supply the dotted ``--key value`` overrides.

    Returns
    -------
    Config
        The fully resolved configuration.
    """
    if not config:
        Logger.error(
            "-c/--config must be set to either a configuration file or a usecase base. Use '[file.yaml]:<section>' to select a section instead of the whole file."
        )
        Logger.error(f"Available usecase bases: {list_bases().keys()}")
        Logger.error(f"Available mixins: {list_mixins().keys()}")
        return

    c = load(config, overrides=ctx, mixin=mixin, interp=True)

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

            fh = logging.FileHandler(l.file, mode="a" if l.append else "w")
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


class PipelineOrder(click.Group):
    """
    Lists subcommands in `isofit retrieval --help` in pipeline order rather than
    alphabetically
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(name="retrieval", cls=PipelineOrder, no_args_is_help=True)
def cli() -> None:
    """
    Staged ISOFIT retrieval pipeline
    """
    pass


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Mixin
def run(ctx: click.Context, **kwargs: Any) -> None:
    """
    Execute every enabled stage of the ISOFIT retrieval pipeline.
    """
    if config := init(ctx=ctx, **kwargs):
        pl.run(config)


def stage_command(name: str, func) -> click.Command:
    """
    Build a command that executes a single stage, regardless of its enabled flag.
    """

    @click.pass_context
    def command(ctx: click.Context, **kwargs: Any) -> None:
        if config := init(ctx=ctx, **kwargs):
            func(config)

    command = Mixin(Config(command))
    return click.command(name=name, context_settings=CS, help=func.__doc__)(command)


for name, func in Stages.items():
    cli.add_command(stage_command(name, func))


if __name__ == "__main__":
    cli()
