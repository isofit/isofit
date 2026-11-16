import logging
from typing import Any

import click

from isofit.utils.retrieval import init
from isofit.utils.retrieval import pipeline as pl

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
Patch = click.option(
    "-p", "--patch", multiple=True, help="Patch the base with additional usecases"
)
# Section = click.option(
#     "-s", "--section", help="Subsection to load rather than the whole file"
# )


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
@Patch
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

    command = Patch(Config(command))
    return click.command(name=name, context_settings=CS, help=func.__doc__)(command)


for name, func in pl.STAGES.items():
    cli.add_command(stage_command(name, func))


if __name__ == "__main__":
    cli()
