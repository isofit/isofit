"""
The ``isofit config`` command group.

Thin click wrappers around the pipeline in :mod:`isofit.configs.utils.interpolate`
for previewing, copying, validating, listing, and browsing configs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import click
import yaml

from .load import load
from .overrides import parse_overrides

Logger = logging.getLogger(__name__)


def load_cli(
    ctx: Optional[click.Context],
    config: str,
    section: Optional[str] = None,
) -> dict:
    """
    Resolve a CLI ``config`` argument that may be a base-config name or a path.

    The CLI commands accept either the name of a base config shipped with ISOFIT
    (e.g. ``vswir``, ``sixs`` — see
    :func:`isofit.configs.loader.list_bases`) or a path to a config
    file. An explicit, existing file path always wins; otherwise the argument is
    treated as a base-config name. In both cases CLI ``--dotted.key value``
    overrides carried on ``ctx`` are applied.

    Parameters
    ----------
    ctx : click.Context or None
        Click context whose extra args carry ``--dotted.key value`` overrides.
    config : str
        A base-config name or a path to a ``.yml`` / ``.yaml`` / ``.json`` file.
    section : str, optional
        Subsection to load; only used when ``config`` is a file path (a
        base-config name already identifies its section).

    Returns
    -------
    dict
        The resolved, interpolated configuration.
    """
    from isofit.configs import loader

    if not Path(config).is_file() and config in loader.list_bases():
        return loader.load(config, overrides=parse_overrides(ctx), interp=True)

    return load(config, section=section, ctx=ctx)


@click.group(name="config", invoke_without_command=True, no_args_is_help=True)
def cli():
    """
    Utility functions for configuration files
    """
    logging.basicConfig(level=logging.DEBUG)


CS = dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
)
Config = click.argument(
    "config"
)  # NOTE: `config` is either the name of a base config shipped with ISOFIT (e.g. vswir`, `sixs`) or a path to a config file; see `load_cli`
Section = click.option(
    "-s", "--section", help="Subsection of the yaml to load rather than the whole file"
)
NoFlow = click.option(
    "--noflow",
    is_flag=True,
    help="Disables the YAML flow style which condenses lists to [a, b] notation",
)


def _format_yaml(box, noflow):
    if noflow:
        return yaml.dump(
            box, default_flow_style=None, sort_keys=False, width=120, allow_unicode=True
        )
    return yaml.dump(box, default_flow_style=False, sort_keys=False, allow_unicode=True)


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
@NoFlow
@click.option("-v", "--validate", is_flag=True, help="Validate the config")
def preview(ctx: click.Context, noflow=False, validate=False, **kwargs: Any) -> None:
    """
    Preview the final, interpolated configuration

    Loads and processes the configuration file (with interpolation and
    overrides) and displays it in YAML format without executing any
    pipeline stages. Useful for debugging config issues.
    """
    box = load_cli(ctx, **kwargs)
    yml = _format_yaml(box, noflow)
    Logger.info(yml)

    if validate:
        from isofit.configs import Config

        try:
            Config.model_validate(box)
        except Exception as e:
            Logger.error(
                f"The above configuration encountered the following errors:\n{e}"
            )
        else:
            Logger.info("The above configuration is valid")


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
@NoFlow
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help="File to write the new configuration to",
)
def copy(ctx: click.Context, output, noflow=False, **kwargs: Any) -> None:
    """
    Copies an existing configuration to a new file
    """
    box = load_cli(ctx, **kwargs)

    if output.suffix in (".yml", ".yaml"):
        data = _format_yaml(box, noflow)
    elif output.suffix == ".json":
        data = json.dumps(box, indent=4, ensure_ascii=False)
    else:
        raise TypeError("Unsupported file extension, expected either .yaml or .json")

    output.write_text(data)
    Logger.info(f"Wrote to {output}")


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
def validate(ctx: click.Context, **kwargs: Any) -> None:
    """
    Validates a configuration
    """
    box = load_cli(ctx, **kwargs)

    Logger.info(
        "Validating the config. If there are no errors, nothing will be printed."
    )

    from isofit.configs import Config

    try:
        Config.model_validate(box)
    except Exception as e:
        Logger.error(f"The above configuration encountered the following errors:\n{e}")
    else:
        Logger.info("The above configuration is valid")


def quiet() -> None:
    """
    Raise the log level to warnings, for the browser views that replace the
    debug logging of loading and interpolation.
    """
    logging.getLogger().setLevel(logging.WARNING)


@cli.command(context_settings=CS, name="list")
def list_sections():
    """
    Lists the available base configs and mixins
    """
    from isofit.configs.utils import browser

    browser.Console().print(browser.overview(browser.entries()))


@cli.command(context_settings=CS)
@click.pass_context
@click.argument("name", required=False)
@click.option(
    "-m", "--mixin", multiple=True, help="Mixins to layer on top of NAME, in order"
)
@Section
@click.option("--raw", is_flag=True, help="Show ${...} references unresolved")
def browse(
    ctx: click.Context,
    name: Optional[str],
    mixin: tuple[str, ...],
    section: Optional[str],
    raw: bool,
):
    """
    Browse the base configs and mixins

    With no NAME, interactively browse them: select a number or name to show
    it, or a base followed by mixins (e.g. `pipeline + emit`) to show the
    composed config. With NAME, show that base or mixin.

    Given mixins, a section, --raw, or extra `--dotted.key value` overrides,
    NAME is instead the base of a composed config: shows what each mixin
    changes, validates the result, and shows the full config.
    """
    from isofit.configs.utils import browser

    quiet()
    console = browser.Console()
    if not name:
        return browser.browse(console)

    items = browser.entries()
    overrides = parse_overrides(ctx)
    if not (mixin or section or raw or overrides):
        if entry := browser.find(items, name):
            return console.print(browser.details(entry))
        raise click.BadParameter(f"Unknown base or mixin {name!r}", param_hint="NAME")

    try:
        base = browser.pick(items, name, "base")
        mixin = [browser.pick(items, ref, "mixin") for ref in mixin]
    except ValueError as e:
        console.print(str(e))
        raise click.exceptions.Exit(2)

    view = browser.compose(
        base, mixin, overrides=overrides, section=section, interp=not raw
    )
    console.print(view)
