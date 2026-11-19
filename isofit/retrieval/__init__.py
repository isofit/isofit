"""
Staged, config-driven replacement for apply_oe.

A retrieval is described by a single YAML config (see
``templates/retrieval.yml`` for every option and its default) and executed as
a sequence of stages (see :mod:`isofit.retrieval.pipeline`), each of
which may also be run on its own from the CLI:

    isofit retrieval run config.yml --n_cores 8
    isofit retrieval presolve config.yml

This package is imported by :mod:`isofit.configs` (for :mod:`.sensor`), so it
must not import :mod:`isofit.configs` itself; the CLI setup lives in :mod:`.cli`.
"""

from rich.console import Console as RichConsole

Console = RichConsole(record=True, force_terminal=True, force_interactive=True)
