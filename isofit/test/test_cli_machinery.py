"""High-level tests to ensure the CLI is constructed properly."""

import subprocess as sp
import sys

import pytest
from click.testing import CliRunner

import isofit
from isofit import __main__


# fmt: off
@pytest.mark.parametrize("executable", [
    ["isofit"],
    [sys.executable, "-m", "isofit"],
])
# fmt: on
def test_subcommand_registration(executable):
    """Ensure all CLI subcommands are registered.

    Test both ``$ isofit`` and ``$ python3 -m isofit``.
    """

    cmd = executable + ["--help"]
    with sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE) as proc:
        proc.wait()
        stdout_txt = proc.stdout.read().decode()
        if proc.returncode != 0:
            print()
            print("stdout:")
            print(stdout_txt, sys.stderr)
            print()
            print("stderr:")
            print(proc.stdout.read().decode(), file=sys.stderr)
            assert False

    subcommand_count = 0
    for subcommand_count, cmd in enumerate(__main__.cli.commands, 1):
        assert cmd in stdout_txt

    # Check to make sure the right number of subcommands are registered
    assert subcommand_count == 10


def test_version():
    """Ensure ``--version`` flag works."""

    runner = CliRunner()
    result = runner.invoke(__main__.cli, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == isofit.__version__
