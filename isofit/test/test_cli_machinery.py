"""High-level tests to ensure the CLI is constructed properly."""

import os
import subprocess as sp
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import isofit
from isofit import __main__

# Determine if 'isofit' is installed via '$PYTHONPATH'. To do this, check to see
# if '$PYTHONPATH' is set, and search for 'isofit' at each location.
ISOFIT_ABSPATH = Path(isofit.__file__).parent.absolute()
IS_INSTALLED_VIA_PYTHONPATH = False
if "PYTHONPATH" in os.environ:
    for p in os.environ["PYTHONPATH"].split(os.pathsep):
        potential_isofit_path = Path(p).absolute() / isofit.__name__
        if potential_isofit_path == ISOFIT_ABSPATH:
            IS_INSTALLED_VIA_PYTHONPATH = True
            break

_NO_EXECUTABLE_MARKER = pytest.mark.skipif(
    IS_INSTALLED_VIA_PYTHONPATH,
    reason="'$ isofit' executable not available when installed via '$PYTHONPATH'",
)


# fmt: off
@pytest.mark.parametrize("executable", [

    # No matter the installation this works: $ python -m isofit
    [sys.executable, "-m", "isofit"],

    # The '$ isofit' executable is not available when installed via
    # '$PYTHONPATH', so sometimes this test is skipped.
    pytest.param(["isofit"], marks=_NO_EXECUTABLE_MARKER),

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
