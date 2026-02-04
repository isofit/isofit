"""
These tests are to ensure any changes to the CLI will be backwards compatible.
"""

import os
import shutil
from pathlib import Path
from time import sleep

import pytest
import ray
from click.testing import CliRunner

from isofit.__main__ import cli, env
from isofit.utils import surface_model

# Mark the entire file as containing slow tests
pytestmark = pytest.mark.slow


# Environment variables
CORES = os.cpu_count()


@pytest.fixture(scope="session")
def cwd(tmp_path_factory):
    """
    Path to the working cube example
    """
    return Path(env.imagecube) / "medium"


@pytest.fixture(scope="session")
def surface(cwd):
    """
    Generates the surface.mat file
    """
    outdir = str(cwd / "data")
    os.makedirs(outdir, exist_ok=True)
    outp = str(cwd / "data" / "surface.mat")

    # Generate the surface.mat using the image_cube example config
    # fmt: off
    surface_model(
        config_path=env.path("examples", "20171108_Pasadena", "configs", "ang20171108t184227_surface.json"),
        wavelength_path=env.path("examples", "20171108_Pasadena", "remote", "20170320_ang20170228_wavelength_fit.txt"),
        output_path=outp
    )
    # fmt: on
    # Return the path to the mat file

    return outp


@pytest.fixture()
def files(cwd):
    """
    Common data files to be used by multiple tests. The return is a list in the
    order: [
        0: Radiance file,
        1: Location file,
        2: Observation file,
        3: Output directory
    ]

    As of 07/24/2023 these are from the medium cube example.
    """
    # Flush dirs if they already exists from a previous test case

    flush_dirs = ["config", "lut_h2o", "lut_full", "output"]
    for flush_dir in flush_dirs:
        shutil.rmtree(cwd / flush_dir, ignore_errors=True)

    return [
        str(cwd / "ang20170323t202244_rdn_7k-8k"),
        str(cwd / "ang20170323t202244_loc_7k-8k"),
        str(cwd / "ang20170323t202244_obs_7k-8k"),
        cwd,
    ]


# fmt: off
@pytest.mark.slow
@pytest.mark.parametrize("args", [
    ["-nn", 10, "-nn", 50,],
    ["-nn", 10, "-nn", 50, "-nn", 10, "--pressure_elevation",],
])
# fmt: on
def test_apply_oe(files, args, surface):
    """
    Executes the isofit apply_oe cli command for various test cases
    """
    ray.shutdown()
    sleep(120)

    emulator = env.path("srtmnet", key="srtmnet.file")
    args = [
        "ang",
        "--presolve",
        "--emulator_base",
        emulator,
        "--n_cores",
        CORES,
        "--analytical_line",
        "--logging_level",
        "DEBUG",
    ] + args

    arguments = [
        "apply_oe",
        *files,
        *args,
        "--surface_path",
        surface,
    ]

    # Passing non-string arguments to click is not allowed.
    arguments = [str(i) for i in arguments]

    runner = CliRunner()
    result = runner.invoke(cli, arguments, catch_exceptions=False)

    assert result.exit_code == 0
