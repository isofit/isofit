"""
These tests are to ensure any changes to the CLI will be backwards compatible.
"""

import io
import os
import shutil
import zipfile
from time import sleep

import pytest
import ray
import requests
from click.testing import CliRunner

from isofit.__main__ import cli
from isofit.utils import surface_model

# Mark the entire file as containing slow tests
pytestmark = pytest.mark.slow


# Environment variables
EMULATOR_PATH = os.environ.get("EMULATOR_PATH", "")
CORES = os.cpu_count()


@pytest.fixture(scope="session")
def cube_example(tmp_path_factory):
    """
    Downloads the medium cube example's data
    """
    url = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data_rev.zip"
    path = tmp_path_factory.mktemp("cube_example")

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

    return path


@pytest.fixture(scope="session")
def surface(cube_example):
    """
    Generates the surface.mat file
    """
    outp = str(cube_example / "surface.mat")

    # Generate the surface.mat using the image_cube example config
    # fmt: off
    surface_model(
        config_path="examples/20171108_Pasadena/configs/ang20171108t184227_surface.json",
        wavelength_path="examples/20171108_Pasadena/remote/20170320_ang20170228_wavelength_fit.txt",
        output_path=outp
    )
    # fmt: on
    # Return the path to the mat file
    return outp


@pytest.fixture()
def files(cube_example):
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
    # Flush the output dir if it already exists from a previous test case
    output = cube_example / "output"
    shutil.rmtree(output, ignore_errors=True)

    return [
        str(cube_example / "medium_chunk/ang20170323t202244_rdn_7k-8k"),
        str(cube_example / "medium_chunk/ang20170323t202244_loc_7k-8k"),
        str(cube_example / "medium_chunk/ang20170323t202244_obs_7k-8k"),
        str(output),
    ]


# fmt: off
@pytest.mark.slow
@pytest.mark.parametrize("args", [
    ["ang", "--presolve", "--emulator_base", EMULATOR_PATH, "--n_cores", CORES, "--analytical_line", "-nn", 10, "-nn", 50,],
    ["ang", "--presolve", "--emulator_base", EMULATOR_PATH, "--n_cores", CORES, "--analytical_line", "-nn", 10, "-nn", 50, "-nn", 10, "--pressure_elevation",],
    ["ang", "--presolve", "--emulator_base", EMULATOR_PATH, "--n_cores", CORES, "--empirical_line", "--surface_category", "additive_glint_surface",],
])
# fmt: on
def test_apply_oe(files, args, surface):
    """
    Executes the isofit apply_oe cli command for various test cases
    """
    ray.shutdown()
    sleep(120)

    arguments = ["apply_oe", *files, *args, "--surface_path", surface]

    # Passing non-string arguments to click is not allowed.
    arguments = [str(i) for i in arguments]

    runner = CliRunner()
    result = runner.invoke(cli, arguments, catch_exceptions=False)

    assert result.exit_code == 0
