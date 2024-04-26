import os
import subprocess as sp
import sys
from unittest import mock

import pytest
from click.testing import CliRunner

from isofit.__main__ import cli
from isofit.core.isofit import Isofit
from isofit.utils import surface_model


# fmt: off
@pytest.mark.examples
@pytest.mark.parametrize("args", [
    ("--level", "DEBUG", "configs/prm20151026t173213_D8W_6s.json"),
    ("--level", "DEBUG", "configs/prm20151026t173213_D8p5W_6s.json"),
    # ("--level", "DEBUG", "configs/prm20151026t173213_D9W_6s.json"),
    ("--level", "DEBUG", "configs/prm20151026t173213_D9p5W_6s.json"),
])
# fmt: on
def test_santa_monica(args, monkeypatch):
    """Run the Santa Monica test dataset."""

    monkeypatch.chdir("examples/20151026_SantaMonica/")
    surface_model("configs/prm20151026t173213_surface_coastal.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


# fmt: off
@pytest.mark.examples
@pytest.mark.parametrize("args", [
    ("--level", "DEBUG", "configs/ang20171108t173546_darklot.json"),
    ("--level", "DEBUG", "configs/ang20171108t173546_horse.json"),
    ("--level", "DEBUG", "configs/ang20171108t184227_astrored.json"),
    ("--level", "DEBUG", "configs/ang20171108t184227_astrogreen.json"),
    ("--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn.json"),
    ("--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn-oversmoothed.json"),
    ("--level", "DEBUG", "configs/ang20171108t184227_beckmanlawn-undersmoothed.json"),
])
# fmt: on
def test_pasadena_modtran(args, monkeypatch):
    """Run Pasadena example dataset."""

    monkeypatch.chdir("examples/20171108_Pasadena/")
    surface_model("configs/ang20171108t184227_surface.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


@pytest.mark.examples
@mock.patch("isofit.radiative_transfer.modtran.ModtranRT.makeSim", new=lambda *_: ...)
def test_pasadena_topoflux(monkeypatch):
    """Run Pasadena topoflux example dataset."""

    monkeypatch.chdir("examples/20171108_Pasadena/")
    surface_model("configs/ang20171108t184227_surface.json")

    model = Isofit("configs/ang20171108t184227_beckmanlawn-multimodtran-topoflux.json")
    model.run()


@pytest.mark.xfail
@pytest.mark.examples
def test_modtran_one(monkeypatch):
    """Run MODTRAN example dataset."""

    monkeypatch.chdir("examples/20190806_ThermalIR/")
    surface_model("configs/surface.json")

    model = Isofit("configs/joint_isofit_with_prof_WATER_nogrid.json", level="DEBUG")
    model.run()


@pytest.mark.xfail
@pytest.mark.examples
def test_profiling_cube_small(monkeypatch):
    """Run profiling datasets."""

    monkeypatch.chdir("examples/profiling_cube/")

    environ = os.environ.copy()
    environ["ISOFIT_DEBUG"] = "1"

    proc = sp.Popen(
        [sys.executable, "run_profiling.py"],
        env=environ,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )

    with proc as proc:
        if proc.returncode != 0:
            print("stdout:")
            print(proc.stdout.read().decode())
            print("stderr:")
            print(proc.stderr.read().decode())
        assert proc.returncode == 0
