import os
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner
from spectral import envi

from isofit.__main__ import cli, env
from isofit.core.common import envi_header, expand_path, json_load_ascii
from isofit.core.isofit import Isofit
from isofit.core.units import micron_to_nm
from isofit.utils import surface_model
from isofit.utils.template_construction import check_surface_model


# TODO - re-engage this test w/ 6c emulator & new glint model
# fmt: off
@pytest.mark.examples
@pytest.mark.parametrize("args", [
    ("--level", "DEBUG", "configs/run/prm20151026t173213_D8W_6s.json"),
    ("--level", "DEBUG", "configs/run/prm20151026t173213_D8p5W_6s.json"),
    # ("--level", "DEBUG", "configs/run/prm20151026t173213_D9W_6s.json"),
    ("--level", "DEBUG", "configs/run/prm20151026t173213_D9p5W_6s.json"),
])
# fmt: on
def test_santa_monica(args, monkeypatch):
    """Run the Santa Monica test dataset."""

    monkeypatch.chdir(env.path("examples", "20151026_SantaMonica/"))
    surface_model("configs/prm20151026t173213_surface_coastal.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


# fmt: off
@pytest.mark.examples
@pytest.mark.parametrize("args", [
    ("--level", "DEBUG", "configs/modtran/ang20171108t173546_darklot.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t173546_horse.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t184227_astrored.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t184227_astrogreen.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t184227_beckmanlawn.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t184227_beckmanlawn-oversmoothed.json"),
    ("--level", "DEBUG", "configs/modtran/ang20171108t184227_beckmanlawn-undersmoothed.json"),
])
# fmt: on
def test_pasadena_modtran(args, monkeypatch):
    """Run Pasadena example dataset."""

    monkeypatch.chdir(env.path("examples", "20171108_Pasadena/"))
    surface_model("configs/ang20171108t184227_surface.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


@pytest.mark.examples
@mock.patch(
    "isofit.radiative_transfer.engines.modtran.ModtranRT.makeSim", new=lambda *_: ...
)
def test_pasadena_topoflux(monkeypatch):
    """Run Pasadena topoflux example dataset."""

    monkeypatch.chdir(env.path("examples", "20171108_Pasadena/"))
    surface_model("configs/ang20171108t184227_surface.json")

    model = Isofit(
        "configs/topoflux/ang20171108t184227_beckmanlawn-multimodtran-topoflux.json"
    )
    model.run()


@pytest.mark.examples
@pytest.mark.parametrize(
    "args",
    [
        ("--level", "DEBUG", "configs/AV320250308t200738_wltest_isofit.json"),
        (
            "--level",
            "DEBUG",
            "configs/AV320250308t200738_wltest_isofit_swir_shift.json",
        ),
        (
            "--level",
            "DEBUG",
            "configs/AV320250308t200738_wltest_isofit_swir_spline.json",
        ),
    ],
)
# fmt: on
def test_av3_calibration(args, monkeypatch):
    """Run the calibration test dataset."""

    monkeypatch.chdir(env.path("examples", "20250308_AV3Cal_wltest/"))
    os.makedirs("output", exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


@pytest.mark.examples
@pytest.mark.parametrize(
    "args",
    [
        (
            "remote/ang20170228_surface_model.mat",
            "remote/20170320_ang20170228_wavelength_fit.txt",
            "",
        ),
        (
            "configs/ang20171108t184227_surface.json",
            "remote/20170320_ang20170228_wavelength_fit.txt",
            "remote/surface.mat",
        ),
        (
            "configs/ang20171108t184227_surface.json",
            "remote/20170320_ang20170228_wavelength_fit.txt",
            "",
        ),
    ],
)
def test_check_surface(args, monkeypatch):
    """Run variations of check_surface_model"""
    monkeypatch.chdir(env.path("examples", "20171108_Pasadena/"))

    if args[0].endswith(".mat"):
        wl = micron_to_nm(np.loadtxt(args[1])[:, 1])
        surface_files = check_surface_model(args[0], wl=wl)

        assert list(surface_files.values())[0] == args[0]

    elif args[0].endswith(".json"):
        surface_files = check_surface_model(
            args[0], output_model_path=args[2], surface_wavelength_path=args[1]
        )
        if args[2]:
            assert list(surface_files.values())[0] == args[2]

        else:
            configdir, _ = os.path.split(os.path.abspath(args[0]))
            config = json_load_ascii(args[0], shell_replace=True)
            output_model_path = expand_path(configdir, config["output_model_file"])
            assert list(surface_files.values())[0] == output_model_path


@pytest.mark.examples
@pytest.mark.parametrize(
    "args",
    [
        (
            "--level",
            "DEBUG",
            "configs/prm20231110t071521_multi_surface_isofit.json",
            "output/prm20231110t071521_multi_surface_state",
            250,
            ("SUN_GLINT", -9999.0),
            True,
        ),
        (
            "--level",
            "DEBUG",
            "configs/prm20231110t071521_single_surface_isofit.json",
            "output/prm20231110t071521_single_surface_state",
            248,
            ("AOT550", 0.1),
            False,
        ),
    ],
)
def test_multisurface_inversions(args, monkeypatch):
    monkeypatch.chdir(env.path("examples", "20231110_Prism_Multisurface/"))

    surface_model("configs/surface_model.json", multisurface=args[6])

    os.makedirs("output", exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run"] + [args[0], args[1], args[2]], catch_exceptions=False
    )

    assert result.exit_code == 0

    # Test that statevector length is correct
    ds = envi.open(envi_header(args[3]))

    assert len(ds.metadata["band names"]) == args[4]

    # # Test that veg pixel don't have glint values
    im = ds.load()
    i = np.where(np.array(ds.metadata["band names"]) == args[5][0])[0]
    val = float(np.squeeze(im[1, 0, i]))

    assert np.isclose(val, args[5][1], atol=0.01)


@pytest.mark.xfail
@pytest.mark.examples
def test_modtran_one(monkeypatch):
    """Run MODTRAN example dataset."""

    monkeypatch.chdir(env.path("examples", "20190806_ThermalIR/"))
    surface_model("configs/surface.json")

    model = Isofit(
        "configs/run/joint_isofit_with_prof_WATER_nogrid.json", level="DEBUG"
    )
    model.run()


# @pytest.mark.xfail
# @pytest.mark.examples
# def test_profiling_cube_small(monkeypatch):
#     """Run profiling datasets."""
#
#     monkeypatch.chdir(env.path("examples", "profiling_cube/"))
#
#     environ = os.environ.copy()
#     environ["ISOFIT_DEBUG"] = "1"
#
#     proc = sp.Popen(
#         [sys.executable, "run_profiling.py"],
#         env=environ,
#         stdout=sp.PIPE,
#         stderr=sp.PIPE,
#     )
#
#     with proc as proc:
#         if proc.returncode != 0:
#             print("stdout:")
#             print(proc.stdout.read().decode())
#             print("stderr:")
#             print(proc.stderr.read().decode())
#         assert proc.returncode == 0
