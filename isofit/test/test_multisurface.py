import os
import subprocess as sp
import sys
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner
from spectral import envi

from isofit.__main__ import cli, env
from isofit.core.common import envi_header
from isofit.core.isofit import Isofit
from isofit.utils import multicomponent_classification, surface_model


@pytest.mark.multisurface
@pytest.mark.parametrize(
    "args",
    [
        (
            "configs/multi_surface_model.json",
            True,
            "remote/multi_surface_test_surface_water.mat",
        ),
        (
            "configs/multi_surface_model.json",
            True,
            "remote/multi_surface_test_surface_land.mat",
        ),
        (
            "configs/single_surface_model.json",
            False,
            "remote/single_surface_test_surface.mat",
        ),
    ],
)
def test_create_surface_model(args, monkeypatch):

    monkeypatch.chdir(env.path("examples", "20231110_Prism_Multisurface/"))

    if os.path.isfile(args[2]):
        os.remove(args[2])

    surface_model(args[0], multisurface=args[1])

    assert os.path.isfile(args[2])


@pytest.mark.multisurface
@pytest.mark.parametrize(
    "args",
    [
        (
            {
                "rdn": "remote/prm20231110t071521_rdn_two_px",
                "obs": "remote/prm20231110t071521_obs_two_px",
                "loc": "remote/prm20231110t071521_loc_two_px",
            },
            "remote/prm20231110t071521_surface_class",
            {
                "multicomponent_surface": "remote/multi_surface_test_surface_land.mat",
                "glint_model_surface": "remote/multi_surface_test_surface_water.mat",
            },
        )
    ],
)
def test_classify_multicomponent(args, monkeypatch):

    monkeypatch.chdir(env.path("examples", "20231110_Prism_Multisurface/"))

    if os.path.isfile(args[1]):
        os.remove(args[1])

    multicomponent_classification(
        args[0]["rdn"],
        args[0]["obs"],
        args[0]["loc"],
        args[1],
        args[2],
        n_cores=1,
        clean=False,
    )

    assert os.path.isfile(args[1])


@pytest.mark.multisurface
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
        ),
        (
            "--level",
            "DEBUG",
            "configs/prm20231110t071521_single_surface_isofit.json",
            "output/prm20231110t071521_single_surface_state",
            248,
            ("AOT550", 0.1),
        ),
    ],
)
def test_multisurface_inversions(args, monkeypatch):
    monkeypatch.chdir(env.path("examples", "20231110_Prism_Multisurface/"))

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
    val = float(np.squeeze(im[0, 0, i]))
    assert np.isclose(val, args[5][1], atol=0.01)
