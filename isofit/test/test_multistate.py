import os
import subprocess as sp
import sys
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner

from isofit.__main__ import cli
from isofit.configs.configs import create_new_config
from isofit.core.common import match_statevector
from isofit.core.forward import ForwardModel
from isofit.core.isofit import Isofit
from isofit.utils import surface_model
from isofit.utils.multistate import construct_full_state


@pytest.mark.multistate
@pytest.mark.parametrize(
    "args",
    [
        ("--level", "DEBUG", "configs/ang20171108t173546_darklot.json"),
    ],
)
def test_single_spectra_multi(args, monkeypatch):
    """Run the Santa Monica test dataset."""

    monkeypatch.chdir(f"{env.examples}/20171108_Pasadena/")
    surface_model("configs/ang20171108t184227_surface.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


@pytest.mark.multistate
@pytest.mark.parametrize(
    "args",
    [
        ("--level", "DEBUG", "configs/prm20151026t173213_D8W_6s.json"),
    ],
)
def test_single_spectra_single(args, monkeypatch):
    """Run the Santa Monica test dataset."""

    monkeypatch.chdir(f"{env.examples}/20151026_SantaMonica/")
    surface_model("configs/prm20151026t173213_surface_coastal.json")

    runner = CliRunner()
    result = runner.invoke(cli, ["run"] + list(args), catch_exceptions=False)

    assert result.exit_code == 0


@pytest.mark.multistate
@pytest.mark.parametrize(
    "args",
    [
        ("configs/ang20171108t173546_darklot.json"),
    ],
)
def test_create_full_state(args, monkeypatch):
    monkeypatch.chdir(f"{env.examples}/20171108_Pasadena/")

    config = create_new_config(args)
    full_state, *_ = construct_full_state(config)

    # Hard coded could pull this from config one day
    num_sv = 429
    assert len(full_state) == num_sv


@pytest.mark.multistate
@pytest.mark.parametrize(
    "args",
    [
        ("configs/ang20171108t173546_darklot.json"),
    ],
)
def test_match_statevector(args, monkeypatch):
    monkeypatch.chdir(f"{env.examples}/20171108_Pasadena/")

    config = create_new_config(args)
    full_state, *_ = construct_full_state(config)

    n = np.random.randint(0, len(full_state) - 1)
    partial_state = [i for i in full_state if i != full_state[n]]
    state_data = [1 for i in partial_state]

    null_value = 99
    full_state_fill = match_statevector(
        state_data, full_state, partial_state, null_value=null_value
    )

    assert full_state_fill[n] == null_value
