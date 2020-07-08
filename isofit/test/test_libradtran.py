#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import json
import os

from isofit.core.common import expand_all_paths
from isofit.core.fileio import IO
from isofit.inversion.inverse import Inversion
from isofit.core.forward import ForwardModel

from isofit.configs.configs import create_new_config


def test_libradtran():
    """This simple unit test simulates an observation of a uniform
    10% reflectance surface, and then performs a retrieval (varying only
    Atmospheric parameters). It then performs an inversion and tests for
    consistency."""

    stateA = run_forward()
    stateB = run_inverse()

    err_h2o = abs(stateA[0] - stateB[0])
    assert err_h2o < 0.05

    err_aod = abs(stateA[1] - stateB[1])
    assert err_aod < 0.01


def run_forward():
    """Simulate the remote measurement of a spectrally uniform surface."""

    # Configure the surface/atmosphere/instrument model
    testdir, fname = os.path.split(os.path.abspath(__file__))
    datadir = os.path.join(testdir, 'data')
    config = create_new_config(os.path.join(datadir, 'config_forward.json'))
    fm = ForwardModel(config)
    iv = Inversion(config, fm)
    io = IO(config, fm, iv, [0], [0])

    # Simulate a measurement and write result
    for row, col, meas, geom in io:
        states = iv.invert(meas, geom)
        io.write_spectrum(row, col, states, meas, geom)

    assert True
    return states[0]


def run_inverse():
    """Invert the remote measurement."""

    # Configure the surface/atmosphere/instrument model
    testdir, fname = os.path.split(os.path.abspath(__file__))
    datadir = os.path.join(testdir, 'data')
    config = create_new_config(os.path.join(datadir, 'config_forward.json'))
    fm = ForwardModel(config)
    iv = Inversion(config, fm)
    io = IO(config, fm, iv, [0], [0])
    geom = None

    # Get our measurement from the simulation results, and invert.
    # Calculate uncertainties at the solution state, write result
    for row, col, meas, geom in io:
        states = iv.invert(meas, geom)
        io.write_spectrum(row, col, states, meas, geom)

    assert True
    return states[-1]
