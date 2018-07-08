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

import sys
import json
from os.path import expandvars, split, abspath, join

# Get directory and file paths

testdir, fname = split(abspath(__file__))
datadir = join(testdir, 'data/')

# Import ISOFIT code

sys.path.append(join(testdir, '../isofit'))
from forward import ForwardModel
from inverse import Inversion
from output import Output
from common import spectrumLoad
from common import expand_all_paths
from rt_libradtran import LibRadTranRT


def test_rt_libradtran():
    """This simple unit test simulates an observation of a uniform
    10% reflectance surface, and then performs a retrieval (varying only
    Atmospheric parameters).  It then performs an inversion and tests for
    consistency."""

    stateA = run_forward()
    stateB = run_inverse()

    err_h2o = abs(stateA[0] - stateB[0])
    assert err_h2o < 0.05

    err_aod = abs(stateA[1] - stateB[1])
    assert err_aod < 0.01


def load_config(config_file):
    """Load a configuration file, expand paths"""

    config_path = join(datadir, config_file)
    config = json.load(open(config_path, 'r'))
    configdir, f = split(abspath(config_path))
    config = expand_all_paths(config, configdir)
    return config


def run_forward():
    """Simulate the remote measurement of a spectrally uniform surface"""

    # Configure the surface/atmosphere/instrument model
    config = load_config('config_forward.json')
    fm = ForwardModel(config['forward_model'])
    iv = Inversion(config['inversion'], fm)
    out = Output(config, iv)
    geom = None

    # Simulate a measurement
    state_est = fm.init_val.copy()
    rdn_est = fm.calc_rdn(state_est, geom)
    rdn_meas = rdn_est.copy()
    rdn_sim = fm.instrument.simulate_measurement(rdn_meas, geom)

    # Calculate uncertainties at the solution state, write result
    rfl_est, rdn_est, path_est, S_hat, K, G =\
        iv.forward_uncertainty(state_est, rdn_meas, geom)
    out.write_spectrum(state_est, rfl_est, rdn_est, path_est,
                       rdn_meas, rdn_sim, geom)
    assert True
    return state_est


def run_inverse():
    """Invert the remote measurement"""

    # Configure the surface/atmosphere/instrument model
    config = load_config('config_inversion.json')
    fm = ForwardModel(config['forward_model'])
    iv = Inversion(config['inversion'], fm)
    out = Output(config, iv)
    geom = None

    # Get our measurement from the simulation results, and invert
    rdn_meas, wl = spectrumLoad(config['input']['measured_radiance_file'])
    state_est = iv.invert(rdn_meas, geom, out)

    # Calculate uncertainties at the solution state, write result
    rfl_est, rdn_est, path_est, S_hat, K, G =\
        iv.forward_uncertainty(state_est, rdn_meas, geom)
    out.write_spectrum(state_est, rfl_est, rdn_est, path_est,
                       rdn_meas, rdn_sim=None, geom=geom)
    assert True
    return state_est
