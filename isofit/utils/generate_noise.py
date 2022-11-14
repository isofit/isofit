#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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

from os.path import abspath, split

import numpy as np

from isofit.core.common import expand_path, json_load_ascii, load_spectrum
from isofit.core.geometry import Geometry
from isofit.core.instrument import Instrument


def generate_noise(config):
    """Add noise to a radiance spectrum or image."""

    config = json_load_ascii(config, shell_replace=True)
    configdir, configfile = split(abspath(config))

    infile = expand_path(configdir, config["input_radiance_file"])
    outfile = expand_path(configdir, config["output_radiance_file"])
    instrument = Instrument(config["instrument_model"])
    geom = Geometry()

    if infile.endswith("txt"):
        rdn, wl = load_spectrum(infile)
        Sy = instrument.Sy(rdn, geom)
        rdn_noise = rdn + np.random.multivariate_normal(np.zeros(rdn.shape), Sy)
        with open(outfile, "w") as fout:
            for w, r in zip(wl, rdn_noise):
                fout.write("%8.5f %8.5f" % (w, r))
    else:
        raise ValueError("Image cubes not yet implemented.")
