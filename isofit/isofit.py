#! /usr/bin/env python
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

import os
import sys
import json
import argparse
import scipy as s
from spectral.io import envi
from scipy.io import savemat
from common import load_config, expand_all_paths, load_spectrum
from forward import ForwardModel
from inverse import Inversion
from inverse_mcmc import MCMCInversion
from geometry import Geometry
from fileio import IO
import cProfile
import logging

# Suppress warnings that don't come from us
import warnings
warnings.filterwarnings("ignore")


def main():

    description = 'Spectroscopic Surface & Atmosphere Fitting'
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--level', default='INFO')
    parser.add_argument('--row_column', default='')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', level=args.level)

    # Load the configuration file.
    config = load_config(args.config_file)

    # Build the forward model and inversion objects.
    fm = ForwardModel(config['forward_model'])
    if 'mcmc_inversion' in config:
        iv = MCMCInversion(config['mcmc_inversion'], fm)
    else:
        iv = Inversion(config['inversion'], fm)

    # We set the row and column range of our analysis. The user can
    # specify: a single number, in which case it is interpreted as a row;
    # a comma-separated pair, in which case it is interpreted as a
    # row/column tuple (i.e. a single spectrum); or a comma-separated
    # quartet, in which case it is interpreted as a row, column range in the
    # order (line_start, line_end, sample_start, sample_end) - all values are
    # inclusive. If none of the above, we will analyze the whole cube.
    rows, cols = None, None

    if len(args.row_column) > 0:
        ranges = args.row_column.split(',')

        if len(ranges) == 1:
            rows, cols = [int(ranges[0])], None

        if len(ranges) == 2:
            row_start, row_end = ranges
            rows, cols = range(int(row_start), int(row_end)), None

        elif len(ranges) == 4:
            row_start, row_end, col_start, col_end = ranges
            line_start, line_end, samp_start, samp_end = ranges
            rows = range(int(row_start), int(row_end))
            cols = range(int(col_start), int(col_end))

    # Iterate over all spectra, reading and writing through the IO
    # object to handle formatting, buffering, and deferred write-to-file.
    # The idea is to avoid reading the entire file into memory, or hitting
    # the physical disk too often.
    io = IO(config, fm, iv, rows, cols)
    for row, col, meas, geom, configs in io:

        if meas is not None and all(meas < -49.0):

            # Bad data flags
            states = []

        else:

            # update model components with new configuration parameters
            # specific to this spectrum.  Typically these would be empty,
            # though they could contain new location-specific prior
            # distributions.
            fm.reconfigure(*configs)

            if args.profile:

                # Profile output
                gbl, lcl = globals(), locals()
                cProfile.runctx('iv.invert(meas, geom, configs)', gbl, lcl)

            else:

                # The inversion returns a list of states, which are
                # intepreted either as samples from the posterior (MCMC case)
                # or as a gradient descent trajectory (standard case). For
                # a trajectory, the last spectrum is the converged solution.
                states = iv.invert(meas, geom)

        io.write_spectrum(row, col, states, meas, geom)


if __name__ == '__main__':
    main()
