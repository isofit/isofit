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
#         Adam Erickson, adam.m.erickson@nasa.gov
#

import logging
import cProfile
from contextlib import suppress
import warnings
from numba.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from .common import load_config
from .forward import ForwardModel
from .inverse import Inversion
from .inverse_mcmc import MCMCInversion
from .fileio import IO

from .. import warnings_enabled


class Isofit:
    """Spectroscopic Surface and Atmosphere Fitting."""

    def __init__(self, config_file, row_column='', profile=False, level='INFO'):
        """Initialize the Isofit class."""

        # Set logging level
        logging.basicConfig(format='%(message)s', level=level)

        self.rows = None
        self.cols = None
        self.config = None
        self.profile = profile
        self.fm = None
        self.iv = None
        self.io = None
        self.states = None

        # Load configuration file
        self.config = load_config(config_file)

        # Build the forward model and inversion objects
        self.fm = ForwardModel(self.config['forward_model'])

        if 'mcmc_inversion' in self.config:
            self.iv = MCMCInversion(self.config['mcmc_inversion'], self.fm)
        else:
            self.iv = Inversion(self.config['inversion'], self.fm)

        # We set the row and column range of our analysis. The user can
        # specify: a single number, in which case it is interpreted as a row;
        # a comma-separated pair, in which case it is interpreted as a
        # row/column tuple (i.e. a single spectrum); or a comma-separated
        # quartet, in which case it is interpreted as a row, column range in the
        # order (line_start, line_end, sample_start, sample_end) - all values are
        # inclusive. If none of the above, we will analyze the whole cube.
        if len(row_column) > 0:
            ranges = row_column.split(',')
            if len(ranges) == 1:
                self.rows, self.cols = [int(ranges[0])], None
            if len(ranges) == 2:
                row_start, row_end = ranges
                self.rows, self.cols = range(
                    int(row_start), int(row_end)), None
            elif len(ranges) == 4:
                row_start, row_end, col_start, col_end = ranges
                line_start, line_end, samp_start, samp_end = ranges
                self.rows = range(int(row_start), int(row_end))
                self.cols = range(int(col_start), int(col_end))

        # Run the model
        self.__call__()

    def __call__(self):
        """
        Iterate over all spectra, reading and writing through the IO
        object to handle formatting, buffering, and deferred write-to-file.
        The idea is to avoid reading the entire file into memory, or hitting
        the physical disk too often. These are our main class variables.
        """

        # Ignore Numba warnings
        if not warnings_enabled:
            warnings.simplefilter(
                action='ignore', category=RuntimeWarning)
            warnings.simplefilter(
                action='ignore', category=NumbaWarning)
            warnings.simplefilter(
                action='ignore', category=NumbaDeprecationWarning)
            warnings.simplefilter(
                action='ignore', category=NumbaPendingDeprecationWarning)

        self.io = IO(self.config, self.fm, self.iv, self.rows, self.cols)
        for row, col, meas, geom, configs in self.io:
            if meas is not None and all(meas < -49.0):
                # Bad data flags
                self.states = []
            else:
                # Update model components with new configuration parameters
                # specific to this spectrum. Typically these would be empty,
                # though they could contain new location-specific prior
                # distributions.

                ### RAISES interp1d() ERROR! ###
                # self.fm.reconfigure(*configs)
                ###

                if self.profile:
                    # Profile output
                    gbl, lcl = globals(), locals()
                    cProfile.runctx(
                        'self.iv.invert(meas, geom, configs)', gbl, lcl)
                else:
                    # The inversion returns a list of states, which are
                    # intepreted either as samples from the posterior (MCMC case)
                    # or as a gradient descent trajectory (standard case). For
                    # a trajectory, the last spectrum is the converged solution.
                    self.states = self.iv.invert(meas, geom)

            # Write the spectra to disk
            self.io.write_spectrum(row, col, self.states, meas, geom)
