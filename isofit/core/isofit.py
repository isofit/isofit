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
import warnings
import cProfile

from isofit.core.common import load_config
from isofit.core.forward import ForwardModel
from isofit.core.inverse import Inversion
from isofit.core.inverse_mcmc import MCMCInversion
from isofit.core.fileio import IO

import multiprocessing

# Suppress warnings that don't come from us
warnings.filterwarnings("ignore")


class Isofit:
    """Spectroscopic Surface and Atmosphere Fitting."""

    rows, cols = None, None
    config = None
    profile = None
    fm = None
    iv = None
    io = None
    states = None

    def __init__(self, config_file, level='INFO', row_column='', profile=False):
        """Initialize the class."""
        self.profile = profile
        # Set logging level
        logging.basicConfig(format='%(message)s', level=level)
        # Load configuration file
        self.config = load_config(config_file)
        # Build the forward model and inversion objects
        self._init_nonpicklable_objects()

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

    def _init_nonpicklable_objects(self):
        self.fm = ForwardModel(self.config['forward_model'])
        if 'mcmc_inversion' in self.config:
            self.iv = MCMCInversion(self.config['mcmc_inversion'], self.fm)
        else:
            self.iv = Inversion(self.config['inversion'], self.fm)

    def _clear_nonpicklable_objects(self):
        self.fm = None
        self.iv = None

    def _run_single_spectra(self, index):
        self._init_nonpicklable_objects()
        io = IO(self.config, self.fm, self.iv, self.rows, self.cols)
        success, row, col, meas, geom, configs = io.get_components_at_index(index)
        # Only run through the inversion if we got some data
        if success:
            if meas is not None and all(meas < -49.0):
                # Bad data flags
                self.states = []
            else:
                # Update model components with new configuration parameters
                # specific to this spectrum. Typically these would be empty,
                # though they could contain new location-specific prior
                # distributions.
                self.fm.reconfigure(*configs)

                # The inversion returns a list of states, which are
                # intepreted either as samples from the posterior (MCMC case)
                # or as a gradient descent trajectory (standard case). For
                # a trajectory, the last spectrum is the converged solution.
                self.states = self.iv.invert(meas, geom)

            # Write the spectra to disk
            self.io.write_spectrum(row, col, self.states, meas, geom)

    def run(self, profile=False):
        """
        Iterate over all spectra, reading and writing through the IO
        object to handle formatting, buffering, and deferred write-to-file.
        The idea is to avoid reading the entire file into memory, or hitting
        the physical disk too often. These are our main class variables.
        """

        io = IO(self.config, self.fm, self.iv, self.rows, self.cols)
        if profile:
            for row, col, meas, geom, configs in io:
                if meas is not None and all(meas < -49.0):
                    # Bad data flags
                    self.states = []
                else:
                    # Update model components with new configuration parameters
                    # specific to this spectrum. Typically these would be empty,
                    # though they could contain new location-specific prior
                    # distributions.
                    self.fm.reconfigure(*configs)
                    # Profile output
                    gbl, lcl = globals(), locals()
                    cProfile.runctx(
                        'self.iv.invert(meas, geom, configs)', gbl, lcl)

                # Write the spectra to disk
                self.io.write_spectrum(row, col, self.states, meas, geom)
        else:
            n_iter = len(io.iter_inds)
            io = None
            self._clear_nonpicklable_objects()
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

            logging.info('Beginning parallel inversions')

            results = []
            for l in range(n_iter):
                results.append(pool.apply_async(self._run_single_spectra, args=(l,)))
            results = [p.get() for p in results]
            pool.close()
            pool.join()

            logging.info('Parallel inversions complete')



