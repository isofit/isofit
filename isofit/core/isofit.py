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

import os
import logging
import cProfile
import time

from isofit.core import common
from .forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_mcmc import MCMCInversion
from .fileio import IO

import multiprocessing
from isofit import configs
from isofit.configs import configs
import numpy as np


class Isofit:
    """Spectroscopic Surface and Atmosphere Fitting."""

    def __init__(self, config_file, row_column='', profile=False, level='INFO'):
        """Initialize the Isofit class."""

        # Explicitly set the number of threads to be 1, so we can make better
        # use of multiprocessing
        os.environ["MKL_NUM_THREADS"] = "1"

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
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

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
        self.fm = ForwardModel(self.config)

        if self.config.implementation.mode == 'mcmc_inversion':
            self.iv = MCMCInversion(self.config, self.fm)
        elif self.config.implementation.mode in ['inversion', 'simulation']:
            self.iv = Inversion(self.config, self.fm)
        else:
            # This should never be reached due to configuration checking
            raise AttributeError('Config implementation mode node valid')

    def _clear_nonpicklable_objects(self):
        self.fm = None
        self.iv = None

    def _run_set_of_spectra(self, index_start, index_stop):
        self._init_nonpicklable_objects()
        io = IO(self.config, self.fm, self.iv, self.rows, self.cols)
        for index in range(index_start, index_stop):
            success, row, col, meas, geom = io.get_components_at_index(
                index)
            # Only run through the inversion if we got some data
            if success:
                if meas is not None and all(meas < -49.0):
                    # Bad data flags
                    self.states = []
                else:
                    # The inversion returns a list of states, which are
                    # intepreted either as samples from the posterior (MCMC case)
                    # or as a gradient descent trajectory (standard case). For
                    # a trajectory, the last spectrum is the converged solution.
                    self.states = self.iv.invert(meas, geom)

                # Write the spectra to disk
                io.write_spectrum(row, col, self.states, meas,
                                  geom, flush_immediately=True)
                if index % 1000 == 0:
                    logging.info(
                        'Completed inversion {}/{}'.format(index, len(io.iter_inds)))

    def run(self, profile=False):
        """
        Iterate over all spectra, reading and writing through the IO
        object to handle formatting, buffering, and deferred write-to-file.
        The idea is to avoid reading the entire file into memory, or hitting
        the physical disk too often. These are our main class variables.
        """

        io = IO(self.config, self.fm, self.iv, self.rows, self.cols)
        if profile:
            for row, col, meas, geom in io:
                if meas is not None and all(meas < -49.0):
                    # Bad data flags
                    self.states = []
                else:
                    # Profile output
                    gbl, lcl = globals(), locals()
                    cProfile.runctx(
                        'self.iv.invert(meas, geom)', gbl, lcl)

                # Write the spectra to disk
                io.write_spectrum(row, col, self.states, meas, geom)
        else:
            n_iter = len(io.iter_inds)
            io = None
            self._clear_nonpicklable_objects()

            if self.config.implementation.n_cores is None:
                n_cores = multiprocessing.cpu_count()
            else:
                n_cores = self.config.implementation.n_cores

            # Don't use more cores than needed
            n_cores = min(n_cores, n_iter)

            if self.config.implementation.runtime_nice_level is None:
                pool = multiprocessing.Pool(processes=n_cores)
            else:
                pool = multiprocessing.Pool(processes=n_cores, initializer=
                                            common.nice_me(self.config.implementation.runtime_nice_level))

            start_time = time.time()
            logging.info('Beginning inversions using {} cores'.format(n_cores))

            results = []
            index_sets = np.linspace(0, n_iter, num=n_cores+1, dtype=int)
            for l in range(len(index_sets)-1):
                if n_cores == 1:
                    self._run_set_of_spectra(index_sets[0], index_sets[-1])
                else:
                    results.append(pool.apply_async(self._run_set_of_spectra, args=(index_sets[l], index_sets[l + 1])))
            results = [p.get() for p in results]
            pool.close()
            pool.join()

            total_time = time.time() - start_time
            logging.info('Parallel inversions complete.  {} s total, {} spectra/s, {}/spectra/core'.format(
                total_time, n_iter/total_time, n_iter/total_time/n_cores))
