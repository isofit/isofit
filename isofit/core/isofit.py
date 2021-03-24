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
# Authors: David R Thompson, david.r.thompson@jpl.nasa.gov
#          Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#          Adam Erickson, adam.m.erickson@nasa.gov
#

import os
import logging
import time

import multiprocessing
import numpy as np
import ray
import ray.services

from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_mcmc import MCMCInversion
from isofit.core.fileio import IO
from isofit.configs import configs


class Isofit:
    """Initialize the Isofit class.

    Args:
        config_file: isofit configuration file in JSON or YAML format
        level: logging level (ERROR, WARNING, INFO, DEBUG)
        logfile: file to write output logs to
    """

    def __init__(self, config_file, row_column='', level='INFO', logfile=None):

        # Explicitly set the number of threads to be 1, so we more effectively
        # run in parallel
        os.environ["MKL_NUM_THREADS"] = "1"

        # Set logging level
        self.loglevel = level
        self.logfile = logfile
        logging.basicConfig(format='%(levelname)s:%(message)s', level=self.loglevel, filename=self.logfile)

        self.rows = None
        self.cols = None
        self.config = None

        # Load configuration file
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

        # Initialize ray for parallel execution
        rayargs = {'address': self.config.implementation.ip_head,
                   '_redis_password': self.config.implementation.redis_password,
                   'ignore_reinit_error': self.config.implementation.ray_ignore_reinit_error,
                   'local_mode': self.config.implementation.n_cores == 1}

        # only specify a temporary directory if we are not connecting to 
        # a ray cluster
        if rayargs['local_mode']:
            rayargs['_temp_dir'] = self.config.implementation.ray_temp_dir
            # Used to run on a VPN
            ray.services.get_node_ip_address = lambda: '127.0.0.1'

        # We can only set the num_cpus if running on a single-node
        if self.config.implementation.ip_head is None and self.config.implementation.redis_password is None:
            rayargs['num_cpus'] = self.config.implementation.n_cores
        ray.init(**rayargs)

        self.workers = None


    def __del__(self):
        ray.shutdown()


    def run(self, row_column = None):
        """
        Iterate over spectra, reading and writing through the IO
        object to handle formatting, buffering, and deferred write-to-file.
        Attempts to avoid reading the entire file into memory, or hitting
        the physical disk too often.

        row_column: The user can specify
            * a single number, in which case it is interpreted as a row
            * a comma-separated pair, in which case it is interpreted as a
              row/column tuple (i.e. a single spectrum)
            * a comma-separated quartet, in which case it is interpreted as
              a row, column range in the order (line_start, line_end, sample_start,
              sample_end) all values are inclusive.

            If none of the above, the whole cube will be analyzed.
        """

        if row_column is not None:
            ranges = row_column.split(',')
            if len(ranges) == 1:
                self.rows, self.cols = [int(ranges[0])], None
            if len(ranges) == 2:
                row_start, row_end = ranges
                self.rows, self.cols = range(
                    int(row_start), int(row_end)), None
            elif len(ranges) == 4:
                row_start, row_end, col_start, col_end = ranges
                self.rows = range(int(row_start), int(row_end) + 1)
                self.cols = range(int(col_start), int(col_end) + 1)
        else:
            io = IO(self.config, ForwardModel(self.config), None)
            self.rows = range(io.n_rows)
            self.cols = range(io.n_cols)
            del io

        index_pairs = np.vstack([x.flatten(order='f') for x in np.meshgrid(self.rows, self.cols)]).T

        n_iter = index_pairs.shape[0]

        if self.config.implementation.n_cores is None:
            n_workers = multiprocessing.cpu_count()
        else:
            n_workers = self.config.implementation.n_cores

        # Max out the number of workers based on the number of tasks
        n_workers = min(n_workers, n_iter)

        if self.workers is None:
            remote_worker = ray.remote(Worker)
            self.workers = ray.util.ActorPool([remote_worker.remote(self.config, self.loglevel, self.logfile)
                                               for n in range(n_workers)])

        start_time = time.time()
        logging.info('Beginning inversions using {} cores'.format(n_workers))

        # Divide up spectra to run into chunks
        index_sets = np.linspace(0, n_iter, num=n_workers+1, dtype=int)

        # Run spectra, in either serial or parallel depending on n_workers
        res = list(self.workers.map_unordered(lambda a, b: a.run_set_of_spectra.remote(b),
                   [index_pairs[index_sets[l]:index_sets[l+1],:] for l in range(len(index_sets)-1)]))

        total_time = time.time() - start_time
        logging.info('Inversions complete.  {} s total, {} spectra/s, {} spectra/s/core'.format(
            round(total_time,2), round(n_iter/total_time,4), round(n_iter/total_time/n_workers,4)))


class Worker(object):
    def __init__(self, config: configs.Config, loglevel: str, logfile: str):
        self.config = config
        self.loglevel= loglevel
        self.logfile= logfile
        self.fm = ForwardModel(self.config)

        if self.config.implementation.mode == 'mcmc_inversion':
            self.iv = MCMCInversion(self.config, self.fm)
        elif self.config.implementation.mode in ['inversion', 'simulation']:
            self.iv = Inversion(self.config, self.fm)
        else:
            # This should never be reached due to configuration checking
            raise AttributeError('Config implementation mode node valid')

        self.io = IO(self.config, self.fm, self.iv)


    def run_set_of_spectra(self, indices: np.array):

        logging.basicConfig(format='%(levelname)s:%(message)s', level=self.loglevel, filename=self.logfile)

        for index in range(0, indices.shape[0], self.config.implementation.io_buffer_size):

            logging.debug("Read chunk of spectra")
            #TODO - expand for multi-pixel buffer size
            # TODO - add IO flag...hard-coded -49 has been placed in IO
            row, col = indices[index,0], indices[index,1]
            success, meas, geom = self.io.get_components_at_index(row, col)

            if success:
                logging.debug("Run model")
                # The inversion returns a list of states, which are
                # intepreted either as samples from the posterior (MCMC case)
                # or as a gradient descent trajectory (standard case). For
                # a trajectory, the last spectrum is the converged solution.
                states = self.iv.invert(meas, geom)

                logging.debug("Write chunk of spectra")
                # Write the spectra to disk
                try:
                    #TODO: Cadapt for multi-pixel buffer size
                    self.io.write_spectrum(row, col, states, meas, geom, flush_immediately=True)
                except ValueError as err:
                    logging.info(
                        f"""
                        Encountered the following ValueError in (row,col) ({row},{col}).
                        Results for this pixel will be all zeros.
                        """
                    )
                    logging.error(err)
                if index % 100 == 0:
                    logging.info(f'Core at start location ({row},{col}) completed {index}/{indices.shape[0]}')




