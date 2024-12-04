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
import logging
import multiprocessing
import os
import time
from itertools import product

# Explicitly set the number of threads to be 1, so we more effectively run in parallel
# Must be executed before importing numpy, otherwise doesn't work
if not os.environ.get("ISOFIT_NO_SET_THREADS"):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import click
import numpy as np

from isofit import checkNumThreads, ray
from isofit.configs import configs
from isofit.core import common
from isofit.core.fileio import IO
from isofit.core.forward import ForwardModel
from isofit.inversion import Inversion

Logger = logging.getLogger(__file__)


class Isofit:
    """Initialize the Isofit class.

    Args:
        config_file: isofit configuration file in JSON or YAML format
        level: logging level (ERROR, WARNING, INFO, DEBUG)
        logfile: file to write output logs to
    """

    def __init__(self, config_file, loglevel="INFO", logfile=None):
        # Check the MKL/OMP env vars and raise a warning if not set properly
        checkNumThreads()

        self.rows = None
        self.cols = None
        self.config = None

        # Load configuration file
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

        # Initialize ray for parallel execution
        rayargs = {
            "address": self.config.implementation.ip_head,
            "_redis_password": self.config.implementation.redis_password,
            "_temp_dir": self.config.implementation.ray_temp_dir,
            "ignore_reinit_error": self.config.implementation.ray_ignore_reinit_error,
            "include_dashboard": self.config.implementation.ray_include_dashboard,
            "local_mode": self.config.implementation.n_cores == 1,
        }

        # We can only set the num_cpus if running on a single-node
        if (
            self.config.implementation.ip_head is None
            and self.config.implementation.redis_password is None
        ):
            rayargs["num_cpus"] = self.config.implementation.n_cores

        ray.init(**rayargs)

    def run(self, row_column=None, process=True):
        """
        Iterate over spectra, reading and writing through the IO object to handle
        formatting, buffering, and deferred write-to-file. Attempts to avoid reading
        the entire file into memory or hitting the physical disk too often.

        Parameters
        ----------
        row_column : int | tuple[int, int] | tuple[int, int, int, int], default=None
            int
                Row
            tuple[int, int]
                Row / column, single spectrum
            tuple[int, int, int, int]
                Row, column in the order
                (line start, line end, sample start, sample end)
                All values are inclusive
        process : bool, default=True
            Process the indices. If False, returns the indices for a user to manually
            process

        Returns
        -------
        indices : list[tuple[int, int]]
            List of indices to process / were processed
        """
        Logger.info("Building first forward model, will generate any necessary LUTs")
        self.fm = fm = ForwardModel(self.config)
        self.io = io = IO(self.config, fm)

        iv = Inversion(self.config, fm)

        if row_column is not None:
            ranges = row_column.split(",")
            if len(ranges) == 1:
                self.rows, self.cols = [int(ranges[0])], [None]
            if len(ranges) == 2:
                row_start, row_end = ranges
                self.rows, self.cols = range(int(row_start), int(row_end)), [None]
            elif len(ranges) == 4:
                row_start, row_end, col_start, col_end = ranges
                self.rows = range(int(row_start), int(row_end) + 1)
                self.cols = range(int(col_start), int(col_end) + 1)
        else:
            self.rows = range(io.n_rows)
            self.cols = range(io.n_cols)

        indices = list(product(self.rows, self.cols))
        cores = ray.cluster_resources()["CPU"]

        Logger.info(f"Beginning {len(indices)} inversions over {cores} cores")

        self.params = [ray.put(obj) for obj in (self.config, fm, iv)]

        # TODO: Smart split based off output file chunking
        limit = 50_000
        split = int(len(indices) / limit) + 1
        batches = np.array_split(indices, split)

        if (total := len(batches)) > 1:
            Logger.info(f"Job limit is {limit}, split work into {total} batches")

        if process:
            for i, batch in enumerate(batches):
                Logger.debug(f"Processing batch {i+1}/{total}")
                self.process(batch)

        return indices

    def process(self, indices):
        """
        Processes a list of indices through the run_spectra function

        Parameters
        ----------
        indices : list[tuple[int, int]]
            List of indices to process
        """
        jobs = [run_spectra.remote(index, *self.params) for index in indices]

        # Report a percentage complete every 10% and flush to disk at those intervals
        errors = []
        report = common.Track(
            jobs,
            step=10,
            reverse=True,
            message="inversions complete",
            print=Logger.info,
        )

        # Update the output as inversions stream in
        while jobs:
            [done], jobs = ray.wait(jobs, num_returns=1)

            # Retrieve the return of the finished job
            ret = ray.get(done)

            if ret:
                index, output, states = ret
                try:
                    self.io.write_datasets(*index, output, states)
                except:
                    errors.append(index)

            if report(len(jobs)):
                self.io.flush_buffers()

        # One last flush, just in case
        self.io.flush_buffers()

        if errors:
            Logger.error(
                f"{len(errors)} spectra encountered an error during inversion, see debug for more"
            )
            Logger.debug(f"Indices that errored: {errors}")

        time = report.elap.total_seconds()

        Logger.info("Inversions completed")
        Logger.info(
            ", ".join(
                [
                    f"{time:.2f}s total",
                    f"{report.total/time:.2f} spectra/s",
                    f"{report.total/time/cores:.2f} spectra/s/core",
                ]
            )
        )


@ray.remote(num_cpus=1)
def run_spectra(
    index: np.array,
    config: configs.Config,
    fm: ForwardModel,
    iv: Inversion,
):
    """
    Inverts a single spectra

    Parameters
    ----------
    index : tuple[int, int]
        Index to retrieve component data
    config : Config
        ISOFIT configuration to initialize an IO object with
    fm : ForwardModel
        ForwardModel to initialize IO with
    iv : Inversion
        Inversion model for inverting the spectra

    Returns
    -------
    index, output, states
        Output is the prepared output to be written to disk
    """
    io = IO(config, fm)
    data = io.get_components_at_index(*index)

    if data is not None:
        states = iv.invert(data.meas, data.geom)
        output = io.build_output(states, io.current_input_data, fm, iv)

        return index, output, states


@click.command(name="run")
@click.argument("config_file")
@click.option(
    "-ll",
    "--loglevel",
    help="Terminal log level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"], case_sensitive=False
    ),
    default="INFO",
)
@click.option("-lf", "--logfile", help="Output log file")
def cli_run(config_file, loglevel, logfile):
    """Execute ISOFIT core"""

    print(f"Running ISOFIT(config_file={config_file!r})")

    Isofit(config_file=config_file, loglevel=loglevel.upper(), logfile=logfile).run()

    print("Done")
