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
from copy import deepcopy

# Explicitly set the number of threads to be 1, so we more effectively run in parallel
# Must be executed before importing numpy, otherwise doesn't work
if not os.environ.get("ISOFIT_NO_SET_THREADS"):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import click
import numpy as np
import scipy

from isofit import checkNumThreads, ray
from isofit.configs import configs
from isofit.core.fileio import IO, SpectrumFile
from isofit.core.forward import ForwardModel
from isofit.core.multistate import (
    construct_full_state,
    index_spectra_by_surface,
    update_config_for_surface,
)
from isofit.data import env
from isofit.inversion import Inversion


class Isofit:
    """Initialize the Isofit class.

    Args:
        config_file: isofit configuration file in JSON or YAML format
        level: logging level (ERROR, WARNING, INFO, DEBUG)
        logfile: file to write output logs to
    """

    def __init__(self, config_file, level="INFO", logfile=None):
        # Check the MKL/OMP env vars and raise a warning if not set properly
        checkNumThreads()

        # Set logging level
        self.loglevel = level
        self.logfile = logfile
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=self.loglevel,
            filename=self.logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.rows = None
        self.cols = None
        self.config = None

        if config_file.endswith(".tmpl"):
            config_file = env.fromTemplate(config_file)

        # Load configuration file
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

        # Construct and track the full statevector (all surfaces)
        self.full_statevector, *_ = construct_full_state(deepcopy(self.config))

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

        self.workers = None

    def run(self, row_column=None):
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

        # Get the number of workers from config
        if self.config.implementation.n_cores is None:
            n_cores = multiprocessing.cpu_count()
        else:
            n_cores = self.config.implementation.n_cores

        rdn = SpectrumFile(self.config.input.measured_radiance_file, write=False)
        self.rows = range(rdn.n_rows)
        self.cols = range(rdn.n_cols)

        # Initialize files in __init__, otherwise workers fail
        IO.initialize_output_files(
            self.config, rdn.n_rows, rdn.n_cols, self.full_statevector
        )
        del rdn

        # Handle case where you only want to run part of an image
        # TODO Clean this, not sure if currently functioning.
        if row_column is not None:
            ranges = row_column.split(",")
            if len(ranges) == 1:
                self.rows, self.cols = [int(ranges[0])], None
            if len(ranges) == 2:
                row_start, row_end = ranges
                self.rows, self.cols = range(int(row_start), int(row_end)), None
            elif len(ranges) == 4:
                row_start, row_end, col_start, col_end = ranges
                self.rows = range(int(row_start), int(row_end) + 1)
                self.cols = range(int(col_start), int(col_end) + 1)

        # Form the row-column pairs (pixels to run)
        # Need to allocate cols of index_pairs together
        # to make them memory contiguous
        # This speeds up the surface indexing
        index_pairs = np.empty(
            (len([i for i in self.rows]) * len([i for i in self.cols]), 2), dtype=int
        )
        meshgrid = np.meshgrid(self.rows, self.cols)
        index_pairs[:, 0] = meshgrid[0].flatten(order="f")
        index_pairs[:, 1] = meshgrid[1].flatten(order="f")
        del meshgrid

        index_pairs = np.vstack(
            [x.flatten(order="f") for x in np.meshgrid(self.rows, self.cols)]
        ).T

        # Save this for logging
        total_samples = index_pairs.shape[0]

        # Keep track of the input version of the config
        input_config = deepcopy(self.config)

        # Loop through index pairs and run workers
        outer_loop_start_time = time.time()

        cache_RT = None
        surface_index = index_spectra_by_surface(input_config, index_pairs)
        for i, (surface_class_str, class_idx_pairs) in enumerate(surface_index.items()):
            logging.info(f"Running surfaces: {surface_class_str}")
            if not len(class_idx_pairs):
                logging.info(
                    f"No pixels found in image for surface: {surface_class_str}"
                )
                continue

            # Don't want more workers than tasks
            n_iter = class_idx_pairs.shape[0]
            n_workers = min(n_cores, n_iter)

            # The number of tasks to be initialized
            n_tasks = min(
                (n_workers * input_config.implementation.task_inflation_factor), n_iter
            )

            # Get indices to pass to each worker
            index_sets = np.linspace(0, n_iter, num=n_tasks, dtype=int)
            if len(index_sets) == 1:
                indices_to_run = [class_idx_pairs[0:1, :]]
            else:
                indices_to_run = [
                    class_idx_pairs[index_sets[l] : index_sets[l + 1], :]
                    for l in range(len(index_sets) - 1)
                ]

            # If multisurface, update config to reflect surface.
            # Otherwise, returns itself
            config = update_config_for_surface(
                deepcopy(input_config), surface_class_str
            )

            # Set forward model
            fm = ForwardModel(config, cache_RT=cache_RT)

            logging.debug(f"Surface: {surface_class_str}")

            # Put worker args into Ray object
            params = [
                ray.put(obj)
                for obj in [
                    config,
                    fm,
                    self.loglevel,
                    self.logfile,
                    self.full_statevector,
                    len(class_idx_pairs),
                    n_workers,
                ]
            ]

            # Initialize Ray actor pool (Worker class)
            self.workers = ray.util.ActorPool(
                [Worker.remote(*params, n) for n in range(n_workers)]
            )

            start_time = time.time()
            logging.info(
                f"Beginning {n_iter} inversions in {n_tasks} chunks "
                f"using {n_workers} cores"
            )

            # Kick off actor pool
            res = list(
                self.workers.map_unordered(
                    lambda a, b: a.run_set_of_spectra.remote(b), indices_to_run
                )
            )

            total_time = time.time() - start_time
            logging.info(f"Pixel class: {surface_class_str} inversions complete.")
            logging.info(f"{round(total_time,2)}s total")
            logging.info(f"{round(n_iter/total_time,4)} spectra/s")
            logging.info(f"{round(n_iter/total_time/n_workers,4)} spectra/s/core")

            # Not sure if it's best practice to null out these vars
            self.workers = None
            params = None

            # Cache RT
            if not i:
                cache_RT = fm.RT

            del fm

        if len(index_pairs):
            outer_loop_total_time = time.time() - outer_loop_start_time
            logging.info(f"All Inversions complete.")
            logging.info(f"{round(outer_loop_total_time,2)}s total")
            logging.info(f"{round(total_samples/outer_loop_total_time,4)} spectra/s")
            logging.info(
                f"{round(total_samples/outer_loop_total_time/n_workers,4)} spectra/s/core"
            )


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: configs.Config,
        forward_model: ForwardModel,
        loglevel: str,
        logfile: str,
        full_statevector: np.array = [],
        total_samples: int = 1,
        total_workers: int = 1,
        worker_id: int = None,
    ):
        """
        Worker class to help run a subset of spectra.

        Args:
            config: isofit configuration
            loglevel: output logging level
            logfile: output logging file
            worker_id: worker ID for logging reference
            total_workers: the total number of workers running, for logging reference
        """

        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        # If full image statevector isn't passed, use forward model
        if not len(full_statevector):
            full_statevector = forward_model.statevec

        self.config = config
        self.fm = forward_model
        self.iv = Inversion(self.config, self.fm)
        self.io = IO(self.config, self.fm, full_statevec=full_statevector)
        self.test = "test"

        self.total_samples = None
        if total_workers is not None:
            self.total_samples = np.floor(total_samples / total_workers)

        self.worker_id = worker_id
        self.completed_spectra = 0

    def run_set_of_spectra(self, indices: np.array):
        for index in range(0, indices.shape[0]):
            logging.debug("Read chunk of spectra")
            row, col = indices[index, 0], indices[index, 1]

            input_data = self.io.get_components_at_index(row, col)

            self.completed_spectra += 1
            if input_data is not None:
                nan_locs = np.where(np.isnan(input_data.meas))
                if len(nan_locs) > 0:
                    non_nan_locs = np.where(np.isnan(input_data.meas) == False)
                    interp = scipy.interpolate.interp1d(
                        self.io.meas_wl[non_nan_locs],
                        input_data.meas[non_nan_locs],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    input_data.meas = interp(self.io.meas_wl)

                logging.debug("Run model")
                # The inversion returns a list of states, which are
                # intepreted either as samples from the posterior (MCMC case)
                # or as a gradient descent trajectory (standard case). For
                # a trajectory, the last spectrum is the converged solution.
                states = self.iv.invert(input_data.meas, input_data.geom)

                logging.debug("Write chunk of spectra")
                # Write the spectra to disk
                try:
                    self.io.write_spectrum(row, col, states, self.fm, self.iv)

                except ValueError as err:
                    logging.exception(
                        f"""
                    Encountered the following ValueError in (row,col) ({row},{col}).
                    Results for this pixel will be all zeros.
                    """
                    )

                if index % 100 == 0:
                    if self.worker_id is not None and self.total_samples is not None:
                        percent = np.round(
                            self.completed_spectra / self.total_samples * 100,
                            2,
                        )
                        logging.info(
                            f"Worker {self.worker_id} completed"
                            f" {self.completed_spectra}/~{self.total_samples}::"
                            f" {percent}% complete"
                        )

        logging.info(
            f"Worker at start location ({row},{col}) completed"
            f" {index}/{indices.shape[0]}"
        )

        self.io.flush_buffers()


@click.command(name="run")
@click.argument("config_file")
@click.option(
    "-l",
    "--level",
    help="Log level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"], case_sensitive=True
    ),
    default="INFO",
)
@click.option("--log_file")
def cli(config_file, level, log_file):
    """Execute ISOFIT core"""

    click.echo(
        f"Running ISOFIT(config_file={config_file!r}, level={level}, logfile={log_file})"
    )
    Isofit(config_file=config_file, level=level, logfile=log_file).run()

    click.echo("Done")
