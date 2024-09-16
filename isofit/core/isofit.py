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

import click
import numpy as np

from isofit import ray
from isofit.configs import configs
from isofit.core.fileio import IO
from isofit.core.forward import ForwardModel
from isofit.inversion import Inversions
from isofit.utils.multistate import construct_full_state, index_image_by_class


class Isofit:
    """Initialize the Isofit class.

    Args:
        config_file: isofit configuration file in JSON or YAML format
        level: logging level (ERROR, WARNING, INFO, DEBUG)
        logfile: file to write output logs to
    """

    def __init__(self, config_file, level="INFO", logfile=None):
        # Explicitly set the number of threads to be 1, so we more effectively
        # run in parallel
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
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

        # Load configuration file
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

        # Set up the multi-state pixel map
        if self.config.forward_model.surface.multi_surface_flag:
            self.state_pixel_index = index_image_by_class(
                self.config.forward_model.surface
            )
        else:
            self.state_pixel_index = []

        # Construct and cache the full statevector (all multistates)
        self.full_statevector, *_ = construct_full_state(self.config)

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

        row_column: TGhe user can specify
            * a single number, in which case it is interpreted as a row
            * a comma-separated pair, in which case it is interpreted as a
              row/column tuple (i.e. a single spectrum)
            * a comma-separated quartet, in which case it is interpreted as
              a row, column range in the order (line_start, line_end, sample_start,
              sample_end) all values are inclusive.

            If none of the above, the whole cube will be analyzed.
        """

        logging.info("Building first forward model, will generate any necessary LUTs")

        # Get the number of workers from config
        if self.config.implementation.n_cores is None:
            n_workers = multiprocessing.cpu_count()
        else:
            n_workers = self.config.implementation.n_cores

        # Get the rows and columns that isofit will run
        # If running only part of the file
        if row_column is not None:
            ranges = row_column.split(",")
            if len(ranges) == 1:
                self.rows, self.cols = [int(ranges[0])], None
            if len(ranges) == 2:
                row_start, row_end = ranges
                self.rows = range(int(row_start), int(row_end))
                self.cols = None
            elif len(ranges) == 4:
                row_start, row_end, col_start, col_end = ranges
                self.rows = range(int(row_start), int(row_end) + 1)
                self.cols = range(int(col_start), int(col_end) + 1)

        # Else running all of the file
        else:
            io = IO(self.config, self.full_statevector)
            self.rows = range(io.n_rows)
            self.cols = range(io.n_cols)
            del io

        # Form the row-column pairs (pixels to run)
        index_pairs = np.vstack(
            [x.flatten(order="f") for x in np.meshgrid(self.rows, self.cols)]
        ).T

        # Save this for logging
        total_samples = index_pairs.shape[0]

        # Split into class if pixel classes are being propogated
        # If this is a multistate run
        if len(self.state_pixel_index):
            index_pairs_class = []
            for class_row_col in self.state_pixel_index:
                if not len(class_row_col):
                    continue

                class_row_col = np.array(class_row_col)
                index_pairs_class.append(index_pairs[class_row_col[:, 0]])

            index_pairs = index_pairs_class

        # Else it's not a multistate run
        else:
            index_pairs = [index_pairs]

        # Some logging that might be nice
        if len(index_pairs):
            logging.info("Multi-state inversion started.")
        else:
            logging.info("Single-state inversion started.")

        """
        Another pair of eyes on the mutiprocessing would be great here.
        There may easily be a better way to do this. Mostly setting 
        worker number on the samples within the loop rather than
        across the entire scene. It seems like we are losing
        some performance.
        """
        # Loop through index pairs and run workers
        class_loop_start_time = time.time()
        for i, index_pair in enumerate(index_pairs):

            # Don't want more workers than tasks
            n_iter = index_pair.shape[0]
            n_workers = min(n_workers, n_iter)

            # The number of tasks to be initialized
            n_tasks = min(
                (n_workers * self.config.implementation.task_inflation_factor), n_iter
            )

            # Get indices to pass to each worker
            index_sets = np.linspace(0, n_iter, num=n_tasks, dtype=int)
            if len(index_sets) == 1:
                indices_to_run = [index_pair[0:1, :]]
            else:
                indices_to_run = [
                    index_pair[index_sets[l] : index_sets[l + 1], :]
                    for l in range(len(index_sets) - 1)
                ]

            # Construct full fm
            self.fm = fm = ForwardModel(self.config)
            # Have to split these out to update the surface dynamically
            self.fm.construct_surface(str(i))
            self.fm.construct_state()

            logging.debug(f"Pixel class: {str(i)}")
            logging.debug(f"Surface: {self.fm.surface}")

            # Put worker args into Ray object
            params = [
                ray.put(obj)
                for obj in [
                    self.config,
                    self.fm,
                    self.full_statevector,
                    self.loglevel,
                    self.logfile,
                    n_workers,
                ]
            ]
            # Initialize Ray actor pool (Worker class)
            self.workers = ray.util.ActorPool(
                [Worker.remote(*params, n) for n in range(n_workers)]
            )

            start_time = time.time()
            logging.info(
                f"Beginning {n_iter} inversions in {n_tasks} chunks"
                f"using {n_workers} cores"
            )

            # Kick off actor pool
            res = list(
                self.workers.map_unordered(
                    lambda a, b: a.run_set_of_spectra.remote(b), indices_to_run
                )
            )

            total_time = time.time() - start_time
            logging.info(
                f"Inversions complete.  {round(total_time,2)}s total,"
                f" {round(n_iter/total_time,4)} spectra/s,"
                f" {round(n_iter/total_time/n_workers,4)} spectra/s/core"
            )

        if len(index_pairs):
            class_loop_total_time = time.time() - class_loop_start_time
            logging.info(
                f"All Inversions complete. {round(class_loop_total_time,2)}s total,"
                f" {round(total_samples/class_loop_total_time,4)} spectra/s,"
                f" {round(total_samples/class_loop_total_time/n_workers,4)} spectra/s/core"
            )


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: configs.Config,
        forward_model: ForwardModel,
        full_statevector: np.array,
        loglevel: str,
        logfile: str,
        total_workers: int = None,
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
        self.config = config
        self.fm = forward_model

        self.io = IO(self.config, full_statevector)

        self.approximate_total_spectra = None
        if total_workers is not None:
            self.approximate_total_spectra = (
                self.io.n_cols * self.io.n_rows / total_workers
            )
        self.worker_id = worker_id
        self.completed_spectra = 0

    def run_set_of_spectra(self, indices: np.array):
        for index in range(0, indices.shape[0]):
            logging.debug("Read chunk of spectra")
            row, col = indices[index, 0], indices[index, 1]

            # Get input data
            input_data = self.io.get_components_at_index(row, col)

            # Get inversion
            iv = Inversions.get(self.config.implementation.mode, None)
            if not iv:
                logging.exception(
                    "Inversion implementation: "
                    f"{self.config.implementation.mode}, "
                    "did not match options"
                )
                raise KeyError
            self.iv = iv(self.config, self.fm)

            self.completed_spectra += 1
            if input_data is not None:
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
                    if (
                        self.worker_id is not None
                        and self.approximate_total_spectra is not None
                    ):
                        percent = np.round(
                            self.completed_spectra
                            / self.approximate_total_spectra
                            * 100,
                            2,
                        )
                        logging.info(
                            f"Worker {self.worker_id} completed"
                            f" {self.completed_spectra}/~{self.approximate_total_spectra}::"
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
def cli_run(config_file, level):
    """Execute ISOFIT core"""

    click.echo(f"Running ISOFIT(config_file={config_file!r}, level={level})")

    logging.basicConfig(format="%(message)s", level=level)
    Isofit(config_file=config_file, level=level).run()

    click.echo("Done")
