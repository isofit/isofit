"""
Manages the generation of new look-up tables (LUTs)
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from isofit import ray
from isofit.core.common import Track, combos

Logger = logging.getLogger(__name__)


class Writer:
    """
    Handles writing RTE engine simulation data to an ISOFIT-compatible LUT store
    """

    # Allows engines to outright disable the parallelized sims if they do nothing
    _disable_makeSim = False

    """Sleep a random amount of time up to max this value
    at the start of each streamSimulation
    Can be set per custom engine"""
    max_buffer_time = 0

    def write(self):
        """
        Initialize a LUT and run simulations
        """
        for attr in ("lut_path", "lut_names", "lut_grid", "wl", "keys"):
            if getattr(self, attr) is None:
                raise AttributeError(
                    f"Missing required attribute to write the LUT: {attr}"
                )

        if not configure_and_exit:
            self.lut = Create(
                file=self.lut_path,
                keys=self.keys,
                wl=self.wl,
                grid=self.lut_grid,
                onedim=self.onedim,
            )
            self.runSimulations()

    def runSimulations(self, lut_names, lut_grid, configure_and_exit=False) -> None:
        """
        Run all simulations for the LUT grid.

        """
        Logger.info(f"Running any pre-sim functions")
        pre = self.preSim()

        if pre:
            Logger.info("Saving pre-sim data to index zero of all dimensions except wl")
            Logger.debug(f"pre-sim data contains keys: {pre.keys()}")

            point = {key: 0 for key in lut_names}
            self.lut.writePoint(point, data=pre)

        # Make the LUT calls (in parallel if specified)
        if not self._disable_makeSim:
            Logger.info("Executing parallel simulations")

            # Place into shared memory space to avoid spilling
            ray_lut_names = ray.put(lut_names)
            makeSim = ray.put(self.makeSim)
            readSim = ray.put(self.readSim)
            ray_lut_path = ray.put(self.lut.file)
            ray_buffer_time = ray.put(self.max_buffer_time)
            ray_configure_and_exit = ray.put(configure_and_exit)

            points = combos(lut_grid.values())

            jobs = [
                self.streamSimulation.remote(
                    point,
                    ray_lut_names,
                    makeSim,
                    readSim,
                    max_buffer_time=ray_buffer_time,
                    configure_and_exit=ray_configure_and_exit,
                )
                for point in points
            ]

            if configure_and_exit:
                # Block until all jobs finish
                ray.get(jobs)

                Logger.warning("Exiting early due to configure_and_exit")
                sys.exit(0)
            else:
                # Report a percentage complete every 10% and flush to disk at those intervals
                report = Track(
                    jobs,
                    step=10,
                    reverse=True,
                    print=Logger.info,
                    message="simulations complete",
                )

                # Update the lut as point simulations stream in
                while jobs:
                    [done], jobs = ray.wait(jobs, num_returns=1)

                    # Retrieve the return of the finished job
                    ret = ray.get(done)

                    # If a simulation fails then it will return None
                    if ret:
                        self.lut.queuePoint(*ret)

                    if report(len(jobs)):
                        Logger.info("Flushing netCDF to disk")
                        self.lut.flush()

                # Shouldn't be hit but just in case
                if self.lut.hold:
                    Logger.warning("Not all points were flushed, doing so now")
                    self.lut.flush()

            del (
                makeSim,
                readSim,
                ray_lut_names,
                ray_lut_path,
                ray_buffer_time,
            )
        else:
            Logger.debug("makeSim is disabled for this engine")

        Logger.info(f"Running any post-sim functions")
        post = self.postSim()

        if post:
            Logger.info(
                "Saving post-sim data to index zero of all dimensions except wl"
            )
            Logger.debug(f"post-sim data contains keys: {post.keys()}")

            point = {key: 0 for key in lut_names}
            self.lut.writePoint(point, data=post)

        self.lut.finalize()

    @ray.remote(num_cpus=1)
    def streamSimulation(
        point: np.array,
        lut_names: list,
        simmer: Callable,
        reader: Callable,
        max_buffer_time: float = 0.5,
        configure_and_exit: bool = False,
    ):
        """Run a simulation for a single point and stream the results to a saved lut file.

        Args:
            point (np.array): conditions to alter in simulation
            lut_names (list): Dimension names aka lut_names
            simmer (function): function to run the simulation
            reader (function): function to read the results of the simulation
            max_buffer_time (float, optional): _description_. Defaults to 0.5.
            configure_and_exit (bool, optional): exit early if not executing simulations
        """
        Logger.debug(f"Simulating(point={point})")

        # Slight delay to prevent all subprocesses from starting simultaneously
        time.sleep(np.random.rand() * max_buffer_time)

        # Execute the simulation
        simmer(point)

        # No data will be produced, just configuration files
        if configure_and_exit:
            return

        # Read the simulation results
        data = reader(point)

        # Save the results to our LUT format
        if data:
            Logger.debug(f"Updating data point {point} for keys: {data.keys()}")

            return point, data
        else:
            Logger.warning(f"No data was returned for point {point}")

        return

    def preSim(self):
        """
        This is an optional function that can be defined by a subclass RTE to be called
        directly before runSim() is executed. A subclass may return a dict containing
        any single or non-dimensional variables to be saved to the LUT file
        """
        ...

    def makeSim(self, point: np.array, template_only: bool = False):
        """
        Prepares and executes a radiative transfer engine's simulations

        Args:
            point (np.array): conditions to alter in simulation
            template_only (bool): only write template file and then stop
        """
        raise NotImplemented(
            "This method must be defined by the subclass RTE, (TODO) see ISOFIT documentation for more information"
        )

    def readSim(self, point: np.array):
        """
        Reads simulation results to standard form

        Args:
            point (np.array): conditions to alter in simulation
        """
        raise NotImplemented(
            "This method must be defined by the subclass RTE, (TODO) see ISOFIT documentation for more information"
        )

    def postSim(self):
        """
        This is an optional function that can be defined by a subclass RTE to be called
        directly after runSim() is finished. A subclass may return a dict containing
        any single or non-dimensional variables to be saved to the LUT file
        """
        ...

    def point_to_filename(self, point: np.array) -> str:
        """Change a point to a base filename

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            str: basename of the file to use for this point
        """
        filename = "_".join(
            ["%s-%6.4f" % (n, x) for n, x in zip(self.lut_names, point)]
        )
        return filename
