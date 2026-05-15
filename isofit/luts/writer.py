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
from isofit.luts.stores import create

Logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
def streamSimulation(
    point: np.array,
    lut_names: list,
    simmer: Callable,
    reader: Callable,
    max_buffer_time: float = 0.5,
    rte_configure_and_exit: bool = False,
):
    """Run a simulation for a single point and stream the results to a saved lut file.

    Args:
        point (np.array): conditions to alter in simulation
        lut_names (list): Dimension names aka lut_names
        simmer (function): function to run the simulation
        reader (function): function to read the results of the simulation
        max_buffer_time (float, optional): _description_. Defaults to 0.5.
        rte_configure_and_exit (bool, optional): exit early if not executing simulations
    """
    # Logger.debug(f"Simulating(point={point})")

    # Slight delay to prevent all subprocesses from starting simultaneously
    time.sleep(np.random.rand() * max_buffer_time)

    # Execute the simulation
    simmer(point)

    # No data will be produced, just configuration files
    if rte_configure_and_exit:
        return

    # Read the simulation results
    data = reader(point)

    # Save the results to our LUT format
    if data:
        # Logger.debug(f"Updating data point {point} for keys: {data.keys()}")
        return point, data
    else:
        Logger.warning(f"No data was returned for point {point}")


@ray.remote(num_cpus=1)
def shardWriter(lut, shard, coord, points, simmer, reader):
    """
    Execute simulations for a set of points and write chunked results to a LUT shard.

    This function is intended to run as a Ray remote task. For each input point,
    it runs a simulation, reads the resulting data, separates chunked and
    non-chunked outputs, and queues the chunked data into a shard buffer. After all
    points are processed, the buffered data is flushed to disk.

    Parameters
    ----------
    lut : dict or LUT-like
        Either a dictionary of arguments used to initialize a LUT via
        ``luts.create`` or an already-instantiated LUT Create object.
    shard : int or str
        Identifier for the shard being processed. Used for logging.
    coord : tuple of slice or similar
        Coordinate slices defining the region of the LUT corresponding to
        this shard.
    points : iterable
        Collection of input points to simulate and process. These points are assumed
        to be in the same shard.
    simmer : callable
        Function to run the simulation.
    reader : callable
        Function to read the results of the simulation.

    Returns
    -------
    tuple
        A tuple ``(point, chunkless)`` where ``point`` is the last processed
        point and ``chunkless`` is a dictionary of non-chunked outputs from
        the last iteration.
    """
    if isinstance(lut, dict):
        lut = create(**lut, mode="a", init=False, buffered=True)

    Logger.info(f"Starting shard {shard}")
    keys = luts["keys"]

    for point in points:
        simmer(point)  # Execute the simulation
        data = reader(point)  # Read the simulation results

        # Remove non-chunk data
        chunkless = {k: v for k, v in data.items() if k not in keys.alldim}
        data = {k: v for k, v in data.items() if k in keys.alldim}
        if data:
            lut.queuePoint(point, data)

    Logger.info(f"Finished points {shard}, flushing to disk")
    lut.flush_buffer(slices=coord)

    Logger.info(f"Finished shard {shard}")
    return point, chunkless


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

    # Optional flag to exit simulations early
    configure_and_exit = False

    # Zarr sharding
    shard_size = None

    def write(self):
        """
        Initialize a LUT and run simulations
        """
        for attr in ("lut_path", "lut_names", "lut_grid", "wl", "lut_keys"):
            if getattr(self, attr) is None:
                raise AttributeError(
                    f"Missing required attribute to write the LUT: {attr}"
                )

        self.lut = None
        if not self.configure_and_exit:
            self.lut = create(
                path=self.lut_path,
                keys=self.lut_keys,
                wl=self.wl,
                grid=self.lut_grid,
                consts=self.consts,
                onedim=self.onedim,
                alldim=self.alldim,
                shard_size=self.shard_size,
                min_shards=self.n_cores,
            )

        self.runSimulations()

    def runSimulations(self):
        """
        Executes an engine object to process simulations using the preSim, makeSim, and
        postSim functions
        """
        Logger.info(f"Running any pre-sim functions")
        if pre := self.preSim():
            Logger.info("Saving pre-sim data to index zero of all dimensions except wl")
            Logger.debug(f"pre-sim data contains keys: {pre.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=pre)

        if not self._disable_makeSim:
            if hasattr(self.lut, "groups"):
                Logger.info("Executing parallel simulations by shards")
                self.parallelize_shards()
            else:
                Logger.info("Executing parallel simulations by points")
                self.parallelize_points()
        else:
            Logger.debug("makeSim is disabled for this engine")

        Logger.info(f"Running any post-sim functions")
        if post := self.postSim():
            Logger.info(
                "Saving post-sim data to index zero of all dimensions except wl"
            )
            Logger.debug(f"post-sim data contains keys: {post.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=post)

        self.lut.finalize()

    def parallelize_shards(self):
        """
        Parallelizes simulations into shard groups to handle very large LUTs: VLLUTS
        """
        groups = self.lut.groups
        coords = self.lut.coords
        simmer = ray.put(self.makeSim)
        reader = ray.put(self.readSim)
        kwargs = ray.put(
            {
                "file": self.lut.file,
                "keys": self.lut_keys,
                "wl": self.lut.wl,
                "grid": self.lut.grid,
                "shards": self.lut.shards,
                "min_shards": self.n_cores,
            }
        )

        jobs = [
            shardWriter.remote(kwargs, shard, coords[shard], points, simmer, reader)
            for shard, points in groups.items()
        ]
        report = Track(
            jobs,
            step=1,
            reverse=True,
            print=Logger.info,
            message="shards complete",
        )

        # Update the lut as point simulations stream in
        saved = set()
        while jobs:
            [done], jobs = ray.wait(jobs, num_returns=1)

            # Retrieve the return of the finished job
            ret = ray.get(done)

            # If a simulation fails then it will return None
            if ret:
                point, data = ret
                data = {k: v for k, v in data.items() if k not in saved}
                saved.update(data)
                if data:
                    Logger.info(f"Saved chunkless: {saved}")
                    self.lut.queuePoint(point, data)

            report(len(jobs))

        Logger.debug("Flushing chunkless data")
        self.lut.flush()

    def parallelize_points(self):
        """
        Parallelizes simulations point-wise to stream finished data into the LUT at set
        completion percentages
        """
        # Place into shared memory space to avoid spilling
        kwargs = dict(
            lut_names=self.lut_names,
            simmer=self.makeSim,
            reader=self.readSim,
            max_buffer_time=self.max_buffer_time,
            rte_configure_and_exit=self.configure_and_exit,
        )
        kwargs = {k: ray.put(v) for k, v in kwargs.items()}

        points = combos(self.lut_grid.values())
        jobs = [streamSimulation.remote(point, **kwargs) for point in points]

        if self.configure_and_exit:
            # Block until all jobs finish
            ray.get(jobs)

            Logger.warning("Exiting early due to rte_configure_and_exit")
            sys.exit(0)

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

            if report(len(jobs)) and self.lut.hold:
                Logger.info("Flushing netCDF to disk")
                self.lut.flush()

        # Shouldn't be hit but just in case
        if self.lut.hold:
            Logger.warning("Not all points were flushed, doing so now")
            self.lut.flush()

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
        raise NotImplemented("This method must be defined by the subclass")

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
