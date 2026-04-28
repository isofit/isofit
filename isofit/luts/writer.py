import atexit
import gc
import logging
import os
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from isofit import __version__

Logger = logging.getLogger(__name__)


class Writer:
    def __init__(self, full_config **kwargs):
        self.compression = full_config.forward_model.atmosphere.lut_compression
        self.complevel = full_config.forward_model.atmosphere.lut_complevel

    def write(self): 
        """Initialize a LUT and run simulations
        """
        # Run sims and write lut
        if not configure_and_exit:
            lut_writer = self.prepare(
                file=self.lut_path,
                wl=self.wl,
                grid=self.lut_grid,
                attrs={"RT_mode": self.rt_mode},
                onedim={"fwhm": self.fwhm},
                compression=self.compression,
                complevel=self.complevel,
            )

        self.runSimulations()

        return 

    def prepare(
        self,
        file: str,
        wl: np.ndarray,
        grid: dict,
        attrs: dict = {},
        consts: dict = {},
        onedim: dict = {},
        alldim: dict = {},
        zeros: List[str] = [],
        compression: str = "zlib",
        complevel: int = None,
    ):
        """Prepare a LUT netCDF

        Parameters
        ----------
        file : str
            NetCDF filepath for the LUT.
        wl : np.ndarray
            The wavelength array.
        grid : dict
            The LUT grid, formatted as {str: Iterable}.
        attrs: dict, defaults={}
            Dict of dataset attributes, ie. {"RT_mode": "transm"}
        consts : dict, optional, default={}
            Dictionary of constant values. Appends/replaces current Create.consts list.
        onedim : dict, optional, default={}
            Dictionary of one-dimensional data. Appends/replaces to the current Create.onedim list.
        alldim : dict, optional, default={}
            Dictionary of multi-dimensional data. Appends/replaces to the current Create.alldim list.
        zeros : List[str], optional, default=[]
            List of zero values. Appends to the current Create.zeros list.
        compression : str, default="zlib"
            Compression method to use to the NetCDF. Check https://unidata.github.io/netcdf4-python/
            for available options. Currently, must use h5py <= 3.14.0
        complevel : int, default=None
            Compression to use. Impact and levels vary per method.
        """
        # Track the ISOFIT version that created this LUT
        attrs["ISOFIT version"] = __version__
        attrs["ISOFIT status"] = "<incomplete>"

        Logger.info(f"No LUT store found, beginning initialization and simulations")

        # Check for duplicates in grid
        duplicates = False
        for dim, vals in grid.items():
            if np.unique(vals).size < len(vals):
                duplicates = True
                Logger.error(
                    f"Duplicates values were detected in the lut_grid for {dim}: {vals}"
                )

        if duplicates:
            raise AttributeError(
                "Input lut_grid detected to have duplicates, please correct them before continuing"
            )

        self.file = file
        self.wl = wl
        self.grid = grid
        self.hold = []

        self.sizes = {key: len(val) for key, val in grid.items()}
        self.attrs = attrs

        self.consts = {**Keys.consts, **consts}
        self.onedim = {**Keys.onedim, **onedim}
        self.alldim = {**Keys.alldim, **alldim}

        self.compression = compression
        self.complevel = complevel

        # Save ds for backwards compatibility (to work with extractGrid, extractPoints)
        self.initialize()

        atexit.register(cleanup, file)

    def initialize(self) -> None:
        """
        Initializes the LUT netCDF by prepopulating it with filler values.
        """

        def createVariable(key, vals, dims=(), fill_value=np.nan, chunksizes=None):
            """
            Reusable createVariable for the Dataset object
            """
            var = ds.createVariable(
                varname=key,
                datatype="f8",
                dimensions=dims,
                fill_value=fill_value,
                chunksizes=chunksizes,
                compression=self.compression,
                complevel=self.complevel,
            )
            var[:] = vals

        with Dataset(self.file, "w", format="NETCDF4") as ds:
            # Dimensions
            ds.createDimension("wl", len(self.wl))
            createVariable("wl", self.wl, ("wl",))

            chunks = [len(self.wl)]
            for key, vals in self.grid.items():
                ds.createDimension(key, len(vals))
                createVariable(key, vals, (key,))
                chunks.append(1)

            # Constants
            dims = ()
            for key, vals in self.consts.items():
                createVariable(key, vals, dims)

            # One dimensional arrays
            dims = ("wl",)
            for key, vals in self.onedim.items():
                createVariable(key, vals, dims)

            # Multi dimensional arrays
            dims += tuple(self.grid)
            for key, vals in self.alldim.items():
                createVariable(key, vals, dims, chunksizes=chunks)

            # Add custom attributes onto the Dataset
            for key, value in self.attrs.items():
                ds.setncattr(key, value)

            ds.sync()
        gc.collect()

    def pointIndices(self, point: np.ndarray) -> List[int]:
        """
        Get the indices of the point in the grid.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.

        Returns
        -------
        List[int]
            Mapped point values to index positions.
        """
        return [
            np.where(self.grid[dim] == val)[0][0] for dim, val in zip(self.grid, point)
        ]

    def queuePoint(self, point: np.ndarray, data: dict) -> None:
        """
        Queues a point and its data to the internal hold list which is used by the
        flush function to write these points to disk.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.
        data : dict
            Data for this point to write.
        """
        self.hold.append((point, data))

    def flush(self, finalize: bool = False) -> None:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT netCDF.

        Parameters
        ----------
        finalize : bool, default=False
            Calls the `finalize` function
        """
        unknowns = set()
        with Dataset(self.file, "a") as ds:
            for point, data in self.hold:
                for key, vals in data.items():
                    if key in self.consts:
                        ds[key].assignValue(vals)
                    elif key in self.onedim:
                        ds[key][:] = vals
                    elif key in self.alldim:
                        index = [slice(None)] + list(self.pointIndices(point))
                        ds[key][index] = vals
                    else:
                        unknowns.update([key])
            ds.sync()

        self.hold = []
        gc.collect()

        # Reduce the number of warnings produced per flush
        for key in unknowns:
            Logger.warning(
                f"Attempted to assign a key that is not recognized, skipping: {key}"
            )

        if finalize:
            self.finalize()

    def writePoint(self, point: np.ndarray, data: dict) -> None:
        """
        Queues a point and immediately flushes to disk.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.
        data : dict
            Data for this point to write.
        """
        self.queuePoint(point, data)
        self.flush()

    def setAttr(self, key: str, value: Any) -> None:
        """
        Sets an attribute in the netCDF

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        self.attrs[key] = value
        with Dataset(self.file, "a") as ds:
            ds.setncattr(key, value)

    def getAttr(self, key: str) -> Any:
        """
        Gets an attribute from the netCDF

        Parameters
        ----------
        key : str
            Key to get

        Returns
        -------
        any | None
            Retrieved attribute from netCDF, if it exists
        """
        with Dataset(self.file, "r") as ds:
            return ds.getncattr(key)

    def finalize(self):
        """
        Finalizes the netCDF by writing any remaining attributes to disk
        """
        self.setAttr("ISOFIT status", "success")

    def runSimulations(self) -> None:
        """
        Run all simulations for the LUT grid.

        """
        Logger.info(f"Running any pre-sim functions")
        pre = self.preSim()

        if pre:
            Logger.info("Saving pre-sim data to index zero of all dimensions except wl")
            Logger.debug(f"pre-sim data contains keys: {pre.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=pre)

        # Make the LUT calls (in parallel if specified)
        if not self._disable_makeSim:
            Logger.info("Executing parallel simulations")

            # Place into shared memory space to avoid spilling
            lut_names = ray.put(self.lut_names)
            makeSim = ray.put(self.makeSim)
            readSim = ray.put(self.readSim)
            lut_path = ray.put(self.lut_path)
            buffer_time = ray.put(self.max_buffer_time)
            rte_configure_and_exit = ray.put(self.engine_config.rte_configure_and_exit)

            jobs = [
                streamSimulation.remote(
                    point,
                    lut_names,
                    makeSim,
                    readSim,
                    lut_path,
                    max_buffer_time=buffer_time,
                    rte_configure_and_exit=self.engine_config.rte_configure_and_exit,
                )
                for point in self.points
            ]

            if self.engine_config.rte_configure_and_exit:
                # Block until all jobs finish
                ray.get(jobs)

                Logger.warning("Exiting early due to rte_configure_and_exit")
                sys.exit(0)
            else:
                # Report a percentage complete every 10% and flush to disk at those intervals
                report = common.Track(
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

            del lut_names, makeSim, readSim, lut_path, buffer_time
        else:
            Logger.debug("makeSim is disabled for this engine")

        Logger.info(f"Running any post-sim functions")
        post = self.postSim()

        if post:
            Logger.info("Saving post-sim data to index zero of all dimensions except wl")
            Logger.debug(f"post-sim data contains keys: {post.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=post)

        self.lut.finalize()

    @ray.remote(num_cpus=1)
    def streamSimulation(
        point: np.array,
        lut_names: list,
        simmer: Callable,
        reader: Callable,
        output: str,
        max_buffer_time: float = 0.5,
        rte_configure_and_exit: bool = False,
    ):
        """Run a simulation for a single point and stream the results to a saved lut file.

        Args:
            point (np.array): conditions to alter in simulation
            lut_names (list): Dimension names aka lut_names
            simmer (function): function to run the simulation
            reader (function): function to read the results of the simulation
            output (str): LUT store to save results to
            max_buffer_time (float, optional): _description_. Defaults to 0.5.
            rte_configure_and_exit (bool, optional): exit early if not executing simulations
        """
        Logger.debug(f"Simulating(point={point})")

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
            Logger.debug(f"Updating data point {point} for keys: {data.keys()}")

            return point, data
        else:
            Logger.warning(f"No data was returned for point {point}")

    def __getitem__(self, key):
        """
        Enables key indexing for easier access to the numpy object store in
        self.lut[key]
        """
        return self.lut[key].load().data

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

