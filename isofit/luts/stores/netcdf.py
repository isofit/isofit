"""
NetCDF implementation for LUTs
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from isofit.luts.stores import Create

Logger = logging.getLogger(__name__)


def cleanup(file):
    """
    Checks the ``ISOFIT status`` attribute on a LUT and removes the file if it is
    incomplete.

    Parameters
    ----------
    file : str
        Path to the file to check
    """
    state = None
    key = "ISOFIT status"
    bad = "<incomplete>"

    if (p := Path(path)).exists():
        store = Dataset(p, mode="r")
        state = store.getncattr(key)

    if state == bad:
        Logger.error(
            f"The LUT status was determined to be incomplete, auto-removing: {p}"
        )
        os.remove(p)


class CreateNetCDF(Create):
    def __init__(
        self,
        *args,
        compression: str = "zlib",
        complevel: int = None,
        **kwargs,
    ):
        """
        Prepare a NetCDF LUT file

        Parameters
        ----------
        file : str
            Filepath for the LUT.
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
        self.compression = compression
        self.complevel = complevel

        super().__init__(*args, **kwargs)

        atexit.register(cleanup, self.path)

    def __getitem__(self, key: str) -> Any:
        """
        Retrieves a value from the NetCDF store

        Parameters
        ----------
        key : str
            The name of the item to retrieve

        Returns
        -------
        Any
            The value of the item retrieved from the NetCDF store
        """
        with Dataset(self.file, "r") as ds:
            return ds[key]

        atexit.register(cleanup, file)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets a variable in the NetCDF store

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        with Dataset(self.file, "a") as ds:
            ds[key][:] = value

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

        with Dataset(self.file, self.mode, format="NETCDF4") as ds:
            # Dimensions
            ds.createDimension("wl", len(self.wl))
            createVariable("wl", self.wl, ("wl",))

            for key, vals in self.grid.items():
                ds.createDimension(key, len(vals))
                createVariable(key, vals, (key,))

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
                createVariable(key, vals, dims, chunksizes=self.chunks)

            # Add custom attributes onto the Dataset
            for key, value in self.attrs.items():
                ds.setncattr(key, value)

            ds.sync()
        gc.collect()

    def _flush(self) -> set:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT netCDF

        Returns
        -------
        unknowns : set
            Set of unknown keys
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

        return unknowns

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
