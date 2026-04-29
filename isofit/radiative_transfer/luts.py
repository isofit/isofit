"""
This is the netCDF4 and Zarr implementations for handling ISOFIT LUT files. For previous
implementations and research, please see https://github.com/isofit/isofit/tree/897062a3dcc64d5292d0d2efe7272db0809a6085/isofit/luts
"""

from __future__ import annotations

import atexit
import gc
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import dask.array
import numpy as np
import xarray as xr
import zarr
from netCDF4 import Dataset
from packaging.version import Version

from isofit import __version__
from isofit.core import common

Logger = logging.getLogger(__name__)


def create(file: str, *args, **kwargs):
    """
    Factory function to return the correct Create subclass

    Parameters
    ----------
    file : str
        File store. Uses the extension to determine the correct subclass
    *args : list
        Arguments to pass to the subclass
    *args : dict
        Key-word arguments to pass to the subclass

    Returns
    -------
    obj
        Create subclass object
    """
    if file.endswith(".zarr"):
        return CreateZarr(file, *args, **kwargs)
    else:
        return CreateNetCDF(file, *args, **kwargs)


# Statically store expected keys of the LUT file and their fill values
class Keys:
    # Constants, not along any dimension
    consts = {
        "coszen": np.nan,
        "solzen": np.nan,
    }

    # Along the wavelength dimension only
    onedim = {
        "fwhm": np.nan,
        "solar_irr": np.nan,
    }

    # Keys along all dimensions, ie. wl and point
    alldim = {
        "rhoatm": np.nan,
        "sphalb": np.nan,
        "transm_down_dir": 0,
        "transm_down_dif": 0,
        "transm_up_dir": 0,
        "transm_up_dif": 0,
        "thermal_upwelling": np.nan,
        "thermal_downwelling": np.nan,
        # add keys for radiances along all optical paths
        "dir-dir": 0,
        "dif-dir": 0,
        "dir-dif": 0,
        "dif-dif": 0,
    }


class Create:
    def __init__(
        self,
        file: str,
        wl: np.ndarray,
        grid: dict,
        mode: str = "w",
        attrs: dict = {},
        consts: dict = {},
        onedim: dict = {},
        alldim: dict = {},
        zeros: List[str] = [],
        chunks: List[int] | str = "auto",
        init: bool = True,
        **kwargs,
    ):
        """
        Prepare a LUT file

        Parameters
        ----------
        file : str
            Filepath for the LUT.
        wl : np.ndarray
            The wavelength array.
        grid : dict
            The LUT grid, formatted as {str: Iterable}.
        mode : str, default="w"
            File mode to open with
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
        chunks : list[int] | "auto", default="auto"
            Chunking strategy to use on alldim variables. "auto" will chunk along the point dimension
        init : bool, default=True
            Call the initialize function
        *kwargs : dict
            Captures any additional key-word arguments and ignores
        """
        # Track the ISOFIT version that created this LUT
        attrs["ISOFIT version"] = __version__
        attrs["ISOFIT status"] = "<incomplete>"

        self.file = file
        self.wl = wl
        self.grid = {key: np.array(vals) for key, vals in grid.items()}
        self.hold = []

        self.sizes = {"wl": len(self.wl)} | {key: len(val) for key, val in grid.items()}
        self.dims = list(self.sizes)
        self.point_dims = list(grid)

        self.attrs = attrs

        self.consts = {**Keys.consts, **consts}
        self.onedim = {**Keys.onedim, **onedim}
        self.alldim = {**Keys.alldim, **alldim}

        self.chunks = chunks
        if chunks == "auto":
            self.chunks = [len(self.wl)] + [1] * len(grid)

        if init:
            self.initialize()

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
        Flushes the (point, data) pairs held in the hold list to the LUT

        Parameters
        ----------
        finalize : bool, default=False
            Calls the `finalize` function
        """
        # Subclass flush
        unknowns = self._flush()

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

    def finalize(self):
        """
        Finalizes the file store by writing any remaining attributes to disk
        """
        self.setAttr("ISOFIT status", "success")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.sizes})"


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


class CreateZarr(Create):
    def __init__(
        self, file, *args, buffered=False, shards=None, min_shards=1, **kwargs
    ):
        """
        Prepare a Zarr v3 LUT store

        Parameters
        ----------
        file : str
            Filepath for the LUT.
        mode : str, default="w"
            File mode to open with
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
        """
        self.store = zarr.storage.LocalStore(file)
        self.z = zarr.open_group(
            store=self.store, mode=kwargs.get("mode", "w"), zarr_format=3
        )

        # TODO: Integrate into config
        self.sharding = "4gb"
        self.sharding = "128gb"  # experimental sRTMnet break the node settings
        self.shards = shards

        # TODO: yep
        self.min_shards = min_shards

        super().__init__(file, *args, **kwargs)

        self.buffer = {}
        if buffered:
            shape = list(self.sizes.values())
            self.buffer = {
                key: np.full(self.shards, vals, "float64")
                for key, vals in self.alldim.items()
            }

        self.data = self.buffer or self.z

    def __getitem__(self, key: str) -> Any:
        """
        Retrieves a value from the Zarr store

        Parameters
        ----------
        key : str
            The name of the item to retrieve

        Returns
        -------
        Any
            The value of the item retrieved from the Zarr store
        """
        return self.zarr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets a variable in the Zarr store

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        # Automatically insert data into the buffer if it's available instead of writing to disk
        self.data[key][...] = value

    def initialize(self, ret=False) -> None | xr.Dataset:
        """
        Initializes the LUT Zarr by prepopulating it with filler values.

        Parameters
        ----------
        ret : bool, default=False
            If True, returns the dataset instead of saving

        Returns
        -------
        None | xr.Dataset
            The initialized dataset
        """
        if self.sharding:
            self.shards, self.groups, self.coords = calc_shards(
                self.grid,
                self.wl,
                self.chunks,
                storage=self.sharding,
                min_shards=self.min_shards,
                scale=len(self.alldim),
            )
            self.setAttr("shards", self.shards.tolist())
            self.setAttr("shard order", list(self.sizes))

        self.z.attrs.update(self.attrs)

        # Coordinates
        array = self.z.create_array(
            name="wl", data=self.wl, fill_value=None, dimension_names=["wl"]
        )

        for key, vals in self.grid.items():
            array = self.z.create_array(
                name=key, data=vals, fill_value=None, dimension_names=[key]
            )

        # Constants
        dims = []
        for key, vals in self.consts.items():
            array = self.z.create_array(
                name=key, data=np.array(vals), fill_value=None, dimension_names=dims
            )

        # One dimensional arrays
        dims = ["wl"]
        shape = (len(self.wl),)
        for key, vals in self.onedim.items():
            if isinstance(vals, Iterable):
                array = self.z.create_array(
                    name=key, data=vals, fill_value=None, dimension_names=dims
                )
            else:
                array = self.z.create_array(
                    name=key,
                    shape=shape,
                    dtype="float64",
                    fill_value=None,
                    dimension_names=dims,
                )

        # Multi dimensional arrays
        dims += list(self.grid)
        shape = list(self.sizes.values())
        for key, vals in self.alldim.items():
            if isinstance(vals, Iterable):
                array = self.z.create_array(
                    key=key,
                    data=vals,
                    fill_value=None,
                    chunks=tuple(self.chunks),
                    shards=tuple(self.shards),
                    dimension_names=dims,
                )
            else:
                array = self.z.create_array(
                    name=key,
                    shape=shape,
                    dtype="float64",
                    fill_value=vals,
                    chunks=self.chunks,
                    shards=tuple(self.shards),
                    dimension_names=dims,
                )

    def _flush(self) -> None:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT Zarr

        Parameters
        ----------
        finalize : bool, default=False
            Calls the `finalize` function
        """
        unknowns = set()
        for point, data in self.hold:
            for key, vals in data.items():
                if key == "wl":
                    continue

                if key in self.consts:
                    self.data[key][...] = vals

                elif key in self.onedim:
                    self.data[key][:] = vals

                elif key in self.alldim:
                    index = self.pointIndices(point)
                    if self.buffer:
                        index = index % self.shards[1:]
                    index = (slice(None),) + tuple(index)
                    self.data[key][index] = vals

                else:
                    unknowns.update([key])

        return unknowns

    def flush_buffer(self, slices):
        """
        Needs to be manually called
        """
        # If the group key is given instead of the slices tuple, retrieve from coords
        if not isinstance(slices[0], slice):
            slices = self.coords[slices]

        for key, vals in self.buffer.items():
            self.z[key][slices] = vals

    def queuePoint(self, *args, **kwargs):
        """
        Overrides the inherited queuePoint to enable flushing immediately for this
        subclass only
        """
        super().queuePoint(*args, **kwargs)

        # Don't hold a queue when buffered
        if self.buffer:
            self.flush()

    def getAttr(self, key: str) -> Any:
        """
        Gets an attribute from the Zarr

        Parameters
        ----------
        key : str
            Key to get

        Returns
        -------
        any | None
            Retrieved attribute from Zarr, if it exists
        """
        return self.z.attrs.get(key)

    def setAttr(self, key: str, value: Any) -> None:
        """
        Sets an attribute in the Zarr

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        self.attrs[key] = value
        self.z.attrs.update({key: value})

    def finalize(self, *args, **kwargs):
        """
        Finalizes the Zarr store by consolidating the metadata at the end to make
        reading it more efficient via xarray
        """
        super().finalize(*args, **kwargs)

        zarr.consolidate_metadata(self.store, zarr_format=3)


def findSlice(dim, val):
    """
    Creates a slice for selecting along a dimension such that a value is encompassed by
    the slice.

    Parameters
    ----------
    dim: array
        Dimension array
    val: float, int
        Value to be encompassed

    Returns
    -------
    slice
        Index slice to encompass the value
    """
    # Increasing is 1, decreasing is -1 for the searchsorted
    orientation = 1
    if dim[0] > dim[-1]:
        orientation = -1

    # Subselect the two points encompassing this interp point
    b = np.searchsorted(dim * orientation, val * orientation)

    # Handle edge cases when val equals first or last lut dim value
    if val <= dim[0]:
        return slice(b, b + 2)

    else:
        return slice(b - 1, b + 1)


def optimizedInterp(ds, strat):
    """
    Optimizes the interpolation step by subselecting along dimensions first then
    interpolating

    Parameters
    ----------
    strat: dict
        Interpolation stategies to perform in the form of {dim: interpolate_values}

    Returns
    -------
    xr.Dataset
        Interpolated dataset
    """
    for key, val in strat.items():
        dim = ds[key]

        if val <= dim[0]:
            Logger.warning(
                f"Scene value for key: {key} of {round(val, 2)} "
                f"is less or equal to minimum LUT value {np.round(dim[0].data, 2)}. "
                "Solutions will use value interpolated to minimum LUT value"
            )

        elif val >= dim[-1]:
            Logger.warning(
                f"Scene value for key: {key} of {round(val, 2)} "
                f"is greater or equal to maximum LUT value {np.round(dim[-1].data, 2)}. "
                "Solutions will use value interpolated to maximum LUT value"
            )

        if isinstance(val, list):
            a = findSlice(dim, val[0])
            b = findSlice(dim, val[-1])

            # Find the correct order
            if a.start < b.start:
                sel = slice(a.start, b.stop)
            else:
                sel = slice(b.start, a.stop)
        else:
            sel = findSlice(dim, val)

        Logger.debug(f"- Subselecting {key}[{sel.start}:{sel.stop}]")
        ds = ds.isel({key: sel})

    Logger.debug("Calling .interp")
    return ds.interp(**strat)


def sel(ds, dim, lt=None, lte=None, gt=None, gte=None, encompass=True):
    """
    Subselects an xarray Dataset object using .sel

    Parameters
    ----------
    ds: xarray.Dataset
        LUT dataset
    dim: str
        Dimension to work on
    lt: float, default=None
        Select along this dim coordinates that are valued less than this
    lte: float, default=None
        Select along this dim coordinates that are valued less than or equal to this
    gt: float, default=None
        Select along this dim coordinates that are valued greater than this
    gte: float, default=None
        Select along this dim coordinates that are valued greater than or equal to this
    encompass: bool, default=True
        Change the values of gte/lte such that these values are encompassed using the
        previous/next valid grid point

    Returns
    -------
    ds: xarray.Dataset
        Subsetted dataset
    """
    assert None in (
        lt,
        lte,
    ), f"Subsetting `lt` and `lte` are mutually exclusive, please only set one for dim {dim}"
    assert None in (
        gt,
        gte,
    ), f"Subsetting `gt` and `gte` are mutually exclusive, please only set one for dim {dim}"

    # Which index in a where to select -- if the dim is in reverse order, this needs to swap
    g, l = -1, 0
    if ds[dim][0] > ds[dim][-1]:
        g, l = 0, -1

    if lt is not None:
        ds = ds.sel({dim: ds[dim] < lt})

    elif lte is not None:
        if encompass:
            where = ds[dim].where(ds[dim] > lte).dropna(dim)
            lte = where[l] if where.size else ds[dim].max()
            Logger.debug(f"Encompass changed lte value to {lte}")

        ds = ds.sel({dim: ds[dim] <= lte})

    if gt is not None:
        ds = ds.sel({dim: gt < ds[dim]})

    elif gte is not None:
        if encompass:
            where = ds[dim].where(ds[dim] < gte).dropna(dim)
            gte = where[g] if where.size else ds[dim].min()
            Logger.debug(f"Encompass changed gte value to {gte}")

        ds = ds.sel({dim: gte <= ds[dim]})

    return ds


def sub(ds: xr.Dataset, dim: str, strat) -> xr.Dataset:
    """
    Subsets a dataset object along a specific dimension in a few supported ways.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to operate on
    dim: str
        Name of the dimension to subset
    strat: float, int, list, dict, str, None
        Strategy to subset the given dimension with

    Returns
    -------
    xr.Dataset
        New subset of the input dataset
    """
    if isinstance(strat, (float, int, list)):
        return ds.sel({dim: strat})

    elif isinstance(strat, str):
        return getattr(ds, strat)(dim)

    elif strat is None:
        return ds  # Take dimension as-is

    elif isinstance(strat, dict):
        if "interp" in strat:
            return ds.interp({dim: strat["interp"]})

        return sel(ds, dim, **strat)

    else:
        Logger.error(f"Unknown subsetting strategy for type: {type(strat)}")
        return ds


def couple(ds, inplace=True):
    """
    Calculates coupled terms on the input Dataset

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to process on
    inplace: bool, default=True
        Insert the coupled terms in-place to the original Dataset. If False, copy the
        Dataset first

    Returns
    -------
    ds: xr.Dataset
        Dataset with coupled terms
    """
    terms = {
        "dir-dir": ("transm_down_dir", "transm_up_dir"),
        "dif-dir": ("transm_down_dif", "transm_up_dir"),
        "dir-dif": ("transm_down_dir", "transm_up_dif"),
        "dif-dif": ("transm_down_dif", "transm_up_dif"),
    }

    # Detect if coupling needs to occur first
    data = ds.get(list(terms))
    calc = False
    if data is None:
        # Not all keys exist
        calc = "missing"
    elif not bool(data.any().to_array().all()):
        # If any key is empty
        calc = "empty"

    if calc:
        Logger.debug(f"A coupled term is {calc}, calculating")
        if not inplace:
            ds = ds.copy()

        for term, (key1, key2) in terms.items():
            try:
                ds[term] = ds[key1] * ds[key2]
            except KeyError:
                ds[term] = 0

    return ds


def load(
    file: str,
    subset: dict = None,
    dask: bool = False,
    mode: str = "a",
    lock: bool = False,
    load: bool = False,
    coupling: str = "after",
    check: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Loads a LUT NetCDF
    Assumes to be a regular grid at this time (auto creates the point dim)

    Parameters
    ----------
    file: str
        LUT file to load
    subset: dict, default={}
        Subset each dimension with a given strategy. Each dimension in the LUT file
        must be specified.
        See examples for more information
    dask: bool, default=False
        Use Dask on the backend of Xarray to lazy load the dataset. This enables
        out-of-core subsetting
    mode: str, default="a"
        File mode to open with, must be set to append="a" for coupled terms to be
        saved back
    lock: bool, default=False
        Set a lock on the input file
    load: bool, default=True
        Calls ds.load() at the end to cast from Dask arrays back into numpy arrays held
        in memory
    coupling: string, default="after"
        Calculates coupling terms, if needed. This may be set one of four ways:
            "before"
                Calculate before subsetting
            "before-save"
                Before + save the coupled terms to the original input file
            "after"
                Calculate after subsetting
            "after-save"
                After + save to a new file (the original input file with the extension
                changed to ".coupled-subset.nc")
    check: bool, default=True
        Checks the dataset for NaNs and replaces them with 0s if the array is not
        entirely NaN

    Examples
    --------
    >>> # Create a test file for the examples to load
    >>> file = 'subsetting_example.nc'
    >>> lut_dims = {
    ...     'AOT550': [0.001, 0.1009, 0.2008, 0.3007, 0.4006, 0.5005, 0.6004, 0.7003, 0.8002, 0.9001, 1.],
    ...     'H2OSTR': [0.2231, 0.4637, 0.7042, 0.9447, 1.1853, 1.4258, 1.6664, 1.9069, 2.1474, 2.388, 2.6285, 2.869, 3.1096, 3.3501],
    ...     'observer_zenith': [7.2155, 9.8900],
    ...     'surface_elevation_km': [0., 0.2361, 0.4721, 0.7082, 0.9442, 1.1803, 1.4164, 1.6524, 1.8885, 2.1245, 2.3606, 2.5966, 2.8327, 3.0688, 3.3048, 3.5409, 3.7769, 4.013],
    ...     'wl': range(285)
    ... }
    >>> ds = xr.Dataset(coords=lut_dims)
    >>> ds.to_netcdf(file)

    >>> # Subset: Exact values along the dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': [1.1853, 2.869],
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None,
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 792})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'H2OSTR': 2, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Interpolate H2OSTR to 1.5
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': {
    ...         'interp': 1.5
    ...     },
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None,
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: 1.1853 < H2OSTR < 2.869
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': {
    ...         'gt': 1.1853,
    ...         'lt': 2.869
    ...     },
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None,
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 2376})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'H2OSTR': 6, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: 1.1853 <= H2OSTR <= 2.869, encompassed
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': {
    ...         'gte': 1.1853,
    ...         'lte': 2.869
    ...     },
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 3960})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'H2OSTR': 10, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: 1.1853 <= H2OSTR <= 2.869, not encompassed
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': {
    ...         'gte': 1.1853,
    ...         'lte': 2.869,
    ...         'encompass': False
    ...     },
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 3168})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'H2OSTR': 8, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Exact value, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 2.869,
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Using mean, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 'mean',
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Using max, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 'max',
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Multiple subsets
    >>> subset = {
    ...     'AOT550': [0.2008, 0.4006, 0.6004],
    ...     'H2OSTR': {
    ...         'gte': 1.1853,
    ...         'lte': 2.869
    ...     },
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 1080})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 3, 'H2OSTR': 10, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Multiple subsets
    >>> subset = {
    ...     'AOT550': [0.2008, 0.4006, 0.6004],
    ...     'H2OSTR': {
    ...         'gte': 1.1853,
    ...         'lte': 2.869
    ...     },
    ...     'observer_zenith': 7.2155,
    ...     'surface_elevation_km': 'mean'
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 30})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 3, 'H2OSTR': 10, 'wl': 285})
    """
    # if coupling not in ("before", "before-save", "after", "after-save"):
    #     raise AttributeError("Coupling must be set to either 'before' or 'after'")

    if file.endswith(".zarr"):
        if dask:
            Logger.debug(f"[Zarr] Using Dask to load: {file}")
            ds = xr.open_zarr(file, **kwargs)
        else:
            Logger.debug(f"[Zarr] Using Xarray to load: {file}")
            ds = xr.open_dataset(file, engine="zarr", **kwargs)
    else:
        kwargs.setdefault("engine", "netcdf4")
        kwargs.setdefault("lock", lock)
        kwargs.setdefault("mode", mode)
        if dask:
            Logger.debug(f"[NetCDF] Using Dask to load: {file}")
            ds = xr.open_mfdataset([file], **kwargs)
        else:
            Logger.debug(f"[NetCDF] Using Xarray to load: {file}")
            ds = xr.open_dataset(file, **kwargs)

    status = ds.attrs.get("ISOFIT status", "<not set>")
    if status != "success":
        Logger.warning(
            f"The LUT status is {status}, there may be issues with it downstream"
        )
        Logger.debug(
            "To fix this error and you know the the LUT is correct, set NetCDF attribute 'ISOFIT status' to 'success'"
        )

    version = ds.attrs.get("ISOFIT version", "<not set>")
    Logger.debug(
        f"This LUT was created with ISOFIT version {version}, you are running ISOFIT {__version__}"
    )

    # Calculate coupling before subsetting
    if "before" in coupling:
        couple(ds)

        # Save back to the original file, only the coupled terms
        if all(
            [
                "save" in coupling,
                dask is False,  # Only works with xarray loader
                mode == "a",  # Original input must have been opened in append
            ]
        ):
            Logger.debug(f"Saving coupled terms back to the original input file")
            terms = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
            ds[terms].to_netcdf(file, mode="a")

    # Special case that doesn't require defining the entire grid subsetting strategy
    if subset is None:
        Logger.debug("Subset was None, using entire file")

    elif isinstance(subset, dict):
        Logger.debug(f"Subsetting with: {subset}")

        # The subset dict must contain all coordinate keys in the lut file
        missing = set(ds.coords) - ({"wl"} | set(subset))
        if missing:
            Logger.error(
                "The following keys are in the LUT file but not specified how to be handled by the config:"
            )
            for key in missing:
                Logger.error(f"- {key}")
            raise AttributeError(
                f"Subset dictionary (engine.lut_names) is missing keys that are present in the LUT dimensions {set(ds.coords)}: {missing=}"
            )

        # Interpolation strategies will be done last for optimization purposes
        interp = {}

        # Apply subsetting strategies
        for dim, strat in subset.items():
            if isinstance(strat, dict) and "interp" in strat:
                interp[dim] = strat["interp"]
            else:
                Logger.debug(f"Subsetting {dim} with {strat}")
                ds = sub(ds, dim, strat)

        if interp:
            Logger.debug("Interpolating")
            ds = optimizedInterp(ds, interp)

        # Save this subsetting strategy to the attributes for future reference
        ds.attrs["subset"] = str(subset)

        Logger.debug("Subsetting finished")
    else:
        Logger.error("The subsetting strategy must be a dictionary")
        raise AttributeError(
            f"Bad subsetting strategy, expected either a dict or a NoneType: {subset}"
        )

    # Calculate coupling after subsetting
    if "after" in coupling:
        couple(ds)

        if "save" in coupling:
            Logger.debug(f"Saving subset with coupling to {file}")

            # Have to save to a different/new file after subsetting
            file = Path(file).with_suffix(".coupled-subset.nc")
            ds.to_netcdf(file)

    dims = ds.drop_dims("wl").dims

    # Create the point dimension -> Coords now len(points)
    ds = ds.stack(point=dims).transpose("point", "wl")

    if load:
        Logger.info("Loading LUT into memory")
        ds.load()

    if check:
        # Handle NaNs in the LUT. Keep as is (if all NaN). Set to 0 (if partial NaN)
        Logger.debug("Attempting to detect NaNs")

        for key, data in ds.items():
            nans = data.isnull()
            if nans.any():
                if nans.all():
                    Logger.warning(f"{key} is fully NaN, leaving as-is")
                    continue

                count = nans.sum().data
                total = data.count().data

                Logger.warning(
                    f"{key} is partially NaN ({count}/{total}, {count/total:.2%}), replacing with 0s"
                )
                ds[key] = data.fillna(0)

    return ds


def extractPoints(ds: xr.Dataset, names: bool = False) -> np.array:
    """
    Extracts the points and point name arrays

    Parameters
    ----------
    ds: xr.Dataset
        LUT Dataset object
    names: bool, default=False
        Return the names of the point coords as well

    Returns
    -------
    points or (points, names)
        Extracted points, plus names if requested
    """
    points = np.array([*ds.point.data])

    if names:
        names = np.array([name for name in ds.point.coords])[1:]
        return (points, names)

    return points


def extractGrid(ds: xr.Dataset) -> dict:
    """
    Extracts the LUT grid from a Dataset

    Parameters
    ----------
    ds: xr.Dataset
        LUT Dataset object. Carried stacked: Dimensions wl, points

    """
    grid = {}
    for dim, vals in ds.coords.items():
        if dim in {"wl", "point"}:
            continue
        if len(vals.data.shape) > 0 and vals.data.shape[0] > 1:
            # Unique call sorts and filters. Faster than unstacking ds.
            grid[dim] = np.unique(vals.data)
    return grid


def saveDataset(file: str, ds: xr.Dataset) -> None:
    """
    Handles saving an xarray.Dataset to a NetCDF file for ISOFIT. Will detect if the
    point dim needs to be unstacked before saving (regular grids) or not (irregular)

    Parameters
    ----------
    file: str
        Path to save the `ds` object to. This will be a NetCDF, recommended extension
        is `.nc`
    ds: xarray.Dataset
        Data object to save
    """
    if "MultiIndex" in str(ds.indexes["point"]):
        ds = ds.unstack("point")

    ds.to_netcdf(file)


def cleanup(file):
    """
    Checks the ``ISOFIT status`` attribute on a LUT and removes the file if it is
    incomplete.

    Parameters
    ----------
    file : str
        Path to the file to check
    """
    if os.path.exists(file):
        with Dataset(file, "r") as ds:
            if ds.getncattr("ISOFIT status") == "<incomplete>":
                Logger.error(
                    f"The LUT status was determined to be incomplete, auto-removing: {file}"
                )
                os.remove(file)


def shard_to_coord(group, shards, shape):
    """
    Converts a shard index to shard coordinates (the slice of the full store for this
    shard).
    """
    start = np.array(group) * shards
    end = np.minimum(start + shards, shape)
    return tuple(slice(s, e) for s, e in zip(start, end))


def calc_shards(grid, wl, chunk, storage="8gb", min_shards=None, scale=1):
    """
    Determine an optimal Zarr sharding strategy for a LUT and group points by shard.

    This function computes candidate shard shapes that:
    - evenly divide the LUT dimensions
    - do not split chunks
    - approximately match a target storage size per shard

    It then selects the shard shape whose number of chunks per shard is closest
    to the target, and groups LUT points into shard-aligned partitions.

    Parameters
    ----------
    grid : dict[str, Iterable]
        LUT grid definition as {dimension_name: values}. These define the
        non-wavelength dimensions of the LUT.
    wl : iterable
        Wavelength dimension values.
    chunk : iterable of int
        Chunk shape used for Zarr storage. Must align with the full LUT shape
        (wl + grid dimensions).
    storage : str or float, default="8gb"
        Target storage size per shard. Determines how many chunks should be grouped
        into a shard.
    min_shards : int, optional
        Minimum number of shards to produce. Candidate shardings that
        result in fewer shards are discarded.
    scale : int, default=1
        Multiplier applied to the estimated chunk size. Useful when multiple arrays
        are written per chunk, to better approximate the actual memory when in buffered
        mode.

    Returns
    -------
    best : np.array[int]
        Selected shard shape, including the wavelength dimension as the first axis.
    groups : dict[tuple[int, ...], list[tuple]]
        Mapping from shard index (multi-dimensional shard ID) to the list of
        grid points (coordinate tuples) that fall within that shard.
    coords : dict[tuple[int, ...], tuple[slice, ...]]
        Mapping from shard index to the corresponding global array slice
        (including wavelength as the leading dimension). These slices can be
        used directly to write shard-aligned data into a Zarr array.
    """
    dtype = np.dtype("float64").itemsize
    if isinstance(storage, str):
        from isofit.core import units

        storage = units.byte_string_to_float(storage)

    space = (
        np.prod(chunk) * dtype * int(scale)
    )  # Determine how much space each chunk takes
    target = int(storage / space)  # Target number of chunks per shard

    # Not technically the fastest but it's small
    divisors = lambda n: np.unique([n // i for i in range(1, n + 1) if not n % i])

    # Viable shard shapes
    lens = [len(v) for v in grid.values()]
    shape = np.array([len(wl)] + lens)
    candidates = [divisors(s) for s in shape[1:]]

    # Of these, how many files are created by each sharding strategy
    shards = common.combos(candidates)

    # Insert the wavelength dimension as a single shard
    shards = np.column_stack([np.full(shards.shape[0], len(wl)), shards])

    # Ensure shards don't split chunks
    shards = shards[np.sum(shards % chunk, axis=1) == 0]

    # 1 shard == 1 file, only consider shardings that reach the minimum number of files
    if min_shards:
        files = np.prod((shape / shards).astype(int), axis=1)
        shards = shards[files >= min_shards]

    # Chunks per shard
    cpf = np.prod(shards / chunk, axis=1).astype(int)

    # Return the strategy that is closest to the target number of chunks per shard
    idx = np.abs(cpf - target).argmin()
    best = shards[idx]

    # Split the points into shard groups
    points = common.combos(grid.values())
    pidxs = common.combos([np.arange(v) for v in lens])
    indices = pidxs // best[1:]
    groups = {}
    coords = {}
    for i, key in enumerate(indices):
        key = tuple(key)
        groups.setdefault(key, []).append(points[i])
        if key not in coords:
            coord = shard_to_coord(key, best[1:], shape[1:])
            coord = (slice(None),) + coord
            coords[key] = coord

    # Useful information
    Logger.info(f"Number of points: {np.prod(shape):,}")
    Logger.info(f"Target chunks per file: {target} ({target * space / 2**30:.2f}gb)")
    Logger.info(f"Shapes:")
    Logger.info(f"  dimensions: {shape}")
    Logger.info(f"    chunking: {np.array(chunk)}")
    Logger.info(f"Recommended")
    Logger.info(f"    sharding: {best}")
    Logger.info(f"Produces:")
    Logger.info(f"  Chunks per file: {cpf[idx]} ({cpf[idx] * space / 2**30:.2f}gb)")
    Logger.info(f"  Number of files: {np.prod((shape / best).astype(int))}")

    return best, groups, coords
