"""
This is the netCDF4 implementation for handling ISOFIT LUT files. For previous
implementations and research, please see https://github.com/isofit/isofit/tree/897062a3dcc64d5292d0d2efe7272db0809a6085/isofit/luts
"""

import gc
import logging
import os
from typing import Any, List, Union

import numpy as np
import xarray as xr
from netCDF4 import Dataset

Logger = logging.getLogger(__file__)


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
    }


class Create:
    def __init__(
        self,
        file: str,
        wl: np.ndarray,
        grid: dict,
        attrs: dict = {},
        consts: dict = {},
        onedim: dict = {},
        alldim: dict = {},
        zeros: List[str] = [],
        reduce: bool = ["fwhm"],
    ):
        """
        Prepare a LUT netCDF

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
        reduce : bool or list, optional, default=['fwhm']
            Reduces the initialized Dataset by dropping the variables to reduce overall memory usage.
            If True, drops all variables. If list, drop everything but these.
        """
        self.file = file
        self.wl = wl
        self.grid = grid
        self.hold = []

        self.sizes = {key: len(val) for key, val in grid.items()}
        self.attrs = attrs

        self.consts = {**Keys.consts, **consts}
        self.onedim = {**Keys.onedim, **onedim}
        self.alldim = {**Keys.alldim, **alldim}

        # Save ds for backwards compatibility (to work with extractGrid, extractPoints)
        self.initialize()

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

    def flush(self) -> None:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT netCDF.
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

    def __getitem__(self, key: str) -> Any:
        """
        Passthrough to __getitem__ on the underlying 'ds' attribute.

        Parameters
        ----------
        key : str
            The name of the item to retrieve.

        Returns
        -------
        Any
            The value of the item retrieved from the 'ds' attribute.
        """
        return self.ds[key]

    def __repr__(self) -> str:
        return f"LUT(wl={self.wl.size}, grid={self.sizes})"


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
    # Retrieve the previous/next values such that gte and lte are encompassed
    if encompass:
        if gte is not None:
            loc_subset = ds[dim].where(ds[dim] < gte).dropna(dim)
            gte = loc_subset[-1] if len(loc_subset) > 0 else ds[dim].min()
        if lte is not None:
            loc_subset = ds[dim].where(ds[dim] > lte).dropna(dim)
            lte = loc_subset[0] if len(loc_subset) > 0 else ds[dim].max()

    if lt is not None:
        ds = ds.sel({dim: ds[dim] < lt})

    if lte is not None:
        ds = ds.sel({dim: ds[dim] <= lte})

    if gt is not None:
        ds = ds.sel({dim: gt < ds[dim]})

    if gte is not None:
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


def load(
    file: str, subset: dict = None, dask=False, mode="r", lock=False, load=True, **kwargs
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
    dask: bool, default=True
        Use Dask on the backend of Xarray to lazy load the dataset. This enables
        out-of-core subsetting
    load: bool, default=True
        Calls ds.load() at the end to cast from Dask arrays back into numpy arrays held
        in memory

    Examples
    --------
    >>> # Create a test file for the examples to load
    >>> file = 'subsetting_example.nc'
    >>> lut_dims = {
    ...     'AOT550': [0.001, 0.1009, 0.2008, 0.3007, 0.4006, 0.5005, 0.6004, 0.7003, 0.8002, 0.9001, 1.],
    ...     'H2OSTR': [0.2231, 0.4637, 0.7042, 0.9447, 1.1853, 1.4258, 1.6664, 1.9069, 2.1474, 2.388, 2.6285, 2.869, 3.1096, 3.3501],
    ...     'observer_zenith': [170.1099, 172.7845],
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
    ...     'observer_zenith': 172.7845,
    ...     'surface_elevation_km': 'mean'
    ... }
    >>> load(file, subset).dims
    Frozen({'wl': 285, 'point': 30})
    >>> load(file, subset).unstack().dims
    Frozen({'AOT550': 3, 'H2OSTR': 10, 'wl': 285})
    """
    ds = Dataset(file, mode="r")
    chunks = {k: 1 for k in ds.dimensions.keys()}
    chunks["wl"] = len(ds.dimensions["wl"])
    ds.close()
    del ds
    if dask:
        ds = xr.open_mfdataset([file], mode=mode, lock=lock, **kwargs, chunks=chunks)
    else:
        #ds = xr.open_dataset(file, mode=mode, lock=lock, **kwargs, chunks=chunks)
        ds = xr.open_dataset(file, mode=mode, lock=lock, **kwargs)

    # Special case that doesn't require defining the entire grid subsetting strategy
    if subset is None:
        Logger.debug("Subset was None, using entire file")

    elif isinstance(subset, dict):
        # The subset dict must contain all coordinate keys in the lut file
        missing = set(ds.coords) - ({"wl"} | set(subset))
        if missing:
            Logger.error(
                "The following keys are in the LUT file but not specified how to be handled by the config:"
            )
            for key in missing:
                Logger.error(f"- {key}")
            raise AttributeError(
                f"Subset dictionary is missing keys that are present in the LUT file: {missing}"
            )

        # Apply subsetting strategies
        for dim, strat in subset.items():
            logging.debug(f"Subsetting {dim}")
            ds = sub(ds, dim, strat)

    else:
        Logger.error("The subsetting strategy must be a dictionary")
        raise AttributeError(
            f"Bad subsetting strategy, expected either a dict or a NoneType: {subset}"
        )

    dims = ds.drop_dims("wl").dims

    # Create the point dimension
    ds = ds.stack(point=dims).transpose("point", "wl")

    for name, nans in ds.isnull().any().items():
        if nans:
            Logger.warning(
                f"Detected NaNs in the following LUT variable and may cause issues: {name}"
            )

    if load:
        Logger.info("Loading LUT into memory")
        ds.load()

    return ds


def extractPoints(ds: xr.Dataset) -> np.array:
    """
    Extracts the points and point name arrays
    """
    points = np.array([*ds.point.data])
    names = np.array([name for name in ds.point.coords])[1:]
    return points


def extractGrid(ds: xr.Dataset) -> dict:
    """
    Extracts the LUT grid from a Dataset
    """
    grid = {}
    for dim, vals in ds.coords.items():
        if dim in {"wl", "point"}:
            continue
        if len(vals.data.shape) > 0 and vals.data.shape[0] > 1:
            grid[dim] = vals.data
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
