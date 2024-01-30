"""
This is the netCDF4 implementation for handling ISOFIT LUT files. For previous
implementations and research, please see https://github.com/isofit/isofit/tree/897062a3dcc64d5292d0d2efe7272db0809a6085/isofit/luts
"""
import logging
import os

import numpy as np
import xarray as xr
from netCDF4 import Dataset

# This resolves race/lock conditions with file opening in updatePoint()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

Logger = logging.getLogger(__file__)

# TODO: Temporary locking lut files for updatePoint until a new solution is found
import fcntl
import hashlib


class SystemMutex:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        lock_id = hashlib.md5(self.name.encode("utf8")).hexdigest()
        self.fp = open(f"/tmp/.lock-{lock_id}.lck", "wb")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def initialize(
    file: str,
    wl: np.array,
    lut_grid: dict,
    chunks: int = 25,  # REVIEW: Good default? Can we calculate it? TODO: Config option?
    consts: list = [],
    onedim: list = [],
    alldim: list = [],
    zeros: list = [],
) -> xr.Dataset:
    """
    Initializes a LUT NetCDF using Xarray

    Parameters
    ----------
    consts: list, defaults=[]
        Keys to fill, these are along  no dimensions, ie. ()
    onedim: list, defaults=[]
        Keys to fill, these are along one dimensions, ie. (wl, )
    alldim: list, defaults=[]
        Keys to fill, these are along all dimensions, ie. (wl, point)
    zeros: list, defaults=[]
        List of keys to default to zeros as the fill value instead of NaNs
    """
    # Initialize with all lut point names as dimensions
    ds = xr.Dataset(coords={"wl": wl} | lut_grid)

    # Insert constants
    filler = np.nan
    for key in consts:
        if isinstance(key, tuple):
            key, filler = key
        ds[key] = filler

    # Insert single dimension keys along wl
    fill = np.full((len(wl),), np.nan)
    for key in onedim:
        if isinstance(key, tuple):
            key, fill = key
        ds[key] = ("wl", fill)

    ## Insert point dimensional keys
    # Filler arrays
    dims = tuple(ds.dims.values())
    nans = np.full(dims, np.nan)
    zero = np.zeros(dims)

    for key in alldim:
        if isinstance(key, tuple):
            key, filler = key
            ds[key] = (ds.coords, filler)
        else:
            if key in zeros:
                ds[key] = (ds.coords, zero)
            else:
                ds[key] = (ds.coords, nans)

    # Must write unstacked
    ds.to_netcdf(file, mode="w", compute=False, engine="netcdf4")

    # Stack to get the common.combos, creates dim "point" = [(v1, v2, ), ...]
    return ds.stack(point=lut_grid).transpose("point", "wl")


def updatePoint(
    file: str, lut_names: list = [], point: tuple = (), data: dict = {}
) -> None:
    """
    Updates a point in a LUT NetCDF

    Parameters
    ----------
    lut_names: list
        List of str (lut_names)
    point: tuple
        Point values
    data: dict
        Input data to write. Aside from the following special keys, all keys in
        the dict should have the shape (len(wl), len(points)). Special keys:
               wl - This is set at lut initialization, will assert np.isclose
                    the existing lut[wl] against data[wl]
        solar_irr - Not along the point dimension, presently clobbers with
                    every new input
    """
    with SystemMutex("lock"):
        with Dataset(file, "a") as nc:
            # Retrieves the index for a point value
            index = lambda key, val: np.argwhere(nc[key][:] == val)[0][0]

            # Assume all keys will have the same dimensions in the same order, so just use a random key
            key, _ = max(nc.variables.items(), key=lambda pair: len(pair[1].dimensions))

            # nc[key].dimensions is ordered, nc.dimensions may be out of order
            dims = nc[key].dimensions
            inds = [-1] * len(dims)
            for i, dim in enumerate(dims):
                if dim == "wl":
                    # Wavelength uses all values
                    inds[i] = slice(None)
                elif dim in lut_names:
                    # Retrieve the index of this point value
                    inds[i] = index(dim, point[lut_names.index(dim)])
                else:
                    # Default to using the first index if key not in the lut_names
                    # This should only happen if you were updating an existing LUT with fewer point dimensions than exists
                    # REVIEW: Should we support this edge case? If we do, ought to add a `defaults={dim: index}` to control
                    # what the default index for dims are
                    # REVIEW: rte.postSim() may call this with lut_names=[], causing this to be hit each time
                    Logger.debug(
                        f"Defaulting to index 0 for dimension {dim!r} because it is not in the lut_names: {lut_names}"
                    )
                    inds[i] = 0

            Logger.debug(f"Writing to point {point!r}, resolved indices: {inds!r}")

            # Now insert the values at this point
            for key, values in data.items():
                if key not in nc.variables:
                    Logger.error(f"Key doesn't exist in LUT file, skipping: {key}")

                elif key == "wl":
                    assert np.isclose(nc[key][:], values).all(), (
                        f"Input data wavelengths do not match existing wavelengths for file: {file}\n"
                        + f"Expected: {nc[key][:]}\n"
                        + f"Received: {values}"
                    )
                else:
                    var = nc[key]
                    dim = len(var.dimensions)

                    # REVIEW: parallel safe to clobber?
                    if dim == 0:
                        # Constant/scalar value
                        var.assignValue(values)
                    elif dim == 1:
                        # Not on the point dimension
                        var[:] = values
                    else:
                        # Not a special case, save as-is
                        var[inds] = values


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
            gte = ds[dim].where(ds[dim] < gte).dropna(dim)[-1]
        if lte is not None:
            lte = ds[dim].where(ds[dim] > lte).dropna(dim)[0]

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


def load(file: str, subset: dict = None) -> xr.Dataset:
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
    ds = xr.open_mfdataset([file], mode="r", lock=False)

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
            ds = sub(ds, dim, strat)

    else:
        Logger.error("The subsetting strategy must be a dictionary")
        raise AttributeError(
            f"Bad subsetting strategy, expected either a dict or a NoneType: {subset}"
        )

    dims = ds.drop_dims("wl").dims
    return ds.stack(point=dims).transpose("point", "wl")


def extractPoints(ds: xr.Dataset) -> (np.array, np.array):
    """
    Extracts the points and point name arrays
    """
    points = np.array([*ds.point.data])
    names = np.array([name for name in ds.point.coords])[1:]

    return (points, names)


def extractGrid(ds: xr.Dataset) -> dict:
    """
    Extracts the LUT grid from a Dataset
    """
    grid = {}
    for dim, vals in ds.coords.items():
        if dim in {"wl", "point"}:
            continue
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
