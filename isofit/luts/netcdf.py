"""
"""
import logging

import dask.array
import numpy as np
import xarray as xr
from netCDF4 import Dataset

Logger = logging.getLogger(__file__)


def initialize(
    file: str,
    wl: np.array,
    lut_grid: dict,
    chunks: int = 25,  # REVIEW: Good default? Can we calculate it? TODO: Config option?
    consts: list = [],
    onedim: list = [],
    alldim: list = [],
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
    filler = dask.array.full((len(wl),), np.nan, chunks=chunks)
    for key in onedim:
        if isinstance(key, tuple):
            key, filler = key
        ds[key] = ("wl", filler)

    # Insert point dimensional keys
    filler = dask.array.full(tuple(ds.dims.values()), np.nan, chunks=chunks)
    for key in alldim:
        if isinstance(key, tuple):
            key, filler = key
        ds[key] = (ds.coords, filler)

    # Must write unstacked
    ds.to_netcdf(file, mode="w", compute=False)

    # Stack to get the common.combos, creates dim "point" = [(v1, v2, ), ...]
    return ds.stack(point=lut_grid).transpose("point", "wl")


def updatePoint(file: str, lut_names: list, point: tuple, data: dict) -> None:
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
                Logger.warning(
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


def load(file: str, lut_names: list = []) -> xr.Dataset:
    """
    Loads a LUT NetCDF
    """
    ds = xr.open_dataset(file, mode="r", lock=False)
    dims = lut_names or ds.drop_dims("wl").coords
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
