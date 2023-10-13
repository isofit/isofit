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
    keys: list,
    wl: np.array,
    fwhm: np.array,
    lut_grid: dict,
    chunks: int = 25,
) -> xr.Dataset:
    """
    Initializes a LUT NetCDF using Xarray
    """
    # Initialize with all lut point names as dimensions
    ds = xr.Dataset({"fwhm": ("wl", fwhm)}, coords={"wl": wl} | lut_grid)

    # Stack them together to get the common.combos, creates dim "point" = [(v1, v2, ), ...]
    ds = ds.stack(point=lut_grid)

    # Easy fill these keys using the stacked (point) form
    filler = dask.array.full((len(wl), ds.point.size), np.nan, chunks=chunks)
    for key in keys:
        ds[key] = (("wl", "point"), filler)

    # Must write unstacked
    ds.unstack().to_netcdf(file, mode="w", compute=False)

    return ds


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

            # Not on the point dimension
            # REVIEW: parallel safe to clobber?
            elif len(nc[key].dimensions) == 1:
                nc[key][:] = values

            else:
                # Not a special case, save as-is
                nc[key][inds] = values


def load(file: str, lut_names: list = []) -> xr.Dataset:
    """
    Loads a LUT NetCDF
    """
    ds = xr.open_dataset(file, mode="r", lock=False)
    dims = lut_names or ds.drop_dims("wl").coords
    return ds.stack(point=dims)


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
