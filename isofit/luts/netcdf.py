"""
"""
import dask.array
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# def initialize(file, keys, wl, points):
#     """
#     file: str
#         File to write the NetCDF to
#     keys: list
#         Keys to fill
#     wl: np.array
#         Wavelength array
#     points: dict
#         {pointName: np.array(pointValues)}
#     """
#     with Dataset(file, "w", format="NETCDF4", clobber=True) as ds:
#         # Initialize the dimensions and set the wavelength values
#         ds.createDimension("point", size=len(list(points.items())[0][1]))
#         ds.createDimension("wl", size=wls)
#         ds.createVariable("wl", np.float64, dimensions=["wl"])
#         ds["wl"][:] = wl
#
#         # Insert the point values as variables
#         for key, values in points.items():
#             ds.createDimension(key, size=len(values))
#             ds.createVariable(key, np.float64, dimensions=["point"])
#             ds[key][:] = values
#
#         # And finally initialize the required LUT variables
#         for key in keys:
#             ds.createVariable(key, np.float64, dimensions=["wl", "point"])


def initialize(file, keys, wl, fwhm, lut_grid, chunks=25):
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


def updatePoint(file, lut_names, point, data):
    """
    Updates a point in a LUT NetCDF

    Parameters
    ----------
    lut_names: list
        List of str (lut_names)
    point: tuple
        Point values
    data: dict
        Input data to write in the form:
            `{key: np.array(shape=(len(wl), len(points)))}`
    """
    with Dataset(file, "a") as nc:
        # Retrieves the index for a point value
        index = lambda key, val: np.argwhere(nc[key][:] == val)[0][0]

        # Assume all keys will have the same dimensions in the same order, so just use the first key
        key = list(data.keys())[0]

        # This nc[key].dimensions is proper order, nc.dimensions may be out of order
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

        print(f"Writing to point {point!r}, resolved indices: {inds!r}")

        skips = ["wl", "solar_irr"]

        # Now insert the values at this point
        for key, values in data.items():
            if key not in nc.variables:
                print(f"Key doesn't exist in LUT file, skipping: {key}")
                continue
            elif key in skips:
                print(f"This key should not be updated by sims, skipping: {key}")
                continue

            print(f"UPDATING: {key!r}, {len(inds)}, dims = {nc[key].dimensions}")
            nc[key][inds] = values


def load(file):
    """
    Loads a LUT NetCDF
    """
    if xr:
        ds = xr.open_dataset(file, mode="r", lock=False)

        # TODO: fix hardcoded point names
        return ds.set_index(point=["AOT550", "H2OSTR"])
    else:
        return Dataset(file, mode="r")


def example():
    from isofit.utils.luts import netcdf as lut

    # First the RTE initializes the lut.nc file using the wavelengths array and points dict
    lut.initialize(file, wl, points)

    # Each runSim will parse its outputs then call updatePoint; done in parallel, must be parallel-write-safe
    lut.updatePoint(file, 0, data, parallel=True)

    # Use our custom loader
    ds = lut.load(file)

    return ds
