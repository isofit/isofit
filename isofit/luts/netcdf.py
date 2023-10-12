import xarray as xr
from netCDF4 import Dataset


def initialize(file, keys, wl, points):
    """
    file: str
        File to write the NetCDF to
    keys: list
        Keys to fill
    wl: np.array
        Wavelength array
    points: dict
        {pointName: np.array(pointValues)}
    """
    with Dataset(file, "w", format="NETCDF4", clobber=True) as ds:
        # Initialize the dimensions and set the wavelength values
        ds.createDimension("point", size=len(list(points.items())[0][1]))
        ds.createDimension("wl", size=wls)
        ds.createVariable("wl", np.float64, dimensions=["wl"])
        ds["wl"][:] = wl

        # Insert the point values as variables
        for key, values in points.items():
            ds.createDimension(key, size=len(values))
            ds.createVariable(key, np.float64, dimensions=["point"])
            ds[key][:] = values

        # And finally initialize the required LUT variables
        for key in keys:
            ds.createVariable(key, np.float64, dimensions=["wl", "point"])


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

    # Append filled data
    ns = ds.unstack()
    ns.to_netcdf(file, mode="w", compute=False)

    return ds


def updatePoint(file, dims, point, data):
    """
    Updates a point in a LUT NetCDF.

    Parameters
    ----------
    dims: list
        List of str (lut_names)
    point: tuple
        Point values
    """
    with Dataset(file, "a", parallel=False) as nc:
        index = lambda key, val: np.argwhere(nc[key][:] == val)[0][0]
        index = [slice(None)] + [index(*item) for item in zip(dims, point)]
        for key, values in data.items():
            nc[key][index] = values


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
