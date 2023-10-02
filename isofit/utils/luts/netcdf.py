from netCDF4 import Dataset

try:
    # Xarray is not an ISOFIT required package
    import xarray as xr
except:
    xr = None


# Required keys to be in the lut file
REQD = [
    "solar_irr",
    "rhoatm",
    "transm",
    "sphalb",
    "thermal_upwelling",
    "thermal_downwelling",
    "t_up_dirs",
    "t_up_difs",
    "t_down_dirs",
    "t_down_difs",
]


def initialize(file, wl, points):
    """
    file: str
        File to write the NetCDF to
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
        for key in REQD:
            ds.createVariable(key, np.float64, dimensions=["wl", "point"])


def updatePoint(file, point, data, parallel=False):
    """
    Updates one point of a NetCDF file

    Experimental
    """
    with Dataset(file, "a", format="NETCDF4", parallel=parallel) as ds:
        for key, values in data.items():
            ds[key][:, point] = values


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
