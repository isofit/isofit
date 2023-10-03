import dask.array
import numpy as np
import xarray as xr

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


def initialize(file, wl, fwhm, points, chunks=25):
    """
    Initializes a zarr store for ISOFIT LUTs

    Parameters
    ----------
    wl: np.array
        Wavelengths array
    points: dict
        {point name: [point values], ...}
    """
    # Filler lazy data, takes no memory, just informs the shape of each key
    size = np.array(list(points.values())).shape[1]
    filler = dask.array.zeros((len(wl), size), chunks=chunks)

    # Initial dataset object to initialize the zarr with
    ds = xr.Dataset(
        coords={"wl": wl} | {key: ("point", value) for key, value in points.items()}
    )
    # fwhm saved as a variable on the wl dim
    ds["fwhm"] = ("wl", fwhm)

    # Write creation mode, save the coordinates
    ds.to_zarr(file, mode="w", compute=True)

    # Add in lazy data for each required key
    for key in REQD:
        ds[key] = (("wl", "point"), filler)

    # Initialize these variables in the zarr store
    ds.to_zarr(file, mode="a", compute=False)

    return ds


def getPointIndex(ds, point):
    """
    Converts a point tuple into the index matching the point in the LUT store
    """
    points = np.array([np.array(point) for point in ds.point.data])
    match = np.all(points == point, axis=1)
    idx = np.where(match)
    if idx:
        idx = idx[0][0]
    else:
        Logger.error(f"Point {point!r} not found in points array: {points}")


def updatePoint(file, point, data):
    """
    Updates a zarr store in place given a point index

    Parameters
    ----------
    file: str
    point: int or tuple
        The point index to write to or a tuple of the point values to discover
        the index with
    data: dict
        Keys to save
    """
    if isinstance(point, tuple):
        point = getPointIndex(ds, point)

    ds = xr.Dataset({key: ("wl", value) for key, value in data.items()})
    ds = ds.expand_dims("point").transpose()

    ds.to_zarr(file, region={"point": slice(point, point + 1)})


def load(file):
    """
    Loads a zarr store
    """
    ds = xr.open_zarr(file)

    # Retrieve the point coordinates and convert them back to a point MultiIndex
    points = list(ds.drop_dims("wl").coords)

    return ds.set_index(point=points)


def extractGrid(ds):
    """
    Extracts the LUT grid from a Dataset
    """
    grid = {}
    for dim, vals in ds.coords.items():
        if dim in ["wl", "point"]:
            continue
        grid[dim] = vals.data
    return grid


def example():
    from isofit.utils.luts import zarr as lut

    # First the RTE initializes the lut.nc file using the wavelengths array and points dict
    lut.initialize(file, wl, points)

    # Each runSim will parse its outputs then call updatePoint; done in parallel, must be parallel-write-safe
    lut.updatePoint(file, 0, data)

    # Use our custom loader
    ds = lut.load(file)

    return ds
