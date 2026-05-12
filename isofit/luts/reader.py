"""
Manages the reading and subsetting of look-up tables (LUTs)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from isofit import __version__
from isofit.core import common

Logger = logging.getLogger(__name__)


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


def subsetting(ds, subset):
    """
    Subsets a Dataset
    """
    # The strategy dict must contain all coordinate keys in the lut file
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

    return ds


def check_nans(ds):
    """
    Checks for NaNs in the Dataset
    """
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


def load(
    self,
    path: str,
    subset: dict = None,
    mf: bool = False,
    check: bool = False,
    **kwargs,
):
    """
    Loads a LUT NetCDF
    Assumes to be a regular grid at this time (auto creates the point dim)

    Parameters
    ----------


    Examples
    --------
    >>> # Create a test path for the examples to load
    >>> path = 'subsetting_example.nc'
    >>> lut_dims = {
    ...     'AOT550': [0.001, 0.1009, 0.2008, 0.3007, 0.4006, 0.5005, 0.6004, 0.7003, 0.8002, 0.9001, 1.],
    ...     'H2OSTR': [0.2231, 0.4637, 0.7042, 0.9447, 1.1853, 1.4258, 1.6664, 1.9069, 2.1474, 2.388, 2.6285, 2.869, 3.1096, 3.3501],
    ...     'observer_zenith': [7.2155, 9.8900],
    ...     'surface_elevation_km': [0., 0.2361, 0.4721, 0.7082, 0.9442, 1.1803, 1.4164, 1.6524, 1.8885, 2.1245, 2.3606, 2.5966, 2.8327, 3.0688, 3.3048, 3.5409, 3.7769, 4.013],
    ...     'wl': range(285)
    ... }
    >>> ds = xr.Dataset(coords=lut_dims)
    >>> ds.to_netcdf(path)

    >>> # Subset: Exact values along the dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': [1.1853, 2.869],
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None,
    ... }
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 792})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 2376})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 3960})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 3168})
    >>> load(path, subset).unstack().dims
    Frozen({'AOT550': 11, 'H2OSTR': 8, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Exact value, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 2.869,
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(path, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Using mean, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 'mean',
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(path, subset).unstack().dims
    Frozen({'AOT550': 11, 'observer_zenith': 2, 'surface_elevation_km': 18, 'wl': 285})

    >>> # Subset: Using max, squeeze dimension
    >>> subset = {
    ...     'AOT550': None,
    ...     'H2OSTR': 'max',
    ...     'observer_zenith': None,
    ...     'surface_elevation_km': None
    ... }
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 396})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 1080})
    >>> load(path, subset).unstack().dims
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
    >>> load(path, subset).dims
    Frozen({'wl': 285, 'point': 30})
    >>> load(path, subset).unstack().dims
    Frozen({'AOT550': 3, 'H2OSTR': 10, 'wl': 285})
    """
    xropen = xr.open_dataset
    if mf:
        xropen = xr.mfopen_dataset

    ds = xropen(path, **kwargs)

    status = ds.attrs.get("ISOFIT status", "<not set>")
    if status != "success":
        Logger.warning(
            f"The LUT status is {status}, there may be issues with it downstream"
        )
        Logger.debug(
            "To fix this error and you know the the LUT is correct, set the attribute 'ISOFIT status' to 'success'"
        )

    version = ds.attrs.get("ISOFIT version", "<not set>")
    Logger.debug(
        f"This LUT was created with ISOFIT version {version}, you are running ISOFIT {__version__}"
    )

    # Special case that doesn't require defining the entire grid subsetting strategy
    if subset is None:
        Logger.debug("Subset was None, using entire store")

    elif isinstance(subset, dict):
        Logger.debug(f"Subsetting with: {subset}")

        ds = subsetting(ds, subset)

        Logger.debug("Subsetting finished")
    else:
        Logger.error("The subsetting strategy must be a dictionary")
        raise AttributeError(
            f"Bad subsetting strategy, expected either a dict or a NoneType: {subset}"
        )

    dims = ds.drop_dims("wl").dims

    # Extract the grid, for convenience
    ds.attrs["lut_grid"] = {dim: ds[dim].data for dim in dims}

    # Create the point dimension -> Coords now len(points)
    ds = ds.stack(point=dims).transpose("point", "wl")

    if check:
        check_nans(ds)

    return ds


# %%
import xarray as xr

ds = xr.open_dataset(
    "/Users/jamesmo/projects/isofit/extras/examples/LakeMary/lut_full/lut.nc"
)

(ds.coords)
ds.coords.to_dataset().to_dict()["coords"]

grid = {key: vals.data for key, vals in ds.coords.items() if key != "wl"}
grid

ds


# %%
class LUT:
    def __init__(
        self,
        ds,
        n_lut_input_dim: int,
        indices: SimpleNamespace,
        interpolators: dict = {},
    ):
        self.n_lut_input_dim = n_lut_input_dim
        self.indices = indices
        self.ds = ds
        self.wl = ds.wl
        self.wl = ds.fwhm
        self.rt_mode = ds.attrs.get("RT_mode", "transm")
        self.interpolators = interpolators
        self.cached = SimpleNamespace(point=np.array([]))

    def __call__(self, x_RT: np.array, geom: Geometry):
        """
        Retrieves the interpolation values for a given point

        Parameters
        ----------
        x_RT: np.array
            Radiative-transfer portion of the statevector
        geom: Geometry
            Local geometry conditions for lookup

        Returns
        -------
        self.interpolate(point): dict
            ...
        """
        point = np.zeros(self.n_lut_input_dim)

        point[self.indices.x_RT] = x_RT
        for i, key in self.indices.geom.items():
            point[i] = getattr(geom, key)

        # convert observer zenith to MODTRAN convention if needed
        if self.indices.convert_observer_zenith:
            point[self.indices.convert_observer_zenith] = (
                180.0 - point[self.indices.convert_observer_zenith]
            )

        return self.interpolate(point)

    def interpolate(self, point: np.array) -> dict:
        """
        Compiles the results of the interpolators for a given point
        """
        if self.cached.point.size and (point == self.cached.point).all():
            return self.cached.value

        # Run the interpolators
        value = {key: lut(point) for key, lut in self.interpolators.items()}

        # Update the cache
        self.cached.point = point
        self.cached.value = value

        return value

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

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets a variable in the netCDF.

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        with Dataset(self.file, "a") as ds:
            ds[key][:] = value

    def __repr__(self):
        lut_dict = {
            coord: np.unique(self.ds.coords[coord].values)
            for coord in self.ds.coords
            if coord not in ["wl", "point"]
        }

        header = f"<LUT>"
        lines = [header]

        for name, values in lut_dict.items():
            n_points = len(values)
            v_min, v_max = values.min(), values.max()

            line = f"{name:<8} - ({n_points} pts): [{v_min}...{v_max}]"
            lines.append(line)

        return "\n".join(lines)


class Reader:
    def __init__(self):
        if not hasattr(self, "lut_path"):
            raise AttributeError("Inheritor class must define 'lut_path'")

        self.lut_path = Path(self.lut_path)
        if self.lut_path.exists():
            Logger.info("Prebuilt LUT provided")
        else:
            Logger.info("No LUT provided, attempting to build it")
            self.write()

        self.lut = load(self.lut_path)

        if build_interpolators:
            self.build_interpolators()

    def write(self):
        raise NotImplemented(
            "This object did not inherit the LUT Writer class and therefore cannot write a LUT, please use a defined engine instead"
        )

    def build_interpolators(self, ds, keys, interpolator_style="mlg_numba"):
        """
        Builds the interpolators using the LUT store

        TODO: optional load from/write to disk
        """
        luts = {}

        ds = ds.unstack("point")

        # Make sure its in expected order, wl at the end
        lut_names = list(self.extractGrid(ds).keys())
        ds = ds.transpose(*lut_names, "wl")

        grid = [ds[key].data for key in lut_names]
        # Create the unique
        for key in keys.alldim:
            luts[key] = common.VectorInterpolator(
                grid_input=grid,
                data_input=ds[key].load().data,
                version=interpolator_style,
            )

        return luts

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def resample_xarray(ds, wl, fwhm, srf_file=""):
        """To support OCI resampling"""

        # Discover variables along the wl dim
        keys = {key for key in ds if "wl" in ds[key].dims} - {"fwhm"}

        # Apply resampling to these keys
        # Use srf_file if OCI
        kwargs = {
            "wl": ds.wl,
            "wl2": wl,
            "fwhm2": fwhm,
        }
        if srf_file:
            kwargs["srf_file"] = srf_file

        conv_ds = xr.apply_ufunc(
            common.resample_spectrum,
            ds[keys],
            kwargs=kwargs,
            input_core_dims=[["wl"]],  # Only operate on keys with this dim
            exclude_dims=set(["wl"]),  # Allows changing the wl size
            output_core_dims=[["wl"]],  # Adds wl to the expected output dims
            keep_attrs="override",
            # on_missing_core_dim = 'copy' # Newer versions of xarray support this
        )
        # If not on newer versions, add keys not on the wl dim
        for key in list(ds.drop_dims("wl")):
            conv_ds[key] = ds[key]

        # Override the fwhm
        conv_ds["fwhm"] = ("wl", fwhm)

        return conv_ds
