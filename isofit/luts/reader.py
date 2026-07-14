"""
Manages the reading and subsetting of look-up tables (LUTs)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from isofit import __version__
from isofit.core import common, units

Logger = logging.getLogger(__name__)


def inspect_lut_dimensions(lut_path: str) -> dict[str, np.ndarray]:
    """Inspect a prebuilt LUT file to determine available dimensions and their grid points.

    Supports both NetCDF (.nc) and Zarr formats.

    Args:
        lut_path: Path to the prebuilt LUT file (NetCDF or Zarr store)

    Returns:
        dict: Mapping dimension names to numpy arrays of grid points

    Examples:
        >>> dims = inspect_lut_dimensions("lut.nc")
        >>> dims.keys()
        dict_keys(['H2OSTR', 'AOT550', 'observer_zenith', 'wl'])
        >>> dims['H2OSTR']
        array([0.5, 1.0, 1.5, 2.0, 2.5])
    """
    lut_path = Path(lut_path)

    # Detect format based on file extension or directory structure
    if lut_path.suffix == ".zarr" or (
        lut_path.is_dir() and not lut_path.suffix == ".nc"
    ):
        return _inspect_zarr_dimensions(lut_path)
    elif lut_path.suffix == ".nc" or lut_path.is_file():
        return _inspect_netcdf_dimensions(lut_path)
    else:
        raise ValueError(
            f"Cannot determine LUT format for: {lut_path}. "
            f"Expected .nc file (NetCDF) or .zarr directory (Zarr)."
        )


def _inspect_netcdf_dimensions(lut_path: Path) -> dict[str, np.ndarray]:
    """Inspect NetCDF LUT dimensions.

    Args:
        lut_path: Path to the NetCDF LUT file

    Returns:
        dict: Mapping dimension names to numpy arrays of grid points
    """
    lut_dimensions = {}

    with Dataset(lut_path, "r") as ncds:
        # Iterate through all variables that could be LUT dimensions
        # These are typically 1-D coordinate variables
        for var_name in ncds.variables:
            var = ncds.variables[var_name]

            # Check if this is a coordinate/dimension variable (1-D)
            if len(var.dimensions) == 1 and var.dimensions[0] == var_name:
                # This is a dimension variable - store the actual grid points
                data = var[:]
                if len(data) > 0:
                    lut_dimensions[var_name] = np.array(data)

    return lut_dimensions


def _inspect_zarr_dimensions(lut_path: Path) -> dict[str, np.ndarray]:
    """Inspect Zarr LUT dimensions using xarray.

    Args:
        lut_path: Path to the Zarr LUT store

    Returns:
        dict: Mapping dimension names to numpy arrays of grid points
    """
    lut_dimensions = {}

    # Open zarr store with xarray (don't load data, just inspect)
    with xr.open_dataset(lut_path, engine="zarr", chunks=None) as ds:
        # Extract all coordinate dimensions
        for coord_name in ds.coords:
            coord_data = ds[coord_name].values
            if len(coord_data) > 0:
                lut_dimensions[coord_name] = np.array(coord_data)

    return lut_dimensions


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
    Logger.debug(f"Estimated dataset size: {ds.nbytes / 2**30} GB")

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

    # Operations to perform
    opts = {"isel": {}, "interp": {}}

    for dim, strat in subset.items():
        # Interpolation
        if (v := strat.get("interp")) is not None:
            i = ds.indexes[dim].get_indexer([v], method="ffill")[0]
            opts["isel"][dim] = slice(i, i + 1)
            opts["interp"][dim] = v
        # Subselect
        else:
            opts["isel"][dim] = slice(
                ds.indexes[dim].get_indexer([strat["gte"]], method="ffill")[0],
                ds.indexes[dim].get_indexer([strat["lte"]], method="bfill")[0] + 1,
            )

    if opts["isel"]:
        Logger.debug(f"Selecting")
        ds = ds.isel(**opts["isel"])

    Logger.debug(f"Estimated isel size: {ds.nbytes / 2**30} GB")

    if opts["interp"]:
        Logger.debug("Loading for interp")
        ds = ds.load()

        Logger.debug("Interpolating")
        ds = ds.interp(**opts["interp"])

    Logger.debug(f"Estimated final size: {ds.nbytes / 2**30} GB")

    # Save this subsetting strategy to the attributes for future reference
    ds.attrs["lut_subset"] = str(subset)

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
    path: str,
    subset: dict = None,
    mf: bool = False,
    check: bool = False,
    stack: bool = True,
    load: bool = True,
    chunks: dict = {},
    **kwargs,
):
    """
    Parameters
    ----------
    path : str
        Path to LUT store to load
    subset : dict, default={}
        Subset each dimension with a given strategy. Each dimension in the LUT file
        must be specified.
        See examples for more information
    mf : bool, default=False
        Uses xr.open_mfdataset instead which enables multi-file support
    check : bool, default=True
        Checks the dataset for NaNs and replaces them with 0s if the array is not
        entirely NaN
    load : bool, default=True
        Loads the final dataset into memory
    chunks : dict, default={}
        Chunks parameter for open_dataset, isofit generally wants to have the dataset
        be chunked by the file's chunks
    **kwargs
        Additional parameters to pass to open_dataset/open_mfdataset


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
    path = Path(path)

    # Windows needs to define the engine as the xarray auto-discover breaks
    engine = None
    if Path(path).is_dir():
        engine = "zarr"

    xropen = xr.open_dataset
    if mf:
        xropen = xr.mfopen_dataset

    # Convert dicts of {dim: None, ...} to None
    if subset and not any(subset.values()):
        subset = None

    # Special case that doesn't require defining the entire grid subsetting strategy
    if load:
        if not subset:
            Logger.debug(
                "With no subset defined and load enabled, disabling default chunking for performance"
            )
            chunks = None

        elif path.is_file() and path.stat().st_size < units.byte_string_to_float("4gb"):
            Logger.debug(
                "LUT store detected less than 4gb and load enabled, disabling default chunking for performance"
            )
            chunks = None

    ds = xropen(path, chunks=chunks, engine=engine, **kwargs)

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
    if stack:
        ds = ds.stack(point=dims).transpose("point", "wl")

    if check:
        check_nans(ds)

    if load:
        ds = ds.load()

    return ds


class Reader:
    """
    LUT reader class to manage the reading and manipulation of LUT stores
    """

    def __init__(
        self,
        build_interpolators=False,
        lut_subset={},
        mode="r",
        load_kwargs={},
        postprocess=True,
        **write_kwargs,
    ):
        """
        Parameters
        ----------
        build_interpolators : bool, default=False
            Call self.build_interpolators() after initialization
        """
        if not hasattr(self, "lut_path"):
            raise AttributeError("Inheritor class must define 'lut_path'")

        self.lut_path = Path(self.lut_path)
        if self.lut_path.exists():
            Logger.info("Prebuilt LUT provided")
        else:
            Logger.info("No LUT provided, attempting to build it")

            if not hasattr(self, "write"):
                raise NotImplemented(
                    "This object did not inherit the LUT Writer class and therefore cannot write a LUT, please use a defined engine instead"
                )

            self.write(**write_kwargs)

        self.lut = load(path=self.lut_path, mode=mode, subset=lut_subset, **load_kwargs)
        self.rt_mode = self.lut.attrs.get("RT_mode", "transm")

        # Write the NetCDF information to the log file so devs have that info during debugging
        # Have to create a fileobj to capture the text because it doesn't return (prints straight to stdout by default)
        info = io.StringIO()
        self.lut.info(info)
        Logger.debug(f"LUT information:\n{info.getvalue()}")

        if not hasattr(self, "lut_grid"):
            self.lut_grid = self.lut.attrs["lut_grid"]

        if not hasattr(self, "lut_names"):
            self.lut_names = list(self.lut_grid)

        # REVIEW: Is this still necessary?
        # remove 'point' if added to lut_names after subsetting
        # if "point" in self.lut_names:
        #     remove = np.where(self.lut_names == "point")
        #     self.lut_names = np.delete(self.lut_names, remove)

        if postprocess:
            self.lut_postprocess()

        if build_interpolators:
            self.build_interpolators()

    def __getitem__(self, key):
        """
        Enables key indexing for easier access to the numpy object store in
        self.lut[key]
        """
        return self.lut[key].load().data

    def lut_postprocess(self):
        """
        Any additional post processing that may need to be applied to the LUT before
        building interpolators
        """
        pass

    def build_interpolators(self, keys=None, interpolator_style="mlg_numba"):
        """
        Builds the interpolators using the LUT store

        TODO: optional load from/write to disk
        """
        if keys is None:
            keys = self.lut_keys.alldim

        self.cached = SimpleNamespace(point=np.array([]))
        self.interpolators = {}

        ds = self.lut.unstack("point")

        # Make sure its in expected order, wl at the end
        lut_names = list(self.lut_names)
        ds = ds.transpose(*list(lut_names), "wl")

        grid = [ds[key].data for key in lut_names]
        for key in keys:
            self.interpolators[key] = common.VectorInterpolator(
                grid_input=grid,
                data_input=ds[key].load().data,
                version=interpolator_style,
            )

        return self.interpolators

    def interpolate(self, point=None):
        """ """
        if self.cached.point.size and (point == self.cached.point).all():
            return self.cached.value

        # Run the interpolators
        value = {key: lut(point) for key, lut in self.interpolators.items()}

        # Convert both the dict and the values to read-only
        value = MappingProxyType(value)
        for v in value.values():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False

        # Update the cache
        self.cached.point = point
        self.cached.value = value

        return value

    def resample_wl(self, wl, fwhm, **kwargs):
        """
        Resamples the LUT wavelengths to a new given wavelengths

        Parameters
        ----------
        wl : array
            Wavelengths to resample to
        fwhm : array
            fwhm for common.resample_spectrum
        srf_file : str, default=None
            OCI srf_file
        """
        ds = self.lut
        ds.load()

        # Discover variables along the wl dim
        keys = {key for key in ds if "wl" in ds[key].dims} - {"fwhm"}

        kwargs |= {
            "wl": ds.wl,
            "wl2": wl,
            "fwhm2": fwhm,
        }

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

        self.lut = conv_ds

    def subset_wl(self, rng):
        """
        Subsets along the wavelength dimension

        Parameters
        ----------
        rng : tuple[float, float]
            Subselects using <= rng[1] >= rng[0]
            Subselects using rng[0] <= wl <= rng[1]
        """
        Logger.info(f"Subsetting wavelengths to range: {rng}")
        self.lut = sub(self.lut, "wl", dict(zip(["gte", "lte"], rng)))
