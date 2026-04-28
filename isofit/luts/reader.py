from __future__ import annotations

import atexit
import gc
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from isofit import __version__
from isofit.core import common
from isofit.luts import optimizedInterp, sub

Logger = logging.getLogger(__name__)


class LUT:
    def __init__(
        ds, n_point: int, indices: SimpleNamespace, lut_interpolators: dict = {}
    ):
        self.n_point = n_point
        self.indices = indices
        self.ds = ds
        self.wl = ds.wl
        self.rt_mode = ds.attrs.get("RT_mode", "transm")
        self.lut_inteprolators = lut_interpolators
        self.cached = SimpleNamespace(point=np.array([]))

    def __call__(x_RT: np.array, geom: Geometry):
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
        point = np.zeros(self.n_point)

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
        value = {key: lut(point) for key, lut in self.lut_inteprolators.items()}

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

    def __repr__(self) -> str:
        return f"LUT(wl={self.wl.size}, grid={self.sizes})"


class Reader:
    def load(
        file,
        mode: str = "a",
        lock: bool = False,
        subset: dict = None,
        dask: bool = False,
        **kwargs,
    ):
        """
        Loads a LUT NetCDF
        Assumes to be a regular grid at this time (auto creates the point dim)

        Parameters
        ----------
        Examples
        --------
        >>> # Create a test file for the examples to load
        >>> file = 'subsetting_example.nc'
        >>> lut_dims = {
        ...     'AOT550': [0.001, 0.1009, 0.2008, 0.3007, 0.4006, 0.5005, 0.6004, 0.7003, 0.8002, 0.9001, 1.],
        ...     'H2OSTR': [0.2231, 0.4637, 0.7042, 0.9447, 1.1853, 1.4258, 1.6664, 1.9069, 2.1474, 2.388, 2.6285, 2.869, 3.1096, 3.3501],
        ...     'observer_zenith': [7.2155, 9.8900],
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
        ...     'observer_zenith': 7.2155,
        ...     'surface_elevation_km': 'mean'
        ... }
        >>> load(file, subset).dims
        Frozen({'wl': 285, 'point': 30})
        >>> load(file, subset).unstack().dims
        Frozen({'AOT550': 3, 'H2OSTR': 10, 'wl': 285})
        """

        if dask:
            Logger.debug(f"Using Dask to load: {file}")
            ds = xr.open_mfdataset([file], mode=mode, lock=lock, **kwargs)
        else:
            Logger.debug(f"Using Xarray to load: {file}")
            ds = xr.open_dataset(file, mode=mode, lock=lock, **kwargs)

        status = ds.attrs.get("ISOFIT status", "<not set>")
        if status != "success":
            Logger.warning(
                f"The LUT status is {status}, there may be issues with it downstream"
            )
            Logger.debug(
                "To fix this error and you know the the LUT is correct, set NetCDF attribute 'ISOFIT status' to 'success'"
            )

        version = ds.attrs.get("ISOFIT version", "<not set>")
        Logger.debug(
            f"This LUT was created with ISOFIT version {version}, you are running ISOFIT {__version__}"
        )

        # Special case that doesn't require defining the entire grid subsetting strategy
        if subset is None:
            Logger.debug("Subset was None, using entire file")

        elif isinstance(subset, dict):
            Logger.debug(f"Subsetting with: {subset}")

            # The subset dict must contain all coordinate keys in the lut file
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

            Logger.debug("Subsetting finished")
        else:
            Logger.error("The subsetting strategy must be a dictionary")
            raise AttributeError(
                f"Bad subsetting strategy, expected either a dict or a NoneType: {subset}"
            )

        dims = ds.drop_dims("wl").dims

        # Create the point dimension -> Coords now len(points)
        ds = ds.stack(point=dims).transpose("point", "wl")

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

        return ds

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
    def resample_xarray(lut, wl, fwhm, srf_file=""):
        """To support OCI resampling"""

        # Discover variables along the wl dim
        keys = {key for key in lut if "wl" in lut[key].dims} - {"fwhm"}

        # Apply resampling to these keys
        # Use srf_file if OCI
        kwargs = {
            "wl": lut.wl,
            "wl2": wl,
            "fwhm2": fwhm,
        }
        if srf_file:
            kwargs["srf_file"] = srf_file
        conv_lut = xr.apply_ufunc(
            common.resample_spectrum,
            lut[keys],
            kwargs=kwargs,
            input_core_dims=[["wl"]],  # Only operate on keys with this dim
            exclude_dims=set(["wl"]),  # Allows changing the wl size
            output_core_dims=[["wl"]],  # Adds wl to the expected output dims
            keep_attrs="override",
            # on_missing_core_dim = 'copy' # Newer versions of xarray support this
        )
        # If not on newer versions, add keys not on the wl dim
        for key in list(self.lut.drop_dims("wl")):
            conv_lut[key] = self.lut[key]
        # Override the fwhm
        conv_lut["fwhm"] = ("wl", fwhm)

        return conv_lut
