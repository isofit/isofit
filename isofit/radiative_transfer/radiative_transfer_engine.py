#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
# Author: Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import xarray as xr

from isofit import ray
from isofit.core import common, units
from isofit.radiative_transfer import luts

Logger = logging.getLogger(__file__)


class RadiativeTransferEngine:
    # Allows engines to outright disable the parallelized sims if they do nothing
    _disable_makeSim = False

    # Sleep a random amount of time up to max this value at the start of each streamSimulation
    # Can be set per custom engine
    max_buffer_time = 0

    # These are retrieved from the geom object
    geometry_input_names = [
        "observer_azimuth",
        "observer_zenith",
        "solar_azimuth",
        "solar_zenith",
        "relative_azimuth",
        "observer_altitude_km",
        "surface_elevation_km",
    ]

    # These properties enable easy access to the lut data
    coszen = property(lambda self: self["coszen"])
    solar_irr = property(lambda self: self["solar_irr"])

    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        lut_path: str = "",
        lut_grid: dict = None,
        wavelength_file: str = None,
        interpolator_style: str = "mlg",
        build_interpolators: bool = True,
        overwrite_interpolator: bool = False,
        wl: np.array = [],  # Wavelength override
        fwhm: np.array = [],  # fwhm override
    ):
        if lut_path is None:
            Logger.error(
                "The lut_path must be a valid path at this time. Either it exists as a valid LUT or a LUT will be generated to that path"
            )

        # Keys of the engine_config are available via self unless overridden
        # eg. engine_config.lut_names == self.lut_names
        self.engine_config = engine_config

        # TODO: mlky should do all this verification stuff
        # Verify either the LUT file exists or a LUT grid is provided
        self.lut_path = lut_path = str(lut_path) or engine_config.lut_path
        logging.debug(f"lut_grid {lut_grid}")
        try:
            logging.debug(f"self.lut_grid {self.lut_grid}")
        except:
            logging.debug("self.lut_grid: None")
        logging.debug(f"lut_grid is none {lut_grid is None}")
        exists = os.path.isfile(lut_path)
        if not exists and lut_grid is None:
            raise AttributeError(
                "Must provide either a prebuilt LUT file or a LUT grid"
            )

        # Save parameters to instance
        self.interpolator_style = interpolator_style
        self.overwrite_interpolator = overwrite_interpolator

        self.treat_as_emissive = engine_config.treat_as_emissive
        self.engine_base_dir = engine_config.engine_base_dir
        self.sim_path = engine_config.sim_path

        # Enable special modes
        self.rt_mode = (
            engine_config.rt_mode if engine_config.rt_mode is not None else "transm"
        )
        self.coupling_terms = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
        self.multipart_transmittance = engine_config.multipart_transmittance
        self.glint_model = engine_config.glint_model
        if self.glint_model and not self.multipart_transmittance:
            raise AttributeError(
                "Using the glint model requires a multipart transmittance LUT table"
            )

        # Specify wavelengths and fwhm to be used for either resampling an existing LUT or building a new instance
        if not any(wl) or not any(fwhm):
            self.wavelength_file = wavelength_file
            if os.path.exists(wavelength_file):
                Logger.info(f"Loading from wavelength_file: {wavelength_file}")
                try:
                    wl, fwhm = common.load_wavelen(wavelength_file)
                except:
                    pass

        # Set for downstream engines may use
        self.wl = wl
        self.fwhm = fwhm

        # ToDo: move setting of multipart rfl values to config
        if self.multipart_transmittance:
            self.test_rfls = [0.0, 0.1, 0.5]

        # Extract from LUT file if available, otherwise initialize it
        if exists:
            Logger.info(f"Prebuilt LUT provided")
            Logger.debug(
                f"Reading from store: {lut_path}, subset={engine_config.lut_names}"
            )
            self.lut = luts.load(lut_path, subset=engine_config.lut_names)
            self.lut_grid = lut_grid or luts.extractGrid(self.lut)
            self.points = luts.extractPoints(self.lut)
            self.lut_names = list(self.lut_grid.keys())
            Logger.info(f"LUT grid loaded from file")
            Logger.debug(f"{self.lut_grid}")

            # remove 'point' if added to lut_names after subsetting
            if "point" in self.lut_names:
                remove = np.where(self.lut_names == "point")
                self.lut_names = np.delete(self.lut_names, remove)

            # Enable special modes - argument: get from prebuilt LUT netCDF if available
            self.rt_mode = self.lut.attrs.get("RT_mode", "transm")
            if self.rt_mode not in ["transm", "rdn"]:
                Logger.error(
                    "Unknown RT mode provided in LUT file. Please use either 'transm' or 'rdn'."
                )
                raise ValueError(
                    "Unknown RT mode provided in LUT file. Please use either 'transm' or 'rdn'."
                )

            # sc - Bandaid for code to know whether to use gaussian assumptions
            #      Currently, if using tsis, then OCI, which is non-gaussian
            srf_file = None
            irr_file = Path(engine_config.irradiance_file)
            if irr_file.stem == "tsis_f0_0p1":
                srf_file = irr_file.parent / "pace_oci_rsr.nc"

            # If necessary, resample prebuilt LUT to desired instrument spectral response
            if (
                not len(wl) == len(self.lut.wl)
                or not all(wl == self.lut.wl)
                or (srf_file is not None)
            ):
                # Discover variables along the wl dim
                keys = {key for key in self.lut if "wl" in self.lut[key].dims} - {
                    "fwhm",
                }
                # Apply resampling to these keys
                conv = xr.apply_ufunc(
                    common.resample_spectrum,
                    self.lut[keys],
                    kwargs={
                        "wl": self.lut.wl,
                        "wl2": wl,
                        "fwhm2": fwhm,
                        "srf_file": srf_file,
                    },  # Use srf_file if OCI
                    input_core_dims=[["wl"]],  # Only operate on keys with this dim
                    exclude_dims=set(["wl"]),  # Allows changing the wl size
                    output_core_dims=[["wl"]],  # Adds wl to the expected output dims
                    keep_attrs="override",
                    # on_missing_core_dim = 'copy' # Newer versions of xarray support this
                )
                # If not on newer versions, add keys not on the wl dim
                for key in list(self.lut.drop_dims("wl")):
                    conv[key] = self.lut[key]
                # Override the fwhm
                conv["fwhm"] = ("wl", fwhm)
                # Exchange the lut with the resampled version
                self.lut = conv
        else:
            Logger.info(f"No LUT store found, beginning initialization and simulations")
            # Check if both wavelengths and fwhm are provided for building the LUT
            if not any(wl) or not any(fwhm):
                Logger.error(
                    "No wavelength information found, please provide either as file or input argument"
                )
                raise AttributeError(
                    "No wavelength information found, please provide either as file or input argument"
                )

            Logger.debug(f"Writing store to: {self.lut_path}")
            self.lut_grid = lut_grid
            self.lut_names = list(lut_grid)
            self.points = common.combos(lut_grid.values())

            # Verify no duplicates exist else downstream functions will fail
            duplicates = False

            for dim, vals in lut_grid.items():
                if np.unique(vals).size < len(vals):
                    duplicates = True
                    Logger.error(
                        f"Duplicates values were detected in the lut_grid for {dim}: {vals}"
                    )

            if duplicates:
                raise AttributeError(
                    "Input lut_grid detected to have duplicates, please correct them before continuing"
                )

            if self.engine_config.rte_configure_and_exit:
                Logger.warning(
                    "rte_configure_and_exit is enabled, the LUT file will not be created"
                )
            else:
                Logger.info(f"Initializing LUT file")
                self.lut = luts.Create(
                    file=self.lut_path,
                    wl=wl,
                    grid=self.lut_grid,
                    attrs={"RT_mode": self.rt_mode},
                    onedim={"fwhm": fwhm},
                )

            # Create and populate a LUT file
            self.runSimulations()

        # Limit the wavelength per the config, does not affect data on disk
        if engine_config.wavelength_range is not None:
            Logger.info(
                f"Subsetting wavelengths to range: {engine_config.wavelength_range}"
            )
            self.lut = luts.sub(
                self.lut,
                "wl",
                dict(zip(["gte", "lte"], engine_config.wavelength_range)),
            )

        self.n_chan = len(self.wl)

        # TODO: This is a bad variable name - change (it's the number of input dimensions of the lut (p) not the number of samples)
        self.n_point = len(self.lut_names)

        # Simple 1-item cache for rte.interpolate()
        self.cached = SimpleNamespace(point=np.array([]))
        Logger.debug(f"LUTs fully loaded")

        # For each point index, determine if that point derives from Geometry or x_RT
        self.indices = SimpleNamespace(geom={}, x_RT=[])

        # Attach interpolators
        if build_interpolators:
            self.build_interpolators()

            geometry_keys = set(engine_config.statevector_names or self.lut_names)

            matches = common.compare(geometry_keys, self.geometry_input_names)
            if matches:
                Logger.warning(
                    "A key in the statevector was detected to be close to keys in the geometry keys list:"
                )
                for key, strings in matches.items():
                    Logger.warning(f"  {key!r} should it be one of {strings}?")

            # Hidden assumption: geometry keys come first, then come RTE keys
            self.geometry_input_names = set(self.geometry_input_names) - geometry_keys
            self.indices.geom = {
                i: key
                for i, key in enumerate(self.lut_names)
                if key in self.geometry_input_names
            }

            # check if values of observer zenith in LUT are given in MODTRAN convention
            self.indices.convert_observer_zenith = None
            if "observer_zenith" in self.lut_grid.keys():
                if any(np.array(self.lut_grid["observer_zenith"]) > 90.0):
                    self.indices.convert_observer_zenith = [
                        i
                        for i in self.indices.geom
                        if self.indices.geom[i] == "observer_zenith"
                    ][0]

            # If it wasn't a geom key, it's x_RT
            self.indices.x_RT = list(set(range(self.n_point)) - set(self.indices.geom))
            Logger.debug(f"Interpolators built")

    def __getitem__(self, key):
        """
        Enables key indexing for easier access to the numpy object store in
        self.lut[key]
        """
        return self.lut[key].load().data

    def build_interpolators(self):
        """
        Builds the interpolators using the LUT store

        TODO: optional load from/write to disk
        """
        self.luts = {}

        ds = self.lut.unstack("point")

        # Make sure its in expected order, wl at the end
        ds = ds.transpose(*self.lut_names, "wl")

        grid = [ds[key].data for key in self.lut_names]

        # Create the unique
        for key in luts.Keys.alldim:
            self.luts[key] = common.VectorInterpolator(
                grid_input=grid,
                data_input=ds[key].load().data,
                version=self.interpolator_style,
            )

    def preSim(self):
        """
        This is an optional function that can be defined by a subclass RTE to be called
        directly before runSim() is executed. A subclass may return a dict containing
        any single or non-dimensional variables to be saved to the LUT file
        """
        ...

    def makeSim(self, point: np.array, template_only: bool = False):
        """
        Prepares and executes a radiative transfer engine's simulations

        Args:
            point (np.array): conditions to alter in simulation
            template_only (bool): only write template file and then stop
        """
        raise NotImplemented(
            "This method must be defined by the subclass RTE, (TODO) see ISOFIT documentation for more information"
        )

    def readSim(self, point: np.array):
        """
        Reads simulation results to standard form

        Args:
            point (np.array): conditions to alter in simulation
        """
        raise NotImplemented(
            "This method must be defined by the subclass RTE, (TODO) see ISOFIT documentation for more information"
        )

    def postSim(self):
        """
        This is an optional function that can be defined by a subclass RTE to be called
        directly after runSim() is finished. A subclass may return a dict containing
        any single or non-dimensional variables to be saved to the LUT file
        """
        ...

    def point_to_filename(self, point: np.array) -> str:
        """Change a point to a base filename

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            str: basename of the file to use for this point
        """
        filename = "_".join(
            ["%s-%6.4f" % (n, x) for n, x in zip(self.lut_names, point)]
        )
        return filename

    # TODO: change this name
    def get(self, x_RT: np.array, geom: Geometry) -> dict:
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
        value = {key: lut(point) for key, lut in self.luts.items()}

        # Update the cache
        self.cached.point = point
        self.cached.value = value

        return value

    def runSimulations(self) -> None:
        """
        Run all simulations for the LUT grid.

        """
        Logger.info(f"Running any pre-sim functions")
        pre = self.preSim()

        if pre:
            Logger.info("Saving pre-sim data to index zero of all dimensions except wl")
            Logger.debug(f"pre-sim data contains keys: {pre.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=pre)

        # Make the LUT calls (in parallel if specified)
        if not self._disable_makeSim:
            Logger.info("Executing parallel simulations")

            # Place into shared memory space to avoid spilling
            lut_names = ray.put(self.lut_names)
            makeSim = ray.put(self.makeSim)
            readSim = ray.put(self.readSim)
            lut_path = ray.put(self.lut_path)
            buffer_time = ray.put(self.max_buffer_time)
            rte_configure_and_exit = ray.put(self.engine_config.rte_configure_and_exit)

            jobs = [
                streamSimulation.remote(
                    point,
                    lut_names,
                    makeSim,
                    readSim,
                    lut_path,
                    max_buffer_time=buffer_time,
                    rte_configure_and_exit=self.engine_config.rte_configure_and_exit,
                )
                for point in self.points
            ]

            if self.engine_config.rte_configure_and_exit:
                # Block until all jobs finish
                ray.get(jobs)

                Logger.warning("Exiting early due to rte_configure_and_exit")
                sys.exit(0)
            else:
                # Report a percentage complete every 10% and flush to disk at those intervals
                report = common.Track(
                    jobs,
                    step=10,
                    reverse=True,
                    print=Logger.info,
                    message="simulations complete",
                )

                # Update the lut as point simulations stream in
                while jobs:
                    [done], jobs = ray.wait(jobs, num_returns=1)

                    # Retrieve the return of the finished job
                    ret = ray.get(done)

                    # If a simulation fails then it will return None
                    if ret:
                        self.lut.queuePoint(*ret)

                    if report(len(jobs)):
                        Logger.info("Flushing netCDF to disk")
                        self.lut.flush()

                # Shouldn't be hit but just in case
                if self.lut.hold:
                    Logger.warning("Not all points were flushed, doing so now")
                    self.lut.flush()

            del lut_names, makeSim, readSim, lut_path, buffer_time
        else:
            Logger.debug("makeSim is disabled for this engine")

        Logger.info(f"Running any post-sim functions")
        post = self.postSim()

        if post:
            Logger.info(
                "Saving post-sim data to index zero of all dimensions except wl"
            )
            Logger.debug(f"post-sim data contains keys: {post.keys()}")

            point = {key: 0 for key in self.lut_names}
            self.lut.writePoint(point, data=post)

        # Reload the LUT now that it's populated
        Logger.debug("Reloading LUT")
        self.lut = luts.load(self.lut_path)

    def summarize(self, x_RT, *_):
        """
        Pretty prints lut_name=value, ...
        """
        pairs = zip(self.lut_names, x_RT)
        return " ".join([f"{name}={val:5.3f}" for name, val in pairs])

    # REVIEW: We need to think about the best place for the two albedo method (here, radiative_transfer.py, utils, etc.)
    @staticmethod
    def two_albedo_method(
        case_0: dict,
        case_1: dict,
        case_2: dict,
        coszen: float,
        rfl_1: float = 0.1,
        rfl_2: float = 0.5,
    ) -> dict:
        """
        Calculates split transmittance values from a multipart file using the
        two-albedo method. See notes for further detail.

        Parameters
        ----------
        case_0: dict
            MODTRAN output for a non-reflective surface (case 0 of the channel file)
        case_1: dict
            MODTRAN output for surface reflectance = rfl_1 (case 1 of the channel file)
        case_2: dict
            MODTRAN output for surface reflectance = rfl_2 (case 2 of the channel file)
        coszen: float
            cosine of the solar zenith angle
        rfl_1: float, defaults=0.1
            surface reflectance for case 1 of the MODTRAN output
        rfl_2: float, defaults=0.5
            surface reflectance for case 2 of the MODTRAN output

        Returns
        -------
        data: dict
            Relevant information

        Notes
        -----
        This implementation follows Guanter et al. (2009)
        (DOI:10.1080/01431160802438555), modified by Nimrod Carmon. It is called
        the "2-albedo" method, referring to running MODTRAN with 2 different
        surface albedos. Alternatively, one could also run the 3-albedo method,
        which is similar to this one with the single difference where the
        "path_radiance_no_surface" variable is taken from a
        zero-surface-reflectance MODTRAN run instead of being calculated from
        2 MODTRAN outputs.

        There are a few argument as to why the 2- or 3-albedo methods are
        beneficial:
            (1) For each grid point on the lookup table you sample MODTRAN 2 or
                3 times, i.e., you get 2 or 3 "data points" for the atmospheric
                parameter of interest. This in theory allows us to use a lower
                band model resolution for the MODTRAN run, which is much faster,
                while keeping high accuracy.
            (2) We use the decoupled transmittance products to expand
                the forward model and account for more physics, currently
                topography and glint.
        """
        # Instrument channel widths
        widths = case_0["width"]
        # Direct upward transmittance
        transm_up_dir = case_0["transm_up_dir"]

        # Top-of-atmosphere solar radiance as a function of solar zenith angle
        L_solar = units.E_to_L(case_0["solar_irr"], coszen)

        # Direct ground reflected radiance at sensor for case 1 (sun->surface->sensor)
        # This includes direct down and direct up transmittance
        L_toa_dir_1 = case_1["drct_rflt"]
        # Total ground reflected radiance at sensor for case 1 (sun->surface->sensor)
        # This includes direct + diffuse down, but only direct up transmittance
        L_toa_1 = case_1["grnd_rflt"]

        # Transforming back to at-surface radiance
        # Since L_toa_dir_1 and L_toa_1 only contain direct upward transmittance,
        # we can safely divide by transm_up_dir without needing transm_up_dif
        # Direct at-surface radiance for case 1 (only direct down transmittance)
        L_down_dir_1 = L_toa_dir_1 / rfl_1 / transm_up_dir
        # Total at-surface radiance for case 1 (direct + diffuse down transmittance)
        L_down_1 = L_toa_1 / rfl_1 / transm_up_dir

        # Total at-surface radiance for case 2 (direct + diffuse down transmittance)
        L_down_2 = case_2["grnd_rflt"] / rfl_2 / transm_up_dir

        # Atmospheric path radiance for case 1
        L_path_1 = case_1["path_rdn"]
        # Atmospheric path radiance for case 2
        L_path_2 = case_2["path_rdn"]

        # Total reflected radiance at surface (before upward atmospheric transmission) for case 1
        L_surf_1 = rfl_1 * L_down_1
        # Total reflected radiance at surface (before upward atmospheric transmission) for case 2
        L_surf_2 = rfl_2 * L_down_2
        # Atmospheric path radiance for non-reflective surface (case 0)
        L_path_0 = ((L_surf_2 * L_path_1) - (L_surf_1 * L_path_2)) / (
            L_surf_2 - L_surf_1
        )

        # Diffuse upward transmittance
        transm_up_dif = (L_path_1 - L_path_0) / L_surf_1

        # Spherical albedo
        salb = (L_down_1 - L_down_2) / (L_surf_1 - L_surf_2)

        # Total at-surface radiance for non-reflective surface (case 0)
        # Only add contribution from atmospheric spherical albedo
        L_down_0 = L_down_1 * (1 - rfl_1 * salb)
        # Diffuse at-surface radiance for non-reflective surface (case 0)
        L_down_dif_0 = L_down_0 - L_down_dir_1

        # Direct downward transmittance
        transm_down_dir = L_down_dir_1 / widths / L_solar
        # Diffuse downward transmittance
        transm_down_dif = L_down_dif_0 / widths / L_solar

        # Return some keys from the first part plus the new calculated keys
        pass_forward = [
            "wl",
            "rhoatm",
            "solar_irr",
            "thermal_upwelling",
            "thermal_downwelling",
        ]
        data = {
            "sphalb": salb,
            "transm_up_dir": transm_up_dir,
            "transm_up_dif": transm_up_dif,
            "transm_down_dir": transm_down_dir,
            "transm_down_dif": transm_down_dif,
        }
        for key in pass_forward:
            data[key] = case_0[key]

        return data


@ray.remote(num_cpus=1)
def streamSimulation(
    point: np.array,
    lut_names: list,
    simmer: Callable,
    reader: Callable,
    output: str,
    max_buffer_time: float = 0.5,
    rte_configure_and_exit: bool = False,
):
    """Run a simulation for a single point and stream the results to a saved lut file.

    Args:
        point (np.array): conditions to alter in simulation
        lut_names (list): Dimension names aka lut_names
        simmer (function): function to run the simulation
        reader (function): function to read the results of the simulation
        output (str): LUT store to save results to
        max_buffer_time (float, optional): _description_. Defaults to 0.5.
        rte_configure_and_exit (bool, optional): exit early if not executing simulations
    """
    Logger.debug(f"Simulating(point={point})")

    # Slight delay to prevent all subprocesses from starting simultaneously
    time.sleep(np.random.rand() * max_buffer_time)

    # Execute the simulation
    simmer(point)

    # No data will be produced, just configuration files
    if rte_configure_and_exit:
        return

    # Read the simulation results
    data = reader(point)

    # Save the results to our LUT format
    if data:
        Logger.debug(f"Updating data point {point} for keys: {data.keys()}")

        return point, data
    else:
        Logger.warning(f"No data was returned for point {point}")
