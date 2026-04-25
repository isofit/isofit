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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#          Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#          Jay E. Fahlen, jay.e.fahlen@jpl.nasa.gov
#
from __future__ import annotations

import io
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import xarray as xr

from isofit import ray
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core import common, units
from isofit.core.common import eps, svd_inv_sqrt
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.engines import Engines

Logger = logging.getLogger(__file__)


class RadiativeTransfer:
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

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
        full_config: Config,
        wl: np.array = [],  # Wavelength override
        fwhm: np.array = [],  # fwhm override
    ):
        config_atmosphere = full_config.forward_model.atmosphere

        self.n_cores = full_config.implementation.n_cores
        # Check to see if this check is necessary
        if self.n_cores is None:
            self.n_cores = multiprocessing.cpu_count()

        # TODO Need to make sure this fits in with inheritance - keep lut_path as somehting that can be fed in?
        self.lut_grid = config_atmosphere.lut_grid
        # self.lut_path = lut_path = str(lut_path) or full_config.atmosphere.lut_path
        self.lut_path = config_atmosphere.lut_path
        self.statevec_names = config_atmosphere.statevector.get_element_names()

        # Irradiance file TODO Check how this relates to OCI
        self.solar_irr = self.engine.solar_irr

        lut_exists = os.path.isfile(self.lut_path)
        if not lut_exists and self.lut_grid is None:
            raise AttributeError(
                "Must provide either a prebuilt LUT file or a LUT grid"
            )

        # TODO Check that this is in fashion
        self.interpolator_style = (
            config_atmosphere.interpolator_style
            if config_atmosphere.interpolator_style
            else config_instrument.get("interpolator_style")
        )
        self.overwrite_interpolator = (
            config_atmosphere.overwrite_interpolator
            if config_atmosphere.overwrite_interpolator
            else config_instrument.get("overwrite_interpolator")
        )

        self.coupling_terms = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
        self.rt_mode = (
            config_atmosphere.rt_mode if engine_config.rt_mode is not None else "transm"
        )

        # Use explicitely passed first
        if not any(wl) or not any(fwhm):
            self.wavelength_file = (
                config_atmosphere.wavelength_file
                if config_atmosphere.wavelength_file
                else config_instrument.get("wavelength_file")
            )
            if os.path.exists(wavelength_file):
                Logger.info(f"Loading from wavelength_file: {wavelength_file}")
                wl, fwhm = common.load_wavelen(wavelength_file)
        if not wl or not fwhm:
            e = "No wavelength information found, please provide either as file or input argument"
            logger.error(e)
            raise AttributeError(e)

        # Set for downstream engines may use
        self.wl = wl
        self.fwhm = fwhm

        # Priors
        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)
        self.Sa_cached = np.diagflat(np.power(self.prior_sigma, 2))
        self.Sa_normalized = self.Sa_cached / np.mean(np.diag(self.Sa_cached))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            self.Sa_normalized
        )

        # Uncertainty
        self.bvec = config_atmosphere.unknowns.get_element_names()
        self.bval = np.array([x for x in config_atmosphere.unknowns.get_elements()[0]])

        # Logic for Pre-built LUT TODO: Move to generic class?
        if lut_exists:
            Logger.info(f"Prebuilt LUT provided")
            Logger.debug(
                f"Reading from store: {lut_path}, subset={engine_config.lut_names}"
            )
            self.lut = luts.load(self.lut_path, subset=config_atmosphere.lut_names)
            self.lut_grid = self.lut_grid or luts.extractGrid(self.lut)
            self.points = luts.extractPoints(self.lut)
            self.lut_names = list(self.lut_grid.keys())
            Logger.info(f"LUT grid loaded from file")

            # remove 'point' if added to lut_names after subsetting
            if "point" in self.lut_names:
                remove = np.where(self.lut_names == "point")
                self.lut_names = np.delete(self.lut_names, remove)

            self.rt_mode = self.lut.attrs.get("RT_mode", "transm")
            if self.rt_mode not in ["transm", "rdn"]:
                Logger.error(
                    "Unknown RT mode provided in LUT file. Please use either 'transm' or 'rdn'."
                )
                raise ValueError(
                    "Unknown RT mode provided in LUT file. Please use either 'transm' or 'rdn'."
                )

            # TODO Clean OCI irr path
            # sc - Bandaid for code to know whether to use gaussian assumptions
            #      Currently, if using tsis, then OCI, which is non-gaussian
            srf_file = None
            irr_file = Path(engine_config.irradiance_file)
            if irr_file.stem == "tsis_f0_0p1":
                srf_file = irr_file.parent / "pace_oci_rsr.nc"

            # TODO Is this ever necessary?? - A lot of this logic is the nested self.lut calls we want to abstract out.
            # If necessary, resample prebuilt LUT to desired instrument spectral response
            if (
                not len(self.wl) == len(self.lut.wl)
                or not all(self.wl == self.lut.wl)
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
            # TODO This may go into the engines classes? That way we could support generic?
            Logger.info(f"No LUT store found, beginning initialization and simulations")
            Logger.debug(f"Writing store to: {self.lut_path}")

            self.lut_names = list(self.lut_grid.keys())
            self.points = common.combos(self.lut_grid.values())

            # Verify no duplicates exist else downstream functions will fail
            duplicates = False

            for dim, vals in self.lut_grid.items():
                if np.unique(vals).size < len(vals):
                    duplicates = True
                    Logger.error(
                        f"Duplicates values were detected in the lut_grid for {dim}: {vals}"
                    )

            if duplicates:
                raise AttributeError(
                    "Input lut_grid detected to have duplicates, please correct them before continuing"
                )

            if config_atmosphere.rte_configure_and_exit:
                Logger.warning(
                    "rte_configure_and_exit is enabled, the LUT file will not be created"
                )
            else:
                # TODO Check keys
                Logger.info(f"Initializing LUT file")
                self.lut = luts.Create(
                    file=self.lut_path,
                    wl=self.wl,
                    grid=self.lut_grid,
                    attrs={"RT_mode": self.rt_mode},
                    onedim={"fwhm": self.fwhm},
                    compression=full_config.implementation.lut_compression,
                    complevel=full_config.implementation.lut_complevel,
                )

            # Create and populate a LUT file
            self.runSimulations()

        # big TODO Abstract into LUT and engines TODO
        # Write the NetCDF information to the log file so devs have that info during debugging
        # Have to create a fileobj to capture the text because it doesn't return (prints straight to stdout by default)
        info = io.StringIO()
        self.lut.info(info)
        Logger.debug(f"LUT information:\n{info.getvalue()}")

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
            # revise call to reference lut
            # TODO Need to correct reference
            self.build_interpolators()

            geometry_keys = set(config_atmosphere.statevector_names or self.lut_names)

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

        # Make sure the length of the config statevectores match the engine's assumed statevectors
        if (expected := len(self.statevec_names)) != (got := len(self.indices.x_RT)):
            error = f"Mismatch between the number of elements for the config and LUT grid: {expected=}, {got=}"
            Logger.error(error)
            raise AttributeError(error)

    def xa(self):
        """Pull the priors from each of the individual RTs."""
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs."""
        return self.Sa_cached

    def Sb(self):
        """Uncertainty due to unmodeled variables."""
        return np.diagflat(np.power(self.bval, 2))

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated modeled atmospheric path radiance.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated modeled atmospheric path radiance
        """

        r = self.engine.get(x_RT, geom)
        if self.engine.rt_mode == "rdn":
            L_atm = r["rhoatm"]
        else:
            rho_atm = r["rhoatm"]
            L_atm = units.transm_to_rdn(rho_atm, geom.coszen, self.solar_irr)
        return L_atm

    def get_upward_transm(self, r: dict, geom: Geometry, max_transm: float = 1.05):
        """
        Get total upward transmittance w/physical check enforced (max_transm) and hand-off between 1c and 4c model.

        This is called for all surfaces to handle thermal downwelling/upwelling component.
        While rt can be either rdn or transm modes, this must be in units of transmittance.

        """
        transm_up_dir = r["transm_up_dir"]
        transm_up_dif = r["transm_up_dif"]

        # NOTE for 1c case transm-up is not a key, and therefore Ls and transup is zero.
        if not isinstance(transm_up_dir, np.ndarray) or len(transm_up_dir) == 1:
            return np.zeros_like(self.solar_irr, dtype=np.float32)
        else:
            transup = transm_up_dir + transm_up_dif

            if np.max(transup) > max_transm:
                raise ValueError(
                    (
                        f"Upward transmittance (max:{np.max(transup)}) is greater than {max_transm}. "
                        f"Verify 'transm_up_dir' and 'transm_up_dif' keys are in units of transmittance."
                    )
                )
            return transup

    def summarize(self, x_RT, geom):
        return self.engine.summarize(x_RT, geom)

    ################ From RTE ####################

    def __getitem__(self, key):
        """
        Enables key indexing for easier access to the numpy object store in
        self.lut[key]
        """
        return self.lut[key].load().data

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
