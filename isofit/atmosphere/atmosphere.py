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

import logging
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isofit.configs import Config
from isofit.core import common, units
from isofit.core.common import json_load_ascii, svd_inv_sqrt
from isofit.core.geometry import Geometry
from isofit.luts.reader import Reader

Logger = logging.getLogger(__name__)


class Keys:
    # Constants, not along any dimension
    consts = {
        "coszen": np.nan,
        "solzen": np.nan,
    }

    # Along the wavelength dimension only
    onedim = {
        "fwhm": np.nan,
        "solar_irr": np.nan,
    }

    # Keys along all dimensions, ie. wl and point
    alldim = {
        "rhoatm": np.nan,
        "sphalb": np.nan,
        "transm_down_dir": 0,
        "transm_down_dif": 0,
        "transm_up_dir": 0,
        "transm_up_dif": 0,
        "thermal_upwelling": np.nan,
        "thermal_downwelling": np.nan,
        # add keys for radiances along all optical paths
        "dir-dir": 0,
        "dif-dir": 0,
        "dir-dif": 0,
        "dif-dif": 0,
    }


class BaseAtmosphere(Reader):
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    lut_keys = Keys

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

    coupling_terms = ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]

    # Defaults for lut compression
    lut_compression = "zlib"
    lut_complevel = None

    @property
    def coszen(self) -> np.ndarray:
        return float(self.lut["coszen"])

    @property
    def solar_irr(self) -> np.ndarray:
        return np.array(self.lut["solar_irr"])

    def __init__(
        self,
        full_config: Config,
        lut_path: str = "",  # lut_path override
        wl: np.array = [],  # Wavelength override
        fwhm: np.array = [],  # fwhm override
        n_cores: int = None,  # n_core override
        build_interpolators: bool = True,
        lut_postprocess: bool = True,
        **kwargs,
    ):
        self.full_config = full_config
        self.config = full_config.forward_model.atmosphere

        self.n_cores = (
            n_cores or full_config.implementation.n_cores or multiprocessing.cpu_count()
        )

        # Initially pull lut information from config
        self.lut_path = lut_path or self.config.lut_path
        self.lut_grid = self.config.lut_grid
        self.lut_names = list(self.lut_grid)
        self.statevec_names = self.config.statevector.get_element_names()

        possible_h2o_names = ["H2OSTR", "h2o"]
        self.h2o_i = [
            i for i, v in enumerate(self.statevec_names) if v in possible_h2o_names
        ]

        # Configure and exit flag
        self.configure_and_exit = self.config.configure_and_exit

        self.engine_base_dir = self.config.engine_base_dir
        self.sim_path = self.config.sim_path
        self.multipart_transmittance = self.config.multipart_transmittance

        if not self.lut_path:
            raise AttributeError(
                "Must provide a lut_path either by parameter or config"
            )
        # self.lut_exists = self.lut_path is not None and os.path.exists(self.lut_path)
        # if self.lut_exists:
        #     Logger.info("Prebuilt LUT provided")
        #
        # elif not self.lut_exists and self.lut_grid is None:
        #     raise AttributeError(
        #         "Must provide either a prebuilt LUT file or a LUT grid"
        #     )

        # TODO: overwrite_interpolator not hooked up. Check if we even want this override
        self.interpolator_style = (
            self.config.interpolator_style
            or full_config.forward_model.instrument.get("interpolator_style")
        )

        self.multipart_transmittance = (
            full_config.forward_model.atmosphere.multipart_transmittance
        )

        self.rt_mode = self.config.rt_mode or "transm"

        # Use explicitely passed first
        if not any(wl) or not any(fwhm):
            self.wavelength_file = (
                self.config.wavelength_file
                or full_config.forward_model.instrument.wavelength_file
            )
            if os.path.exists(self.wavelength_file):
                Logger.info(f"Loading from wavelength_file: {self.wavelength_file}")
                wl, fwhm = common.load_wavelen(self.wavelength_file)

        if not any(wl) or not any(fwhm):
            e = "No wavelength information found, please provide either as file or input argument"
            Logger.error(e)
            raise AttributeError(e)

        # Set for downstream engines
        self.wl = wl
        self.fwhm = fwhm

        # Priors
        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*self.config.statevector.get_elements()):
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

        # Day of year of the scene this run targets, read from the template IDAY.
        # None when unavailable (e.g. no template, or an engine using a non-MODTRAN
        # template format). Used both to tag a LUT at build time and to reconcile a
        # prebuilt LUT's build date against the current scene date.
        self.dayofyear = None
        if self.config.template_file:
            modtran_input = json_load_ascii(self.config.template_file)["MODTRAN"][0][
                "MODTRANINPUT"
            ]
            self.atmosphere_type = modtran_input["ATMOSPHERE"].get(
                "M1", "ATM_MIDLAT_SUMMER"
            )
            iday = modtran_input.get("GEOMETRY", {}).get("IDAY")
            self.dayofyear = int(iday) if iday is not None else None
        else:
            self.atmosphere_type = "ATM_MIDLAT_SUMMER"

        self.h2o_bounds_polynomial = modtran_water_upperbound_polynomials()[
            self.atmosphere_type
        ]

        # Uncertainty
        self.bvec = self.config.unknowns.get_element_names()
        self.bval = np.array([x for x in self.config.unknowns.get_elements()[0]])

        # Initialize the LUT
        self.consts = {}
        self.onedim = {"fwhm": fwhm}
        self.alldim = {}
        super().__init__(
            build_interpolators=build_interpolators,
            lut_subset=self.config.lut_names,
            **kwargs,
        )

        self.set_esd_correction()

        self.get_indices()

    def set_esd_correction(self):
        """Set esd_correction factor onto self. Presumes that there is only one
        ESD per scene being processed.
        """
        self.esd_correction = 1.0

        lut_doy = self.lut.attrs.get("dayofyear")

        # Double check to make sure we have everything we need
        if lut_doy is None:
            Logger.warning(
                "LUT has no 'dayofyear' attribute; skipping Earth Sun Distance "
                "correction. Rebuild the LUT to enable it for off-date scenes."
            )
            return

        if self.dayofyear is None:
            Logger.warning(
                "Current day of year unavailable (no template IDAY); skipping "
                "Earth-Sun-distance correction."
            )
            return

        lut_doy = int(lut_doy)
        if lut_doy == self.dayofyear:
            return

        esd = common.load_esd()
        self.esd_correction = (esd[lut_doy - 1, 1] / esd[self.dayofyear - 1, 1]) ** 2
        Logger.info(
            f"Applying Earth Sun Distance correction to LUT radiances: "
            f"LUT day of year={lut_doy}, current day of year={self.dayofyear}, "
            f"factor={self.esd_correction:.6f}"
        )

    def lut_postprocess(self):
        """
        Checks for additional postprocessing that may need to be applied to the loaded
        LUT
        """
        self.lut = self.couple(self.lut)

        # Special treatment for PACE OCI
        srf_file = None
        irr_file = Path(self.config.irradiance_file)
        if irr_file.stem == "tsis_f0_0p1":
            srf_file = irr_file.parent / "pace_oci_rsr.nc"

        if (
            not len(self.wl) == len(self.lut.wl)
            or srf_file is not None
            or not all(self.wl == self.lut.wl)
        ):
            self.resample_wl(wl=self.wl, fwhm=self.fwhm, srf_file=srf_file)

        # Limit the wavelength per the config, does not affect data on disk
        if self.config.wavelength_range is not None:
            self.subset_wl(rng=self.config.wavelength_range)

    def get_indices(self):
        """
        Retrieves the point indices for keys that source from different locations, such
        as statevector or geom
        """
        self.indices = SimpleNamespace(geom={}, x_RT=[])

        # Check if an input statevector key is a typo for a geom key
        geometry_keys = set(self.config.statevector_names or self.lut_names)
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
        if "observer_zenith" in self.lut_grid:
            if any(np.array(self.lut_grid["observer_zenith"]) > 90.0):
                self.indices.convert_observer_zenith = {
                    key: index for key, index in self.indices.geom()
                }["observer_zenith"]

        # If it wasn't a geom key, it's x_RT
        self.indices.x_RT = list(
            set(range(len(self.lut_names))) - set(self.indices.geom)
        )

        # Make sure the length of the config statevectores match the engine's assumed statevectors
        if (expected := len(self.statevec_names)) != (got := len(self.indices.x_RT)):
            error = f"Mismatch between the number of elements for the config and LUT grid: {expected=}, {got=}"
            Logger.error(error)
            raise AttributeError(error)

    def update_heuristic_prior_means(self, x_atmosphere, geom):
        xa = self.prior_mean.copy()
        xa[self.h2o_i] = x_atmosphere[self.h2o_i]

        return xa

    def xa(self, x_atmosphere, geom):
        """
        Use the image-wide prior mean
        """
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs."""
        return self.Sa_cached

    def Sb(self):
        """Uncertainty due to unmodeled variables."""
        return np.diagflat(np.power(self.bval, 2))

    def get(self, x_RT: np.array, geom: Geometry) -> dict:
        """Interpolate the LUT at the given RT statevector and geometry.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            dict of interpolated LUT quantities
        """
        point = np.zeros(len(self.lut_names))

        point[self.indices.x_RT] = x_RT
        for i, key in self.indices.geom.items():
            point[i] = getattr(geom, key)

        # convert observer zenith to MODTRAN convention if needed
        if self.indices.convert_observer_zenith:
            point[self.indices.convert_observer_zenith] = (
                180.0 - point[self.indices.convert_observer_zenith]
            )

        return self.interpolate(point)

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated modeled atmospheric path radiance.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated modeled atmospheric path radiance
        """
        r = self.get(x_RT, geom)
        if self.rt_mode == "rdn":
            L_atm = r["rhoatm"]
        else:
            rho_atm = r["rhoatm"]
            L_atm = units.transm_to_rdn(rho_atm, geom.coszen, self.solar_irr)
        return L_atm * self.esd_correction

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

    @staticmethod
    def couple(ds, inplace=True):
        """
        Calculates coupled terms on the input Dataset

        Parameters
        ----------
        ds: xr.Dataset
            Dataset to process on
        inplace: bool, default=True
            Insert the coupled terms in-place to the original Dataset. If False, copy the
            Dataset first

        Returns
        -------
        ds: xr.Dataset
            Dataset with coupled terms
        """
        terms = {
            "dir-dir": ("transm_down_dir", "transm_up_dir"),
            "dif-dir": ("transm_down_dif", "transm_up_dir"),
            "dir-dif": ("transm_down_dir", "transm_up_dif"),
            "dif-dif": ("transm_down_dif", "transm_up_dif"),
        }

        # Detect if coupling needs to occur first
        data = ds.get(list(terms))
        calc = False
        if data is None:
            # Not all keys exist
            calc = "missing"
        elif not bool(data.any().to_array().all()):
            # If any key is empty
            calc = "empty"

        if calc:
            Logger.debug(f"A coupled term is {calc}, calculating")
            if not inplace:
                ds = ds.copy()

            for term, (key1, key2) in terms.items():
                try:
                    ds[term] = ds[key1] * ds[key2]
                except KeyError:
                    ds[term] = 0

        return ds

    def summarize(self, x_RT, *_):
        """
        Pretty prints lut_name=value, ...
        """
        pairs = zip(self.lut_names, x_RT)
        return " ".join([f"{name}={val:5.3f}" for name, val in pairs])


def modtran_water_upperbound_polynomials() -> dict:
    """Polynomials as a function of ground altitude (km) to estimate upperbound of water column vapor (g/cm2). Generated from MODTRAN.

    Returns:
        dict: 3rd degree polynomials to estimate upperbound of water column vapor
    """

    # Capping polynomial to not allow negative, or increasing trend, at very high ground altitudes (>6km).
    min_value = 0.25

    polynomials = {
        "ATM_TROPICAL": lambda x: np.maximum(
            6.74256 + (-2.37052 * x) + (0.313829 * x**2) + (-0.0159003 * x**3),
            min_value,
        ),
        "ATM_MIDLAT_SUMMER": lambda x: np.maximum(
            5.350046 + (-1.839548 * x) + (2.296582e-01 * x**2) + (-1.020594e-02 * x**3),
            min_value,
        ),
        "ATM_MIDLAT_WINTER": lambda x: np.maximum(
            1.371226 + (-0.442087 * x) + (4.485325e-02 * x**2) + (-1.130163e-03 * x**3),
            min_value,
        ),
        "ATM_SUBARC_SUMMER": lambda x: np.maximum(
            3.121272 + (-1.171145 * x) + (1.704094e-01 * x**2) + (-9.701062e-03 * x**3),
            min_value,
        ),
        "ATM_SUBARC_WINTER": lambda x: np.maximum(
            0.630406 + (-0.176336 * x) + (8.286409e-03 * x**2) + (8.138824e-04 * x**3),
            min_value,
        ),
        "ATM_US_STANDARD_1976": lambda x: np.maximum(
            2.869655 + (-1.227473 * x) + (2.039059e-01 * x**2) + (-1.274801e-02 * x**3),
            min_value,
        ),
    }

    return polynomials


def modtran_aot_lowerbound_polynomials() -> dict:
    """Polynomials as a function of ground altitude (km) to estimate lowerbound of AOT at 550nm. Generated from MODTRAN.

    Returns:
        dict: 3rd degree polynomials to estimate lowerbound of AOT
    """

    polynomials = {
        "ATM_TROPICAL": lambda x: 0.042090
        + (-0.003120 * x)
        + (4.462979e-18 * x**2)
        + (-4.260469e-19 * x**3),
        "ATM_MIDLAT_SUMMER": lambda x: 0.042090
        + (-0.003120 * x)
        + (4.462979e-18 * x**2)
        + (-4.260469e-19 * x**3),
        "ATM_MIDLAT_WINTER": lambda x: 0.024748
        + (-0.001654 * x)
        + (-5.083805e-07 * x**2)
        + (7.252007e-08 * x**3),
        "ATM_SUBARC_SUMMER": lambda x: 0.042090
        + (-0.003120 * x)
        + (4.462979e-18 * x**2)
        + (-4.260469e-19 * x**3),
        "ATM_SUBARC_WINTER": lambda x: 0.024748
        + (-0.001654 * x)
        + (-5.083805e-07 * x**2)
        + (7.252007e-08 * x**3),
        "ATM_US_STANDARD_1976": lambda x: 0.042090
        + (-0.003120 * x)
        + (4.462979e-18 * x**2)
        + (-4.260469e-19 * x**3),
    }

    return polynomials
