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

import logging
import os
import time
from typing import Callable

import numpy as np
import ray

import isofit
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core import common
from isofit.core.geometry import Geometry
from isofit.luts import netcdf as luts

Logger = logging.getLogger(__file__)


class RadiativeTransferEngine:
    ## LUT Keys --
    # Constants, not along any dimension
    consts = ["coszen", "solzen"]

    # Along the wavelength dimension only
    onedim = ["solar_irr"]

    # Keys along all dimensions, ie. wl and point
    alldim = [
        "rhoatm",
        "sphalb",
        "transm_down_dir",
        "transm_down_dif",
        "transm_up_dir",
        "transm_up_dif",
        "thermal_upwelling",
        "thermal_downwelling",
    ]
    ## End LUT keys --

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
    # ...
    angular_lut_keys_degrees = [
        "observer_azimuth",
        "observer_zenith",
        "solar_azimuth",
        "solar_zenith",
        "relative_azimuth",
    ]
    angular_lut_keys_radians = []

    # Informs the VectorInterpolator the units for a given key
    angular_lut_keys = {
        # Degrees
        "observer_azimuth": "d",
        "observer_zenith": "d",
        "solar_azimuth": "d",
        "solar_zenith": "d",
        "relative_azimuth": "d",
        # Radians
        #   "key": "r",
        # All other keys default to "n" = Not angular
    }

    earth_sun_distance_path = os.path.join(
        isofit.root, "data", "earth_sun_distance.txt"
    )

    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        lut_path: str = None,
        lut_grid: dict = None,
        wavelength_file: str = None,
        interpolator_style: str = "mlg",
        overwrite_interpolator: bool = False,
        cache_size: int = 16,
    ):
        # Verify either the LUT file exists or a LUT grid is provided
        self.lut_path = lut_path = str(lut_path)
        exists = os.path.isfile(lut_path)
        if not exists and lut_grid is None:
            raise AttributeError(
                "Must provide either a prebuilt LUT file or a LUT grid"
            )

        # Save parameters to instance
        self.engine_config = engine_config

        self.interpolator_style = interpolator_style
        self.overwrite_interpolator = overwrite_interpolator
        self.cache_size = cache_size

        self.treat_as_emissive = engine_config.treat_as_emissive
        self.engine_base_dir = engine_config.engine_base_dir

        # ToDo: discuss how to handle sim_path and lut_path, perhaps even using temp dirs for simulation?
        self.lut_dir = engine_config.sim_path  # Backwards compatibility
        self.sim_path = engine_config.sim_path  # REVIEW: New way

        # Enable special modes - argument: get from HDF5
        self.multipart_transmittance = engine_config.multipart_transmittance
        self.topography_model = engine_config.topography_model

        # ToDo: move setting of multipart rfl values to config
        if self.multipart_transmittance:
            self.test_rfls = [0.1, 0.5]

        # Extract from LUT file if available, otherwise initialize it
        if exists:
            Logger.info(f"Prebuilt LUT provided")
            Logger.debug(f"Reading from store: {lut_path}")
            self.lut = luts.load(lut_path, lut_grid)
            self.wl = self.lut.wl.data
            self.fwhm = self.lut.fwhm.data

            self.points, self.lut_names = luts.extractPoints(self.lut)
            self.lut_grid = lut_grid or luts.extractGrid(self.lut)
        else:
            Logger.info(f"No LUT store found, beginning initialization and simulations")
            Logger.debug(f"Writing store to: {lut_path}")
            Logger.debug(f"Using wavelength file: {wavelength_file}")

            self.lut_names = engine_config.lut_names or lut_grid.keys()
            self.lut_grid = {
                key: lut_grid[key] for key in self.lut_names if key in lut_grid
            }

            self.points = common.combos(
                self.lut_grid.values()
            )  # 2d numpy array.  rows = points, columns = lut_names

            self.wl, self.fwhm = common.load_wavelen(wavelength_file)
            self.lut = luts.initialize(
                file=lut_path,
                wl=self.wl,
                lut_grid=lut_grid,
                consts=self.consts,
                onedim=self.onedim + [("fwhm", self.fwhm)],
                alldim=self.alldim,
            )

            # Populate the newly created LUT file
            self.run_simulations()

        # Limit the wavelength per the config
        if engine_config.wavelength_range is not None:
            valid_wl = np.logical_and(
                self.wl >= engine_config.wavelength_range[0],
                self.wl <= engine_config.wavelength_range[1],
            )
            self.wl = self.wl[valid_wl]
            self.fwhm = self.fwhm[valid_wl]

        self.n_chan = len(self.wl)

        # This is a bad variable name - change (it's the number of input dimensions of the lut (p) not the number of samples)
        self.n_point = len(self.lut_names)

        # Attach interpolators
        self.build_interpolators()

        # TODO: These are definitely wrong, what should they initialize to?
        self.solar_irr = [1]
        self.coszen = [1]  # TODO: get from call

        # Hidden assumption: geometry keys come first, then come RTE keys
        self.geometry_lut_indices = np.array(
            [
                self.geometry_input_names.index(key)
                for key in self.lut_names
                if key in self.geometry_input_names
            ]
        )
        self.x_RT_lut_indices = np.array(
            [x for x in range(self.n_point) if x not in self.geometry_lut_indices]
        )

        # Prepare a cache for self._lookup_lut(), setting cache_size to 0 will disable
        self.cache = {}
        # Initialize a 1 value hash dict
        self.last_point_looked_up = np.zeros(self.n_point)
        self.last_point_lookup_values = np.zeros(self.n_point)

        self.earth_sun_distance_reference = np.loadtxt(self.earth_sun_distance_path)

    @property
    def lut_interp_types(self):
        return np.array([self.angular_lut_keys.get(key, "n") for key in self.lut_names])

    def build_interpolators(self):
        """
        Builds the interpolators using the LUT store

        TODO: optional load from/write to disk
        """
        self.luts = {}

        # Create the unique
        for key in self.alldim:
            self.luts[key] = common.VectorInterpolator(
                grid_input=self.lut_grid.values(),
                data_input=self.lut[key].load().data,
                lut_interp_types=self.lut_interp_types,
                version=self.interpolator_style,
            )

    def make_simulation_call(self, point: np.array, template_only: bool = False):
        """Write template(s) and run simulation.

        Args:
            point (np.array): conditions to alter in simulation
            template_only (bool): only write template file and then stop
        """
        raise AssertionError("Must populate simulation call")

    def read_simulation_results(self, point: np.array):
        """Read simulation results to standard form.

        Args:
            point (np.array): conditions to alter in simulation
        """
        raise AssertionError("Must populate simulation call")

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

    def get_coszen(self, point: np.array) -> float:
        """Get solar zenith cosine from point

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            float: cosine of solar zenith angle at the given point
        """
        if "solar_zenith" in self.lut_names:
            return [np.cos(np.deg2rad(point[self.lut_names.index("solar_zenith")]))]
        else:
            return [0.2]
            # TODO: raise AttributeError("Havent implemented this yet....should have a default read from template")

    # TODO: change this name
    def get(self, x_RT: np.array, geom: Geometry):
        """Retrieve point from LUT interpolator
        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated RTE result
        """
        point = np.zeros((self.n_point,))
        point[self.x_RT_lut_indices] = x_RT
        geom_keys = [key for key in self.lut_names if key in self.geometry_input_names]
        if len(geom_keys) > 0:
            point[self.geometry_lut_indices] = np.array(
                [getattr(geom, key) for key in geom_keys]
            )
        return self._lookup_lut(point)

    def _lookup_lut(self, point):
        if np.all(np.equal(point, self.last_point_looked_up)):
            return self.last_point_lookup_values
        else:
            ret = {}
            for key, lut in self.luts.items():
                ret[key] = np.array(lut(point)).ravel()

            self.last_point_looked_up = point
            self.last_point_lookup_values = ret
            return ret

    def run_simulations(self) -> None:
        """
        Run all simulations for the LUT grid.

        """
        # "points" contains all combinations of grid points
        # We will have one filename prefix per point

        Logger.info(f"Executing {len(self.points)} simulations")

        # Make the LUT calls (in parallel if specified)
        results = ray.get(
            [
                stream_simulation.remote(
                    point,
                    self.lut_names,
                    self.make_simulation_call,
                    self.read_simulation_results,
                    self.lut_path,
                )
                for point in self.points
            ]
        )

        # Reload the LUT now that it's populated
        self.lut = luts.load(self.lut_path, self.lut_names)

    def summarize(self, x_RT, *_):
        """ """
        pairs = zip(self.lut_grid.keys(), x_RT)
        return " ".join([f"{name}={val:5.3f}" for name, val in pairs])

    # ToDo: we need to think about the best place for the two albedo method (here, radiative_transfer.py, utils, etc.)
    @staticmethod
    def two_albedo_method(
            case0: dict,
            case1: dict,
            case2: dict,
            coszen: float,
            rfl1: float = 0.1,
            rfl2: float = 0.5,
    ) -> dict:
        """
        Calculates split transmittance values from a multipart file using the
        two-albedo method. See notes for further detail.

        Parameters
        ----------
        case0: dict
            MODTRAN output for a non-reflective surface (case 0 of the channel file)
        case1: dict
            MODTRAN output for surface reflectance = rfl1 (case 1 of the channel file)
        case2: dict
            MODTRAN output for surface reflectance = rfl2 (case 2 of the channel file)
        coszen: float
            ...
        rfl1: float, defaults=0.1
            Surface reflectance  for case 1 of the MODTRAN output
        rfl2: float, defaults=0.5
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
        widths = case0["width"]
        # Direct upward transmittance
        t_up_dir = case0["transm_up_dir"]

        # REVIEW: two_albedo_method-v1 used a single solar_irr value, but now we have an array of values
        # The last value in the new array is the same as the old v1, so for backwards compatibility setting that here
        # Top-of-atmosphere solar irradiance as a function of sun zenith angle
        E0 = case0["solar_irr"][-1] * coszen / np.pi

        # Direct ground reflected radiance at sensor for case 1 (sun->surface->sensor)
        # This includes direct down and direct up transmittance
        Ltoa_dir1 = case1["drct_rflt"]
        # Total ground reflected radiance at sensor for case 1 (sun->surface->sensor)
        # This includes direct + diffuse down, but only direct up transmittance
        Ltoa1 = case1["grnd_rflt"]

        # Transforming back to at-surface irradiance
        # Since Ltoa_dir1 and Ltoa1 only contain direct upward transmittance,
        # we can safely divide by t_up_dir without needing t_up_dif
        # Direct at-surface irradiance for case 1 (only direct down transmittance)
        E_down_dir1 = Ltoa_dir1 * np.pi / rfl1 / t_up_dir
        # Total at-surface irradiance for case 1 (direct + diffuse down transmittance)
        E_down1 = Ltoa1 * np.pi / rfl1 / t_up_dir

        # Total at-surface irradiance for case 2 (direct + diffuse down transmittance)
        E_down2 = case2["grnd_rflt"] * np.pi / rfl2 / t_up_dir

        # Atmospheric path radiance for case 1
        Lp1 = case1["path_rdn"]
        # Atmospheric path radiance for case 2
        Lp2 = case2["path_rdn"]

        # Total reflected radiance at surface (before upward atmospheric transmission) for case 1
        Lsurf1 = rfl1 * E_down1
        # Total reflected radiance at surface (before upward atmospheric transmission) for case 2
        Lsurf2 = rfl2 * E_down2
        # Atmospheric path radiance for non-reflective surface (case 0)
        Lp0 = ((Lsurf2 * Lp1) - (Lsurf1 * Lp2)) / (Lsurf2 - Lsurf1)

        # Diffuse upward transmittance
        t_up_dif = np.pi * (Lp1 - Lp0) / (rfl1 * E_down1)

        # Spherical albedo
        salb = (E_down1 - E_down2) / (Lsurf1 - Lsurf2)

        # Total at-surface irradiance for non-reflective surface (case 0)
        # Only add contribution from atmospheric spherical albedo
        E_down0 = E_down1 * (1 - rfl1 * salb)
        # Diffuse at-surface irradiance for non-reflective surface (case 0)
        E_down_dif0 = E_down0 - E_down_dir1

        # Direct downward transmittance
        t_down_dir = E_down_dir1 / widths / np.pi / E0
        # Diffuse downward transmittance
        t_down_dif = E_down_dif0 / widths / np.pi / E0

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
            "transm_up_dir": t_up_dir,
            "transm_up_dif": t_up_dif,
            "transm_down_dir": t_down_dir,
            "transm_down_dif": t_down_dif,
        }
        for key in pass_forward:
            data[key] = case0[key]

        return data

    # ToDo: to be removed in the future, for now still available to ensure compatibility with ModtranRTv1
    def two_albedo_method_old(
        self,
        transups: list,
        drct_rflts_1: list,
        grnd_rflts_1: list,
        grnd_rflts_2: list,
        lp_1: list,
        lp_2: list,
        coszen: float,
        widths: list,
    ):
        """This implementation follows Guanter et al. (2009) (DOI:10.1080/01431160802438555),
        with modifications by Nimrod Carmon. It is called the "2-albedo" method, referring to running
        MODTRAN with 2 different surface albedos. Alternatively, one could also run the 3-albedo method,
        which is similar to this one with the single difference where the "path_radiance_no_surface"
        variable is taken from a zero-surface-reflectance MODTRAN run instead of being calculated from
        2 MODTRAN outputs.

        There are a few argument as to why the 2- or 3-albedo methods are beneficial:
        (1) for each grid point on the lookup table you sample MODTRAN 2 or 3 times, i.e., you get
        2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
        to use a lower band model resolution for the MODTRAN run, which is much faster, while keeping
        high accuracy. (2) we use the decoupled transmittance products to expand
        the forward model and account for more physics, currently topography and glint.

        Args:
            transups:     upwelling direct transmittance
            drct_rflts_1: direct path ground reflected radiance for reflectance case 1
            grnd_rflts_1: total ground reflected radiance for reflectance case 1
            grnd_rflts_2: total ground reflected radiance for reflectance case 2
            lp_1:         path radiance (sum of single and multiple scattering) for reflectance case 1
            lp_2:         path radiance (sum of single and multiple scattering) for reflectance case 2
            coszen:       cosine of solar zenith angle
            widths:       fwhm of radiative transfer simulations

        Returns:
            t_down_dirs:  downwelling direct transmittance
            t_down_difs:  downwelling diffuse transmittance
            t_up_dirs:    upwelling direct transmittance
            t_up_difs:    upwelling diffuse transmittance
            sphalbs:      atmospheric spherical albedo
        """
        t_up_dirs = np.array(transups)
        direct_ground_reflected_1 = np.array(drct_rflts_1)
        total_ground_reflected_1 = np.array(grnd_rflts_1)
        total_ground_reflected_2 = np.array(grnd_rflts_2)
        path_radiance_1 = np.array(lp_1)
        path_radiance_2 = np.array(lp_2)
        # ToDo: get coszen from LUT and assign as attribute to self
        TOA_Irad = np.array(self.solar_irr) * coszen / np.pi
        rfl_1 = self.test_rfls[0]
        rfl_2 = self.test_rfls[1]

        direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
        global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs

        global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

        path_radiance_no_surface = (
            rfl_2 * path_radiance_1 * global_flux_2
            - rfl_1 * path_radiance_2 * global_flux_1
        ) / (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

        # Diffuse upwelling transmittance
        t_up_difs = (
            np.pi
            * (path_radiance_1 - path_radiance_no_surface)
            / (rfl_1 * global_flux_1)
        )

        # Spherical Albedo
        sphalbs = (global_flux_1 - global_flux_2) / (
            rfl_1 * global_flux_1 - rfl_2 * global_flux_2
        )
        direct_flux_radiance = direct_flux_1 / coszen

        global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
        diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen

        t_down_dirs = (
            direct_flux_radiance * coszen / np.array(widths) / np.pi
        ) / TOA_Irad
        t_down_difs = (diffuse_flux_no_surface / np.array(widths) / np.pi) / TOA_Irad

        return t_down_dirs, t_down_difs, t_up_dirs, t_up_difs, sphalbs


@ray.remote
def stream_simulation(
    point: np.array,
    lut_names: list,
    simmer: Callable,
    reader: Callable,
    output: str,
    max_buffer_time: float = 0.5,
):
    """Run a simulation for a single point and stream the results to a saved lut file.

    Args:
        point (np.array): conditions to alter in simulation
        lut_names (list): Dimension names aka lut_names
        simmer (function): function to run the simulation
        reader (function): function to read the results of the simulation
        output (str): LUT store to save results to
        max_buffer_time (float, optional): _description_. Defaults to 0.5.
    """
    Logger.debug(f"Simulating(point={point})")

    # Slight delay to prevent all subprocesses from starting simultaneously
    time.sleep(np.random.rand() * max_buffer_time)

    # Execute the simulation
    simmer(point)

    # Read the simulation results
    data = reader(point)

    # Save the results to our LUT format
    luts.updatePoint(output, lut_names, point, data)
