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

        self.lut_dir = engine_config.sim_path  # Backwards compatibility
        self.sim_path = engine_config.sim_path  # REVIEW: New way

        # Enable special modes - argument: get from HDF5
        self.multipart_transmittance = engine_config.multipart_transmittance
        self.topography_model = engine_config.topography_model

        if self.multipart_transmittance:
            # ToDo: check if we're running the 2- or 3-albedo method
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
            p0: dict,
            p1: dict,
            p2: dict,
            coszen: float,
            rfl1: float = 0.1,
            rfl2: float = 0.5,
    ) -> dict:
        """
        Calculates split transmittance values from a multipart file using the
        two-albedo method. See notes for further detail

        Parameters
        ----------
        p0: dict
            Part 0 of the channel file
        p1: dict
            Part 1 of the channel file
        p2: dict
            Part 2 of the channel file
        coszen: float
            ...
        rfl1: float, defaults=0.1
            Reflectance scaler 1
        rfl2: float, defaults=0.5
            Reflectance scaler 2

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
        # Extract relevant columns
        widths = p0["width"]
        t_up_dirs = p0["transup"]

        # REVIEW: two_albedo_method-v1 used a single solar_irr value, but now we have an array of values
        # The last value in the new array is the same as the old v1, so for backwards compatibility setting that here
        toa_irad = p0["solar_irr"][-1] * coszen / np.pi

        # Calculate some fluxes
        directRflt1 = p1["drct_rflt"]
        groundRflt1 = p1["grnd_rflt"]

        directFlux1 = directRflt1 * np.pi / rfl1 / t_up_dirs
        globalFlux1 = groundRflt1 * np.pi / rfl1 / t_up_dirs

        # diffuseFlux = globalFlux1 - directFlux1 # Unused

        globalFlux2 = p2["grnd_rflt"] * np.pi / rfl2 / t_up_dirs

        # Path radiances
        rdn1 = p1["path_rdn"]
        rdn2 = p2["path_rdn"]

        # Path Radiance No Surface = prns
        val1 = rfl1 * globalFlux1  # TODO: Needs a better name
        val2 = rfl2 * globalFlux2  # TODO: Needs a better name
        prns = ((val2 * rdn1) - (val1 * rdn2)) / (val2 - val1)

        # Diffuse upwelling transmittance
        t_up_difs = np.pi * (rdn1 - prns) / (rfl1 * globalFlux1)

        # Spherical Albedo
        sphalbs = (globalFlux1 - globalFlux2) / (val1 - val2)
        dFluxRN = directFlux1 / coszen  # Direct Flux Radiance

        globalFluxNS = globalFlux1 * (1 - rfl1 * sphalbs)  # Global Flux No Surface
        diffusFluxNS = globalFluxNS - dFluxRN * coszen  # Diffused Flux No Surface

        t_down_dirs = dFluxRN * coszen / widths / np.pi / toa_irad
        t_down_difs = diffusFluxNS / widths / np.pi / toa_irad

        transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

        # Return some keys from the first part plus the new calculated keys
        pass_forward = [
            "wl",
            "rhoatm",
            "solar_irr",
            "thermal_upwelling",
            "thermal_downwelling",
        ]
        data = {
            "sphalb": sphalbs,
            "transm_up_dir": t_up_dirs,
            "transm_up_dif": t_up_difs,
            "transm_down_dir": t_down_dirs,
            "transm_down_dif": t_down_difs,
        }
        for key in pass_forward:
            data[key] = p0[key]

        return data


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
