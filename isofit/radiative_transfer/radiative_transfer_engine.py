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
        "transm_up_dif",
        "transm_up_dir",
        "transm_down_dif",
        "transm_down_dir",
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
        lut_path: str,
        lut_grid: dict = None,
        wavelength_file: str = None,
        interpolator_style: str = "mlg",
        overwrite_interpolator: bool = False,
        cache_size: int = 16,
    ):
        # Verify either the LUT file exists or a LUT grid is provided
        self.lut_path = lut_path
        exists = os.path.exists(lut_path)
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
        self.sim_path = engine_config.sim_path  # New way

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
                data_input=self.lut[key].load().data.T,
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
    # REVIEW: This function seems to be inspired by sRTMnet.get() but is super broken
    def _get(self, x_RT: np.array, geom: Geometry):
        """Retrieve point from LUT interpolator
        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated RTE result
        """
        point = np.zeros((self.n_point,))
        point[self.x_RT_lut_indices] = x_RT
        point[self.geometry_lut_indices] = np.array(
            [getattr(geom, key) for key in self.lut_names]
        )
        return self._lookup_lut(point)

    def get(self, point, *_):
        """
        Temporarily circumventing the geom obj and passing x_RT as the point to
        interpolate
        point == x_RT
        Combines the get and _lookup_lut into one function (why have two?)
        """
        data = {key: self.luts[key](point) for key in self.luts}
        # TODO: These are clearly wrong. This temporarily alleviates issues with functions on the physics side expecting certain keys to exist
        # I just creatively chose the values. Please either fix these or the functions themselves.
        # Known issues in:
        # - isofit.inversion.invert_simple
        data["transdown"] = data["transm_down_dir"] + data["transm_down_dif"]
        data["transup"] = data["transm_up_dir"] + data["transm_up_dif"]
        data["transm"] = data["transdown"] * data["transup"]

        return data

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
