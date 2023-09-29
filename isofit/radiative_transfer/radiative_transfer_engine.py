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
import subprocess
import time
from collections import OrderedDict

import numpy as np
import ray

from isofit import get_isofit_path
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.configs.sections.statevector_config import StateVectorElementConfig
from isofit.core import common
from isofit.core.geometry import Geometry
from isofit.utils.luts import readHDF5


class RadiativeTransferEngine:
    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        interpolator_style: str,
        instrument_wavelength_file: str = None,
        overwrite_interpolator: bool = False,
        cache_size: int = 16,
        lut_grid: dict = None,
    ):
        self.engine_config = engine_config
        self.interpolator_style = interpolator_style
        self.overwrite_interpolator = overwrite_interpolator
        self.cache_size = cache_size

        if os.path.isfile(self.prebuilt_lut_file) is False and lut_grid is None:
            raise AttributeError(
                "Must provide either a prebuilt LUT file or a LUT grid"
            )

        self.emission_mode = engine_config.emission_mode
        self.engine_base_dir = engine_config.engine_base_dir
        self.angular_lut_keys_degrees = [
            "observer_azimuth",
            "observer_zenith",
            "solar_azimuth",
            "solar_zenith",
            "relative_azimuth",
        ]
        self.angular_lut_keys_radians = []

        self.lut_dir = engine_config.lut_path

        self.geometry_lut_indices = None
        self.geometry_lut_names = None
        self.x_RT_lut_indices = None

        self.prebuilt_lut_file = engine_config.engine_lut_file
        # Read prebuilt LUT from HDF5 if existing
        if os.path.isfile(self.prebuilt_lut_file):
            self.lut = readHDF5(file=self.prebuilt_lut_file)
        else:
            self.lut = None

        # Enable special modes - argument: get from HDF5
        self.multipart_transmittance = engine_config.multipart_transmittance
        self.topography_model = engine_config.topography_model

        self.earth_sun_distance_path = os.path.join(
            get_isofit_path(), "data", "earth_sun_distance.txt"
        )
        self.earth_sun_distance_reference = np.loadtxt(self.earth_sun_distance_path)

        # This is only for checking the validity of the read in sim names
        self.possible_lut_output_names = [
            "rhoatm",
            "transm_down_dir",
            "transm_down_dif",
            "transm_up_dir",
            "transm_up_dif",
            "sphalb",
            "thermal_upwelling",
            "thermal_downwelling",
        ]

        if self.multipart_transmittance:
            self.test_rfls = [0, 0.1, 0.5]

        # Get instrument wavelengths and FWHM
        if engine_config.wavelength_file is not None:
            wavelength_file = engine_config.wavelength_file
        else:
            wavelength_file = instrument_wavelength_file

        self.wl, self.fwhm = common.load_wavelen(wavelength_file)

        if engine_config.wavelength_range is not None:
            valid_wl = np.logical_and(
                self.wl >= engine_config.wavelength_range[0],
                self.wl <= engine_config.wavelength_range[1],
            )
            self.wl = self.wl[valid_wl]
            self.fwhm = self.fwhm[valid_wl]

        self.n_chan = len(self.wl)

        # Prepare a cache for self._lookup_lut(), setting cache_size to 0 will disable
        self.cache = {}

        # If prebuilt LUT is available, read a few important LUT parameters and functions
        if self.lut:
            # TODO: ensure consistency with group keys in LUT file
            self.solar_irr = self.lut["MISCELLANEOUS"]["sols"]
            # We use a sorted dictionary here so that filenames for lookup
            # table (LUT) grid points are always constructed the same way (with
            # consistent dimension ordering). Every state vector element has
            # a lookup table dimension, but some lookup table dimensions
            # (like geometry parameters) may not be in the state vector.
            # TODO: ensure consistency with group keys in LUT file
            full_lut_grid = self.lut["sample_space"]
        else:
            self.solar_irr = None
            full_lut_grid = lut_grid

        # Set up LUT grid
        self.lut_grid_config = OrderedDict()
        if engine_config.lut_names is not None:
            lut_names = engine_config.lut_names
        else:
            lut_names = full_lut_grid.keys()

        for key, value in full_lut_grid.items():
            if key in lut_names:
                self.lut_grid_config[key] = value
        del lut_names

        self.n_point = len(self.lut_grid_config)

        self.luts = {}

        self.lut_dims = []
        self.lut_grids = []
        self.lut_names = []
        self.lut_interp_types = []

        for key, grid_values in self.lut_grid_config.items():
            # TODO: make sure 1-d grids can be handled
            if grid_values != sorted(grid_values):
                raise ValueError("Lookup table grid needs ascending order")

            # Store the values
            self.lut_grids.append(grid_values)
            self.lut_dims.append(len(grid_values))
            self.lut_names.append(key)

            # Store in an indication of the type of value each key is
            # (normal - n, degree - d, radian - r)
            if key in self.angular_lut_keys_radians:
                self.lut_interp_types.append("r")
            elif key in self.angular_lut_keys_degrees:
                self.lut_interp_types.append("d")
            else:
                self.lut_interp_types.append("n")

        # Cast as array for faster reference later
        self.lut_interp_types = np.array(self.lut_interp_types)

        # Initialize a 1 value hash dict
        self.last_point_looked_up = np.zeros(self.n_point)
        self.last_point_lookup_values = np.zeros(self.n_point)

        self.interpolator_disk_paths = [
            engine_config.interpolator_base_path + "_" + rtq + ".pkl"
            for rtq in self.lut_names
        ]

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
        point[self.geometry_lut_indices] = np.array(
            [getattr(geom, key) for key in self.geometry_lut_names]
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
        points = common.combos(self.lut_grids)

        # Make the LUT calls (in parallel if specified)
        results = ray.get(
            [
                stream_simulation.remote(
                    point,
                    self.make_simulation_call,
                    self.read_simulation_results,
                    self.prebuilt_lut_file,
                )
                for point in points
            ]
        )


@ray.remote
def stream_simulation(
    point: np.array,
    simmulation_call: function,
    reader: function,
    save_file: str,
    max_buffer_time: float = 0.5,
):
    """Run a simulation for a single point and stream the results to a saved lut file.

    Args:
        point (np.array): conditions to alter in simulation
        simmulation_call (function): function to run the simulation
        reader (function): function to read the results of the simulation
        save_file (str): netcdf file to save results to
        max_buffer_time (float, optional): _description_. Defaults to 0.5.
    """

    logging.debug("Running simulation for point: %s" % point)

    # Add a very slight timing offset to prevent all subprocesses
    # starting simultaneously
    time.sleep(float(np.random.random(1)) * max_buffer_time)

    simmulation_call(point)
    res = reader(point)
    # TODO: access this from new netcdf lut helper
    append_to_lut(point, res, save_file)
