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

from collections import OrderedDict

import numpy as np

from isofit.configs import Config
from isofit.configs.sections.implementation_config import ImplementationConfig
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.configs.sections.statevector_config import StateVectorElementConfig
from isofit.core import common
from isofit.core.geometry import Geometry


class RadiativeTransferEngine:
    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        full_config: Config,
    ):
        self.engine_config = engine_config
        self.full_config = full_config
        self.implementation_config: ImplementationConfig = (
            self.full_config.implementation
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

        self.geometry_lut_indices = None
        self.geometry_lut_names = None
        self.x_RT_lut_indices = None

        self.prebuilt_lut_file = engine_config.engine_lut_file

        # Enable special modes - argument: get from HDF5
        self.multipart_transmittance = engine_config.multipart_transmittance
        self.topography_model = engine_config.topography_model

        # TBD
        self.lut_names = ["rhoatm", "transm", "sphalb", "transup"]
        if self.emission_mode:
            self.lut_names = [
                "thermal_upwelling",
                "thermal_downwelling",
            ] + self.lut_names

        if self.multipart_transmittance:
            self.test_rfls = [0, 0.1, 0.5]
            self.lut_names = self.lut_names + [
                "t_down_dir",
                "t_down_dif",
                "t_up_dir",
                "t_up_dif",
            ]

        self.solar_irr = None  # TODO - get from HDF5

        if engine_config.wavelength_file is not None:
            wavelength_file = engine_config.wavelength_file
        else:
            wavelength_file = full_config.forward_model.instrument.wavelength_file

        self.wl, self.fwhm = common.load_wavelen(wavelength_file)

        if engine_config.wavelength_range is not None:
            valid_wl = np.logical_and(
                self.wl >= engine_config.wavelength_range[0],
                self.wl <= engine_config.wavelength_range[1],
            )
            self.wl = self.wl[valid_wl]
            self.fwhm = self.fwhm[valid_wl]

        self.n_chan = len(self.wl)

        self.implementation_mode = full_config.implementation.mode

        self.interpolator_style = (
            full_config.forward_model.radiative_transfer.interpolator_style
        )

        # Defaults False, where True will overwrite any existing interpolator pickles
        self.overwrite_interpolator = (
            full_config.forward_model.radiative_transfer.overwrite_interpolator
        )

        # Prepare a cache for self._lookup_lut(), setting cache_size to 0 will disable
        self.cache = {}
        self.cache_size = full_config.forward_model.radiative_transfer.cache_size

        # We use a sorted dictionary here so that filenames for lookup
        # table (LUT) grid points are always constructed the same way (with
        # consistent dimension ordering). Every state vector element has
        # a lookup table dimension, but some lookup table dimensions
        # (like geometry parameters) may not be in the state vector.
        full_lut_grid = full_config.forward_model.radiative_transfer.lut_grid

        # selectively get lut components that are in this particular RTE
        self.lut_grid_config = OrderedDict()

        for key, value in full_lut_grid.items():
            if key in self.lut_names:
                self.lut_grid_config[key] = value

        # selectively get statevector components that are in this particular RTE
        self.statevector_names = (
            full_config.forward_model.radiative_transfer.statevector.get_element_names()
        )

        self.n_point = len(self.lut_grid_config)
        self.n_state = len(self.statevector_names)

        self.luts = {}

        # We establish scaling, bounds, and initial guesses for each free parameter
        # of the state vector. The free state vector elements are optimized during the
        # inversion, and must have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for key in self.statevector_names:
            element: StateVectorElementConfig = full_config.forward_model.radiative_transfer.statevector.get_single_element_by_name(
                key
            )
            self.bounds.append(element.bounds)
            self.scale.append(element.scale)
            self.init.append(element.init)
            self.prior_sigma.append(element.prior_sigma)
            self.prior_mean.append(element.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        self.lut_dims = []
        self.lut_grids = []
        self.lut_names = []
        self.lut_interp_types = []

        for key, grid_values in self.lut_grid_config.items():
            # do some quick checks on the values
            # For forward (simulation) mode, 1-dimensional LUT grids are OK!
            if len(grid_values) == 1 and not self.implementation_mode == "simulation":
                err = (
                    "Only 1 value in LUT grid {}. ".format(key)
                    + "1-d LUT grids cannot be interpreted."
                )
                raise ValueError(err)
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

        # "points" contains all combinations of grid points
        # We will have one filename prefix per point
        self.points = common.combos(self.lut_grids)
        self.files = []

        for point in self.points:
            outf = self.point_to_filename(point)
            self.files.append(outf)

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

    # TODO - change this name
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

    def read_simulation_results(self, point: np.array):
        raise AssertionError("Must populate simulation call")
