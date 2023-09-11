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


import numpy as np

from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.geometry import Geometry


class RadiativeTransferEngine:
    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        full_config: Config,
    ):
        self.engine_config = engine_config
        self.full_config = full_config

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

        # Enable special modes - argument: get from from HDF5
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

        # Initialize a 1 value hash dict
        self.last_point_looked_up = np.zeros(self.n_point)
        self.last_point_lookup_values = np.zeros(self.n_point)

        self.interpolator_disk_paths = [
            engine_config.interpolator_base_path + "_" + rtq + ".pkl"
            for rtq in self.lut_names
        ]

    def make_simulation_call(point: np.array, template_only: bool = False):
        """Write template(s) and run simulation.

        Args:
            point (np.array): conditions to alter in simulation
            template_only (bool): only write template file and then stop
        """
        raise AssertionError("Must populate simulation call")

    def point_to_filename(point: np.array) -> str:
        """Change a point to a base filename

        Args:
            point (np.array): conditions to alter in simulation

        Returns:
            str: basename of the file to use for this point
        """
        filename = ""  # TODO make up name based on point
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

    def read_simulation_results(point: np.array):
        raise AssertionError("Must populate simulation call")
