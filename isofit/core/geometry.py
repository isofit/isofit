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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import logging
import os
from datetime import datetime

import numpy as np

import isofit

from .sunposition import sunpos


class Geometry:
    """The geometry of the observation, all we need to calculate sensor,
    surface, and solar positions."""

    def __init__(
        self,
        obs: np.array = None,
        loc: np.array = None,
        dt: datetime = None,
        bg_rfl=None,
    ):
        # Set some benign defaults...
        self.observer_zenith = (
            0  # REVIEW: pytest/test_geometry asserts 0, change to None?
        )
        self.observer_azimuth = (
            0  # REVIEW: pytest/test_geometry asserts 0, change to None?
        )
        self.solar_zenith = None
        self.solar_azimuth = None
        self.observer_altitude_km = None
        self.surface_elevation_km = None
        self.earth_sun_distance = None
        self.esd_factor = None

        self.earth_sun_file = None
        self.earth_sun_distance_path = os.path.join(
            isofit.root, "data", "earth_sun_distance.txt"
        )
        try:
            self.earth_sun_distance_reference = np.loadtxt(self.earth_sun_distance_path)
        except FileNotFoundError:
            logging.warning(
                "Earth-sun-distance file not found on system. "
                "Proceeding without might cause some inaccuracies down the line."
            )
            self.earth_sun_distance_reference = np.ones((366, 2))
            self.earth_sun_distance_reference[:, 0] = np.arange(1, 367, 1)

        self.bg_rfl = bg_rfl
        self.cos_i = None

        # The 'obs' object is observation metadata that follows a historical
        # AVIRIS-NG format.  It arrives to our initializer in the form of
        # a list-like object...
        if obs is not None:
            self.path_length_km = obs[0] / 1000
            self.observer_azimuth = obs[1]  # 0 to 360 clockwise from N
            self.observer_zenith = obs[2]  # 0 to 90 from zenith
            self.solar_azimuth = obs[3]  # 0 to 360 clockwise from N
            self.solar_zenith = obs[4]  # 0 to 90 from zenith
            self.cos_i = obs[8]  # cosine of eSZA
            # calculate relative to-sun azimuth
            delta_phi = np.abs(self.solar_azimuth - self.observer_azimuth)
            self.relative_azimuth = np.minimum(delta_phi, 360 - delta_phi)  # 0 to 180

        # The 'loc' object is a list-like object that optionally contains
        # latitude and longitude information about the surface being
        # observed.
        self.latitude = None
        self.longitude = None
        if loc is not None:
            self.surface_elevation_km = loc[2] / 1000.0
            self.latitude = loc[1]  # Northing
            self.longitude = loc[0]  # Westing
            if self.longitude < 0:
                self.longitude = 360.0 - self.longitude

        if loc is not None and obs is not None:
            self.observer_altitude_km = (
                self.surface_elevation_km
                + self.path_length_km * np.cos(np.deg2rad(self.observer_zenith))
            )

        if dt is not None:
            self.esd_factor = self.get_esd_factor(dt)

    def get_esd_factor(self, date_time: datetime):
        """Get distance ratio from sun based on time of year, relative to day 1
        Args:
            date_time: datetime to search

        Returns:
            float: ratio of earth sun distnace based on datetime.
        """

        day_of_year = date_time.timetuple().tm_yday
        return float(self.earth_sun_distance[day_of_year - 1, 1])
