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
from datetime import datetime

import numpy as np

from isofit.core import units


class Geometry:
    """The geometry of the observation, all we need to calculate sensor,
    surface, and solar positions."""

    def __init__(
        self,
        obs: np.array = None,
        loc: np.array = None,
        dt: datetime = None,
        esd: np.array = None,
        bg_rfl: np.array = None,
        svf: float = 1,
        terrain_style: str = "flat",
        max_slope: float = 45.0,
    ):
        """Initialize geometry object.
        Args:
            obs: Observation metadata array.
            loc: Location metadata array.
            dt: Date time of observation.
            esd: Earth sun distance array.
            bg_rfl: Background reflectance spectrum.
            svf: Sky view factor.
        """
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

        if esd is None:
            logging.warning(
                "Earth sun distance not provided. Proceeding without might cause some inaccuracies down the line"
            )
            esd = np.ones((366, 2))
            esd[:, 0] = np.arange(1, 367, 1)
        self.earth_sun_distance_reference = esd

        self.bg_rfl = bg_rfl
        self.cos_i = None
        self.skyview_factor = svf

        # The 'obs' object is observation metadata that follows a historical
        # AVIRIS-NG format.  It arrives to our initializer in the form of
        # a list-like object...
        if obs is not None:
            self.path_length_km = units.m_to_km(obs[0])
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
            self.surface_elevation_km = units.m_to_km(loc[2])
            self.latitude = loc[1]  # Northing
            self.longitude = loc[0]  # Easting

        if loc is not None and obs is not None:
            self.observer_altitude_km = (
                self.surface_elevation_km
                + self.path_length_km * np.cos(np.deg2rad(self.observer_zenith))
            )

        if dt is not None:
            self.esd_factor = self.get_esd_factor(dt)

        # Bring in config settings for terrain everytime geom is loaded
        self.max_slope = max_slope
        self.terrain_style = terrain_style
        self.verify(
            coszen=np.nan, max_slope=self.max_slope, terrain_style=self.terrain_style
        )

    def get_esd_factor(self, date_time: datetime):
        """Get distance ratio from sun based on time of year, relative to day 1
        Args:
            date_time: datetime to search

        Returns:
            float: ratio of earth sun distnace based on datetime.
        """

        day_of_year = date_time.timetuple().tm_yday
        return float(self.earth_sun_distance[day_of_year - 1, 1])

    def verify(self, coszen, max_slope, terrain_style):
        """Verify important geometry data such as coszen, cos_i, slope, aspect, and sky view prior to inversion."""
        valid_data = {}

        # coszen should ideally come from RT because of how simulation constructed.
        # However, if SZA is in the lut grid, then falling back to self.solar_zenith is best.
        # This method is called again in get_coszen(), right before computing radiative transfer quantities.
        if not np.isnan(coszen):
            pass
        elif self.solar_zenith is not None and not np.isnan(self.solar_zenith):
            coszen = np.cos(np.radians(self.solar_zenith))
        else:
            valid_data["coszen"] = np.nan
            valid_data["cos_i"] = np.nan
            valid_data["skyview_factor"] = np.nan
            return

        # set min cosi (which is at max slope facing away from sun)
        self.min_cosi = max(
            0,
            np.sin(np.arccos(coszen))
            * np.sin(np.radians(max_slope))
            * np.cos(np.radians(180))
            + coszen * np.cos(np.radians(max_slope)),
        )

        # Pretend that the surface is flat, regardless of input geometry
        if terrain_style == "flat":
            cos_i = coszen
        else:
            # Local solar zenith angle as a function of surface slope and aspect
            cos_i = self.cos_i
            if cos_i is None or np.isnan(cos_i):
                raise ValueError("Verify cos_i in the input OBS data.")

        # Check bounds
        valid_data["coszen"] = max(self.min_cosi, min(coszen, 1.0))
        valid_data["cos_i"] = max(self.min_cosi, min(cos_i, 1.0))
        valid_data["skyview_factor"] = (
            1.0 if not 0 < self.skyview_factor <= 1 else self.skyview_factor
        )

        self.coszen = valid_data["coszen"]
        self.cos_i = valid_data["cos_i"]
        self.skyview_factor = valid_data["skyview_factor"]
