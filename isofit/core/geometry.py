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
from isofit.configs import configs


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
        coszen: float = None,
        full_config: configs.Config = {},
    ):
        """Initialize geometry object.
        Args:
            obs: Observation metadata array.
            loc: Location metadata array.
            dt: Date time of observation.
            esd: Earth sun distance array.
            bg_rfl: Background reflectance spectrum.
            svf: Sky view factor.
            coszen: Cosine of the solar zenith angle for top of atmosphere.
            config: isofit config.
        """
        # Set some benign defaults...
        self.observer_zenith = None
        self.observer_azimuth = None
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

        self.max_slope = 0.0
        self.terrain_style = "flat"

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

        # Determine how to treat coszen
        self.use_universal_coszen = True

        # Allow for backwards compatibility where config is not given
        if not full_config:
            # If no config given, prioritize using the OBS data
            if self.solar_zenith is not None:
                self.coszen = np.cos(np.radians(self.solar_zenith))
                self.use_universal_coszen = False
            # If OBS data is not present, then fall back to using the coszen input
            else:
                self.coszen = coszen

        # In the more common case that Isofit config is provided...
        else:
            # Update terrain parameters from config
            self.max_slope = full_config.forward_model.surface.max_slope
            self.terrain_style = full_config.forward_model.surface.terrain_style
            self.lut_grid = full_config.forward_model.atmosphere.lut_grid

            # 1. If user has a lut grid that contains solar_zenith this takes priority
            if self.lut_grid is not None and "solar_zenith" in self.lut_grid:
                self.coszen = np.cos(np.radians(self.solar_zenith))
                self.use_universal_coszen = False

            # 2. If it's not in the lut grid, and user doesn't give a coszen, we should
            # raise a helpful warning, but still allow them to use the OBS data
            elif coszen is None and self.solar_zenith is not None:
                self.coszen = np.cos(np.radians(self.solar_zenith))
                self.use_universal_coszen = False
                logging.warning(
                    "coszen was not defined and solar zenith was not found in the lut grid. "
                    "This will proceed with the OBS data, however, this may cause small errors in the forward model."
                )

            # 3. In this case, the user does not have solar zenith in the lut grid
            # and cozen is correctly defined (based on the atmospheric RT Engine)
            elif coszen is not None:
                self.coszen = coszen

            else:
                raise ValueError(
                    "coszen is not defined and valid solar zenith not found in OBS data."
                )

        if self.use_universal_coszen:
            logging.info(f"The coszen will be universal:   coszen={coszen}")
        else:
            logging.info(
                f"The coszen is pixel dependent and will based on the OBS data"
            )

        if self.coszen is not None:

            # Pretend that the surface is flat, regardless of input geometry
            if self.terrain_style == "flat":
                self.cos_i = self.coszen

            # Set min cosi (which is at max slope facing away from sun)
            self.min_cosi = max(
                0,
                np.sin(np.arccos(self.coszen))
                * np.sin(np.radians(self.max_slope))
                * np.cos(np.radians(180))
                + self.coszen * np.cos(np.radians(self.max_slope)),
            )

            # Check bounds
            self.coszen = max(self.min_cosi, min(self.coszen, 1.0))
            self.cos_i = max(self.min_cosi, min(self.cos_i, 1.0))
            self.skyview_factor = (
                1.0 if not 0 < self.skyview_factor <= 1 else self.skyview_factor
            )
        else:
            logging.warning(
                "Unable to determine coszen. Proceeding without will cause errors during the inversion."
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
