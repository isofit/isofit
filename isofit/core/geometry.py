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

from .sunposition import sunpos


class Geometry:
    """The geometry of the observation, all we need to calculate sensor,
      surface, and solar positions."""

    def __init__(self, obs=None, glt=None, loc=None, ds=None,
                 esd=None, pushbroom_column=None, bg_rfl=None):

        # Set some benign defaults...
        self.earth_sun_file = None
        self.observer_zenith = 0
        self.observer_azimuth = 0
        self.observer_altitude_km = None
        self.surface_elevation_km = None
        self.datetime = None
        self.day_of_year = None
        self.latitude = None
        self.longitude = None
        self.longitudeE = None
        self.gmtime = None
        self.earth_sun_distance = None
        self.OBSZEN = 180.0
        self.RELAZ = 0.0
        self.TRUEAZ = 0.0
        self.H1ALT = None
        self.umu = 1.0
        self.pushbroom_column = pushbroom_column
        self.bg_rfl = bg_rfl

        # The 'obs' object is observation metadata that follows a historical
        # AVIRIS-NG format.  It arrives to our initializer in the form of
        # a list-like object...
        if obs is not None:
            self.path_length = obs[0]
            self.observer_azimuth = obs[1]  # 0 to 360 clockwise from N
            self.observer_zenith = obs[2]  # 0 to 90 from zenith
            self.solar_azimuth = obs[3]  # 0 to 360 clockwise from N
            self.solar_zenith = obs[4]  # 0 to 90 from zenith
            self.OBSZEN = 180.0 - abs(obs[2])  # MODTRAN convention?
            self.RELAZ = obs[1] - obs[3] + 180.0
            self.TRUEAZ = obs[1]  # MODTRAN convention?
            self.umu = np.cos(obs[2]/360.0*2.0*np.pi)  # Libradtran

        # The 'loc' object is a list-like object that optionally contains
        # latitude and longitude information about the surface being
        # observed.
        if loc is not None:
            self.surface_elevation_km = loc[2] / 1000.0
            self.latitude = loc[1]
            self.longitude = loc[0]
            self.longitudeE = -loc[0]
            if self.longitude < 0:
                self.longitude = 360.0 - self.longitude

            logging.debug('Geometry lat: %f lon: %f' %
                          (self.latitude, self.longitude))
            logging.debug('Geometry observer OBSZEN: %f RELAZ: %f GNDALT: %f' %
                          (self.OBSZEN, self.RELAZ, self.surface_elevation_km))
        
        if loc is not None and obs is not None:
            self.H1ALT = self.surface_elevation_km + self.path_length*np.cos(np.deg2rad(self.observer_zenith))
            self.observer_altitude_km = self.surface_elevation_km + self.path_length*np.cos(np.deg2rad(self.observer_zenith))

        # The ds object is an optional date object, defining the time of
        # the observation.
        if ds is not None:
            self.datetime = datetime.strptime(ds, '%Y%m%dt%H%M%S')
            self.day_of_year = self.datetime.timetuple().tm_yday

        # Finally, the earth sun distance is an array that maps the day of the
        # year (zero-indexed!) onto the mean-relative distance to the sun.
        if esd is not None:
            self.earth_sun_distance = esd.copy()

    def coszen(self):
        """ Return the cosine of the solar zenith."""
        self.dt = self.datetime
        az, zen, ra, dec, h = sunpos(self.datetime, self.latitude,
                                     self.longitudeE,
                                     self.surface_elevation_km * 1000.0,
                                     radians=True)
        return np.cos(zen)

    def sundist(self):
        '''Return the mean-relative distance to the sun as defined by the
        day of the year.  Note that we use zero-indexed table, offset by one 
        from the actual cardenality, per Python conventions...'''
        return float(self.earth_sun_distance[self.day_of_year-1, 1])
