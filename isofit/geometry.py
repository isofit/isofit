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

import scipy as s
from sunposition import sunpos
from datetime import datetime


class Geometry:
    """The geometry of the observation, all we need to calculate sensor,
      surface and solar positions"""

    def __init__(self, obs=None, glt=None, loc=None, ds=None,
                 esd=None, pushbroom_column=None):

        self.earth_sun_file = None
        self.observer_zenith = None
        self.observer_azimuth = None
        self.observer_altitude_km = None
        self.surface_elevation_km = None
        self.datetime = None
        self.day_of_year = None
        self.latitude = None
        self.longitude = None
        self.longitudeE = None
        self.gmtime = None
        self.earth_sun_distance = None
        self.pushbroom_column = pushbroom_column

        if obs is not None:
            self.path_length = obs[0]
            self.observer_azimuth = obs[1]  # 0 to 360 clockwise from N
            self.observer_zenith = obs[2]  # 0 to 90 from zenith
            self.solar_azimuth = obs[3]  # 0 to 360 clockwise from N
            self.solar_zenith = obs[4]  # 0 to 90 from zenith
            self.OBSZEN = 180.0 - abs(obs[2])  # MODTRAN convention?
            self.RELAZ = obs[1] - obs[3] + 180.0
            self.TRUEAZ = self.RELAZ  # MODTRAN convention?
            self.umu = s.cos(obs[2]/360.0*2.0*s.pi)  # Libradtran
        else:
            self.observer_azimuth = 0
            self.observer_zenith = 0
            self.OBSZEN = 180.0
            self.RELAZ = 0.0
            self.TRUEAZ = 0.0
            self.umu = 1.0

        if loc is not None:
            self.GNDALT = loc[2]
            self.altitude = loc[2]
            self.surface_elevation_km = loc[2] / 1000.0
            self.latitude = loc[1]
            self.longitude = loc[0]
            self.longitudeE = -loc[0]
            if self.longitude < 0:
                self.longitude = 360.0 - self.longitude

            print('Geometry lat: %f, lon: %f' %
                  (self.latitude, self.longitude))
            print('observer OBSZEN: %f, RELAZ: %f' % (self.OBSZEN, self.RELAZ))

        if ds is not None:
            self.datetime = datetime.strptime(ds, '%Y%m%dt%H%M%S')
            self.day_of_year = self.datetime.timetuple().tm_yday

        if esd is not None:
            self.earth_sun_distance = esd.copy()

    def coszen(self):
        self.dt = self.datetime
        az, zen, ra, dec, h = sunpos(self.datetime, self.latitude,
                                     self.longitudeE, self.surface_elevation_km * 1000.0,
                                     radians=True)
        return s.cos(zen)

    def sundist(self):
        '''Use zero-indexed table'''
        return float(self.earth_sun_distance[self.day_of_year-1, 1])
