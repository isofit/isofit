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
# Authors: David R Thompson, david.r.thompson@jpl.nasa.gov
#

from isofit.core.geometry import Geometry


def test_Geometry():
    geom = Geometry()
    assert geom.earth_sun_file == None
    assert geom.observer_zenith == 0
    assert geom.observer_azimuth == 0
    assert geom.observer_altitude_km == None
    assert geom.surface_elevation_km == None
    assert geom.datetime == None
    assert geom.day_of_year == None
    assert geom.latitude == None
    assert geom.longitude == None
    assert geom.longitudeE == None
    assert geom.gmtime == None
    assert geom.earth_sun_distance == None
    assert geom.OBSZEN == 180.0
    assert geom.RELAZ == 0.0
    assert geom.TRUEAZ == 0.0
    assert geom.umu == 1.0
