#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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
#

import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header

def approx_pixel_size(lat, lon):
    """Approximate pixel size (meters) using the haversine formula."""
    R = 6371000.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    def haversine(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    dx = haversine(lat_r[:, :-1], lon_r[:, :-1], lat_r[:, 1:], lon_r[:, 1:])
    dy = haversine(lat_r[:-1, :], lon_r[:-1, :], lat_r[1:, :], lon_r[1:, :])

    return (dx + dy) / 2

def get_adjacency_range(sensor_alt_asl, ground_alt_asl, Rsat=1.0, min_range=0.2):
    """Estimate adjacency range based on Richter et al 2011"""

    # for spaceborne case
    if sensor_alt_asl>95.0:
        return Rsat
    
    else:
        # for airborne case
        z_rel = sensor_alt_asl - ground_alt_asl
        R1 = Rsat * (1 - np.exp(-z_rel / 8))
        R2 = Rsat * (1 - np.exp(-ground_alt_asl / 8))
        r = R1 - R2
        
        # assume min adj range
        r[r<min_range] = min_range

        return r


def background_reflectance(input_radiance, input_loc, input_obs, bgrfl_path):
    """Produces background reflectance for each pixel."""

    # open datasets
    rdn = envi.open(envi_header(input_radiance), input_radiance).open_memmap()
    loc = envi.open(envi_header(input_loc), input_radiance).open_memmap()
    obs = envi.open(envi_header(input_obs), input_radiance).open_memmap()

    # estimate pixel size and get adj range
    pixel_size = approx_pixel_size(loc[:,:,0], loc[:,:,1])
    r = get_adjacency_range(sensor_alt_asl, loc[:,:,2]/1000., Rsat=1.0, min_range=0.2)

    

    return