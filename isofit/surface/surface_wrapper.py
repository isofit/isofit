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

import numpy as np
from scipy.interpolate import interp1d
from spectral.io import envi

from isofit.configs import Config

from ..core.common import envi_header, load_spectrum, load_wavelen
from .surfaces import Surfaces


class SurfaceWrapper:
    """A wrapper for the specific surface models"""

    def __init__(self, full_config: Config, subs: bool = True):
        # Save the full config to the surface object
        # Is it running in subs-mode
        surface_config = full_config.forward_model.surface

        self.surface_params = surface_config.surface_params
        self.surf_lookup = surface_config.Surfaces
        for i, surf_dict in self.surf_lookup.items():
            surf_category = surf_dict["surface_category"]
            self.surf_lookup[i]["surface_model"] = Surfaces[surf_category](
                surf_dict, self.surface_params
            )

        # Check if this run is a multi-surface run. If so, index the pixels
        if surface_config.multi_surface_flag:
            if subs:
                self.class_file = surface_config.sub_surface_class_file
            else:
                self.class_file = surface_config.surface_class_file

            classes = envi.open(envi_header(self.class_file)).open_memmap(
                interleave="bip"
            )

            self.class_groups = []
            for c in self.surf_lookup.keys():
                pixel_list = np.argwhere(classes == int(c)).astype(int).tolist()
                self.class_groups.append(pixel_list)

            del classes

    def match_class(self, row, col):
        matches = np.zeros((len(self.class_groups))).astype(int)
        for i, group in enumerate(self.class_groups):
            if [row, col, 0] in group:
                matches[i] = 1
            else:
                matches[i] = 0

        if len(matches[np.where(matches)]) < 1:
            logging.exception(
                "Pixel did not match any class. \
                             Something is wrong"
            )
            raise ValueError

        elif len(matches[np.where(matches)]) > 1:
            logging.exception(
                "Pixel matches too many classes. \
                             Something is wrong"
            )
            raise ValueError

        return matches[np.where(matches)][0]

    def retrieve_pixel_surface_class(self, row, col):
        # Easy case, no classification is propogated through
        if len(self.surf_lookup) == 1 or not self.class_file:
            return str(0), self.surf_lookup[str(0)]["surface_model"]

        # Case where there is more than one surface
        elif len(self.surf_lookup) > 1:
            surf_i = str(self.match_class(row, col))
            return surf_i, self.surf_lookup[surf_i]["surface_model"]
