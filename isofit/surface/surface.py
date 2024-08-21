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
from isofit.core.common import envi_header, load_spectrum, load_wavelen
from isofit.surface import Surfaces


class Surface:
    """A wrapper for the specific surface models"""

    def __init__(self, full_config: Config):
        config = full_config.forward_model.surface

        surfaces = make_surface_config(paths)

        self.surfaces = config
        for i, surf_dict in config.items():
            self.surfaces[i] = {}
            self.surfaces[i]["surface_model"] = Surfaces[surf_dict["surface_category"]]

        # Is there a way to not have to open this every operation?
        if self.surfaces[0]["surface_class_file"]:
            classes = envi.open(
                envi_header(surfaces[0]["surface_class_file"])
            ).open_memmap(interleave="bip")

            for c in config.keys():
                test = np.argwhere(classes == c)
                break

    def pixel_surface(self, row, col):
        # Easy case, no classification is propogated through
        if len(self.surfaces) == 1 or not self.surfaces[0]["surface_class_file"]:
            return self.surfaces[0]["surface_mode"]

        elif len(self.surfaces) > 1:
            class_file = envi.open(envi_header(self.surfaces[0]["surface_class_file"]))

        single_surface = {0: surfaces[0]}
