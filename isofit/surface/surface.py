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
        self.full_config = full_config

        config = full_config.forward_model.surface
        self.surfaces = config
        for i, surf_dict in config.items():
            self.surfaces[i]["surface_model"] = Surfaces[surf_dict["surface_category"]]
        # surfaces = config
        for i, surf_dict in config.items():
            config[i]["surface_model"] = Surfaces[surf_dict["surface_category"]]

        # Set up pixel groups in the init to only read file once
        if config[0]["surface_class_file"]:
            classes = envi.open(
                envi_header(config[0]["surface_class_file"])
            ).open_memmap(interleave="bip")

            self.groups = []
            for c in self.surfaces.keys():
                self.groups.append(np.argwhere(classes == c).astype(int).tolist())

    def match_class(self, row, col):
        matches = np.zeros((len(self.groups))).astype(int)
        for i, group in enumerate(self.groups):
            if [row, col, 0] in group:
                matches[i] = 1
            else:
                matches[i] = 0

        if len(matches[np.where(matches)]) > 1:
            raise ValueError(
                "Pixel did not match any class. \
                             Something is wrong"
            )

        elif len(matches[np.where(matches)]) > 1:
            raise ValueError(
                "Pixel matches too many classes. \
                             Something is wrong"
            )

        return matches[np.where(matches)][0]

    def call_rowcol_surface(self, row, col):
        # Easy case, no classification is propogated through
        if len(self.surfaces) == 1 or not self.surfaces[0]["surface_class_file"]:
            return self.surfaces[0]["surface_model"](self.full_config)

        elif len(self.surfaces) > 1:
            return self.surfaces[self.match_class(groups, row, col)]["surface_model"](
                self.full_config
            )
