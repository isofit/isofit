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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import os
from typing import Dict, List, Type

import numpy as np

from isofit.configs.base_config import BaseConfigSection


class SurfaceConfig(BaseConfigSection):
    """
    Surface configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._multi_surface_flag_type = bool
        self.multi_surface_flag = False

        self._surface_file_type = str
        self.surface_file = None

        self._surface_category_type = str
        self.surface_category = None

        self._surface_class_file_type = str
        self.surface_class_file = None

        self._sub_surface_class_file_type = str
        self.sub_surface_class_file = None

        self._Surfaces_type = dict
        self.Surfaces = {}

        self._wavelength_file_type = str
        self.wavelength_file = None

        # Multicomponent Surface
        self._select_on_init_type = bool
        self.select_on_init = True
        """bool: This field, if present and set to true, forces us to use any initialization state and never change.
        The state is preserved in the geometry object so that this object stays stateless"""

        self._selection_metric_type = str
        self.selection_metric = "Euclidean"

        self._select_on_init_type = bool
        self.select_on_init = True

        self._full_glint_type = bool
        self.full_glint = False

        self._glint_model_type = bool
        self.glint_model = False

        # Surface Thermal
        self._emissivity_for_surface_T_init_type = float
        self.emissivity_for_surface_T_init = 0.98
        """ Initial Value recommended by Glynn Hulley."""

        self._surface_T_prior_sigma_degK_type = float
        self.surface_T_prior_sigma_degK = 1.0

        self._sun_glint_prior_sigma_type = float
        self.sun_glint_prior_sigma = 0.1

        self._sky_glint_prior_sigma_type = float
        self.sky_glint_prior_sigma = 0.01

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_surface_categories = [
            "surface",
            "multicomponent_surface",
            "glint_model_surface",
            "thermal_surface",
            "lut_surface",
        ]
        if (self.surface_category is None) and (self.Surfaces is None):
            errors.append("surface->surface_category or Surfaces must be specified")

        elif self.surface_category not in valid_surface_categories:
            errors.append(
                "surface->surface_category: {} not in valid surface categories: {}".format(
                    self.surface_category, valid_surface_categories
                )
            )

        if self.surface_category is None:
            errors.append("surface->surface_category must be specified")

        valid_metrics = ("Euclidean", "Mahalanobis")
        if self.selection_metric not in valid_metrics:
            errors.append(f"surface->selection_metric must be one of: {valid_metrics}")

        return errors
