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
from scipy.io import loadmat

from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import (
    StateVectorConfig,
    StateVectorElementConfig,
)
from isofit.core.common import recursive_get
from isofit.surface.surface import DefaultState
from isofit.surface.surface_glint_model import (
    DefaultSkyGlintPrior,
    DefaultSunGlintPrior,
)
from isofit.surface.surface_thermal import DefaultSurfTempKPrior


class SurfaceStateVectorConfig(StateVectorConfig):
    """
    Surface State vector configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        super().__init__()

        self._SURF_TEMP_K_type = StateVectorElementConfig
        self.SURF_TEMP_K: StateVectorElementConfig = StateVectorElementConfig(
            DefaultSurfTempKPrior._asdict()
        )

        self._SKY_GLINT_type = StateVectorElementConfig
        self.SKY_GLINT: StateVectorElementConfig = StateVectorElementConfig(
            DefaultSkyGlintPrior._asdict()
        )

        self._SUN_GLINT_type = StateVectorElementConfig
        self.SUN_GLINT: StateVectorElementConfig = StateVectorElementConfig(
            DefaultSunGlintPrior._asdict()
        )

        assert len(self.get_all_elements()) == len(self._get_nontype_attributes())

        self._set_statevector_config_options(sub_configdic)


class SurfaceConfig(BaseConfigSection):
    """
    Surface configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        super().__init__()

        self._multi_surface_flag_type = bool
        self.multi_surface_flag = False

        self._surface_file_type = str
        self.surface_file = None

        self._surface_category_type = str
        self.surface_category = None

        self._surface_class_file_type = str
        self.surface_class_file = None

        self._base_surface_class_file_type = str
        self.base_surface_class_file = None

        self._Surfaces_type = dict
        self.Surfaces = {}

        self._wavelength_file_type = str
        self.wavelength_file = None

        self._surface_lut_file_type = str
        self.surface_lut_file = None

        """bool: This field, if present and set to true, forces us to use any initialization state and never change.
        The state is preserved in the geometry object so that this object stays stateless"""
        self._select_on_init_type = bool
        self.select_on_init = True

        self._selection_metric_type = str
        self.selection_metric = "Euclidean"

        self._statevector_type = SurfaceStateVectorConfig
        self.statevector: StateVectorConfig = SurfaceStateVectorConfig({})

        # Surface Thermal
        """ Initial Value recommended by Glynn Hulley."""
        self._emissivity_for_surface_T_init_type = float
        self.emissivity_for_surface_T_init = 0.98

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()
        warnings = list()

        if (self.surface_file is None) and not len(self.Surfaces):
            errors.append(
                "surface_file not specified. All current supported surface models require surface_file to be provided"
            )

        valid_surface_categories = [
            "surface",
            "multicomponent_surface",
            "glint_model_surface",
            "thermal_surface",
            "lut_surface",
        ]
        if (self.surface_category is None) and not len(self.Surfaces):
            errors.append("surface->surface_category or Surfaces must be specified")

        elif self.surface_category not in valid_surface_categories and not len(
            self.Surfaces
        ):
            errors.append(
                "surface->surface_category: {} not in valid surface categories: {}".format(
                    self.surface_category, valid_surface_categories
                )
            )

        valid_metrics = ["Euclidean", "SGA"]
        if self.selection_metric not in valid_metrics:
            errors.append(f"surface->selection_metric must be one of: {valid_metrics}")

        # multistate checks
        if len(self.Surfaces):
            missing_cats, missing_files, missing_ints = [], [], []
            for name, sub_surface in self.Surfaces.items():
                if not sub_surface.get("surface_category"):
                    missing_cats.append(name)
                if not sub_surface.get("surface_file"):
                    missing_files.append(name)
                if sub_surface.get("surface_int", -1) < 0:
                    missing_ints.append(name)

            if len(missing_cats):
                errors.append(
                    "Multi-surface config given and no "
                    f"surface-category specified for keys: {missing_cats}"
                )

            if len(missing_files):
                errors.append(
                    "Multi-surface config given and "
                    f"no surface-file specified for keys: {missing_files}"
                )

            if len(missing_ints):
                errors.append(
                    "Multi-surface config given and no surface-"
                    f"int mapping specified for keys: {missing_ints}"
                )

        # Check statevector
        mat_files = list(
            filter(None, recursive_get(self.get_config_as_dict(), "surface_file"))
        )
        statevec = {
            key: value
            for di in recursive_get(self.get_config_as_dict(), "statevector")
            for key, value in di.items()
        }

        mismatch = {}
        for f in mat_files:
            model_dict = loadmat(f)
            for i, name in enumerate(model_dict.get("statevec_names", [])):
                for key in DefaultState._fields:
                    if not (
                        np.all(statevec[name][key] == model_dict[key].squeeze()[i])
                    ):
                        mismatch[key] = name

        if len(mismatch):
            message = (
                "Configured non-reflectance surface statevector "
                + "elements do not match .mat file.\nRun will use configured "
                + "values in the .json config."
                + "\nMismatching values:"
            )
            for key, value in mismatch.items():
                message += f"\n{value}: {key}"

            warnings.append(message)

        return errors, warnings
