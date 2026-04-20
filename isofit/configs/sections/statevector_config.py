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

import logging
from typing import Dict, List, Type

import numpy as np

from isofit.configs.base_config import BaseConfigSection
from isofit.surface.surface_glint_model import (
    DefaultSkyGlintPrior,
    DefaultSunGlintPrior,
)
from isofit.surface.surface_thermal import DefaultSurfTempKPrior


class StateVectorElementConfig(BaseConfigSection):
    """
    State vector element configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        super().__init__()

        self._bounds_type = list()
        self.bounds = [np.nan, np.nan]

        self._scale_type = float
        self.scale = np.nan

        self._prior_mean_type = float
        self.prior_mean = np.nan

        self._prior_sigma_type = float
        self.prior_sigma = np.nan

        self._init_type = float
        self.init = np.nan

        self.set_config_options(sub_configdic)


class StateVectorConfig(BaseConfigSection):
    def __init__(self, sub_configdic: dict = None):
        super().__init__()

    def _set_statevector_config_options(self, configdic):
        # TODO: update using methods below
        if configdic is not None:
            for key in configdic:
                sv = StateVectorElementConfig(configdic[key])
                setattr(self, key, sv)

    def get_all_bounds(self):
        bounds = []
        for element, name in zip(*self.get_elements()):
            bounds.append(element.bounds)
        return bounds

    def get_all_scales(self):
        scales = []
        for element, name in zip(*self.get_elements()):
            scales.append(element.scale)
        return scales

    def get_all_inits(self):
        inits = []
        for element, name in zip(*self.get_elements()):
            inits.append(element.init)
        return inits

    def get_all_prior_means(self):
        prior_means = []
        for element, name in zip(*self.get_elements()):
            prior_means.append(element.prior_mean)
        return prior_means

    def get_all_prior_sigmas(self):
        prior_sigmas = []
        for element, name in zip(*self.get_elements()):
            prior_sigmas.append(element.prior_sigma)
        return prior_sigmas
