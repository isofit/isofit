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

from sys import platform
import json
import os
import re
import scipy as s
from common import json_load_ascii, combos, VectorInterpolator
from common import recursive_replace
from copy import deepcopy
from scipy.stats import norm as normal
from scipy.interpolate import interp1d
from rt_lut import TabularRT, FileExistsError
from rt_modtran import ModtranRT

eps = 1e-5  # used for finite difference derivative calculations


class TelluricRT(ModtranRT):
    """A model of photon transport including the atmosphere, for upward-
       looking spectra."""

    def __init__(self, config):

        TabularRT.__init__(self, config)
        self.modtran_dir = self.find_basedir(config)
        self.filtpath = os.path.join(self.lut_dir, 'wavelengths.flt')
        self.template = deepcopy(json_load_ascii(
            config['modtran_template_file'])['MODTRAN'])
        self.build_lut()

