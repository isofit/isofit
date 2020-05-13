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

from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig
import os


class InstrumentUnknowns(BaseConfigSection):
    """
    Instrument Unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._channelized_radiometric_uncertainty_file_type = str
        self.channelized_radiometric_uncertainty_file = None

        self._uncorrelated_radiometric_uncertainty_type = float
        self.uncorrelated_radiometric_uncertainty = None

        self._wavelength_calibration_uncertainty_type = float
        self.wavelength_calibration_uncertainty = None

        self._stray_srf_uncertainty_type = float
        self.stray_srf_uncertainty = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        file_params = [self.channelized_radiometric_uncertainty_file]
        for param in file_params:
            if param is not None:
                if os.path.isfile(param) is False:
                    errors.append('Instrument unknown file: {} not found'.format(param))

        return errors


class InstrumentConfig(BaseConfigSection):
    """
    Instrument configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._wavelength_file_type = str
        self.wavelength_file = None

        self._integrations_type = int
        self.integrations = None
        """Number of integrations comprising the measurement.  
        Noise diminishes with the square root of this number.  Applicable in concert with parametric_noise_file 
        or pushbroom_noise_file"""

        self._unknowns_type = InstrumentUnknowns
        self.unknowns: InstrumentUnknowns = None

        self._fast_resample_type = bool
        self.fast_resample = True
        """bool: Approximates a complete resampling by a convolution with a uniform FWHM."""

        self._statevector_type = StateVectorConfig
        self.statevector: StateVectorConfig = None

        self._SNR_type = float
        self.SNR = None
        """float: We have several ways to define the instrument noise.  The simplest model is based on a single uniform 
        SNR number that is signal-independnet and applied uniformly to all wavelengths"""

        self._parametric_noise_file_type = str
        self.parametric_noise_file = None
        """str: We have several ways to define the instrument noise.
        The second option is a parametric, signal- and wavelength-
        dependent noise function. This is given by a four-column
        ASCII Text file.  Rows represent, respectively, the reference
        wavelength, and coefficients A, B, and C that define the
        noise-equivalent radiance via NeDL = A * sqrt(B+L) + C
        For the actual radiance L."""

        self._pushbroom_noise_file_type = str
        self.pushbroom_noise_file = None
        """str: We have several ways to define the instrument noise.
        The third option is a full pushbroom noise model that
        specifies noise columns and covariances independently for
        each cross-track location via an ENVI-format binary data file."""

        self._nedt_noise_file_type = str
        self.nedt_noise_file = None
        """str: We have several ways to define the instrument noise.  The last is NEDT noise"""

        self.set_config_options(sub_configdic)

        # If necessary, initialize some blank options
        if self.statevector is None:
            self.statevector = StateVectorConfig({})

    def _check_config_validity(self) -> List[str]:
        errors = list()

        noise_options = [self.SNR, self.parametric_noise_file,
                         self.pushbroom_noise_file, self.nedt_noise_file]
        used_noise_options = [x for x in noise_options if x is not None]

        if len(used_noise_options) == 0:
            errors.append('Instrument noise not defined.')

        if len(used_noise_options) > 1:
            errors.append('Multiple instrument noise options selected - please choose only 1.')

        file_params = [self.parametric_noise_file, self.pushbroom_noise_file, self.nedt_noise_file]
        for param in file_params:
            if param is not None:
                if os.path.isfile(param) is False:
                    errors.append('Instrument config file: {} not found'.format(param))

        return errors
