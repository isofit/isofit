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

import numpy as np

from isofit.configs.base_config import BaseConfigSection


class OutputConfig(BaseConfigSection):
    """
    Output file(s) configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._estimated_state_file_header = (
            "statevector",
            "{State Parameter, Value}",
            "{}",
        )
        self._estimated_state_file_type = str
        self.estimated_state_file = None

        self._estimated_reflectance_file_header = (
            "wavelength",
            "{Wavelength (nm), Lambertian Reflectance}",
            "{0.0,1.0}",
        )
        self._estimated_reflectance_file_type = str
        self.estimated_reflectance_file = None

        self._estimated_emission_file_header = (
            "wavelength",
            "{Wavelength (nm), Emitted Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        )
        self._estimated_emission_file_type = str
        self.estimated_emission_file = None

        self._modeled_radiance_file_header = (
            "wavelength",
            "{Wavelength (nm), Modeled Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        )
        self._modeled_radiance_file_type = str
        self.modeled_radiance_file = None

        self._apparent_reflectance_file_header = (
            "wavelength",
            "{Wavelength (nm), Apparent Surface Reflectance}",
            "{}",
        )
        self._apparent_reflectance_file_type = str
        self.apparent_reflectance_file = None

        self._path_radiance_file_header = (
            "wavelength",
            "{Wavelength (nm), Path Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        )
        self._path_radiance_file_type = str
        self.path_radiance_file = None

        self._simulated_measurement_file_header = (
            "wavelength",
            "{Wavelength (nm), Simulated Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        )
        self._simulated_measurement_file_type = str
        self.simulated_measurement_file = None

        self._algebraic_inverse_file_header = (
            "wavelength",
            "{Wavelength (nm), Apparent Surface Reflectance}",
            "{}",
        )
        self._algebraic_inverse_file_type = str
        self.algebraic_inverse_file = None

        self._atmospheric_coefficients_file_header = (
            "atm_coeffs",
            "{Wavelength (nm), Atmospheric Optical Parameters}",
            "{}",
        )
        self._atmospheric_coefficients_file_type = str
        self.atmospheric_coefficients_file = None

        self._radiometry_correction_file_header = (
            "wavelength",
            "{Wavelength (nm), Radiometric Correction Factors}",
            "{}",
        )
        self._radiometry_correction_file_type = str
        self.radiometry_correction_file = None

        self._spectral_calibration_file_header = ("wavelength", "{}", "{}")
        self._spectral_calibration_file_type = str
        self.spectral_calibration_file = None

        self._posterior_uncertainty_file_header = (
            "statevector",
            "{State Parameter, Value}",
            "{}",
        )
        self._posterior_uncertainty_file_type = str
        self.posterior_uncertainty_file = None

        self._plot_surface_components_type = bool
        self.plot_surface_components = False

        self._mcmc_samples_file_type = str
        self.mcmc_samples_file = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors

    def get_all_output_file_names(self):
        keys = []
        for key in self._get_nontype_attributes():
            if hasattr(self, "_{}_header".format(key)):
                keys.append(key)
        return keys

    def get_output_files(self):
        names = self.get_all_output_file_names()
        elements = [getattr(self, name) for name in names]
        headers = [getattr(self, "_{}_header".format(name)) for name in names]

        valid = [x is not None for x in elements]
        elements = [elements[x] for x in range(len(elements)) if valid[x]]
        headers = [headers[x] for x in range(len(headers)) if valid[x]]
        names = [names[x] for x in range(len(names)) if valid[x]]

        order = np.argsort(names)
        elements = [elements[idx] for idx in order]
        headers = [headers[idx] for idx in order]
        names = [names[idx] for idx in order]

        return elements, headers, names
