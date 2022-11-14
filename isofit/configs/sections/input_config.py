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

from isofit.configs.base_config import BaseConfigSection


class InputConfig(BaseConfigSection):
    def __init__(self, sub_configdic: dict = None):
        """
        Input file(s) configuration.
        """

        self._measured_radiance_file_type = str
        self.measured_radiance_file = None
        """
        str: Input radiance file.  Can be either a .mat, .txt, or ENVI formatted binary cube. 
        Used for inverse-modeling (radiance -> reflectance).
        """

        self._reference_reflectance_file_type = str
        self.reference_reflectance_file = None
        """
        str: Input reference reflectance (to be compared against) file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Used for radiometric calibration.
        """

        self._reflectance_file_type = str
        self.reflectance_file = None
        """
        str: Input reflectance file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Used for forward-modeling (reflectance -> radiance).
        """

        self._obs_file_type = str
        self.obs_file = None
        """
        str: Input 'obs', or observation, file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Provides information about the conditions during observaiton.  Assumed to be in the band-wise format:
        {path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth, to-sun zenith, phase, slope, aspect, cosine i, 
        UTC time}
        """

        self._glt_file_type = str
        self.glt_file = None
        """
        str: Input glt file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Provides (x,y) offset information for the spatial location of raw-space input files
        """

        self._loc_file_type = str
        self.loc_file = None
        """
        str: Input 'loc', or location, file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Provides per-pixel lat, long, and elevation information.
        """

        self._background_reflectance_file_type = str
        self.background_reflectance_file = None
        """
        str: Background reflectance file used for spatial inference.
        Should be an ENVI formatted binary cube that is the same
        dimensions as the input data cube.
        """

        self._radiometry_correction_file_type = str
        self.radiometry_correction_file = None
        """
        str: Input radiometric correction file.  Can be either a .mat, .txt, or ENVI formatted binary cube.
        Used to make minor channelized corrections to account for slight systematic errors not captured in calibration.
        """

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # Check that all input files exist
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            if value is not None:
                if os.path.isfile(value) is False:
                    errors.append(
                        "Config value Input->{}: {} not found".format(key, value)
                    )

        # TODO: check that the right combination of input files exists

        # Recursive call to any sub-config-module errors
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if callable(key):
                errors.extend(value.check_config_validity())

        return errors
