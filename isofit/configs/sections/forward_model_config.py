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
from isofit.configs.sections.instrument_config import InstrumentConfig
from isofit.configs.sections.surface_config import SurfaceConfig
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferConfig


class ForwardModelConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._instrument_type = InstrumentConfig
        self.instrument: InstrumentConfig = None
        """
        Instrument: instrument config section. 
        """

        self._surface_type = SurfaceConfig
        self.surface: SurfaceConfig = None
        """
        Instrument: instrument config section. 
        """

        self._radiative_transfer_type = RadiativeTransferConfig
        self.radiative_transfer: RadiativeTransferConfig = None
        """
        RadiativeTransfer: radiative transfer config section.
        """

        self._model_discrepancy_file_type = str
        self.model_discrepancy_file = None
        """
        Points to an numpy-format covariance matrix. 
        """

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        return errors
