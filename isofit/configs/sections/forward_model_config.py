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

from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.atmosphere_config import AtmosphereConfig
from isofit.configs.sections.instrument_config import InstrumentConfig
from isofit.configs.sections.surface_config import SurfaceConfig


class ForwardModelConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        super().__init__()

        self._surface_type = SurfaceConfig
        self.surface: SurfaceConfig = None
        """
        Surface: surface config section. 
        """

        self._atmosphere_type = AtmosphereConfig
        self.atmosphere: AtmosphereConfig = None
        """
        Atmosphere: atmospheric radiative transfer config section.
        """

        self._instrument_type = InstrumentConfig
        self.instrument: InstrumentConfig = None
        """
        Instrument: instrument config section. 
        """

        self._model_discrepancy_file_type = str
        self.model_discrepancy_file = None
        """
        Points to an numpy-format covariance matrix. 
        """

        # Backward compatibility: old configs used forward_model.radiative_transfer
        # with nested radiative_transfer_engines. Flatten the first engine into
        # forward_model.atmosphere so the rest of the code finds it.
        if (
            sub_configdic
            and "radiative_transfer" in sub_configdic
            and "atmosphere" not in sub_configdic
        ):
            rt = sub_configdic["radiative_transfer"]
            engines = rt.get("radiative_transfer_engines", {})
            # Take the first engine (usually the only one, e.g. "vswir")
            first_engine = next(iter(engines.values()), {}) if engines else {}
            atmosphere = {**first_engine}
            # Hoist statevector / lut_grid / unknowns from the RT container
            for key in ("statevector", "lut_grid", "unknowns"):
                if key in rt:
                    atmosphere.setdefault(key, rt[key])
            sub_configdic = {**sub_configdic, "atmosphere": atmosphere}

        self.set_config_options(sub_configdic)
