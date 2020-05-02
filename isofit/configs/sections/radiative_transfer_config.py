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
from collections import OrderedDict


class RadiativeTransferEngineConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None, name: str = None):
        self._name_type = str
        self.name = name

        self._engine_name_type = str
        self.engine_name = None

        self._engine_base_dir_type = str
        self.engine_base_dir = None

        self._wavelength_range_type = list()
        self.wavelength_range = None

        self._lut_path_type = str
        self.lut_path = None

        self._template_file_type = str
        self.template_file = None

        self._lut_names_type = list()
        self.lut_names = None

        self._statevector_names_type = list()
        self.statevector_names = None

        self._configure_and_exit_type = bool
        self.configure_and_exit = False

        self._auto_rebuild_type = bool
        self.auto_rebuild = True

        # MODTRAN parameters
        self._aerosol_template_file_type = str
        self.aerosol_template_file = None

        self._aerosol_model_file_type = str
        self.aerosol_model_file = None

        # 6S parameters - not the corcommemnd
        # TODO: these should come from a template file, as in modtran
        self._day_type = int
        self.day = None

        self._month_type = int
        self.month = None

        self._elev_type = float
        self.elev = None

        self._alt_type = float
        self.alt = None

        self._obs_file_type = str
        self.obs_file = None

        self._solzen_type = float
        self.solzen = None

        self._solaz_type = float
        self.solaz = None

        self._viewzen_type = float
        self.viewzen = None

        self._viewaz_type = float
        self.viewaz = None

        self._earth_sun_distance_file_type = str
        self.earth_sun_distance_file = None

        self._irradiance_file_type = str
        self.irradiance_file = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_rt_engines = ['modtran', 'libradtran', '6s']
        if self.engine_name not in valid_rt_engines:
            errors.append('radiative_transfer->raditive_transfer_model: {} not in one of the available models: {}'.
                          format(self.engine_name, valid_rt_engines))

        if self.earth_sun_distance_file is None and self.engine_name == '6s':
            errors.append('6s requires earth_sun_distance_file to be specified')

        if self.irradiance_file is None and self.engine_name == '6s':
            errors.append('6s requires irradiance_file to be specified')

        files = [self.earth_sun_distance_file, self.irradiance_file,
                 self.obs_file, self.aerosol_model_file, self.aerosol_template_file]
        for f in files:
            if f is not None:
                if os.path.isfile(f) is False:
                    errors.append('Radiative transfer engine file not found on system: {}'.
                                  format(self.earth_sun_distance_file))
        return errors

    def get_lut_names(self):
        self.lut_names.sort()
        return self.lut_names


class RadiativeTransferUnknownsConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._H2O_ABSCO_type = float
        self.H2O_ABSCO = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        return errors


class RadiativeTransferConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._statevector_type = StateVectorConfig
        self.statevector: StateVectorConfig = None

        self._lut_grid_type = OrderedDict
        self.lut_grid = None

        self._unknowns_type = RadiativeTransferUnknownsConfig
        self.unknowns: RadiativeTransferUnknownsConfig = None

        self.set_config_options(sub_configdic)

        # sort lut_grid
        for key, value in self.lut_grid.items():
            self.lut_grid[key] = sorted(self.lut_grid[key])
        self.lut_grid = OrderedDict(sorted(self.lut_grid.items(), key=lambda t: t[0]))

        # Hold this parameter for after the config_options, as radiative_transfer_engines
        # have a special (dynamic) load
        self._radiative_transfer_engines_type = list()
        self.radiative_transfer_engines = []

        self._set_rt_config_options(sub_configdic['radiative_transfer_engines'])

    def _set_rt_config_options(self, subconfig):
        if type(subconfig) is list:
            for rte in subconfig:
                rt_model = RadiativeTransferEngineConfig(rte)
                self.radiative_transfer_engines.append(rt_model)
        elif type(subconfig) is dict:
            for key in subconfig:
                rt_model = RadiativeTransferEngineConfig(subconfig[key], name=key)
                self.radiative_transfer_engines.append(rt_model)

    def get_ordered_radiative_transfer_engines(self):

        self.radiative_transfer_engines.sort(key=lambda x: x.wavelength_range[0])
        return self.radiative_transfer_engines

    def _check_config_validity(self) -> List[str]:
        errors = list()

        for key, item in self.lut_grid.items():
            if len(item) < 2:
                errors.append('lut_grid item {} has less than the required 2 elements'.format(key))

        return errors
