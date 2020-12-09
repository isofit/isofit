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
import logging


class RadiativeTransferEngineConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None, name: str = None):
        self._name_type = str
        self.name = name
        """str: Name of config - optional, and not currently used."""

        self._engine_name_type = str
        self.engine_name = None
        """str: Name of radiative transfer engine to use - options ['modtran', 'libradtran', '6s']."""

        self._engine_base_dir_type = str
        self.engine_base_dir = None
        """str: base directory of the given radiative transfer engine on user's OS."""

        self._wavelength_range_type = list()
        self.wavelength_range = None
        """List: The wavelength range to execute this radiative transfer engine over."""

        self._environment_type = str
        self.environment = None
        """str: Additional environment directives for the shell script."""

        self._lut_path_type = str
        self.lut_path = None
        """str: The path to the look up table directory used by the radiative transfer engine."""

        self._template_file_type = str
        self.template_file = None
        """str: A template file to be used as the base-configuration for the given radiative transfer engine."""

        self._lut_names_type = list()
        self.lut_names = None
        """List: Names of the elements to run this radiative transfer element on.  Must be a subset
        of the keys in radiative_transfer->lut_grid.  If not specified, uses all keys from 
        radiative_transfer-> lut_grid.  Auto-sorted (alphabetically) below."""

        self._statevector_names_type = list()
        self.statevector_names = None
        """List: Names of the statevector elements to use with this radiative transfer engine.  Must be a subset
        of the keys in radiative_transfer->statevector.  If not specified, uses all keys from 
        radiative_transfer->statevector.  Auto-sorted (alphabetically) below."""

        # MODTRAN parameters
        self._aerosol_template_file_type = str
        self.aerosol_template_file = None
        """str: Aerosol template file, currently only implemented for MODTRAN."""

        self._aerosol_model_file_type = str
        self.aerosol_model_file = None
        """str: Aerosol model file, currently only implemented for MODTRAN."""

        self._multipart_transmittance_type = bool
        self.multipart_transmittance = False
        """str: Use True to specify triple-run diffuse & direct transmittance 
           estimation.  Only implemented for MODTRAN."""

        # MODTRAN simulator
        self._emulator_file_type = str
        self.emulator_file = None
        """str: Path to emulator model file"""

        self._emulator_aux_file_type = str
        self.emulator_aux_file = None
        """str: path to emulator auxiliary data - expected npz format"""

        self._interpolator_base_path_type = str
        self.interpolator_base_path = None
        """str: path to emulator interpolator base - will dump multiple pkl extensions to this location"""

        # 6S parameters - not the corcommemnd
        # TODO: these should come from a template file, as in modtran
        self._day_type = int
        self.day = None
        """int: 6s-only day parameter."""

        self._month_type = int
        self.month = None
        """int: 6s-only month parameter."""

        self._elev_type = float
        self.elev = None
        """float: 6s-only elevation parameter."""

        self._alt_type = float
        self.alt = None
        """float: 6s-only altitude parameter."""

        self._obs_file_type = str
        self.obs_file = None
        """str: 6s-only observation file."""

        self._solzen_type = float
        self.solzen = None
        """float: 6s-only solar zenith."""

        self._solaz_type = float
        self.solaz = None
        """float: 6s-only solar azimuth."""

        self._viewzen_type = float
        self.viewzen = None
        """float: 6s-only view zenith."""

        self._viewaz_type = float
        self.viewaz = None
        """float: 6s-only view azimuth."""

        self._earth_sun_distance_file_type = str
        self.earth_sun_distance_file = None
        """str: 6s-only earth-to-sun distance file."""

        self._irradiance_file_type = str
        self.irradiance_file = None
        """str: 6s-only irradiance file."""

        self.set_config_options(sub_configdic)

        if self.lut_names is not None:
            self.lut_names.sort()

        if self.statevector_names is not None:
            self.statevector_names.sort()
        
        if self.interpolator_base_path is None and self.emulator_file is not None:
            self.interpolator_base_path = self.emulator_file + '_interpolator'
            logging.info('No interpolator base path set, and emulator used, so auto-setting interpolator path at: {}'.format(self.interpolator_base_path))


    def _check_config_validity(self) -> List[str]:
        errors = list()

        # Check that all input files exist
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            if value is not None and key[-5:] == '_file' and key != 'emulator_file':
                if os.path.isfile(value) is False:
                    errors.append('Config value radiative_transfer->{}: {} not found'.format(key, value))

        valid_rt_engines = ['modtran', 'libradtran', '6s', 'simulated_modtran']
        if self.engine_name not in valid_rt_engines:
            errors.append('radiative_transfer->raditive_transfer_model: {} not in one of the available models: {}'.
                          format(self.engine_name, valid_rt_engines))

        if self.multipart_trasmittance and self.engine_name != 'modtran':
            errors.append('Multipart transmittance is supported for MODTRAN only')

        if self.earth_sun_distance_file is None and self.engine_name == '6s':
            errors.append('6s requires earth_sun_distance_file to be specified')

        if self.irradiance_file is None and self.engine_name == '6s':
            errors.append('6s requires irradiance_file to be specified')

        if self.engine_name == 'simulated_modtran' and self.emulator_file is None:
            errors.append('The Modtran Simulator requires an emulator_file to be specified.')

        if self.engine_name == 'simulated_modtran' and self.emulator_aux_file is None:
            errors.append('The Modtran Simulator requires an emulator_aux_file to be specified.')

        files = [self.earth_sun_distance_file, self.irradiance_file,
                 self.obs_file, self.aerosol_model_file, self.aerosol_template_file]
        for f in files:
            if f is not None:
                if os.path.isfile(f) is False:
                    errors.append('Radiative transfer engine file not found on system: {}'.
                                  format(self.earth_sun_distance_file))
        return errors


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
        self.statevector: StateVectorConfig = StateVectorConfig({})

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

    def _check_config_validity(self) -> List[str]:
        errors = list()

        for key, item in self.lut_grid.items():
            if len(item) < 2:
                errors.append('lut_grid item {} has less than the required 2 elements'.format(key))
        
        for rte in self.radiative_transfer_engines:
            errors.extend(rte.check_config_validity())

        return errors
