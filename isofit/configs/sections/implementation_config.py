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
from isofit.configs.sections.inversion_config import InversionConfig, McmcInversionConfig, GridInversionConfig
import os


class ImplementationConfig(BaseConfigSection):

    def __init__(self, sub_configdic: dict = None):
        """
        Input file(s) configuration.
        """

        self._mode_type = str
        self.mode = 'inversion'
        """
        str: Defines the operating mode for isofit. Current options are: inversion, grid_inversion, inversion_mcmc.
        """

        self._inversion_type = InversionConfig
        self.inversion: InversionConfig = None
        """InversionConfig: optional config for running in inversion mode."""

        self._mcmc_inversion_type = McmcInversionConfig
        self.mcmc_inversion: McmcInversionConfig = None
        """McmcInversionConfig: optional config for running in MCMC inversion mode."""

        self._grid_inversion_type = GridInversionConfig
        self.grid_inversion: GridInversionConfig = None
        """GridInversionConfig: optional config for running grid inversion mode."""

        self._n_cores_type = int
        self.n_cores = None
        """int: number of cores to use."""

        self._runtime_nice_level_type = int
        self.runtime_nice_level = None
        """int: nice level to run multiprocessing at.  If None, will use all available.  If 1, will run without 
        multiprocessing (good for debugging)"""

        self._rte_configure_and_exit_type = bool
        self.rte_configure_and_exit = False
        """bool: Indicates that code should terminate as soon as all radiative transfer engine configuration files are
        written (without running them)"""

        self._rte_auto_rebuild_type = bool
        self.rte_auto_rebuild = True
        """bool: Flag indicating whether radiative transfer engines should automatically rebuild."""


        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_implementation_modes = ['inversion', 'grid_inversion', 'mcmc_inversion']
        if self.mode not in valid_implementation_modes:
            errors.append('Invalid implementation mode: {}.  Valid options are: {}'.
                          format(self.mode, valid_implementation_modes))

        # TODO: recursive implmentation

        return errors
