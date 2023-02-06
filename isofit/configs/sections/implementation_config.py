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
from isofit.configs.sections.inversion_config import InversionConfig


class ImplementationConfig(BaseConfigSection):
    def __init__(self, sub_configdic: dict = None):
        """
        Input file(s) configuration.
        """

        self._mode_type = str
        self.mode = "inversion"
        """
        str: Defines the operating mode for isofit. Current options are: inversion, inversion_mcmc, 
        and 'simulation'.
        """

        self._inversion_type = InversionConfig
        self.inversion: InversionConfig = None
        """InversionConfig: optional config for running in inversion mode."""

        self._n_cores_type = int
        self.n_cores = None
        """int: number of cores to use."""

        self._task_inflation_factor_type = int
        self.task_inflation_factor = 10
        """int: Submit task_inflation_factor*n_cores number of tasks."""

        self._ip_head_type = str
        self.ip_head = None
        """str: Ray - parameter.  IP-head (for multi-node runs)."""

        self._redis_password_type = str
        self.redis_password = None
        """str: Ray - parameter.  Redis-password (for multi-node runs)."""

        self._ray_include_dashboard_type = bool
        self.ray_include_dashboard = False
        """str: Ray - parameter.  Boolean to include dashboard."""

        self._rte_configure_and_exit_type = bool
        self.rte_configure_and_exit = False
        """bool: Indicates that code should terminate as soon as all radiative transfer engine configuration files are
        written (without running them)"""

        self._rte_auto_rebuild_type = bool
        self.rte_auto_rebuild = True
        """bool: Flag indicating whether radiative transfer engines should automatically rebuild."""

        self._ray_temp_dir_type = str
        self.ray_temp_dir = "/tmp/ray"
        """str: Overrides the standard ray temporary directory.  Useful for multiuser systems."""

        self._ray_ignore_reinit_error_type = bool
        self.ray_ignore_reinit_error = True
        """bool: Boolean to tell ray to ignore re-initilaization.  Can be convenient for multiple Isofit instances."""

        self._io_buffer_size_type = int
        self.io_buffer_size = 100
        """bool: Integer indicating how large (how many spectra) of chunks to read/process/write.  A
        buffer size of 1 means pixels are processed independently.  Large buffers can help prevent IO choke points, 
        especially if the """

        self._max_hash_table_size_type = int
        self.max_hash_table_size = 50
        """int: The maximum size of inversion hash tables.  Can provide speedups with redundant surfaces, but comes
        with increased memory costs.
        """

        self._debug_mode_type = bool
        self.debug_mode = False
        """bool: A flag to run the code in debug mode, which circumvents ray.
        """

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_implementation_modes = ["inversion", "mcmc_inversion", "simulation"]
        if self.mode not in valid_implementation_modes:
            errors.append(
                "Invalid implementation mode: {}.  Valid options are: {}".format(
                    self.mode, valid_implementation_modes
                )
            )

        if self.mode != "simulation" and self.inversion is None:
            errors.append(
                "If running outside of simulation mode, Inversion must be defined"
            )
        elif self.mode == "simulation" and self.inversion is None:
            # TODO: fix this
            errors.append(
                "Even in simulation mode, and inversion config must be defined, though"
                "it may be blank."
            )

        if int(self.ip_head is not None) + int(self.redis_password is not None) == 1:
            errors.append(
                "If either ip_head or redis_password are specified, both must be"
                " specified"
            )

        return errors
