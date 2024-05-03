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

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Type

import numpy as np

from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig


class RadiativeTransferEngineConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None, name: str = None):
        super().__init__()
        self._name_type = str
        self.name = name
        """str: Name of config - optional, and not currently used."""

        self._engine_name_type = str
        self.engine_name = None
        """str: Name of radiative transfer engine to use - options ['modtran', '6s', 'sRTMnet']."""

        self._engine_base_dir_type = str
        self.engine_base_dir = None
        """str: base directory of the given radiative transfer engine on user's OS."""

        self._engine_lut_file_type = str
        self.engine_lut_file = None
        """str: File containing the prebuilt LUT file (hdf5)."""

        self._wavelength_file_type = str
        self.wavelength_file = None
        """str: Optional path to wavelength file for high-res atmospheric calculations"""

        self._wavelength_range_type = list()
        self.wavelength_range = None
        """List: The wavelength range to execute this radiative transfer engine over."""

        self._environment_type = str
        self.environment = None
        """str: Additional environment directives for the shell script."""

        self._lut_path_type = str
        self.lut_path = None
        """str: The path to the look up table directory used by the radiative transfer engine."""

        self._sim_path_type = str
        self.sim_path = None
        """str: Path to the simulation outputs for the radiative transfer engine."""

        self._template_file_type = str
        self.template_file = None
        """str: A template file to be used as the base-configuration for the given radiative transfer engine."""

        self._treat_as_emissive_type = bool
        self.treat_as_emissive = False
        """bool: Run the simulation in emission mode"""

        self._topography_model_type = bool
        self.topography_model = False
        """
        Flag to indicated whether to use a topographic-flux (topoflux)
        implementation of the forward model.
        """

        self._glint_model_type = bool
        self.glint_model = False
        """
        Flag to indicate whether to use the sun and sky glint model from Gege (2012, 2014) in the forward model. 
        Only currently functional with multipart MODTRAN.
        """

        self._rt_mode_type = str
        self.rt_mode = None
        """str: Radiative transfer mode of LUT simulations. 
        'transm' for transmittances, 'rdn' for reflected radiance."""

        self._lut_names_type = dict()
        self.lut_names = None
        """Dictionary: Names of the elements to run this radiative transfer element on.  Must be a subset
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
            keys = list(self.lut_names.keys())
            keys.sort()
            self.lut_names = {i: self.lut_names[i] for i in keys}

        if self.statevector_names is not None:
            self.statevector_names.sort()

        if self.interpolator_base_path is None and self.emulator_file is not None:
            self.interpolator_base_path = self.emulator_file + "_interpolator"
            logging.info(
                "No interpolator base path set, and emulator used, so auto-setting"
                " interpolator path at: {}".format(self.interpolator_base_path)
            )

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # Check that all input files exist
        for key in self._get_nontype_attributes():
            value = getattr(self, key)
            if value is not None and key[-5:] == "_file" and key != "emulator_file":
                if os.path.isfile(value) is False:
                    errors.append(
                        "Config value radiative_transfer->{}: {} not found".format(
                            key, value
                        )
                    )

        valid_rt_engines = ["modtran", "6s", "sRTMnet", "KernelFlowsGP"]
        if self.engine_name not in valid_rt_engines:
            errors.append(
                "radiative_transfer->raditive_transfer_model: {} not in one of the"
                " available models: {}".format(self.engine_name, valid_rt_engines)
            )

        valid_rt_modes = ["transm", "rdn"]
        if self.rt_mode not in valid_rt_modes:
            errors.append(
                "radiative_transfer->raditive_transfer_mode: {} not in one of the"
                " available modes: {}".format(self.rt_mode, valid_rt_modes)
            )

        if self.multipart_transmittance and self.engine_name != "modtran":
            errors.append("Multipart transmittance is supported for MODTRAN only")

        if self.earth_sun_distance_file is None and self.engine_name == "6s":
            errors.append("6s requires earth_sun_distance_file to be specified")

        if self.irradiance_file is None and self.engine_name == "6s":
            errors.append("6s requires irradiance_file to be specified")

        if self.engine_name == "sRTMnet" and self.emulator_file is None:
            errors.append("The sRTMnet requires an emulator_file to be specified.")

        if self.engine_name == "sRTMnet" and self.emulator_aux_file is None:
            errors.append("The sRTMnet requires an emulator_aux_file to be specified.")

        if self.engine_name == "sRTMnet" and self.emulator_file is not None:
            if os.path.splitext(self.emulator_file)[1] != ".h5":
                errors.append(
                    "sRTMnet now requires the emulator_file to be of type .h5.  Please download an updated version from:\n https://zenodo.org/records/10831425"
                )

        files = [
            self.earth_sun_distance_file,
            self.irradiance_file,
            self.obs_file,
            self.aerosol_model_file,
            self.aerosol_template_file,
        ]
        for f in files:
            if f is not None:
                if os.path.isfile(f) is False:
                    errors.append(
                        "Radiative transfer engine file not found on system: {}".format(
                            self.earth_sun_distance_file
                        )
                    )
        return errors


class RadiativeTransferUnknownsConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        super().__init__()
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

        self._interpolator_style_type = str
        self.interpolator_style = "mlg"
        """str: Style of interpolation.
        - mlg   = Multilinear Grid
        - rg    = RegularGrid
        Speed performance:
            mlg >> stacked rg >> unstacked rg
        Caching provides significant gains for rg, marginal for mlg"""

        self._overwrite_interpolator_type = bool
        self.overwrite_interpolator = False
        """bool: Overwrite any existing interpolator pickles"""

        self._cache_size_type = int
        self.cache_size = 16
        """int: Size of the cache to store interpolation lookups. Defaults to 16 which
        provides the most significant gains. Setting higher may provide marginal gains."""

        self.set_config_options(sub_configdic)

        # sort lut_grid
        for key, value in self.lut_grid.items():
            self.lut_grid[key] = sorted(self.lut_grid[key])
        self.lut_grid = OrderedDict(sorted(self.lut_grid.items(), key=lambda t: t[0]))

        # Hold this parameter for after the config_options, as radiative_transfer_engines
        # have a special (dynamic) load
        self._radiative_transfer_engines_type = list()
        self.radiative_transfer_engines = []

        self._set_rt_config_options(sub_configdic["radiative_transfer_engines"])

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
                errors.append(
                    "lut_grid item {} has less than the required 2 elements".format(key)
                )

        if self.topography_model:
            for rtm in self.radiative_transfer_engines:
                if rtm.engine_name != "modtran":
                    errors.append(
                        "All self.forward_model.radiative_transfer.radiative_transfer_engines"
                        ' must be of type "modtran" if forward_model.topograph_model is'
                        " set to True"
                    )
                if rtm.multipart_transmittance is False:
                    errors.append(
                        "All self.forward_model.radiative_transfer.radiative_transfer_engines"
                        " must have multipart_transmittance set as True if"
                        " forward_model.topograph_model is set to True"
                    )

        for rte in self.radiative_transfer_engines:
            errors.extend(rte.check_config_validity())

        kinds = ["rg", "nds", "mlg"]  # Implemented kinds of interpolator functions
        kind = self.interpolator_style[:3]
        degrees = self.interpolator_style[4:]

        if kind not in kinds:
            errors.append(
                f"Interpolator style {self.interpolator_style} must be one of: {kinds}"
            )
        if kind == "nds":
            try:
                degree = int(degrees)
                assert degree >= 0 and np.isfinite(degree)
            except:
                errors.append(
                    "Invalid degree number. Should be an integer, e.g. nds-3, got"
                    f" {degrees!r} from {self.interpolator_style!r}[4:]"
                )

        return errors
