
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig
import logging
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

        self._wavelength_range = list()
        self.wavelength_range = None

        self._lut_path_type = str
        self.lut_path = None

        self._template_file_type = str
        self.template_file = None

        self._lut_names_type = str
        self.lut_names = None

        self._statevector_names_type = str
        self.statevector_names = None

        self._aerosol_template_file_type = str
        self.aerosol_template_file = None

        self._aerosol_model_file_type = str
        self.aerosol_model_file = None

        self._configure_and_exit_type = bool
        self.configure_and_exit = False

        self._auto_rebuild_type = bool
        self.auto_rebuild = True

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_rt_engines = ['modtran','libradtran','6s']
        if self.engine_name not in valid_rt_engines:
            errors.append('radiative_transfer->raditive_transfer_model: {} not in one of the available models: {}'.
                          format(self.engine_name, valid_rt_engines))


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

    def get_all_unknowns(self):
        return [self.H2O_ABSCO], ['H2O_ABSCO']


class RadiativeTransferConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._statevector_type = StateVectorConfig
        self.statevector:StateVectorConfig = None

        self._lut_grid_type = OrderedDict
        self.lut_grid = None

        self._unknowns_type = RadiativeTransferUnknownsConfig
        self.unknowns: RadiativeTransferUnknownsConfig = None

        self.set_config_options(sub_configdic)

        # sort lut_grid
        for key, value in self.lut_grid.items():
            self.lut_grid[key] = self.lut_grid[key].sort()
        self.lut_grid = OrderedDict(sorted(self.lut_grid.items(), key=lambda t:t[0]))

        # Hold this parameter for after the config_options, as radiative_transfer_engines have a special (dynamic) load
        self._radiative_transfer_engines_type = List[RadiativeTransferEngineConfig]
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

        #TODO: figure out submodule checking
        return errors

