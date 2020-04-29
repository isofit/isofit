
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig
import logging


class RadiativeTransferEngineConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, sub_configdic: dict = None, name: str = None):
        self._name_type = str
        self.name = name

        self._engine_name_type = str
        self.engine_name = None

        self._wavelength_file_type = str
        self.wavelength_file = None

        self._lut_path_type = str
        self.lut_path = None

        self._template_file_type = str
        self.template_file = None

        self._lut_names_type = str
        self.lut_names = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_rt_engines = ['modtran','libradtran','6s']
        if self.engine_name not in valid_rt_engines:
            errors.append('radiative_transfer->raditive_transfer_model: {} not in one of the available models: {}'.
                          format(self.engine_name, valid_rt_engines))


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

    def get_all_unknowns(self):
        return [self.H2O_ABSCO], ['H2O_ABSCO']


class RadiativeTransferConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._statevector_type = StateVectorConfig
        self.statevector:StateVectorConfig = None

        self._lut_grid_type = List
        self.lut_grid = None

        self._radiative_transfer_template_file_type = str
        self.radiative_transfer_template_file = None

        self._aerosol_template_file_type = str
        self.aerosol_template_file = None

        self._aerosol_model_file_type = str
        self.aerosol_model_file = None

        self._unknowns_type = RadiativeTransferUnknownsConfig
        self.unknowns: RadiativeTransferUnknownsConfig = None


        #self._vswir_model_type = RadiativeTransferEngineConfig
        #self.vswir_model: RadiativeTransferEngineConfig = None

        #self._tir_model_type = RadiativeTransferEngineConfig
        #self.tir_model: RadiativeTransferEngineConfig = None

        #try:
        #    self._set_rt_config_options(sub_configdic['radiative_transfer_models'])
        #except KeyError:
        #    logging.debug('No radiative_transfer_models section found, skipping')

        self.set_config_options(sub_configdic)

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

    def get_radiative_transfer_model_names(self):
        name_list = []
        for rt_model in self.radiative_transfer_models:
            name_list.append(rt_model.model_name)
        return name_list

    def _check_config_validity(self) -> List[str]:
        errors = list()

        #TODO: figure out submodule checking

        return errors

