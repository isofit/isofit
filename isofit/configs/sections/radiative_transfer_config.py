
from typing import Dict, List, Type
from isofit.configs import BaseConfigSection
from isofit.configs.sections.statevector_config import StateVectorConfig
import logging


class DomainConfig(BaseConfigSection):
    """
    Domain configuration.
    """

    def __init__(self, sub_configdic: dict = None):

        self._start_type = float
        self.start = None

        self._end_type = float
        self.end = None

        self._step_type = float
        self.step = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()
        if self.end < self.start:
            errors.append('Domain end: {} is < domain start: {}'.format(self.end,self.start))
        return errors


class RadiativeTransferModelConfig(BaseConfigSection):
    """
    Radiative transfer unknowns configuration.
    """

    def __init__(self, model_name: str, sub_configdic: dict = None):
        self._model_name_type = str
        self.model_name = model_name

        self._wavelength_file_type = str
        self.wavelength_file = None

        self._lut_path = str
        self.lut_path = None

        self._modtran_template_file_str = str
        self.modtran_template_file = None

        self._domain_type = DomainConfig
        self.domain = None

        self._lut_names_type = str
        self.lut_names = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()
        return errors

    def get_all_unknowns(self):
        return [self.H2O_ABSCO], ['H2O_ABSCO']



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

        self.model_name_type = str
        self.model_name = str

        self._statevector_type = StateVectorConfig
        self.statevector = None

        self._lut_grid_type = List
        self.lut_grid = None

        self._radiative_transfer_template_file_type = str
        self.radiative_transfer_template_file = None

        self._aerosol_template_file_type = str
        self.aerosol_template_file = None

        self._aerosol_model_file_type = str
        self.aerosol_model_file = None

        self._unknowns_type = RadiativeTransferUnknownsConfig
        self.unknowns = None

        #self._radiative_transfer_models_type = List
        #self.radiative_transfer_models = []

        self._vswir_model_type = RadiativeTransferModelConfig
        self.vswir_model = None

        self._tir_model_type = RadiativeTransferModelConfig
        self.tir_model = None

        #try:
        #    self._set_rt_config_options(sub_configdic['radiative_transfer_models'])
        #except KeyError:
        #    logging.debug('No radiative_transfer_models section found, skipping')

        self.set_config_options(sub_configdic)

    def _set_rt_config_options(self, configdict):
        for key in configdict:
            rt_model = RadiativeTransferModelConfig(configdict[key], name=key)
            self.radiative_transfer_models.append(rt_model)

    def get_radiative_transfer_model_names(self):
        name_list = []
        for rt_model in self.radiative_transfer_models:
            name_list.append(rt_model.model_name)
        return name_list

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_rt_models = ['modtran','libradtran','6s']
        if self.model_name not in valid_rt_models:
            errors.append('radiative_transfer->raditive_transfer_model: {} not in one of the available models: {}'.
                          format(self.model_name, valid_rt_models))


        #TODO: figure out submodule checking

        return errors

