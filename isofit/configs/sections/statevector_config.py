
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
import logging


class StateVectorElementConfig(BaseConfigSection):
    """
    State vector element configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._bounds_type = list()
        self.bounds = None

        self._scale_type = float
        self.scale = None

        self._prior_mean_type = float
        self.prior_mean = None

        self._prior_sigma_type = float
        self.prior_sigma = None

        self._init_type = float
        self.init = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        return errors

class StateVectorConfig(BaseConfigSection):
    """
    State vector configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._H2OSTR_type = StateVectorElementConfig
        self.H2OSTR: StateVectorElementConfig = None

        self._AOT550_type = StateVectorElementConfig
        self.AOT550: StateVectorElementConfig = None

        self._AERFRAC_1_type = StateVectorElementConfig
        self.AERFRAC_1: StateVectorElementConfig = None

        self._AERFRAC_2_type = StateVectorElementConfig
        self.AERFRAC_2: StateVectorElementConfig = None

        self._AERFRAC_3_type = StateVectorElementConfig
        self.AERFRAC_3: StateVectorElementConfig = None

        assert(len(self.get_elements()) == len(self.__dict__)/2)

        self._set_statevector_config_options(sub_configdic)

    def _check_config_validity(self):
        errors = list()

        return errors

    def _set_statevector_config_options(self, configdic):
        #TODO: update using methods below
        if configdic is not None:
            for key in configdic:
                sv = StateVectorElementConfig(configdic[key])
                setattr(self, key, sv)

    def get_elements(self):
        return [self.H2OSTR, self.AOT550, self.AERFRAC_1, self.AERFRAC_2, self.AERFRAC_3]

    def get_all_bounds(self):
        bounds = []
        for element in self.get_elements():
            bounds.append(element.bounds)
        return bounds

    def get_all_scales(self):
        scales = []
        for element in self.get_elements():
            scales.append(element.scale)
        return scales

    def get_all_inits(self):
        inits = []
        for element in self.get_elements():
            inits.append(element.init)
        return inits

    def get_all_prior_means(self):
        prior_means = []
        for element in self.get_elements():
            prior_means.append(element.prior_mean)
        return prior_means

    def get_all_prior_sigmas(self):
        prior_sigmas = []
        for element in self.get_elements():
            prior_sigmas.append(element.prior_sigma)
        return prior_sigmas




