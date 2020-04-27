
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

        self._sigma_mean_type = float
        self.sigma_mean = None

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

        self.set_statevector_config_options(sub_configdic)

    def set_statevector_config_options(self, configdic):
        if configdic is not None:
            for key in configdic:
                sv = StateVectorElementConfig(configdic[key])
                setattr(self, key, sv)



