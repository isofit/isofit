
from collections import OrderedDict
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection

class InversionConfig(BaseConfigSection):
    """
    Inversion configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._windows_type = list()
        self.windows = None

        self._simulation_mode_type = bool
        self.simulation_mode = False

        self._state_indep_S_hat_type = bool
        self.state_indep_S_hat = False

        self.set_config_options(sub_configdic)


    def _check_config_validity(self) -> List[str]:
        errors = list()

        #TODO: add some checking to windows

        #TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors


class McmcInversionConfig(BaseConfigSection):
    """
    Inversion configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._iterations_type = int
        self.interations = 10000

        self._burnin_type = int
        self.burnin = 200

        self._method_type = str
        self.method = 'MCMC'

        self._regularizer_type = float
        self.regularizer = 1e-3

        self._proposal_scaling_type = 0.01
        self.proposal_scaling = 0.01

        self._verbose_type = bool
        self.verbose = True

        self._restart_every_type = int
        self.restart_every = 2000

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors


class GridInversionConfig(BaseConfigSection):
    """
    Inversion configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._integration_grid_type = OrderedDict
        self.integration_grid = None

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors
