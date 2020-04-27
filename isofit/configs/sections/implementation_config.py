
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

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        valid_implementation_modes = ['inversion','grid_inversion','mcmc_inversion']
        if self.mode not in valid_implementation_modes:
            errors.append('Invalid implementation mode: {}.  Valid options are: {}'.
                          format(self.mode, valid_implementation_modes))

        #TODO: recursive implmentation

        return errors
