
from typing import Dict, List, Type
from isofit.configs import BaseConfigSection
from isofit.configs.sections.instrument_config import InstrumentConfig
#from isofit.configs.sections import SurfaceConfig, RadiativeTransferConfig



class ForwardModelConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self.instrument = None
        self._instrument_type = InstrumentConfig
        """
        Instrument: instrument config section. 
        """

        #self.surface = None
        #self._surface_type = SurfaceConfig
        """
        Instrument: instrument config section. 
        """

        #self.radiative_transfer = None
        #self._radiative_transfer_type = RadiativeTransferConfig
        """
        RadiativeTransfer: radiative transfer config section.
        """

        self.set_config_options(sub_configdic)


    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        #TODO: figure out submodule checking

        return errors

