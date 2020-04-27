
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.instrument_config import InstrumentConfig
from isofit.configs.sections.surface_config import SurfaceConfig
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferConfig



class ForwardModelConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._instrument_type = InstrumentConfig
        self.instrument: InstrumentConfig = None
        """
        Instrument: instrument config section. 
        """

        self._surface_type = SurfaceConfig
        self.surface: SurfaceConfig = None
        """
        Instrument: instrument config section. 
        """

        self._radiative_transfer_type = RadiativeTransferConfig
        self.radiative_transfer: RadiativeTransferConfig = None
        """
        RadiativeTransfer: radiative transfer config section.
        """
        self.set_config_options(sub_configdic)


    def _check_config_validity(self) -> List[str]:
        errors = list()

        #TODO: figure out submodule checking

        return errors

