
from typing import Dict, List, Type
from isofit.configs.configs import BaseConfigSection
from isofit.configs.sections import InstrumentConfig, SurfaceConfig, RadiativeTransferConfig



class ForwardModelConfig(BaseConfigSection):
    """
    Forward model configuration.
    """

    instrument = None
    _instrument_type = InstrumentConfig
    """
    Instrument: instrument config section. 
    """

    surface = None
    _surface_type = SurfaceConfig
    """
    Instrument: instrument config section. 
    """

    radiative_transfer = None
    _radiative_transfer_type = RadiativeTransferConfig
    """
    RadiativeTransfer: radiative transfer config section.
    """


    def _check_config_validity(self) -> List[str]:
        self.get_option_keys()
        errors = list()

        #TODO: figure out submodule checking

        return errors

