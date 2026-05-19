import logging

from isofit.atmosphere.atmosphere import BaseAtmosphere, Keys
from isofit.atmosphere.engines import Engines

Logger = logging.getLogger(__name__)


class Atmosphere:
    def __new__(self, full_config):
        engine_name = full_config.forward_model.atmosphere.engine_name
        engine = Engines.get(engine_name)

        # Always default to base atmosphere
        engine = engine or BaseAtmosphere

        Logger.debug(f"Using atmosphere class: {engine.__name__}")
        return engine(full_config)
