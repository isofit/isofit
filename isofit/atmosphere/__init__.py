from isofit.atmosphere.atmosphere import BaseAtmosphere, Keys
from isofit.atmosphere.engines import Engines


class Atmosphere:
    def __new__(self, full_config):
        engine_name = full_config.forward_model.atmosphere.engine_name
        engine = Engines.get(engine_name)

        # Always default to base atmosphere
        engine = engine or BaseAtmosphere

        return engine(full_config)
