from .atmosphere import Atmosphere
from isofit.atmosphere.engines import Engines


class Atmospheres:
    def __new__(full_config):
        engine = Engines.get(
            full_config.forward_model.atmosphere.engine_name,
            Atmosphere
        )
        return engine(full_config)
