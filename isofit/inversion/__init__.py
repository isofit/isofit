from isofit.inversion.inverse import Inversion as ClassicInversion
from isofit.inversion.inverse_mcmc import MCMCInversion


def Inversion(config, fm):
    """
    Retrieves the correct Inversion model to initialize and returns
    """
    if config.implementation.mode == "mcmc_inversion":
        return MCMCInversion(config, fm)

    elif config.implementation.mode in ("inversion", "simulation"):
        return ClassicInversion(config, fm)

    else:
        # This should never be reached due to configuration checking
        raise AttributeError("Config implementation mode node valid")
