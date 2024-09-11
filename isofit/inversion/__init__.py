from .inverse_mcmc import MCMCInversion
from .inversion import Inversion

Inversions = {
    "mcmc_inversion": MCMCInversion,
    "inversion": Inversion,
    "simulation": Inversion,
}
