from .inverse import Inversion
from .inverse_mcmc import MCMCInversion

Inversions = {
    "mcmc_inversion": MCMCInversion,
    "inversion": Inversion,
    "simulation": Inversion,
}
