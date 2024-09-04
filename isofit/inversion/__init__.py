from .inverse import Inverse
from .inverse_mcmc import MCMCInversion

Inversions = {
    "mcmc_inversion": MCMCInversion,
    "inversion": Inverse,
    "simulation": Inverse,
}
