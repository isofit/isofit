import logging
from copy import deepcopy

import numpy as np


class StateVector:
    def __init__(self, instrument, RT, surface):
        if surface.n_wl != len(RT.wl) or not np.all(
            np.isclose(surface.wl, RT.wl, atol=0.01)
        ):
            Logger.warning(
                "Surface and RTM wavelengths differ - if running at higher RTM"
                " spectral resolution or with variable wavelength position, this"
                " is expected.  Otherwise, consider checking the surface model."
            )

        # Build combined vectors from surface, RT, and instrument
        bounds, scale, init, statevec, bvec, bval = ([] for i in range(6))
        for obj_with_statevec in [surface, RT, instrument]:
            bounds.extend([deepcopy(x) for x in obj_with_statevec.bounds])
            scale.extend([deepcopy(x) for x in obj_with_statevec.scale])
            init.extend([deepcopy(x) for x in obj_with_statevec.init])
            statevec.extend([deepcopy(x) for x in obj_with_statevec.statevec_names])

            bvec.extend([deepcopy(x) for x in obj_with_statevec.bvec])
            bval.extend([deepcopy(x) for x in obj_with_statevec.bval])

        # Persist to class object
        self.bounds = tuple(np.array(bounds).T)
        self.scale = np.array(scale)
        self.init = np.array(init)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        self.bvec = np.array(bvec)
        self.nbvec = len(self.bvec)
        self.bval = np.array(bval)
        self.Sb = np.diagflat(np.power(self.bval, 2))

        """Set up state vector indices - 
        MUST MATCH ORDER FROM ABOVE ASSIGNMENT"""
        self.idx_surface = np.arange(len(surface.statevec_names), dtype=int)

        # surface reflectance portion
        self.idx_surf_rfl = self.idx_surface[: len(surface.idx_lamb)]

        # non-reflectance surface parameters
        self.idx_surf_nonrfl = self.idx_surface[len(surface.idx_lamb) :]

        # radiative transfer portion
        self.idx_RT = np.arange(len(RT.statevec_names), dtype=int) + len(
            self.idx_surface
        )

        # instrument portion
        self.idx_instrument = (
            np.arange(len(instrument.statevec_names), dtype=int)
            + len(self.idx_surface)
            + len(self.idx_RT)
        )

        self.surface_b_inds = np.arange(len(surface.bvec), dtype=int)

        self.RT_b_inds = np.arange(len(RT.bvec), dtype=int) + len(self.surface_b_inds)

        self.instrument_b_inds = (
            np.arange(len(instrument.bvec), dtype=int)
            + len(self.surface_b_inds)
            + len(self.RT_b_inds)
        )
