from collections import OrderedDict

import numpy as np

from isofit.configs import Config
from isofit.configs.sections.implementation_config import InversionConfig
from isofit.core.forward import ForwardModel


class Inversion:
    def __init__(self, full_config: Config, forward: ForwardModel):
        """Initialization for inversion including:
        retrieval subwindows for calculating measurement cost distributions.
        """

        # Initialize things that are shared across pixels
        self.full_config = full_config
        config: InversionConfig = full_config.implementation.inversion
        self.config = config

        # Moved from inverse.py - Things that aren't contingent on statevec
        self.hashtable = OrderedDict()  # Hash table for caching inverse matrices
        self.max_table_size = full_config.implementation.max_hash_table_size
        self.state_indep_S_hat = False

        self.windows = config.windows  # Retrieval windows
        self.mode = full_config.implementation.mode
        self.state_indep_S_hat = config.cressie_map_confidence

        # We calculate the instrument channel indices associated with the
        # retrieval windows using the initial instrument calibration.  These
        # window indices never change throughout the life of the object.
        self.winidx = np.array((), dtype=int)  # indices of retrieval windows
        for lo, hi in self.windows:
            idx = np.where(
                np.logical_and(
                    forward.instrument.wl_init > lo, forward.instrument.wl_init < hi
                )
            )[0]
            self.winidx = np.concatenate((self.winidx, idx), axis=0)

        self.outside_ret_windows = np.ones(forward.instrument.n_chan, dtype=bool)
        self.outside_ret_windows[self.winidx] = False

        self.counts = 0
        self.inversions = 0

        self.integration_grid = OrderedDict(config.integration_grid)
        self.grid_as_starting_points = config.inversion_grid_as_preseed

    def construct_inverse(self, forward: ForwardModel):
        # This can't be imported on init. Introduces circular package error
        from isofit.inversion import Inversions

        iv = Inversions.get(self.full_config.implementation.mode, None)

        return iv(self.full_config, forward)
