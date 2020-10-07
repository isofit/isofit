#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

from collections import OrderedDict
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection


class InversionConfig(BaseConfigSection):
    """
    Inversion configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._windows_type = list()
        self.windows = None
        """List[List[float]]: inversion retrieval windows to operate over."""

        self._cressie_map_confidence_type = bool
        self.cressie_map_confidence = False
        """bool: N. Cressie [ASA 2018] suggests an alternate definition of S_hat for
        more statistically-consistent posterior confidence estimation, this flag runs in this mode"""

        self._mcmc_type = McmcConfig
        self.mcmc = McmcConfig({})
        """MCMC parameters, only used if mode = mcmc."""

        self._integration_grid_type = OrderedDict
        self.integration_grid = OrderedDict({})
        """Grid of inversion points to execute if mode='grid'.  Either fixed, or starting points, depending
        on self.fixed_inversion_grid"""

        self._inversion_grid_as_preseed_type = bool
        self.inversion_grid_as_preseed = False
        """Parameter indicating whether to treat the inversion grid as:
         (True) - a series of seeds for the optimization (variable by the optimization algorithm).
         (False) - a set of fixed points (not variable by the optimization algorithm)
        """

        self._least_squares_params_type = LeastSquaresConfig
        self.least_squares_params = LeastSquaresConfig({})
        """
        Least squares parameters for core inversion solve.  
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        for details.
        """

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add some checking to windows
        if self.windows is None and self.mode != 'simulation':
            errors.append('windows is a required parameters inside of inversion')
        else:
            for subset in self.windows:
                if isinstance(subset, List) is False:
                    errors.append('windows parameter must be a list of lists of wavelength ranges')
                elif subset[0] > subset[1]:
                    errors.append('In inversion window subset {}, wavelength ranges must be in order'.format(subset))
        
        return errors


class McmcConfig(BaseConfigSection):
    """
    MCMC inversion configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._iterations_type = int
        self.iterations = 10000
        """int: Number of MCMC iterations to run."""

        self._burnin_type = int
        self.burnin = 200

        self._regularizer_type = float
        self.regularizer = 1e-3

        self._proposal_scaling_type = float
        self.proposal_scaling = 0.01

        self._verbose_type = bool
        self.verbose = True

        self._restart_every_type = int
        self.restart_every = 2000

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add flags for rile overright, and make sure files don't exist if not checked?

        return errors


class LeastSquaresConfig(BaseConfigSection):
    """
    Least squares config parameters.
    """

    def __init__(self, sub_configdic: dict = None):
        self._method_type = str
        self.method = 'trf'
        """str: Optimzation method to use. Default 'trf'."""

        self._max_nfev_type = int
        self.max_nfev = 20
        """int: Maximum number of function evaluations before the termination. 
        If None (default), the value is chosen automatically. Default 20."""

        self._xtol_type = float
        self.xtol = None
        """float: Tolerance for termination by the change of the independent variables.  
        Default is None, which disables termination from this criteria."""

        self._ftol_type = float
        self.ftol = 0.01
        """float: Tolerance for termination by the change of the cost function.
        Default is 0.01"""

        self._gtol_type = float
        self.gtol = None
        """float: Tolerance for termination by the norm of the gradient. 
        Default is None, which disables termination from this criteria."""

        self._tr_solver_type = str
        self.tr_solver = 'lsmr'
        """str: Method for solving trust-region subproblems, relevant only for ‘trf’ and ‘dogbox’ methods.
        Options are None, 'exact', or 'lsmr'"""

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        # TODO: add flags for rile overright, and make sure files don't exist if not checked?
        if self.tr_solver is not None:
            if self.tr_solver not in [None, 'exact', 'lsmr']:
                errors.append('Least squares tr_solver must be in [None, \'exact\', \'lsmr\']')

        return errors


