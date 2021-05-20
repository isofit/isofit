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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import time
import logging
import numpy as np
from collections import OrderedDict
from scipy.optimize import least_squares
import scipy.linalg

from isofit.core.common import svd_inv, svd_inv_sqrt, eps, combos, conditional_gaussian
from .inverse_simple import invert_simple
from isofit.configs import Config
from isofit.core.forward import ForwardModel
from isofit.configs.sections.implementation_config import InversionConfig


### Variables ###

error_code = -1


### Classes ###

class Inversion:

    def __init__(self, full_config: Config, forward: ForwardModel):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions."""

        config: InversionConfig = full_config.implementation.inversion

        self.lasttime = time.time()
        self.fm = forward
        self.hashtable = OrderedDict()  # Hash table for caching inverse matrices
        self.max_table_size = 500
        self.state_indep_S_hat = False

        self.windows = config.windows  # Retrieval windows
        self.mode = full_config.implementation.mode
        self.state_indep_S_hat = config.cressie_map_confidence

        # We calculate the instrument channel indices associated with the
        # retrieval windows using the initial instrument calibration.  These
        # window indices never change throughout the life of the object.
        self.winidx = np.array((), dtype=int)  # indices of retrieval windows
        for lo, hi in self.windows:
            idx = np.where(np.logical_and(self.fm.instrument.wl_init > lo,
                                          self.fm.instrument.wl_init < hi))[0]
            self.winidx = np.concatenate((self.winidx, idx), axis=0)
        self.counts = 0
        self.inversions = 0

        self.integration_grid = OrderedDict(config.integration_grid)
        self.grid_as_starting_points = config.inversion_grid_as_preseed

        if self.grid_as_starting_points:
            # We're using the integration grid to preseed, not fix values.  So
            # Track the grid, but don't fix the integration grid points
            self.inds_fixed = []
            self.inds_preseed = np.array([self.fm.statevec.index(k) for k in
                               self.integration_grid.keys()])
            self.inds_free = np.array([i for i in np.arange(self.fm.nstate, dtype=int) if
                              not (i in self.inds_fixed)])

        else:
            # We're using the integration grid to fix values.  So
            # Get set up to fix the integration grid points
            self.inds_fixed = np.array([self.fm.statevec.index(k) for k in
                               self.integration_grid.keys()])
            self.inds_free = np.array([i for i in np.arange(self.fm.nstate, dtype=int) if
                              not (i in self.inds_fixed)])
            self.inds_preseed = []

        self.x_fixed = None

        # Set least squares params that come from the forward model
        self.least_squares_params = {
            'method': 'trf',
            'max_nfev': 20,
            'bounds': (self.fm.bounds[0][self.inds_free],
                       self.fm.bounds[1][self.inds_free]),
            'x_scale': self.fm.scale[self.inds_free],
        }

        # Update the rest from the config
        for key, item in config.least_squares_params.get_config_options_as_dict().items():
            self.least_squares_params[key] = item

    def full_statevector(self, x_free):
        x = np.zeros(self.fm.nstate)
        if self.x_fixed is not None:
            x[self.inds_fixed] = self.x_fixed
        x[self.inds_free] = x_free
        return x

    def calc_conditional_prior(self, x_free, geom):
        """Calculate prior distribution of radiance. This depends on the
        location in the state space. Return the inverse covariance and
        its square root (for non-quadratic error residual calculation)."""

        x = self.full_statevector(x_free)
        xa = self.fm.xa(x, geom)
        Sa = self.fm.Sa(x, geom)

        # If there aren't any fixed parameters, we just directly
        if self.x_fixed is None or self.grid_as_starting_points:
            Sa_inv, Sa_inv_sqrt = svd_inv_sqrt(Sa, hashtable=self.hashtable)
            return xa, Sa, Sa_inv, Sa_inv_sqrt

        else:
            # otherwise condition on fixed variables
            #TODO: could make the below calculation without the svd_inv (using full initial inversion),
            # which would be way cheaper
            xa_free, Sa_free = conditional_gaussian(xa, Sa, self.inds_free,
                                                    self.inds_fixed, self.x_fixed)
            Sa_free_inv, Sa_free_inv_sqrt = svd_inv_sqrt(Sa_free,
                                                         hashtable=self.hashtable)
            return xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt


    def calc_prior(self, x, geom):
        """Calculate prior distribution of radiance. This depends on the 
        location in the state space. Return the inverse covariance and 
        its square root (for non-quadratic error residual calculation)."""

        xa = self.fm.xa(x, geom)
        Sa = self.fm.Sa(x, geom)
        Sa_inv, Sa_inv_sqrt = svd_inv_sqrt(Sa, hashtable=self.hashtable)
        return xa, Sa, Sa_inv, Sa_inv_sqrt

    def calc_posterior(self, x, geom, meas):
        """Calculate posterior distribution of state vector. This depends 
        both on the location in the state space and the radiance (via noise)."""

        xa = self.fm.xa(x, geom)
        Sa = self.fm.Sa(x, geom)
        Sa_inv = svd_inv(Sa, hashtable=self.hashtable)
        K = self.fm.K(x, geom)
        Seps = self.fm.Seps(x, meas, geom)
        Seps_inv = svd_inv(Seps, hashtable=self.hashtable)

        # Gain matrix G reflects current state, so we use the state-dependent
        # Jacobian matrix K
        S_hat = svd_inv(K.T.dot(Seps_inv).dot(
            K) + Sa_inv, hashtable=self.hashtable)
        G = S_hat.dot(K.T).dot(Seps_inv)

        # N. Cressie [ASA 2018] suggests an alternate definition of S_hat for
        # more statistically-consistent posterior confidence estimation
        if self.state_indep_S_hat:
            Ka = self.fm.K(xa, geom)
            S_hat = svd_inv(Ka.T.dot(Seps_inv).dot(Ka) + Sa_inv,
                            hashtable=self.hashtable)
        return S_hat, K, G

    def calc_Seps(self, x, meas, geom):
        """Calculate (zero-mean) measurement distribution in radiance terms.
        This depends on the location in the state space. This distribution is 
        calculated over one or more subwindows of the spectrum. Return the 
        inverse covariance and its square root."""

        Seps = self.fm.Seps(x, meas, geom)
        wn = len(self.winidx)
        Seps_win = np.zeros((wn, wn))
        for i in range(wn):
            Seps_win[i, :] = Seps[self.winidx[i], self.winidx]
        return svd_inv_sqrt(Seps_win, hashtable=self.hashtable)


    def jacobian(self, x_free, geom, Seps_inv_sqrt) -> np.ndarray:
        """Calculate measurement Jacobian and prior Jacobians with
        respect to cost function. This is the derivative of cost with
        respect to the state, commonly known as the gradient or loss
        surface. The cost is expressed as a vector of 'residuals'
        with respect to the prior and measurement, expressed in absolute
        (not quadratic) terms for the solver; It is the square root of
        the Rodgers (2000) Chi-square version. All measurement
        distributions are calculated over subwindows of the full
        spectrum.
        Args:
            x_free: decision variables - portion of the statevector not fixed by a static integration grid
            geom: Geometry to use for inversion
            Seps_inv_sqrt: Inverse square root of the covariance of "observation noise",
             including both measurement noise from the instrument as well as variability due to
             unknown variables.

        Returns:
            total_jac: The complete (measurement and prior) jacobian
        """
        x = self.full_statevector(x_free)

        # jacobian of measurment cost term WRT full state vector.
        K = self.fm.K(x, geom)[self.winidx, :]
        K = K[:, self.inds_free]
        meas_jac = Seps_inv_sqrt.dot(K)

        # jacobian of prior cost term with respect to state vector.
        xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt = \
            self.calc_conditional_prior(x_free, geom)
        prior_jac = Sa_free_inv_sqrt

        # The total cost vector (as presented to the solver) is the
        # concatenation of the "residuals" due to the measurement
        # and prior distributions. They will be squared internally by
        # the solver.
        total_jac = np.real(np.concatenate((meas_jac, prior_jac), axis=0))

        return total_jac

    def loss_function(self, x_free, geom, Seps_inv_sqrt, meas) -> (np.array, np.array):
        """Calculate cost function expressed here in absolute (not
        quadratic) terms for the solver, i.e., the square root of the
        Rodgers (2000) Chi-square version. We concatenate 'residuals'
        due to measurment and prior terms, suitably scaled.
        All measurement distributions are calculated over subwindows
        of the full spectrum.

        Args:
            x_free: decision variables - portion of the statevector not fixed by a static integration grid
            geom: Geometry to use for inversion
            Seps_inv_sqrt: Inverse square root of the covariance of "observation noise",
             including both measurement noise from the instrument as well as variability due to
             unknown variables.
            meas: a one-D scipy vector of radiance in uW/nm/sr/cm2

        Returns:
            total_residual: the complete, calculated residual
            x: the complete (x_free + any x_fixed augmented in)
        """

        # set up full-sized state vector
        x = self.full_statevector(x_free)

        # Measurement cost term.  Will calculate reflectance and Ls from
        # the state vector.
        est_meas = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
        est_meas_window = est_meas[self.winidx]
        meas_window = meas[self.winidx]
        meas_resid = (est_meas_window - meas_window).dot(Seps_inv_sqrt)

        # Prior cost term
        xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt = \
            self.calc_conditional_prior(x_free, geom)
        prior_resid = (x_free - xa_free).dot(Sa_free_inv_sqrt)

        # Total cost
        total_resid = np.concatenate((meas_resid, prior_resid))

        return np.real(total_resid), x

    def invert(self, meas, geom):
        """Inverts a meaurement and returns a state vector.
        Args:
            meas: a one-D scipy vector of radiance in uW/nm/sr/cm2
            geom: a geometry object

        Returns:
            final_solution: a converged state vector solution
        """
        self.counts = 0
        costs, solutions = [], []

        # Simulations are easy - return the initial state vector
        if self.mode == 'simulation' or meas is None:
            return np.array([self.fm.init.copy()])

        if len(self.integration_grid.values()) == 0:
            combo_values = [None]
        else:
            combo_values = combos(self.integration_grid.values()).copy()

        for combo in combo_values:

            if self.grid_as_starting_points is False:
                self.x_fixed = combo
            trajectory = []

            # Calculate the initial solution, if needed.
            x0 = invert_simple(self.fm, meas, geom)
            x0 = x0[self.inds_free]

            # Catch any state vector elements outside of bounds
            lower_bound_violation = x0 < self.fm.bounds[0][self.inds_free]
            x0[lower_bound_violation] = \
                self.fm.bounds[0][self.inds_free][lower_bound_violation] + eps

            upper_bound_violation = x0 > self.fm.bounds[1][self.inds_free]
            x0[upper_bound_violation] = \
                self.fm.bounds[1][self.inds_free][upper_bound_violation] - eps
            del lower_bound_violation, upper_bound_violation

            # Find the full state vector with bounds checked
            x = self.full_statevector(x0)

            # Regardless of anything we did for the heuristic guess, bring the
            # static preseed back into play (only does anything if inds_preseed
            # is not blank)
            if len(self.inds_preseed) > 0:
                x0[self.inds_preseed] = combo

            # Record initializaation state
            geom.x_surf_init = x[self.fm.idx_surface]
            geom.x_RT_init = x[self.fm.idx_RT]

            # Seps is the covariance of "observation noise" including both
            # measurement noise from the instrument as well as variability due to
            # unknown variables. For speed, we will calculate it just once based
            # on the initial solution (a potential minor source of inaccuracy).
            Seps_inv, Seps_inv_sqrt = self.calc_Seps(x, meas, geom)

            def jac(x_free):
                """Short wrapper function for use with scipy opt"""
                return self.jacobian(x_free, geom, Seps_inv_sqrt)

            def err(x_free):
                """Short wrapper function for use with scipy opt and logging"""
                residual, x = self.loss_function(x_free, geom, Seps_inv_sqrt, meas)

                trajectory.append(x)

                it = len(trajectory)
                rs = float(np.sum(np.power(residual, 2)))
                sm = self.fm.summarize(x, geom)
                logging.debug('Iteration: %02i  Residual: %12.2f %s' %
                              (it, rs, sm))

                return np.real(residual)

            # Initialize and invert
            try:
                xopt = least_squares(err, x0, jac=jac, **self.least_squares_params)
                x_full_solution = self.full_statevector(xopt.x)
                trajectory.append(x_full_solution)
                solutions.append(trajectory)
                costs.append(np.sqrt(np.power(xopt.fun, 2).sum()))
            except scipy.linalg.LinAlgError:
                logging.warning('Optimization failed to converge')
                solutions.append(trajectory)
                costs.append(9e99)

        final_solution = np.array(solutions[np.argmin(costs)])
        return final_solution


    def forward_uncertainty(self, x, meas, geom):
        """
        Args:
            x: statevector
            meas: a one-D scipy vector of radiance in uW/nm/sr/cm2
            geom: a geometry object
        Returns:
            lamb: the converged lambertian surface reflectance
            path: the converged path radiance estimate
            mdl: the modeled radiance estimate
            S_hat: the posterior covariance of the state vector
            K: the derivative matrix d meas_x / d state_x
            G: the G matrix from the CD Rodgers 2000 formalism
        """

        dark_surface = np.zeros(self.fm.surface.wl.shape)
        path = self.fm.calc_meas(x, geom, rfl=dark_surface)
        mdl = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
        lamb = self.fm.calc_lamb(x, geom)
        S_hat, K, G = self.calc_posterior(x, geom, meas)
        return lamb, mdl, path, S_hat, K, G
