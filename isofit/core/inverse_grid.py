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

import sys
import time
import logging
import xxhash
import scipy as s
from collections import OrderedDict
from scipy.optimize import least_squares
from scipy.linalg import inv, norm, det, cholesky, qr, svd, LinAlgError
from hashlib import md5
from numba import jit

from .common import svd_inv, svd_inv_sqrt, eps, combos, conditional_gaussian
from .inverse_simple import invert_simple
from .inverse import Inversion


### Variables ###

error_code = -1


### Classes ###

class GridInversion(Inversion):

    def __init__(self, config, forward):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions."""

        super().__init__(config, forward)
        self.integration_grid = OrderedDict(config['integration_grid'])
        self.inds_fixed = [self.fm.statevec.index(k) for k in
                           self.integration_grid.keys()]
        self.inds_free = [i for i in s.arange(self.fm.nstate, dtype=int) if
                          not (i in self.inds_fixed)]
        self.x_fixed = None
        self.least_squares_params['bounds'] = \
            (self.fm.bounds[0][self.inds_free]+eps,
             self.fm.bounds[1][self.inds_free]-eps)
        self.least_squares_params['x_scale'] = \
            self.fm.scale[self.inds_free]

    def full_statevector(self, x_free):
        x = s.zeros(self.fm.nstate)
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

        # condition on fixed variables
        xa_free, Sa_free = conditional_gaussian(xa, Sa, self.inds_free,
                                                self.inds_fixed, self.x_fixed)
        Sa_free_inv, Sa_free_inv_sqrt = svd_inv_sqrt(Sa_free,
                                                     hashtable=self.hashtable)
        return xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt

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
        Seps_win = s.zeros((wn, wn))
        for i in range(wn):
            Seps_win[i, :] = Seps[self.winidx[i], self.winidx]
        return svd_inv_sqrt(Seps_win, hashtable=self.hashtable)

    def invert(self, meas, geom):
        """Inverts a meaurement and returns a state vector.

           Parameters:

             meas           - a one-D scipy vector of radiance in uW/nm/sr/cm2
             geom           - a geometry object.
             plot_directory - the base directory to which diagnostics are
                                writtena preinitialized ForwardModel object

           Returns a tuple consisting of the following:

             lamb           - the converged lambertian surface reflectance
             path           - the converged path radiance estimate
             mdl            - the modeled radiance estimate
             S_hat          - the posterior covariance of the state vector
             G              - the G matrix from the CD Rodgers 2000 formalism
             xopt           - the converged state vector solution"""

        self.counts = 0
        costs, solutions = [], []

        # Simulations are easy - return the initial state vector
        if self.simulation_mode or meas is None:
            return s.array([self.fm.init.copy()])

        for combo in combos(self.integration_grid.values()):

            self.x_fixed = combo.copy()
            trajectory = []

            # Calculate the initial solution, if needed.
            x0 = invert_simple(self.fm, meas, geom)
            x0 = x0[self.inds_free]
            x = self.full_statevector(x0)

            # Record initializaation state
            geom.x_surf_init = x[self.fm.idx_surface]
            geom.x_RT_init = x[self.fm.idx_RT]

            # Seps is the covariance of "observation noise" including both
            # measurement noise from the instrument as well as variability due to
            # unknown variables. For speed, we will calculate it just once based
            # on the initial solution (a potential minor source of inaccuracy).
            Seps_inv, Seps_inv_sqrt = self.calc_Seps(x, meas, geom)

            def jac(x_free):
                """Calculate measurement Jacobian and prior Jacobians with 
                respect to cost function. This is the derivative of cost with
                respect to the state, commonly known as the gradient or loss
                surface. The cost is expressed as a vector of 'residuals'
                with respect to the prior and measurement, expressed in absolute
                (not quadratic) terms for the solver; It is the square root of
                the Rodgers (2000) Chi-square version. All measurement
                distributions are calculated over subwindows of the full 
                spectrum."""

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
                total_jac = s.concatenate((meas_jac, prior_jac), axis=0)

                return s.real(total_jac)

            def err(x_free):
                """Calculate cost function expressed here in absolute (not
                quadratic) terms for the solver, i.e., the square root of the 
                Rodgers (2000) Chi-square version. We concatenate 'residuals'
                due to measurment and prior terms, suitably scaled.
                All measurement distributions are calculated over subwindows 
                of the full spectrum."""

                # set up full-sized state vector
                x = self.full_statevector(x_free)

                # Measurement cost term.  Will calculate reflectance and Ls from
                # the state vector.
                est_meas = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
                est_meas_window = est_meas[self.winidx]
                meas_window = meas[self.winidx]
                meas_resid = (est_meas_window-meas_window).dot(Seps_inv_sqrt)

                # Prior cost term
                xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt = \
                    self.calc_conditional_prior(x_free, geom)
                prior_resid = (x_free - xa_free).dot(Sa_free_inv_sqrt)

                # Total cost
                total_resid = s.concatenate((meas_resid, prior_resid))

                trajectory.append(x)

                it = len(trajectory)
                rs = sum(pow(total_resid, 2))
                sm = self.fm.summarize(x, geom)
                logging.debug('Iteration: %02i  Residual: %12.2f %s' %
                              (it, rs, sm))

                return s.real(total_resid)

            # Initialize and invert
            try:
                xopt = least_squares(err, x0, jac=jac,
                                     **self.least_squares_params)
                x_full_solution = self.full_statevector(xopt.x)
                trajectory.append(x_full_solution)
                solutions.append(trajectory)
                costs.append(s.sqrt(pow(xopt.fun, 2).sum()))
            except LinAlgError:
                logging.warning('Levenberg Marquardt failed to converge')
                solutions.append(trajectory)
                costs.append(9e99)
        return s.array(solutions[s.argmin(costs)])

    def forward_uncertainty(self, x, meas, geom):
        """Converged estimates of path radiance, radiance, reflectance.

        Also calculate the posterior distribution and Rodgers K, G matrices.
        """

        dark_surface = s.zeros(self.fm.surface.wl.shape)
        path = self.fm.calc_meas(x, geom, rfl=dark_surface)
        mdl = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
        lamb = self.fm.calc_lamb(x, geom)
        S_hat, K, G = self.calc_posterior(x, geom, meas)
        return lamb, mdl, path, S_hat, K, G
