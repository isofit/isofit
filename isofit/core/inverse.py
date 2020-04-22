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

from .common import svd_inv, svd_inv_sqrt, eps
from .inverse_simple import invert_simple


### Variables ###

error_code = -1


### Classes ###

class Inversion:

    def __init__(self, config, forward):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions."""

        self.lasttime = time.time()
        self.fm = forward
        self.method = 'GradientDescent'
        self.hashtable = OrderedDict()  # Hash table for caching inverse matrices
        self.max_table_size = 500
        self.windows = config['windows']  # Retrieval windows
        self.state_indep_S_hat = False
        self.simulation_mode = None
        if 'simulation_mode' in config:
            self.simulation_mode = config['simulation_mode']
        if 'Cressie_MAP_confidence' in config:
            self.state_indep_S_hat = config['Cressie_MAP_confidence']

        # We calculate the instrument channel indices associated with the
        # retrieval windows using the initial instrument calibration.  These
        # window indices never change throughout the life of the object.
        self.winidx = s.array((), dtype=int)  # indices of retrieval windows
        for lo, hi in self.windows:
            idx = s.where(s.logical_and(self.fm.instrument.wl_init > lo,
                                        self.fm.instrument.wl_init < hi))[0]
            self.winidx = s.concatenate((self.winidx, idx), axis=0)
        self.counts = 0
        self.inversions = 0

        # Configure Levenberg-Marquardt
        self.least_squares_params = {
            'method': 'trf',
            'max_nfev': 20,
            'bounds': (self.fm.bounds[0]+eps, self.fm.bounds[1]-eps),
            'x_scale': self.fm.scale,
            'xtol': 1e-4,
            'ftol': 1e-4,
            'gtol': 1e-4,
            'tr_solver': 'exact'
        }

        for k, v in config.items():
            if k in self.least_squares_params:
                self.least_squares_params[k] = v

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

        self.lasttime = time.time()
        self.trajectory = []
        self.counts = 0

        # Simulations are easy - return the initial state vector
        if self.simulation_mode or meas is None:
            return s.array([self.fm.init.copy()])

        # Calculate the initial solution, if needed.
        x0 = invert_simple(self.fm, meas, geom)

        # Catch any instances outside of bounds
        lower_bound_violation = x0 < self.fm.bounds[0]
        x0[lower_bound_violation] = self.fm.bounds[0][lower_bound_violation] + eps

        upper_bound_violation = x0 > self.fm.bounds[1]
        x0[upper_bound_violation] = self.fm.bounds[1][upper_bound_violation] - eps
        del lower_bound_violation, upper_bound_violation

        # Seps is the covariance of "observation noise" including both
        # measurement noise from the instrument as well as variability due to
        # unknown variables. For speed, we will calculate it just once based
        # on the initial solution (a potential minor source of inaccuracy).
        Seps_inv, Seps_inv_sqrt = self.calc_Seps(x0, meas, geom)

        def jac(x):
            """Calculate measurement Jacobian and prior Jacobians with 
            respect to cost function. This is the derivative of cost with
            respect to the state, commonly known as the gradient or loss
            surface. The cost is expressed as a vector of 'residuals'
            with respect to the prior and measurement, expressed in absolute
            (not quadratic) terms for the solver; It is the square root of
            the Rodgers (2000) Chi-square version. All measurement
            distributions are calculated over subwindows of the full spectrum."""

            # jacobian of measurment cost term WRT state vector.
            K = self.fm.K(x, geom)[self.winidx, :]
            meas_jac = Seps_inv_sqrt.dot(K)

            # jacobian of prior cost term with respect to state vector.
            xa, Sa, Sa_inv, Sa_inv_sqrt = self.calc_prior(x, geom)
            prior_jac = Sa_inv_sqrt

            # The total cost vector (as presented to the solver) is the
            # concatenation of the "residuals" due to the measurement
            # and prior distributions. They will be squared internally by
            # the solver.
            total_jac = s.concatenate((meas_jac, prior_jac), axis=0)

            return s.real(total_jac)

        def err(x):
            """Calculate cost function expressed here in absolute (not
            quadratic) terms for the solver, i.e., the square root of the 
            Rodgers (2000) Chi-square version. We concatenate 'residuals'
            due to measurment and prior terms, suitably scaled.
            All measurement distributions are calculated over subwindows 
            of the full spectrum."""

            # Measurement cost term.  Will calculate reflectance and Ls from
            # the state vector.
            est_meas = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
            est_meas_window = est_meas[self.winidx]
            meas_window = meas[self.winidx]
            meas_resid = (est_meas_window-meas_window).dot(Seps_inv_sqrt)

            # Prior cost term
            xa, Sa, Sa_inv, Sa_inv_sqrt = self.calc_prior(x, geom)
            prior_resid = (x - xa).dot(Sa_inv_sqrt)

            # Total cost
            total_resid = s.concatenate((meas_resid, prior_resid))

            # How long since the last call?
            newtime = time.time()
            secs = newtime-self.lasttime
            self.lasttime = newtime

            self.trajectory.append(x)

            it = len(self.trajectory)
            tm = newtime - self.lasttime
            rs = sum(pow(total_resid, 2))
            sm = self.fm.summarize(x, geom)
            logging.debug('Iteration: %02i  Residual: %12.2f %s' %
                          (it, rs, sm))

            return s.real(total_resid)

        # Initialize and invert
        try:
            xopt = least_squares(err, x0, jac=jac, **self.least_squares_params)
            self.trajectory.append(xopt.x)
        except LinAlgError:
            logging.warning('Levenberg Marquardt failed to converge')
        return s.array(self.trajectory)

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
