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
"""Optimal-estimation inversion for ISOFIT.

Contains the `Inversion` class, which implements a nonlinear
least-squares retrieval using the Rodgers (2000) optimal-estimation
framework.  The solver minimizes the sum of a measurement residual term
(scaled by the observation-noise covariance) and a prior regularization
term (scaled by the prior covariance), iterating with a trust-region
reflective algorithm from `scipy.optimize`.

See Also:
    `isofit.inversion.inverse_mcmc` for a Bayesian MCMC alternative.
    `isofit.inversion.inverse_simple` for the heuristic initialiser.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares

from isofit.core.common import combos, conditional_gaussian, eps, svd_inv, svd_inv_sqrt
from isofit.inversion.inverse_simple import invert_simple

error_code = -1


class Inversion:
    """Nonlinear least-squares optimal-estimation inversion.

    Solves for the state vector (surface reflectance, atmospheric parameters,
    and instrument calibration) that best explains a measured radiance
    spectrum given prior knowledge encoded in the forward model.  The cost
    function is based on the Rodgers (2000) formalism and is minimised with
    a trust-region reflective (TRF) algorithm.

    When an *integration grid* is configured, the inversion is repeated for
    every grid point and the solution with the lowest final cost is returned.
    Grid points can either fix certain state-vector elements (the default)
    or be used as alternative starting points (`inversion_grid_as_preseed`).
    """

    def __init__(self, full_config: Config, forward: ForwardModel):
        """Initialise the inversion and compute retrieval-window channel indices.

        Args:
            full_config: Top-level ISOFIT configuration object.  Inversion
                settings are read from
                `full_config.implementation.inversion`.
            forward: Configured forward model that provides the Jacobian,
                prior, noise covariance, and measurement simulation.
        """

        config: InversionConfig = full_config.implementation.inversion
        self.config = config

        self.lasttime = time.time()
        self.fm = forward
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
                    self.fm.instrument.wl_init > lo, self.fm.instrument.wl_init < hi
                )
            )[0]
            self.winidx = np.concatenate((self.winidx, idx), axis=0)
        self.outside_ret_windows = np.ones(self.fm.n_meas, dtype=bool)
        self.outside_ret_windows[self.winidx] = False

        self.counts = 0
        self.inversions = 0

        self.integration_grid = OrderedDict(config.integration_grid)
        self.grid_as_starting_points = config.inversion_grid_as_preseed

        if self.grid_as_starting_points:
            # We're using the integration grid to preseed, not fix values.  So
            # Track the grid, but don't fix the integration grid points
            self.inds_fixed = []
            self.inds_preseed = np.array(
                [self.fm.statevec.index(k) for k in self.integration_grid.keys()]
            )
            self.inds_free = np.array(
                [
                    i
                    for i in np.arange(self.fm.nstate, dtype=int)
                    if not (i in self.inds_fixed)
                ]
            )

        else:
            # We're using the integration grid to fix values.  So
            # Get set up to fix the integration grid points
            self.inds_fixed = np.array(
                [self.fm.statevec.index(k) for k in self.integration_grid.keys()]
            )
            self.inds_free = np.array(
                [
                    i
                    for i in np.arange(self.fm.nstate, dtype=int)
                    if not (i in self.inds_fixed)
                ]
            )
            self.inds_preseed = []

        self.x_fixed = None

        # Set least squares params that come from the forward model
        self.least_squares_params = {
            "method": "trf",
            "max_nfev": 20,
            "bounds": (
                self.fm.bounds[0][self.inds_free],
                self.fm.bounds[1][self.inds_free],
            ),
            "x_scale": self.fm.scale[self.inds_free],
        }

        # Update the rest from the config
        for (
            key,
            item,
        ) in config.least_squares_params.get_config_options_as_dict().items():
            self.least_squares_params[key] = item

    def full_statevector(self, x_free):
        """Reconstruct the full state vector from the free (non-fixed) elements.

        Inserts `x_free` into the positions given by `self.inds_free` and
        fills any fixed positions from `self.x_fixed`.

        Args:
            x_free: Values for the free state-vector elements, shape `(n_free,)`.

        Returns:
            x: Complete state vector of length `nstate`.
        """
        x = np.zeros(self.fm.nstate)
        if self.x_fixed is not None:
            x[self.inds_fixed] = self.x_fixed
        x[self.inds_free] = x_free
        return x

    def calc_conditional_prior(self, x_free, geom):
        """Calculate the prior distribution conditioned on any fixed state-vector elements.

        When an integration grid is active (i.e., some state-vector elements
        are held fixed), the prior mean and covariance of the free elements
        are obtained by conditioning the joint prior on the fixed values.
        Otherwise the full prior is returned unchanged.

        Args:
            x_free: Current values of the free state-vector elements, shape `(n_free,)`.
            geom: Geometry object for the current observation.

        Returns:
            xa_free: Conditional prior mean for the free elements.
            Sa_free: Conditional prior covariance matrix for the free elements.
            Sa_free_inv: Inverse of `Sa_free`.
            Sa_free_inv_sqrt: Square root of `Sa_free_inv` (used for the
                residual-based cost formulation).
        """

        x = self.full_statevector(x_free)
        xa = self.fm.xa(x, geom)
        Sa, Sa_inv, Sa_inv_sqrt = self.fm.Sa(x, geom)

        # If there aren't any fixed parameters, we just directly
        if self.x_fixed is None or self.grid_as_starting_points:
            return xa, Sa, Sa_inv, Sa_inv_sqrt
        else:
            # otherwise condition on fixed variables
            # TODO: could make the below calculation without the svd_inv (using full initial inversion),
            # which would be way cheaper
            xa_free, Sa_free = conditional_gaussian(
                xa, Sa, self.inds_free, self.inds_fixed, self.x_fixed
            )
            Sa_free_inv, Sa_free_inv_sqrt = svd_inv_sqrt(
                Sa_free, hashtable=self.hashtable, max_hash_size=self.max_table_size
            )
            return xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt

    def calc_prior(self, x, geom):
        """Evaluate the prior distribution at a full state vector.

        Returns the prior mean and covariance (together with its pre-computed
        inverse and inverse square root) as provided by the forward model.

        Args:
            x: Full state vector, shape `(nstate,)`.
            geom: Geometry object for the current observation.

        Returns:
            xa: Prior mean state vector.
            Sa: Prior covariance matrix.
            Sa_inv: Inverse of `Sa`.
            Sa_inv_sqrt: Square root of `Sa_inv`.
        """

        xa = self.fm.xa(x, geom)
        Sa, Sa_inv, Sa_inv_sqrt = self.fm.Sa(x, geom)

        return xa, Sa, Sa_inv, Sa_inv_sqrt

    def calc_posterior(self, x, geom, meas):
        """Calculate the posterior distribution of the state vector.

        Computes the posterior covariance `S_hat`, the Jacobian `K`, and
        the gain matrix `G` following Rodgers (2000).  If
        `cressie_map_confidence` is enabled, `S_hat` is recomputed at the
        prior mean as suggested by Cressie (ASA 2018) for statistically
        consistent posterior confidence intervals.

        Args:
            x: Current state vector, shape `(nstate,)`.
            geom: Geometry object for the current observation.
            meas: Measured radiance in uW/nm/sr/cm2, shape `(nbands,)`.

        Returns:
            S_hat: Posterior covariance matrix of the state vector.
            K: Jacobian matrix `d meas / d x`, shape `(nbands, nstate)`.
            G: Gain matrix (`S_hat @ K.T @ Seps_inv`), shape `(nstate, nbands)`.
        """

        xa = self.fm.xa(x, geom)
        Sa, Sa_inv, Sa_inv_sqrt = self.fm.Sa(x, geom)
        K = self.fm.K(x, geom)
        Seps = self.fm.Seps(x, meas, geom)

        Seps_inv = svd_inv(
            Seps, hashtable=self.hashtable, max_hash_size=self.max_table_size
        )

        # Gain matrix G reflects current state, so we use the state-dependent
        # Jacobian matrix K
        S_hat = svd_inv(
            K.T.dot(Seps_inv).dot(K) + Sa_inv,
            hashtable=self.hashtable,
            max_hash_size=self.max_table_size,
        )
        G = S_hat.dot(K.T).dot(Seps_inv)

        # N. Cressie [ASA 2018] suggests an alternate definition of S_hat for
        # more statistically-consistent posterior confidence estimation
        if self.state_indep_S_hat:
            Ka = self.fm.K(xa, geom)
            S_hat = svd_inv(
                Ka.T.dot(Seps_inv).dot(Ka) + Sa_inv,
                hashtable=self.hashtable,
                max_hash_size=self.max_table_size,
            )
        return S_hat, K, G

    def calc_Seps(self, x, meas, geom):
        """Compute the observation-noise covariance restricted to the retrieval windows.

        `Seps` captures both instrument measurement noise and variability
        from unresolved forward-model parameters.  This method extracts the
        sub-matrix corresponding to `self.winidx` and returns its inverse
        and inverse square root for use in the cost function.

        Args:
            x: Current state vector, shape `(nstate,)`.
            meas: Measured radiance in uW/nm/sr/cm2, shape `(nbands,)`.
            geom: Geometry object for the current observation.

        Returns:
            Seps_inv: Inverse of the windowed noise covariance, shape `(n_win, n_win)`.
            Seps_inv_sqrt: Square root of `Seps_inv`, shape `(n_win, n_win)`.
        """

        Seps = self.fm.Seps(x, meas, geom)
        wn = len(self.winidx)
        Seps_win = np.zeros((wn, wn))
        for i in range(wn):
            Seps_win[i, :] = Seps[self.winidx[i], self.winidx]
        return svd_inv_sqrt(
            Seps_win, hashtable=self.hashtable, max_hash_size=self.max_table_size
        )

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
        xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt = self.calc_conditional_prior(
            x_free, geom
        )
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
        est_meas = self.fm.calc_meas(x, geom)
        est_meas_window = est_meas[self.winidx]
        meas_window = meas[self.winidx]
        meas_resid = (est_meas_window - meas_window).dot(Seps_inv_sqrt)

        # Prior cost term
        xa_free, Sa_free, Sa_free_inv, Sa_free_inv_sqrt = self.calc_conditional_prior(
            x_free, geom
        )
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
        if self.mode == "simulation":
            self.fm.surface.rfl = meas
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

            # Update regions outside retrieval windows to match priors
            if self.config.priors_in_initial_guess:
                prior_subset_idx = np.arange(len(x0))[self.fm.idx_surf_rfl][
                    self.outside_ret_windows
                ]
                x0[prior_subset_idx] = self.fm.surface.xa(x0, geom)[prior_subset_idx]

            trajectory.append(x0)

            x0 = x0[self.inds_free]

            # Catch any state vector elements outside of bounds
            lower_bound_violation = x0 < self.fm.bounds[0][self.inds_free]
            x0[lower_bound_violation] = (
                self.fm.bounds[0][self.inds_free][lower_bound_violation] + eps
            )

            upper_bound_violation = x0 > self.fm.bounds[1][self.inds_free]
            x0[upper_bound_violation] = (
                self.fm.bounds[1][self.inds_free][upper_bound_violation] - eps
            )
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
                logging.debug("Iteration: %02i  Residual: %12.2f %s" % (it, rs, sm))

                return np.real(residual)

            # Initialize and invert
            try:
                xopt = least_squares(err, x0, jac=jac, **self.least_squares_params)
                x_full_solution = self.full_statevector(xopt.x)
                trajectory.append(x_full_solution)
                solutions.append(trajectory)
                costs.append(np.sqrt(np.power(xopt.fun, 2).sum()))
            except scipy.linalg.LinAlgError:
                logging.warning("Optimization failed to converge")
                solutions.append(trajectory)
                costs.append(9e99)

        final_solution = np.array(solutions[np.argmin(costs)])
        return final_solution
