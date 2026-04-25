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

"""Markov Chain Monte Carlo (MCMC) inversion for ISOFIT.

Provides a Metropolis-Hastings MCMC sampler that draws samples from the
posterior distribution of the state vector given a measured radiance.
This is an alternative to the gradient-based `Inversion` solver
and is useful for uncertainty quantification.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.stats

from isofit.core.common import eps
from isofit.inversion.inverse import Inversion


class MCMCInversion(Inversion):
    """Inversion via Markov Chain Monte Carlo sampling.

    Draws samples from the posterior distribution of the state vector using
    a Metropolis-Hastings algorithm.  The sampler is initialized at the
    MAP (maximum a posteriori) solution found by the parent
    `Inversion` solver and then explores
    the posterior using a Gaussian proposal distribution derived from the
    posterior covariance.
    """

    def __init__(self, full_config: Config, forward: ForwardModel):
        """Initialize the MCMC inversion and apply configuration defaults.

        Args:
            full_config: Top-level ISOFIT configuration object.  MCMC
                settings are read from
                `full_config.implementation.inversion.mcmc`.
            forward: Configured forward model used to evaluate the
                likelihood of proposed state vectors.
        """

        Inversion.__init__(self, full_config, forward)
        mcmc_config = full_config.implementation.inversion.mcmc

        self.iterations = mcmc_config.iterations
        self.burnin = mcmc_config.burnin
        self.regularizer = mcmc_config.regularizer
        self.proposal_scaling = mcmc_config.proposal_scaling
        self.verbose = mcmc_config.verbose
        self.restart_every = mcmc_config.restart_every

    def stable_mvnpdf(self, mean, cov, x):
        """Evaluate a multivariate Gaussian log-PDF using a numerically stable SVD inverse.

        Uses Singular Value Decomposition and retains only eigenvectors
        corresponding to strictly positive eigenvalues, avoiding numerical
        issues with near-singular covariance matrices.

        Args:
            mean: Mean vector of the distribution, shape `(n,)`.
            cov: Covariance matrix of the distribution, shape `(n, n)`.
            x: Query point at which to evaluate the log-PDF, shape `(n,)`.

        Returns:
            log_pdf: Scalar log-probability density at ``x``.
        """

        U, V, D = scipy.linalg.svd(cov)
        use = np.where(V > 0)[0]
        Cinv = (D[use, :].T).dot(np.diag(1.0 / V[use])).dot(U[:, use].T)
        logCdet = np.sum(np.log(2.0 * np.pi * V[use]))

        # Multivariate Gaussian PDF
        lead = -0.5 * logCdet
        dist = (x - mean)[:, np.newaxis]
        diverg = -0.5 * (dist.T).dot(Cinv).dot(dist)
        return lead + diverg

    def log_density(self, x, rdn_meas, geom, bounds):
        """Compute the unnormalized log posterior density for a state vector.

        The log posterior is the sum of the log prior and the log data
        likelihood.  Returns ``-inf`` immediately if any element of ``x``
        violates ``bounds``.

        Args:
            x: Candidate state vector, shape `(nstate,)`.
            rdn_meas: Measured radiance in uW/nm/sr/cm², shape `(nbands,)`.
            geom: Geometry object for the current observation.
            bounds: Two-row array of lower and upper bounds on the state
                vector, shape `(2, nstate)`, or `None` to disable
                bounds checking.

        Returns:
            log_dens: Scalar unnormalized log posterior density, or
            `-inf` if `x` is out of bounds.
        """

        # First check bounds
        if bounds is not None and any(np.logical_or(x < bounds[0], x > bounds[1])):
            return -np.Inf

        # Prior term
        Sa, _, _ = self.fm.Sa(x, geom)
        xa = self.fm.xa(x, geom)
        pa = self.stable_mvnpdf(xa, Sa, x)

        # Data likelihood term
        Seps = self.fm.Seps(x, rdn_meas, geom)
        Seps_win = np.array([Seps[i, self.winidx] for i in self.winidx])

        # Get RT quantities
        rdn_est = self.fm.calc_meas(x=x, geom=geom)
        pm = self.stable_mvnpdf(rdn_est[self.winidx], Seps_win, rdn_meas[self.winidx])
        return pa + pm

    def invert(self, rdn_meas, geom):
        """Invert a measured radiance and return posterior state vector samples.

        Runs the Metropolis-Hastings MCMC sampler.  The chain is initialized
        at the MAP solution from `Inversion.invert()`
        and periodically restarted to guard against getting stuck in
        low-probability regions.  Samples collected before each burnin
        period are discarded.

        Args:
            rdn_meas: Measured radiance in uW/nm/sr/cm², shape `(nbands,)`.
            geom: Geometry object for the current observation.

        Returns:
            samples: Array of accepted state vector samples drawn from the
            posterior distribution, shape `(iterations, nstate)`.
        """

        # We will truncate non-surface parameters to their bounds, but leave
        # Surface reflectance unconstrained so it can dip slightly below zero
        # in a channel without invalidating the whole vector
        bounds = np.array([self.fm.bounds[0].copy(), self.fm.bounds[1].copy()])
        bounds[:, self.fm.idx_surface] = np.array([[-np.inf], [np.inf]])

        # Initialize to conjugate gradient solution
        x_MAP = Inversion.invert(self, rdn_meas, geom)[-1]

        # Proposal is based on the posterior uncertainty
        S_hat, K, G = self.calc_posterior(x_MAP, geom, rdn_meas)
        proposal_Cov = S_hat * self.proposal_scaling
        proposal = scipy.stats.multivariate_normal(cov=proposal_Cov)

        # We will use this routine for initializing
        def initialize():
            x = scipy.stats.multivariate_normal(mean=x_MAP, cov=S_hat).rvs()
            too_low = x < bounds[0]
            x[too_low] = bounds[0][too_low] + eps
            too_high = x > bounds[1]
            x[too_high] = bounds[1][too_high] - eps
            dens = self.log_density(x, rdn_meas, geom, bounds)
            return x, dens

        # Sample from the posterior using Metropolis/Hastings MCMC
        samples, acpts, rejs, x = [], 0, 0, None
        for i in range(self.iterations):
            if i % self.restart_every == 0:
                x, dens = initialize()

            xp = x + proposal.rvs()
            dens_new = self.log_density(xp, rdn_meas, geom, bounds=bounds)

            # Test vs. the Metropolis / Hastings criterion
            if np.isfinite(dens_new) and np.log(np.random.rand()) <= min(
                (dens_new - dens, 0.0)
            ):
                x = xp
                dens = dens_new
                acpts = acpts + 1
                if self.verbose:
                    print(
                        "%8.5e %8.5e ACCEPT! rate %4.2f"
                        % (dens, dens_new, np.mean(acpts / (acpts + rejs)))
                    )
            else:
                rejs = rejs + 1
                if self.verbose:
                    print(
                        "%8.5e %8.5e REJECT  rate %4.2f"
                        % (dens, dens_new, np.mean(acpts / (acpts + rejs)))
                    )

            # Make sure we have not wandered off the map
            if not np.isfinite(dens_new):
                x, dens = initialize()

            if i % self.restart_every < self.burnin:
                samples.append(x)

        return np.array(samples)
