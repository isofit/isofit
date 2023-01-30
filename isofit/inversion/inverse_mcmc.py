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

import numpy as np
import scipy.linalg
import scipy.stats

from isofit.configs import Config
from isofit.core.common import eps
from isofit.core.forward import ForwardModel

from .inverse import Inversion


class MCMCInversion(Inversion):
    def __init__(self, full_config: Config, forward: ForwardModel):
        """Initialize and apply defaults."""

        Inversion.__init__(self, full_config, forward)
        mcmc_config = full_config.implementation.inversion.mcmc

        self.iterations = mcmc_config.iterations
        self.burnin = mcmc_config.burnin
        self.regularizer = mcmc_config.regularizer
        self.proposal_scaling = mcmc_config.proposal_scaling
        self.verbose = mcmc_config.verbose
        self.restart_every = mcmc_config.restart_every

    def stable_mvnpdf(self, mean, cov, x):
        """Stable inverse via Singular Value Decomposition, using only the significant eigenvectors."""

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
        """Log probability density combines prior and likelihood terms."""

        # First check bounds
        if bounds is not None and any(np.logical_or(x < bounds[0], x > bounds[1])):
            return -np.Inf

        # Prior term
        Sa = self.fm.Sa(x, geom)
        xa = self.fm.xa(x, geom)
        pa = self.stable_mvnpdf(xa, Sa, x)

        # Data likelihood term
        Seps = self.fm.Seps(x, rdn_meas, geom)
        Seps_win = np.array([Seps[i, self.winidx] for i in self.winidx])
        rdn_est = self.fm.calc_rdn(x, geom, rfl=None, Ls=None)
        pm = self.stable_mvnpdf(rdn_est[self.winidx], Seps_win, rdn_meas[self.winidx])
        return pa + pm

    def invert(self, rdn_meas, geom):
        """Inverts a meaurement. Returns an array of state vector samples.
        Similar to Inversion.invert() but returns a list of samples."""

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
