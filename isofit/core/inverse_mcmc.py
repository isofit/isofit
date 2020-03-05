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
import scipy as s
import scipy.optimize
from scipy.linalg import inv, norm, sqrtm, det, cholesky, qr, svd
from scipy.stats import multivariate_normal
from hashlib import md5

from .common import eps
from .inverse import Inversion


class MCMCInversion(Inversion):

    def __init__(self, config, forward):
        """Initialize and apply defaults."""

        Inversion.__init__(self, config, forward)
        defaults = {
            'iterations': 10000,
            'burnin': 200,
            'method': 'MCMC',
            'regularizer': 1e-3,
            'proposal_scaling': 0.01,
            'verbose': True,
            'restart_every': 2000
        }
        for key, val in defaults.items():
            if key in config:
                setattr(self, key, config[key])
            else:
                setattr(self, key, val)

    def stable_mvnpdf(self, mean, cov, x):
        """Stable inverse via Singular Value Decomposition, using only the significant eigenvectors."""

        U, V, D = svd(cov)
        use = s.where(V > 0)[0]
        Cinv = (D[use, :].T).dot(s.diag(1.0/V[use])).dot(U[:, use].T)
        logCdet = s.sum(s.log(2.0 * s.pi * V[use]))

        # Multivariate Gaussian PDF
        lead = -0.5 * logCdet
        dist = (x-mean)[:, s.newaxis]
        diverg = -0.5 * (dist.T).dot(Cinv).dot(dist)
        return lead + diverg

    def log_density(self, x, rdn_meas, geom, bounds):
        """Log probability density combines prior and likelihood terms."""

        # First check bounds
        if bounds is not None and any(s.logical_or(x < bounds[0], x > bounds[1])):
            return -s.Inf

        # Prior term
        Sa = self.fm.Sa(x, geom)
        xa = self.fm.xa(x, geom)
        pa = self.stable_mvnpdf(xa, Sa, x)

        # Data likelihood term
        Seps = self.fm.Seps(x, rdn_meas, geom)
        Seps_win = s.array([Seps[i, self.winidx] for i in self.winidx])
        rdn_est = self.fm.calc_rdn(x, geom, rfl=None, Ls=None)
        pm = self.stable_mvnpdf(rdn_est[self.winidx], Seps_win,
                                rdn_meas[self.winidx])
        return pa+pm

    def invert(self, rdn_meas, geom):
        """Inverts a meaurement. Returns an array of state vector samples.
           Similar to Inversion.invert() but returns a list of samples."""

        # We will truncate non-surface parameters to their bounds, but leave
        # Surface reflectance unconstrained so it can dip slightly below zero
        # in a channel without invalidating the whole vector
        bounds = s.array([self.fm.bounds[0].copy(), self.fm.bounds[1].copy()])
        bounds[:, self.fm.idx_surface] = s.array([[-s.inf], [s.inf]])

        # Initialize to conjugate gradient solution
        x_MAP = Inversion.invert(self, rdn_meas, geom)[-1]

        # Proposal is based on the posterior uncertainty
        S_hat, K, G = self.calc_posterior(x_MAP, geom, rdn_meas)
        proposal_Cov = S_hat * self.proposal_scaling
        proposal = multivariate_normal(cov=proposal_Cov)

        # We will use this routine for initializing
        def initialize():
            x = multivariate_normal(mean=x_MAP, cov=S_hat).rvs()
            too_low = x < bounds[0]
            x[too_low] = bounds[0][too_low]+eps
            too_high = x > bounds[1]
            x[too_high] = bounds[1][too_high]-eps
            dens = self.log_density(x, rdn_meas, geom, bounds)
            return x, dens

        # Sample from the posterior using Metropolis/Hastings MCMC
        samples, acpts, rejs, x = [], 0, 0, None
        for i in range(self.iterations):

            if i % self.restart_every == 0:
                x, dens = initialize()

            xp = x + proposal.rvs()
            dens_new = self.log_density(xp, rdn_meas,  geom, bounds=bounds)

            # Test vs. the Metropolis / Hastings criterion
            if s.isfinite(dens_new) and\
                    s.log(s.rand()) <= min((dens_new - dens, 0.0)):
                x = xp
                dens = dens_new
                acpts = acpts + 1
                if self.verbose:
                    print('%8.5e %8.5e ACCEPT! rate %4.2f' %
                          (dens, dens_new, s.mean(acpts/(acpts+rejs))))
            else:
                rejs = rejs + 1
                if self.verbose:
                    print('%8.5e %8.5e REJECT  rate %4.2f' %
                          (dens, dens_new, s.mean(acpts/(acpts+rejs))))

            # Make sure we have not wandered off the map
            if not s.isfinite(dens_new):
                x, dens = initialize()

            if i % self.restart_every < self.burnin:
                samples.append(x)

        return s.array(samples)
