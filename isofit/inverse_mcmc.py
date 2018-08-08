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
from common import spectrumLoad, chol_inv, eps
from inverse import Inversion
import scipy.optimize
from scipy.linalg import inv, norm, sqrtm, det, cholesky, qr, svd
from scipy.stats import multivariate_normal
from hashlib import md5


class MCMCInversion(Inversion):

    def __init__(self, config, forward):
        """Initialize and apply defaults"""
        Inversion.__init__(self, config, forward)

        defaults = {'iterations': 10000, 'burnin': 2000,
                    'regularizer': 1e-3, 'proposal_scaling': 0.01,
                    'verbose': True}

        for key, val in defaults.items():
            if key in config:
                setattr(self, key, config[key])
            else:
                setattr(self, key, val)

    def density(self, x, rdn_meas,  geom):
        """Probability density combines prior and likelihood terms"""

        # Prior distribution
        Sa = self.fm.Sa(x, geom)
        xa = self.fm.xa(x, geom)

        # Stable inverse via Singular Value Decomposition, using only the
        # significant eigenvectors
        U, V, D = svd(Sa)
        maxeig = max(V)
        mineig = maxeig * 1e-6
        use = s.where(V > mineig)[0]
        SaInv = (D[use, :].T).dot(s.diag(V[use])).dot(U[:, use].T)
        SaDet = s.prod(V[use])

        # Probability density of prior
        z = s.sqrt(SaDet * 2.0 * s.pi)
        pa = z * ((x-xa)[s.newaxis, :]).dot(SaInv).dot((x-xa)[:, s.newaxis])

        # Probability density of measurement noise distribution
        # i.e. the likelihood
        Seps = self.fm.Seps(rdn_meas, geom)
        Seps_win = s.array([Seps[i, self.winidx] for i in self.winidx])
        est_rdn = self.fm.calc_rdn(x, geom, rfl=None, Ls=None)
        pm = multivariate_normal(mean=rdn_meas[self.winidx], cov=Seps_win)

        # Complete posterior density is proportional to prior term times
        # the likelihood term
        return pa * pm.pdf(est_rdn[self.winidx])

    def invert(self, rdn_meas, geom, out=None, init=None):
        """Inverts a meaurement. Returns an array of state vector samples.
           Similar to Inversion.invert() but returns a list of samples."""

        # Initialize to conjugate gradient solution
        init = Inversion.invert(self, rdn_meas, geom, out, init)
        x = init.copy()
        dens = self.density(x, rdn_meas,  geom)

        # Proposal is based on the posterior uncertainty
        # We truncate non-surface parameters to their bounds
        bounds = s.array([self.fm.bounds[0].copy(), self.fm.bounds[1].copy()])
        bounds[:, self.fm.surface_inds] = s.array([[-s.inf], [s.inf]])
        S_hat, K, G = self.calc_posterior(x, geom, rdn_meas)
        proposal_Cov = S_hat * self.proposal_scaling
        proposal = multivariate_normal(cov=proposal_Cov)

        # Sample from the posterior using Metropolis/Hastings MCMC
        samples, acpts, rejs = [], 0, 0
        for i in range(self.iterations):
            xp = s.ones(x.shape) * s.inf
            max_tries, count = 10000, 0
            while any(xp < bounds[0, :]) or any(xp > bounds[1, :]):
                xp = x + proposal.rvs()
                count = count + 1
                if count > max_tries:
                    raise RuntimeError(
                        'Could not generate proposal distribution')
            dens_new = self.density(xp, rdn_meas,  geom)

            # Test vs. the Metropolis / Hastings criterion
            if s.isfinite(dens_new) and \
                    dens_new > 0 and \
                    s.rand() <= min((dens_new / dens, 1.0)):
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
            if not s.isfinite(dens_new) or dens_new < 0:
                x = init.copy()
                dens = self.density(x, rdn_meas,  geom)
            samples.append(x)

        return init.copy(), s.array(samples)


if __name__ == '__main__':
    main()
