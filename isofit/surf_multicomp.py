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

import scipy as s
from scipy.io import loadmat, savemat
from common import recursive_replace, emissive_radiance, svd_inv, eps
from scipy.linalg import block_diag, det, norm, pinv, sqrtm, inv
from surf import Surface


class MultiComponentSurface(Surface):
    """A model of the surface based on a collection of multivariate 
       Gaussians, with one or more equiprobable components and full 
       covariance matrices. 

       To evaluate the probability of a new spectrum, we calculate the
       Mahalanobis distance to each component cluster, and use that as our
       Multivariate Gaussian surface model."""

    def __init__(self, config, RT):

        Surface.__init__(self, config, RT)
        # Models are stored as dictionaries in .mat format
        model_dict = loadmat(config['surface_file'])
        self.components = list(zip(model_dict['means'], model_dict['covs']))
        self.ncomp = len(self.components)

        # Set up normalization method
        self.normalize = model_dict['normalize']
        if self.normalize == 'Euclidean':
            self.norm = lambda r: norm(r)
        elif self.normalize == 'RMS':
            self.norm = lambda r: s.sqrt(s.mean(pow(r, 2)))
        elif self.normalize == 'None':
            self.norm = lambda r: 1.0
        else:
            raise ValueError('Unrecognized Normalization: %s\n' %
                             self.normalize)

        try:
            self.selection_metric = config['selection_metric']
        except KeyError:
            self.selection_metric = 'Mahalanobis'

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = s.squeeze(model_dict['refwl'])
        self.refidx = [s.argmin(abs(self.wl-w)) for w in s.squeeze(self.refwl)]
        self.refidx = s.array(self.refidx)

        # Cache some important computations
        self.Covs, self.Cinvs, self.mus = [], [], []
        for i in range(self.ncomp):
            Cov = self.components[i][1]
            self.Covs.append(s.array([Cov[j, self.refidx]
                                      for j in self.refidx]))
            self.Cinvs.append(svd_inv(self.Covs[-1]))
            self.mus.append(self.components[i][0][self.refidx])

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 1.2
        self.statevec = ['RFL_%04i' % int(w) for w in self.wl]
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init_val = [0.15 * (rmax-rmin)+rmin for v in self.wl]
        self.lrfl_inds = s.arange(len(self.wl))

    def component(self, x_surface, geom):
        """ We pick a surface model component using the Mahalanobis distance.
            This always uses the Lambertian (non-specular) version of the 
            surface reflectance."""

        if len(self.components) <= 1:
            return 0

        # Get the (possibly normalized) reflectance
        lrfl = self.calc_lrfl(x_surface, geom)
        ref_lrfl = lrfl[self.refidx]
        ref_lrfl = ref_lrfl / self.norm(ref_lrfl)

        # Mahalanobis or Euclidean distances
        mds = []
        for ci in range(self.ncomp):
            ref_mu = self.mus[ci]
            ref_Cinv = self.Cinvs[ci]
            if self.selection_metric == 'Mahalanobis':
                md = (ref_lrfl - ref_mu).T.dot(ref_Cinv).dot(ref_lrfl - ref_mu)
            else:
                md = sum(pow(ref_lrfl - ref_mu, 2))
            mds.append(md)
        closest = s.argmin(mds)
        return closest

    def xa(self, x_surface, geom):
        '''Mean of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function.  This always uses the
        Lambertian (non-specular) version of the surface reflectance.'''

        lrfl = self.calc_lrfl(x_surface, geom)
        ref_lrfl = lrfl[self.refidx]
        mu = s.zeros(len(self.statevec))
        ci = self.component(x_surface, geom)
        mu_lrfl = self.components[ci][0]
        mu_lrfl = mu_lrfl * self.norm(ref_lrfl)
        mu[self.lrfl_inds] = mu_lrfl
        return mu

    def Sa(self, x_surface, geom):
        '''Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function.'''

        lrfl = self.calc_lrfl(x_surface, geom)
        ref_lrfl = lrfl[self.refidx]

        ci = self.component(x_surface, geom)
        Cov = self.components[ci][1]
        Cov = Cov * (self.norm(ref_lrfl)**2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec) == len(self.lrfl_inds):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.lrfl_inds[0]
        nsuffix = len(self.statevec)-self.lrfl_inds[-1]-1
        Cov_prefix = s.zeros((nprefix, nprefix))
        Cov_suffix = s.zeros((nsuffix, nsuffix))
        return block_diag(Cov_prefix, Cov, Cov_suffix)

    def heuristic_surface(self, rfl_meas, Ls, geom):
        '''Given a reflectance estimate, fit a state vector.'''

        x_surface = s.zeros(len(self.statevec))
        if len(rfl_meas) != len(self.lrfl_inds):
            raise ValueError('Mismatched reflectances')
        for i, r in zip(self.lrfl_inds, rfl_meas):
            x_surface[i] = max(self.bounds[i][0]+0.001,
                               min(self.bounds[i][1]-0.001, r))
        return x_surface

    def calc_rfl(self, x_surface, geom):
        '''Non-Lambertian reflectance'''

        return self.calc_lrfl(x_surface, geom)

    def calc_lrfl(self, x_surface, geom):
        '''Lambertian reflectance'''

        return x_surface[self.lrfl_inds]

    def drfl_dx(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface.'''

        return self.dlrfl_dx(x_surface, geom)

    def dlrfl_dx(self, x_surface, geom):
        '''Partial derivative of Lambertian reflectance with respect to 
           state vector, calculated at x_surface.'''
        dlrfl = s.eye(self.nwl, dtype=float)
        nprefix = self.lrfl_inds[0]
        nsuffix = len(self.statevec)-self.lrfl_inds[-1]-1
        prefix = s.zeros((self.nwl, nprefix))
        suffix = s.zeros((self.nwl, nsuffix))
        return s.concatenate((prefix, dlrfl, suffix), axis=1)

    def calc_Ls(self, x_surface, geom):
        '''Emission of surface, as a radiance'''
        return s.zeros(self.nwl, dtype=float)

    def dLs_dx(self, x_surface, geom):
        '''Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface.'''
        dLs = s.zeros((self.nwl, self.nwl), dtype=float)
        nprefix = self.lrfl_inds[0]
        nsuffix = len(self.statevec)-self.lrfl_inds[-1]-1
        prefix = s.zeros((self.nwl, nprefix))
        suffix = s.zeros((self.nwl, nsuffix))
        return s.concatenate((prefix, dLs, suffix), axis=1)

    def summarize(self, x_surface, geom):
        '''Summary of state vector'''
        return 'Component: %i' % self.component(x_surface, geom)
