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
#

import numpy as np
from scipy.linalg import block_diag, norm
from scipy.io import loadmat
from scipy.stats import multivariate_normal

from ..core.common import svd_inv
from .surface import Surface
from isofit.configs import Config


class Gaussian:

    def __init__(self,mean,covs):
        self.mean = mean
        self.covs = covs

    def pdf(self,x):
        return multivariate_normal.pdf(x, self.mean, self.covs)

    def log_pdf(self,x):
        return multivariate_normal.log_pdf(x, self.mean, self.covs)

    def grad_x_log_pdf(self,x):
        return -1.*np.linalg.solve(self.covs, (x - self.mean).T).T


class MixtureOfGaussianSurface(Surface):
    """A model of the surface based on a mixture of guassians.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        # TODO: inforce surface_file existence in the case of multicomponent_surface
        model_dict = loadmat(config.surface_file)
        self.mean = model_dict['mean']
        self.covs = model_dict['covs']
        self.weights = model_dict['weights']
        self.n_comp = model_dict['n_components']
        self.components = [Gaussian(self.mean[i], self.covs[i]) for i in range(self.n_comp)]
        self.wl = model_dict['wl'][0]
        self.n_wl = len(self.wl)

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = np.squeeze(model_dict['refwl'])
        self.idx_ref = [np.argmin(abs(self.wl-w))
                        for w in np.squeeze(self.refwl)]
        self.idx_ref = np.array(self.idx_ref)

        ## Cache some important computations
        #self.Covs, self.Cinvs, self.mus = [], [], []
        #for i in range(self.n_comp):
        #    Cov = self.components[i][1]
        #    self.Covs.append(np.array([Cov[j, self.idx_ref]
        #                               for j in self.idx_ref]))
        #    self.Cinvs.append(svd_inv(self.Covs[-1]))
        #    self.mus.append(self.components[i][0][self.idx_ref])

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ['RFL_%04i' % int(w) for w in self.wl]
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax-rmin)+rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

    def pdf(self, x):
        return np.sum(self.weights * self.component_pdfs(x), axis=1)

    def log_pdf(self, x):
        return np.log(self.pdf(x))

    def component_pdfs(self, x):
        return np.array([self.components[i].pdf(x) for i in range(self.n_comp)]).T

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance."""

        return self.calc_lamb(x_surface, geom)

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        return x_surface[self.idx_lamb]

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface."""

        dlamb = np.eye(self.n_wl, dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = self.n_state - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dlamb, suffix), axis=1)

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_wl), dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dLs, suffix), axis=1)

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ''
        return 'Component: %i' % self.component(x_surface, geom)
