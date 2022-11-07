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
import os
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import block_diag, norm
from scipy.optimize import least_squares

from ..core.common import svd_inv, table_to_array
from .surface import Surface
from isofit.configs import Config


class MultiComponentSurface(Surface):
    """A model of the surface based on a collection of multivariate 
    Gaussians, with one or more equiprobable components and full 
    covariance matrices. 

    To evaluate the probability of a new spectrum, we calculate the
    Mahalanobis distance to each component cluster, and use that as our
    Multivariate Gaussian surface model.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        # TODO: inforce surface_file existence in the case of multicomponent_surface
        model_dict = loadmat(config.surface_file)
        self.components = list(zip(model_dict['means'], model_dict['covs']))
        self.n_comp = len(self.components)
        self.wl = model_dict['wl'][0]
        self.n_wl = len(self.wl)

        # Set up normalization method
        self.normalize = model_dict['normalize']
        if self.normalize == 'Euclidean':
            self.norm = lambda r: norm(r)
        elif self.normalize == 'RMS':
            self.norm = lambda r: np.sqrt(np.mean(pow(r, 2)))
        elif self.normalize == 'None':
            self.norm = lambda r: 1.0
        else:
            raise ValueError('Unrecognized Normalization: %s\n' %
                             self.normalize)

        self.selection_metric = config.selection_metric
        self.select_on_init = config.select_on_init

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = np.squeeze(model_dict['refwl'])
        self.idx_ref = [np.argmin(abs(self.wl-w))
                        for w in np.squeeze(self.refwl)]
        self.idx_ref = np.array(self.idx_ref)

        # Cache some important computations
        self.Covs, self.Cinvs, self.mus = [], [], []
        for i in range(self.n_comp):
            Cov = self.components[i][1]
            self.Covs.append(np.array([Cov[j, self.idx_ref]
                                       for j in self.idx_ref]))
            self.Cinvs.append(svd_inv(self.Covs[-1]))
            self.mus.append(self.components[i][0][self.idx_ref])

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ['RFL_%04i' % int(w) for w in self.wl]
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax-rmin)+rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

        # params needed for liquid water fitting
        self.lw_feature_left = np.argmin(abs(850 - self.wl))
        self.lw_feature_right = np.argmin(abs(1100 - self.wl))
        self.wl_sel = self.wl[self.lw_feature_left:self.lw_feature_right + 1]

        # init and bounds for liquid water as well as for
        # intercept and slope of the linear reflectance continuum
        self.lw_init = [0.02, 0.3, 0.0002]
        self.lw_bounds = [[0, 0.5], [0, 1.0], [-0.0004, 0.0004]]

        # load imaginary part of liquid water refractive index
        # and calculate wavelength dependent absorption coefficient
        # __file__ should live at isofit/isofit/surface/
        self.isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.path_k = os.path.join(self.isofit_path, "data", "iop", "k_liquid_water_ice.xlsx")
        self.k_wi = pd.read_excel(io=self.path_k, sheet_name='Sheet1', engine='openpyxl')
        self.wl_water, self.k_water = table_to_array(k_wi=self.k_wi, a=0, b=982, col_wvl="wvl_6", col_k="T = 20°C")
        self.kw = np.interp(x=self.wl_sel, xp=self.wl_water, fp=self.k_water)
        self.abs_co_w = 4 * np.pi * self.kw / self.wl_sel

    def component(self, x, geom):
        """We pick a surface model component using the Mahalanobis distance.

        This always uses the Lambertian (non-specular) version of the 
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the 
        geometry object.
        """

        if self.n_comp <= 1:
            return 0
        elif hasattr(geom, 'surf_cmp_init'):
            return geom.surf_cmp_init
        elif self.select_on_init and hasattr(geom, 'x_surf_init'):
            x_surface = geom.x_surf_init
        else:
            x_surface = x

        # Get the (possibly normalized) reflectance
        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        lamb_ref = lamb_ref / self.norm(lamb_ref)

        # Mahalanobis or Euclidean distances
        mds = []
        for ci in range(self.n_comp):
            ref_mu = self.mus[ci]
            ref_Cinv = self.Cinvs[ci]
            if self.selection_metric == 'Mahalanobis':
                md = (lamb_ref - ref_mu).T.dot(ref_Cinv).dot(lamb_ref - ref_mu)
            else:
                md = sum(pow(lamb_ref - ref_mu, 2))
            mds.append(md)
        closest = np.argmin(mds)

        if self.select_on_init and hasattr(geom, 'x_surf_init') and \
                (not hasattr(geom, 'surf_cmp_init')):
            geom.surf_cmp_init = closest
        return closest

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function. This always uses the
        Lambertian (non-specular) version of the surface reflectance."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        mu = np.zeros(self.n_state)
        ci = self.component(x_surface, geom)
        lamb_mu = self.components[ci][0]
        lamb_mu = lamb_mu * self.norm(lamb_ref)
        mu[self.idx_lamb] = lamb_mu
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        ci = self.component(x_surface, geom)
        Cov = self.components[ci][1]
        Cov = Cov * (self.norm(lamb_ref)**2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))
        return block_diag(Cov_prefix, Cov, Cov_suffix)

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError('Mismatched reflectances')
        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(self.bounds[i][0]+0.001,
                               min(self.bounds[i][1]-0.001, r))
        return x_surface

    def fit_liquid_water(self, rfl_meas):
        """Given a reflectance estimate, fit a state vector including liquid water
        based on a simple Beer-Lambert surface model."""

        rfl_meas_sel = rfl_meas[self.lw_feature_left:self.lw_feature_right+1]

        x_opt = least_squares(
            fun=self.err_obj,
            x0=self.lw_init,
            jac='2-point',
            method='trf',
            bounds=(np.array([self.bounds[ii][0] for ii in range(3)]),
                    np.array([self.bounds[ii][1] for ii in range(3)])),
            max_nfev=15,
            args=(rfl_meas_sel,)
        )

        return x_opt.x

    def err_obj(self, x, y):
        """Function, which computes the vector of residuals between measured and modeled surface reflectance optimizing
        for path length of surface liquid water based on the Beer-Lambert attenuation law."""

        attenuation = np.exp(-x[0] * 1e7 * self.abs_co_w)
        rho = (x[1] + x[2] * self.wl_sel) * attenuation
        resid = rho - y
        return resid

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
