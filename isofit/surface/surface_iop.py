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
from scipy.linalg import block_diag
from scipy.interpolate import interp1d

from .surface import Surface
from ..core.common import load_wavelen, eps, srf


class IOPSurface(Surface):
    """A model of the surface based on a collection of multivariate 
       Gaussians, extended with a surface glint term."""

    def __init__(self, config):
        """."""

        Surface.__init__(self, config)
        self.wl, fwhm = load_wavelen(config['wavelength_file'])
        self.statevec, self.bounds, self.scale, self.init = [], [], [], []

        # Each channel maps to a nonnegative absorption residual
        if 'absorption_resid_file' in config:
            abs_file = config['absorption_resid_file']
            self.C = loadmat(abs_file)['C']
            self.abs_inds = s.arange(len(self.wl))
            amin, amax = -1.0, 1.0
            self.statevec.extend(['ABS_%04i' % int(w) for w in self.wl])
            self.bounds.extend([[amin, amax] for w in self.wl])
            self.scale.extend([0.01 for w in self.wl])
            self.init.extend([0 for v in self.wl])
            ind_start = len(self.statevec)
        else:
            self.abs_inds = []
            self.C = s.array([[]], dtype=float)
            ind_start = 0

        # Other retrieved variables
        nonabs_sv = ['X', 'G', 'P', 'Y', 'GLINT', 'FLH']
        self.statevec.extend(nonabs_sv)
        self.nonabs_inds = ind_start + s.arange(len(nonabs_sv), dtype=int)
        self.X_ind = ind_start
        self.G_ind = ind_start+1
        self.P_ind = ind_start+2
        self.Y_ind = ind_start+3
        self.glint_ind = ind_start+4
        self.flh_ind = ind_start+5
        self.scale.extend([0.1, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.init.extend([0.1, 0.1, 0.1, 0.1, 0.1, 0.01])
        self.bounds.extend(
            [[0, 1.0], [0, 1.0], [0, 10], [0, 2.5], [0, 1], [0, 10]])

        aw_wl, aw, q, q2 = s.loadtxt(config['h2oabs_file'], comments='#').T
        self.aw = interp1d(aw_wl, aw, fill_value='extrapolate')(self.wl)
        bbw_wl, bbw, q = s.loadtxt(config['h2oscatter_file'], comments='#').T
        self.bbw = interp1d(bbw_wl, bbw, fill_value='extrapolate')(self.wl)
        ap_wl, ap1, ap2 = s.loadtxt(config['pigments_file'], comments='#').T
        aphi = [interp1d(ap_wl, ap1, fill_value='extrapolate')(self.wl),
                interp1d(ap_wl, ap2, fill_value='extrapolate')(self.wl)]
        self.aphi_coeffs = s.array(aphi).T

        self.g0 = config.get('G0', 0.0895)  # Lee's first constant
        self.g1 = config.get('G1', 0.1247)  # Lee's second constant
        self.b1000 = s.argmin(abs(self.wl-1000))
        self.b900 = s.argmin(abs(900-self.wl))
        self.b440 = s.argmin(abs(self.wl-440))
        self.b490 = s.argmin(abs(self.wl-490))
        self.b550 = s.argmin(abs(self.wl-550))
        self.b640 = s.argmin(abs(self.wl-640))

        # Phytoplankton fluorescence peak center and width
        self.fl_mu = config.get('fl_mu', 683.0)
        self.fl_sigma = config.get('fl_fwhm', 25.0)/2.355

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        return s.array(self.init)

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        scales = s.array(self.scale)
        if len(self.abs_inds) > 0:
            Sa = block_diag(self.C, s.diag(pow(scales[self.nonabs_inds], 2)))
        else:
            Sa = s.diag(pow(scales[self.nonabs_inds], 2))
        return Sa

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters, 
        fit a state vector."""

        init = s.array((self.init))
        init[self.glint_ind] = min(max(rfl_meas[self.b1000],
                                       self.bounds[self.glint_ind][0]+eps),
                                   self.bounds[self.glint_ind][1]-eps)
        init[self.Y_ind] = 1.0
        init[self.X_ind] = 0.1
        init[self.P_ind] = 0.1
        init[self.G_ind] = 0.1
        init[self.flh_ind] = eps*2
        return init

    def qaa(self, x_surface):
        """."""

        X, G, P, Y, glint, flh = x_surface[self.nonabs_inds]

        # total backscatter from particle and water contributions
        bbp = X * pow(400.0/self.wl, Y)
        bb = self.bbw + bbp

        # total absorptions from Gelbstoff, phytoplankton, water and residual
        ag = G * s.exp(-0.015*(self.wl-440))
        aphi = (self.aphi_coeffs[:, 0] + self.aphi_coeffs[:, 1]*s.log(P))*P
        chla = s.exp(s.log(P/0.06)/0.65)
        a = self.aw + aphi + ag
        if len(self.abs_inds) > 0:
            a = a + x_surface[self.abs_inds]

        # Remote sensing reflectance below water surface
        # Values as described in Lee et al., Applied Optics 2002
        # Vol. 41, No. 27 pg. 5757
        u = bb / (a+bb)
        rrs = (self.g0 * u) + pow(self.g1 * u, 2)

        # Water surface effects
        Rrs = 0.518*rrs / (1-1.562*rrs)
        lamb = Rrs * s.pi
        return rrs, lamb, bb, a, u

    def calc_lamb(self, x_surface, geom):
        """Lambertian-equivalent reflectance."""

        rrs, lamb, bb, a, u = self.qaa(x_surface)
        return lamb

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface."""

        rrs, lamb, bb, a, u = self.qaa(x_surface)
        X, G, P, Y, glint, flh = x_surface[self.nonabs_inds]

        # total backscatter from particle and water contributions
        dbb_dsurface = s.zeros((len(self.wl), len(self.statevec)))
        w = 400.0/self.wl
        dbb_dsurface[:, self.X_ind] = pow(w, Y)
        dbb_dsurface[:, self.Y_ind] = X * pow(w, Y)*s.log(w)

        # total absorptions from Gelbstoff, phytoplankton, water and residual
        da_dsurface = s.zeros((len(self.wl), len(self.statevec)))
        factor1 = (self.aphi_coeffs[:, 0] + self.aphi_coeffs[:, 1]*s.log(P))
        factor2 = (self.aphi_coeffs[:, 1]/P)*P
        da_dsurface[:, self.P_ind] = factor1 + factor2
        da_dsurface[:, self.G_ind] = s.exp(-0.015*(self.wl-440))
        for i in self.abs_inds:
            da_dsurface[i, i] = 1.0

        du_da = -bb/pow(a+bb, 2)
        du_dbb = -bb/pow(a+bb, 2) + 1.0/(a+bb)
        du_dsurface = ((da_dsurface).T * du_da + (dbb_dsurface).T * du_dbb).T
        drrs_du = self.g0 + 2 * self.g1 * u
        drrs_dsurface = ((du_dsurface).T * drrs_du).T
        dRrs_drrs = 0.518 / (1-1.562*rrs) + \
            (0.518*rrs) * -1.0/(pow(1-1.562*rrs, 2)*-1.562)
        dRrs_dsurface = ((drrs_dsurface).T * dRrs_drrs).T
        dlamb_dsurface = dRrs_dsurface * s.pi
        return dlamb_dsurface

    def calc_rfl(self, x_surface, geom):
        """Reflectance (includes specular glint)."""

        return self.calc_lamb(x_surface, geom) + x_surface[self.glint_ind]

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.glint_ind] = 1
        return drfl

    def calc_Ls(self, x_surface, geom):
        """Emission at surface includes fluorescence (here, a Gaussian)."""

        ngauss = srf(self.wl, self.fl_mu, self.fl_sigma)
        ngauss = ngauss/max(ngauss)
        return ngauss * x_surface[self.flh_ind]

    def dLs_dsurface(self, x_surface, geom):
        """Emission at surface includes fluorescence (here, a Gaussian)."""

        dLs = s.zeros((len(self.wl), len(self.statevec)))
        ngauss = srf(self.wl, self.fl_mu, self.fl_sigma)
        ngauss = ngauss/max(ngauss)
        dLs[:, self.flh_ind] = ngauss
        return dLs
