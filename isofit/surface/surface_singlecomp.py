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

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from isofit.configs import Config
from isofit.core.common import load_spectrum, load_wavelen, svd_inv


class SingleComponentSurface:
    """A model of the surface.
    This isn't built out at all. Could re-tool to produce a single
    component surface prior.

    This is a version that uses a single component .mat file
    in the same format as the multi-component
    """

    def __init__(self, full_config: Config):
        config = full_config.forward_model.surface

        # Check to see if .mat surface file exists
        if exists(vars(config).get("surface_file", "")):
            model_dict = loadmat(config.surface_file)
        else:
            raise FileNotFoundError("No surface .mat file exists")

        self.component = (model_dict["means"], model_dict["covs"])
        self.n_comp = 1
        self.wl = model_dict["wl"][0]
        self.n_wl = len(self.wl)

        # Set up normalization method
        self.normalize = model_dict["normalize"]
        if self.normalize == "Euclidean":
            self.norm = lambda r: norm(r)
        elif self.normalize == "RMS":
            self.norm = lambda r: np.sqrt(np.mean(pow(r, 2)))
        elif self.normalize == "None":
            self.norm = lambda r: 1.0
        else:
            raise ValueError("Unrecognized Normalization: %s\n" % self.normalize)

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = np.squeeze(model_dict["refwl"])
        self.idx_ref = [np.argmin(abs(self.wl - w)) for w in np.squeeze(self.refwl)]
        self.idx_ref = np.array(self.idx_ref)

        # Cache some important computations
        # First pass, doesn't look like these are being used
        Cov = self.component[1][0, ...]
        Self.Covs = np.array([Cov[j, self.idx_ref] for j in self.idx_ref])

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function. This always uses the
        Lambertian (non-specular) version of the surface reflectance."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        mu = np.zeros(self.n_state)
        lamb_mu = self.component[0]
        lamb_mu = lamb_mu * self.norm(lamb_ref)
        mu[self.idx_lamb] = lamb_mu

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x. We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        Cov = self.component[1]
        Cov = Cov * (self.norm(lamb_ref) ** 2)

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):
            return Cov

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))

        return block_diag(Cov_prefix, Cov, Cov_suffix)

    def fit_params(self, rfl_meas):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError("Mismatched reflectances")

        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(
                self.bounds[i][0] + 0.001, min(self.bounds[i][1] - 0.001, r)
            )

        return x_surface

    def calc_lamb(self, x_surface, geom):
        """Calculate a Lambertian surface reflectance for this state vector."""

        return x_surface[self.idx_lamb]

    def calc_rfl(self, x_surface, geom):
        """Calculate the directed reflectance (specifically the HRDF) for this
        state vector."""

        return self.calc_lamb(x_surface, geom)

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface. In the case that there are no free
        paramters our convention is to return the vector of zeros."""

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
            return ""
