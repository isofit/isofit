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

from isofit.configs import Config

from ..core.common import eps
from .surface_multicomp import MultiComponentSurface


class GlintModelSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
    Gaussians, extended with two surface glint terms (sun + sky glint)."""

    def __init__(self, full_config: Config):
        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(["SUN_GLINT", "SKY_GLINT"])
        self.scale.extend([1.0, 1.0])
        self.init.extend([0.02, 1 / np.pi])
        self.bounds.extend([[-1, 10], [0, 10]])  # Gege (2021), WASI user manual
        self.n_state = self.n_state + 2
        self.glint_ind = len(self.statevec_names) - 2
        self.f = np.array([[(1000000 * np.array(self.scale[self.glint_ind :])) ** 2]])

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.glint_ind :] = self.init[self.glint_ind :]
        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        Cov[self.glint_ind :, self.glint_ind :] = self.f
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters,
        fit a state vector."""
        # first guess suggestion: E_dd => see below, E_ds => ~0.01
        glint_band = np.argmin(abs(900 - self.wl))
        glint = np.mean(rfl_meas[(glint_band - 2) : glint_band + 2])
        glint = max(
            self.bounds[self.glint_ind][0] + eps,
            min(self.bounds[self.glint_ind][1] - eps, glint),
        )
        lamb_est = rfl_meas - glint
        x = MultiComponentSurface.fit_params(self, lamb_est, geom)
        x[self.glint_ind] = glint
        x[self.glint_ind + 1] = 0.01
        return x

    def calc_rfl(self, x_surface, geom):
        """Reflectance (includes specular glint)."""

        return self.calc_lamb(x_surface, geom) + x_surface[self.glint_ind]

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.glint_ind :] = 1
        return drfl

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface.  We append two columns of zeros to handle
        the extra glint parameter"""

        dLs_dsurface = super().dLs_dsurface(x_surface, geom)
        return dLs_dsurface

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(
            self, x_surface, geom
        ) + " Sun Glint: %5.3f, Sky Glint: %5.3f" % (x_surface[-2], x_surface[-1])
