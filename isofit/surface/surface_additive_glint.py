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
from __future__ import annotations

import numpy as np

from isofit.core.common import eps
from isofit.surface.surface_thermal import ThermalSurface


class AdditiveGlintSurface(ThermalSurface):
    """A model of the surface based on a collection of multivariate
    Gaussians, extended with a surface glint term."""

    def __init__(self, full_config: Config):
        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(["GLINT"])
        self.idx_surface = np.arange(len(self.statevec_names))
        self.scale.extend([1.0])
        self.init.extend([0.005])
        self.bounds.extend([[0, 0.2]])
        self.n_state = self.n_state + 1
        self.glint_ind = len(self.statevec_names) - 1

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        mu = ThermalSurface.xa(self, x_surface, geom)
        mu[self.glint_ind] = self.init[self.glint_ind]

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        Cov = ThermalSurface.Sa(self, x_surface, geom)
        f = np.array([[(10.0 * self.scale[self.glint_ind]) ** 2]])
        Cov[self.glint_ind, self.glint_ind] = f

        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters,
        fit a state vector."""

        glint_band = np.argmin(abs(900 - self.wl))
        glint = np.mean(rfl_meas[(glint_band - 2) : glint_band + 2])
        water_band = np.argmin(abs(400 - self.wl))
        water = np.mean(rfl_meas[(water_band - 2) : water_band + 2])
        if glint > 0.05 or water < glint:
            glint = 0
        glint = max(
            self.bounds[self.glint_ind][0] + eps,
            min(self.bounds[self.glint_ind][1] - eps, glint),
        )
        lamb_est = rfl_meas - glint
        x = ThermalSurface.fit_params(self, lamb_est, geom)
        x[self.glint_ind] = glint

        return x

    def calc_rfl(self, x_surface, geom):
        """Reflectance (includes specular glint)."""
        rfl = self.calc_lamb(x_surface, geom) + x_surface[self.glint_ind]

        return rfl, rfl

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.glint_ind] = 1

        return drfl

    def drdn_dglint(self, L_tot, s_alb, rho_dif_dir):
        """Partial derivative of radiance with respect to
        additive glint"""

        return L_tot / ((1.0 - (s_alb * rho_dif_dir)) ** 2)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface.  We append a column of zeros to handle
        the extra glint parameter"""

        dLs_dsurface = super().dLs_dsurface(x_surface, geom)
        dLs_dglint = np.zeros((dLs_dsurface.shape[0], 1))
        dLs_dsurface = np.hstack([dLs_dsurface, dLs_dglint])

        return dLs_dsurface

    def drdn_dsurface(
        self,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_tot,
        L_down_dir,
    ):
        """Derivative of radiance with respect to
        full surface vector. Everything should be at RT wavelength
        resolution entering this function.
        """

        # Construct the output matrix
        # Dimensions should be (len(RT.wl), len(x_surface))
        # which is correctly handled by the instrument resampling
        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(L_tot, s_alb, rho_dif_dir)
        drdn_dsurface[:, : self.n_wl] = np.multiply(
            drdn_drfl[:, np.newaxis], drfl_dsurface[:, : self.n_wl]
        )

        # Glint derivatives
        drdn_dglint = self.drdn_dglint(L_tot, s_alb, rho_dif_dir)
        # Last columns is glint derivative
        drdn_dsurface[:, -1] = drdn_dglint * drfl_dsurface[:, -1]

        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return (
            ThermalSurface.summarize(self, x_surface, geom)
            + " Glint: %5.3f" % x_surface[self.glint_ind]
        )
