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

from isofit.core.common import emissive_radiance
from isofit.surface.surface_multicomp import MultiComponentSurface


class ThermalSurface(MultiComponentSurface):
    """A model of the surface based on a Mixture of a hot Black Body and
    Multicomponent cold surfaces."""

    def __init__(self, full_config: Config):
        """."""

        config = full_config.forward_model.surface

        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        # Handle additional state vector elements
        if "SURF_TEMP_K" not in self.statevec_names:
            self.statevec_names.extend(["SURF_TEMP_K"])
            self.init.extend([300.0])  # This is overwritten below
            self.scale.extend([100.0])
            self.bounds.extend([[250.0, 400.0]])
        self.idx_surface = np.arange(len(self.statevec_names))
        self.surf_temp_ind = len(self.statevec_names) - 1
        self.emissive = True
        self.n_state = len(self.init)

        self.emissivity_for_surface_T_init = config.emissivity_for_surface_T_init
        self.surface_T_prior_sigma_degK = config.surface_T_prior_sigma_degK

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function."""

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.surf_temp_ind] = self.init[self.surf_temp_ind]

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        Cov[self.surf_temp_ind, self.surf_temp_ind] = self.surface_T_prior_sigma_degK**2

        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, find the surface reflectance"""

        x_surface = MultiComponentSurface.fit_params(self, rfl_meas, geom)
        x_surface[self.surf_temp_ind] = self.init[self.surf_temp_ind]

        return x_surface

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to state
        vector, calculated at x_surface."""

        dlamb = MultiComponentSurface.dlamb_dsurface(self, x_surface, geom)
        dlamb[:, self.surf_temp_ind] = 0

        return dlamb

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        T = x_surface[self.surf_temp_ind]
        rfl = self.calc_rfl(x_surface, geom)
        # ToDo: direct and diffuse reflectance vectors not supported yet
        rfl[0][rfl[0] > 1.0] = 1.0
        emissivity = 1 - rfl[0]
        Ls, dLs_dT = emissive_radiance(emissivity, T, self.wl)

        return Ls

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        T = x_surface[self.surf_temp_ind]
        lambertian_rfl = self.calc_lamb(x_surface, geom)
        emissivity = 1 - lambertian_rfl
        Ls, dLs_dT = emissive_radiance(emissivity, T, self.wl)
        dLs_drfl = np.diag(-1 * Ls)
        dLs_dsurface = np.vstack([dLs_drfl, dLs_dT]).T

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
        full surface vector"""

        # Construct the output matrix:
        # Dimensions should be (len(RT.wl), len(x_surface))
        # which is correctly handled by the instrument resampling
        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(L_tot, s_alb, rho_dif_dir)

        drdn_dsurface[:, : self.n_wl] = np.multiply(
            drdn_drfl[:, np.newaxis], drfl_dsurface[:, : self.n_wl]
        )

        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        mcm = MultiComponentSurface.summarize(self, x_surface, geom)
        msg = " Kelvins: %5.1f " % x_surface[self.surf_temp_ind]

        return msg + mcm
