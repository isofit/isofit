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
from isofit.surface.surface_multicomp import MultiComponentSurface


class GlintModelSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate
    Gaussians, extended with two surface glint terms (sun + sky glint)."""

    def __init__(self, full_config: Config):
        super().__init__(full_config)

        # TODO: Enforce this attribute in the config, not here (this is hidden)
        self.statevec_names.extend(["SUN_GLINT", "SKY_GLINT"])
        self.scale.extend([1.0, 1.0])
        self.init.extend(
            [0.02, 1 / np.pi]
        )  # Numbers from Marcel Koenig; used for prior mean
        self.bounds.extend([[-1, 10], [0, 10]])  # Gege (2021), WASI user manual
        self.n_state = self.n_state + 2
        self.glint_ind = len(self.statevec_names) - 2
        self.f = np.array(
            [[(1000000 * np.array(self.scale[self.glint_ind :])) ** 2]]
        )  # Prior covariance, *very* high...

        if "full_glint" in (full := full_config.forward_model.surface.__dict__.keys()):
            self.full_glint = full

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
        # Unclear if this should be a fully correlated block or a diagonal
        Cov[self.glint_ind :, self.glint_ind :] = np.eye(2) * self.f
        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate and one or more emissive parameters,
        fit a state vector."""
        # Estimate reflectance, assuming all signal around 1020 nm == glint
        glint_band = np.argmin(abs(1020 - self.wl))
        glint_est = np.mean(rfl_meas[(glint_band - 2) : glint_band + 2])
        bounds_glint_est = [
            0,
            0.2,
        ]  # Stealing the bounds for this from additive_glint_model
        glint_est = max(
            bounds_glint_est[0] + eps,
            min(bounds_glint_est[1] - eps, glint_est),
        )
        lamb_est = rfl_meas - glint_est
        x = MultiComponentSurface.fit_params(self, lamb_est, geom)  # Bounds reflectance

        # Get estimate for g_dd and g_dsf parameters, given signal at 900 nm
        g_dsf_est = (
            0.01  # Set to a static number; don't need to apply bounds because static
        )
        # Use nadir fresnel coeffs (0.02) and t_down_dir = 0.83, t_down_diff = 0.14 for initialization
        # Transmission values taken from MODTRAN sim with AERFRAC_2 = 0.5, H2OSTR = 0.5
        g_dd_est = ((glint_est * 0.97 / 0.02) - 0.14 * g_dsf_est) / 0.83
        g_dd_est = max(
            self.bounds[self.glint_ind][0] + eps,
            min(self.bounds[self.glint_ind][1] - eps, g_dd_est),
        )
        x[self.glint_ind] = g_dd_est  # SUN_GLINT g_dd
        x[self.glint_ind + 1] = g_dsf_est  # SKY_GLINT g_dsf
        return x

    def calc_rfl(self, x_surface, geom, L_down_dir=None, L_down_dif=None):
        """Direct and diffuse Reflectance (includes sun and sky glint)."""

        rho_ls = 0.02  # fresnel reflectance factor (approx. 0.02 for nadir view)
        sun_glint = rho_ls * (x_surface[-2] * L_down_dir / (L_down_dir + L_down_dif))
        sky_glint = rho_ls * (x_surface[-1] * L_down_dif / (L_down_dir + L_down_dif))

        rho_dir_dir = self.calc_lamb(x_surface, geom) + sun_glint
        rho_dif_dir = self.calc_lamb(x_surface, geom) + sky_glint

        return rho_dir_dir, rho_dif_dir

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        drfl = self.dlamb_dsurface(x_surface, geom)
        drfl[:, self.glint_ind :] = 0
        return drfl

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface.  We append two columns of zeros to handle
        the extra glint parameter"""

        dLs_dsurface = super().dLs_dsurface(x_surface, geom)
        return dLs_dsurface

    def drdn_drfl(
        self,
        rho_scaled_for_mltiscattering_drfl,
        L_down_tot,
        L_down_dir,
        L_down_dif,
        t_total_up,
    ):
        """Partial derivative of radiance with respect to
        surface reflectance"""
        drdn_drfl = super().drdn_drfl(
            rho_scaled_for_mltiscattering_drfl,
            L_down_tot,
            L_down_dir,
            L_down_dif,
            t_total_up,
        )
        return drdn_drfl

    def drdn_dglint(
        self, drho_scaled_for_multiscattering_drfl, t_total_up, L_down_dir, L_down_dif
    ):
        """Derivative of radiance with respect to
        the direct and diffuse glint terms"""
        drdn_dgdd = L_down_dir * t_total_up * drho_scaled_for_multiscattering_drfl
        drdn_dgdsf = L_down_dif * t_total_up * drho_scaled_for_multiscattering_drfl
        return drdn_dgdd, drdn_dgdsf

    def drdn_dsurface(
        self,
        rho_dir_dir,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_down_tot,
        L_down_dir,
        L_down_dif,
    ):
        """Derivative of radiance with respect to
        full surface vector"""

        drho_scaled_for_multiscattering_drfl = 1.0 / (1.0 - s_alb * rho_dif_dir) ** 2

        # Reflectance derivatives
        drdn_drfl = self.drdn_drfl(
            drho_scaled_for_multiscattering_drfl,
            L_down_tot,
            L_down_dir,
            L_down_dif,
            t_total_up,
        )
        # Glint derivatives
        drdn_dgdd, drdn_dgdsf = self.drdn_dglint(
            drho_scaled_for_multiscattering_drfl, t_total_up, L_down_dir, L_down_dif
        )
        # Emission derivatives
        drdn_dLs = self.drdn_dLs(t_total_up)

        k_surface = (
            drdn_drfl[:, np.newaxis] * drfl_dsurface
            + drdn_dLs[:, np.newaxis] * dLs_dsurface
        )

        # Direct glint term is second to last surface index
        k_surface[:, -2] = k_surface[:, -2] * drdn_dgdd
        # Diffuse glint term is last surface index
        k_surface[:, -1] = k_surface[:, -1] * drdn_dgdsf
        return k_surface

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(
            self, x_surface, geom
        ) + " Sun Glint: %5.3f, Sky Glint: %5.3f" % (x_surface[-2], x_surface[-1])
