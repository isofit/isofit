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

        # Special glint bounds
        rmin, rmax = -0.05, 2.0
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.bounds.extend([[-1, 10], [0, 10]])  # Gege (2021), WASI user manual
        self.n_state = self.n_state + 2

        # Useful indexes to track
        self.glint_ind = len(self.statevec_names) - 2
        self.idx_surface = np.arange(len(self.statevec_names))

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

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
        fit a state vector.
        """
        # Estimate reflectance, assuming all signal around 1020 nm == glint
        glint_band = np.argmin(abs(1020 - self.wl))
        glint_est = np.mean(rfl_meas[(glint_band - 2) : glint_band + 2])

        # Stealing the bounds for this from additive_glint_model
        bounds_glint_est = [
            0,
            0.2,
        ]
        glint_est = max(
            bounds_glint_est[0] + eps,
            min(bounds_glint_est[1] - eps, glint_est),
        )
        lamb_est = rfl_meas - glint_est
        x = MultiComponentSurface.fit_params(self, lamb_est, geom)  # Bounds reflectance

        # Get estimate for g_dd and g_dsf parameters, given signal at 900 nm
        # Set to a static number; don't need to apply bounds because static
        g_dsf_est = 0.01

        # Use nadir fresnel coeffs (0.02) and t_down_dir = 0.83, t_down_diff = 0.14 for initialization
        # Transmission values taken from MODTRAN sim with AERFRAC_2 = 0.5, H2OSTR = 0.5
        g_dd_est = ((glint_est * 0.97 / 0.02) - 0.14 * g_dsf_est) / 0.83
        g_dd_est = max(
            self.bounds[self.glint_ind][0] + eps,
            min(self.bounds[self.glint_ind][1] - eps, g_dd_est),
        )

        # SUN_GLINT g_dd
        x[self.glint_ind] = g_dd_est
        # SKY_GLINT g_dsf
        x[self.glint_ind + 1] = g_dsf_est
        return x

    def calc_rfl(self, x_surface, geom):
        """Direct and diffuse Reflectance (includes sun and sky glint).

        Inputs:
        x_surface : np.ndarray
            Surface portion of the statevector element
        geom : Geometry
            Isofit geometry object

        Outputs:
        rho_dir_dir : np.ndarray
            Reflectance quantity for downward direct photon paths
        rho_dif_dir : np.ndarray
            Reflectance quantity for downward diffuse photon paths

        NOTE:
            Here, we treat direct and diffuse photon path reflectance
            differently. The sun and sky glint magnitudes are statevector
            elements that interact with the two reflectance quantities
            independently.
        """
        # fresnel reflectance factor (approx. 0.02 for nadir view)
        rho_ls = self.fresnel_rf(geom.observer_zenith)

        sun_glint = x_surface[-2] * rho_ls
        sky_glint = x_surface[-1] * rho_ls

        rho_dir_dir = self.calc_lamb(x_surface, geom) + sun_glint
        rho_dif_dir = self.calc_lamb(x_surface, geom) + sky_glint

        return rho_dir_dir, rho_dif_dir

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface.
        """

        drfl = self.dlamb_dsurface(x_surface, geom)

        rho_ls = self.fresnel_rf(geom.observer_zenith)
        # TODO make the indexing better for the surface state elements
        drfl[:, self.glint_ind] = rho_ls
        drfl[:, self.glint_ind + 1] = rho_ls

        return drfl

    def drdn_dglint(self, L_tot, L_down_dir, s_alb, rho_dif_dir):
        """Derivative of radiance with respect to
        the direct and diffuse glint terms"""
        drdn_dgdd = L_down_dir
        drdn_dgdsf = (L_tot / ((1.0 - (s_alb * rho_dif_dir)) ** 2)) - drdn_dgdd

        return drdn_dgdd, drdn_dgdsf

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
        # Construct the output matrix
        # Dimensions should be (len(RT.wl), len(x_surface))
        # which is correctly handled by the instrument resampling
        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(L_tot, s_alb, rho_dif_dir)
        drdn_dsurface[:, : self.n_wl] = np.multiply(
            drdn_drfl[:, np.newaxis], drfl_dsurface[:, : self.n_wl]
        )

        # Glint derivatives
        drdn_dgdd, drdn_dgdsf = self.drdn_dglint(L_tot, L_down_dir, s_alb, rho_dif_dir)

        # Store the glint derivatives as last two rows in drdn_drfl
        drdn_dsurface[:, -2] = drdn_dgdd * drfl_dsurface[:, -2]
        drdn_dsurface[:, -1] = drdn_dgdsf * drfl_dsurface[:, -1]

        # Get the derivative w.r.t. surface emission
        drdn_dLs = np.multiply(self.drdn_dLs(t_total_up)[:, np.newaxis], dLs_dsurface)

        return np.add(drdn_dsurface, drdn_dLs)

    def analytical_model(
        self,
        background,
        L_down_dir,
        L_down_dif,
        L_tot,
        geom,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        """
        Linearization of the glint terms to use in AOE inner loop.
        Function will fetch the linearization of the rho terms and
        add the matrix components for the direct glint term.
        Currently we set the diffuse glint scaling term to constant
        value, which makes the AOE inner loop inversion possible.
        """
        rho_ls = self.fresnel_rf(geom.observer_zenith)

        # Construct the H matrix from:
        # theta (rho portion)
        # gam (sun glint portion)
        # ep (sky glint portion)
        H = super().analytical_model(
            background,
            L_down_dir,
            L_down_dif,
            L_tot,
            geom,
            L_dir_dir,
            L_dir_dif,
            L_dif_dir,
            L_dif_dif,
        )

        gam = (L_dir_dir + L_dir_dif) * rho_ls
        gam = np.reshape(gam, (len(gam), 1))
        H = np.append(H, gam, axis=1)

        # Diffuse portion
        # ep = ((L_dif_dir + L_dif_dif) + ((L_tot * background) / (1 - background))) * rho_ls
        # If you ignore multi-scattering
        ep = (L_dif_dir + L_dif_dif) * rho_ls
        ep = np.reshape(ep, (len(ep), 1))
        H = np.append(H, ep, axis=1)

        return H

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return MultiComponentSurface.summarize(
            self, x_surface, geom
        ) + " Sun Glint: %5.3f, Sky Glint: %5.3f" % (x_surface[-2], x_surface[-1])

    @staticmethod
    def fresnel_rf(vza):
        """Calculates reflectance factor of sky radiance based on the
        Fresnel equation for unpolarized light as a function of view zenith angle (vza).
        """
        if vza > 0.0:
            n_w = 1.33  # refractive index of water
            theta = np.deg2rad(vza)

            # calculate angle of refraction using Snell′s law
            theta_i = np.arcsin(np.sin(theta) / n_w)

            # reflectance factor of sky radiance based on the Fresnel equation for unpolarized light
            rho_ls = 0.5 * np.abs(
                ((np.sin(theta - theta_i) ** 2) / (np.sin(theta + theta_i) ** 2))
                + ((np.tan(theta - theta_i) ** 2) / (np.tan(theta + theta_i) ** 2))
            )
        else:
            rho_ls = 0.02  # the reflectance factor converges to 0.02 for view angles equal to 0.0°

        return rho_ls
