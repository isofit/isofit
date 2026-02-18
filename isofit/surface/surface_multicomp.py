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
from scipy.linalg import block_diag, norm
from scipy.ndimage import gaussian_filter1d

from isofit.core.common import svd_inv_sqrt
from isofit.surface.surface import Surface


class MultiComponentSurface(Surface):
    """A model of the surface based on a collection of multivariate
    Gaussians, with one or more equiprobable components and full
    covariance matrices.

    To evaluate the probability of a new spectrum, we calculate the
    distance to each component cluster.
    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        # TODO: enforce surface_file existence in the case of multicomponent_surface
        self.component_means = self.model_dict["means"]
        self.component_covs = self.model_dict["covs"]

        self.n_comp = len(self.component_means)
        self.wl = self.model_dict["wl"][0]
        self.n_wl = len(self.wl)

        # Set up normalization method
        self.normalize = self.model_dict["normalize"]
        if self.normalize == "Euclidean":
            self.norm = lambda r: norm(r)
        elif self.normalize == "RMS":
            self.norm = lambda r: np.sqrt(np.mean(pow(r, 2)))
        elif self.normalize == "None":
            self.norm = lambda r: 1.0
        else:
            raise ValueError("Unrecognized Normalization: %s\n" % self.normalize)

        self.selection_metric = self.model_dict.get(
            "selection_metric", config.selection_metric
        )
        self.select_on_init = config.select_on_init

        # Reference values are used for normalizing the reflectances.
        # in the VSWIR regime, reflectances are normalized so that the model
        # is agnostic to absolute magnitude.
        self.refwl = np.squeeze(self.model_dict["refwl"])
        self.idx_ref = [np.argmin(abs(self.wl - w)) for w in np.squeeze(self.refwl)]
        self.idx_ref = np.array(self.idx_ref)

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = -0.05, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        self.idx_surface = np.arange(len(self.statevec_names))

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

        # Surface specific attributes. Can override in inheriting classes
        self.full_glint = False

        # Cache some important computations
        self.Covs, self.Cinvs, self.mus = [], [], []
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = [], []
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        for i in range(self.n_comp):
            Cov = self.component_covs[i]
            self.Covs.append(np.array([Cov[j, self.idx_ref] for j in self.idx_ref]))
            self.Cinvs.append(svd_inv_sqrt(self.Covs[-1])[0])
            self.mus.append(self.component_means[i][self.idx_ref])

            # Caching the normalized Sa inv and Sa inv sqrt
            Cov_full = block_diag(
                np.zeros((nprefix, nprefix)),
                self.component_covs[i],
                np.zeros((nsuffix, nsuffix)),
            )
            Cov_normalized = Cov_full / np.mean(np.diag(Cov_full))
            Cinv_normalized, Cinv_sqrt_normalized = svd_inv_sqrt(Cov_normalized)
            self.Sa_inv_normalized.append(Cinv_normalized)
            self.Sa_inv_sqrt_normalized.append(Cinv_sqrt_normalized)

    def component(self, x, geom):
        """We pick a surface model component using a distance metric.

        This always uses the Lambertian (non-specular) version of the
        surface reflectance. If the forward model initialize via heuristic
        (i.e. algebraic inversion), the component is only calculated once
        based on that first solution. That state is preserved in the
        geometry object.
        """

        if self.n_comp <= 1:
            return 0
        elif hasattr(geom, "surf_cmp_init"):
            return geom.surf_cmp_init
        elif self.select_on_init and hasattr(geom, "x_surf_init"):
            x_surface = geom.x_surf_init
        else:
            x_surface = x

        # Get the (possibly normalized) reflectance
        lamb = self.calc_lamb(x_surface, geom)
        lamb_ref = lamb[self.idx_ref]
        lamb_ref = lamb_ref / self.norm(lamb_ref)

        # Only support euclidean distance comparrison for now
        if self.selection_metric == "SGA":
            mds = self.spectral_gradient_angle(lamb_ref, np.array(self.mus))
        elif self.selection_metric == "Euclidean":
            mds = self.euclidean_distance(
                lamb_ref,
                np.array(self.mus),
            )
        else:
            raise ValueError(
                "Surface component selection metric not valid:", self.selection_metric
            )

        closest = np.argmin(mds)

        if (
            self.select_on_init
            and hasattr(geom, "x_surf_init")
            and (not hasattr(geom, "surf_cmp_init"))
        ):
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
        lamb_mu = self.component_means[ci]
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
        Cov = self.component_covs[ci]
        Sa_unnormalized = Cov * (self.norm(lamb_ref) ** 2)

        # select the Sa inverse from the list of components
        Sa_inv_normalized = self.Sa_inv_normalized[ci]
        Sa_inv_sqrt_normalized = self.Sa_inv_sqrt_normalized[ci]

        # If there are no other state vector elements, we're done.
        if len(self.statevec_names) == len(self.idx_lamb):

            return Sa_unnormalized, Sa_inv_normalized, Sa_inv_sqrt_normalized

        # Embed into a larger state vector covariance matrix
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        Cov_prefix = np.zeros((nprefix, nprefix))
        Cov_suffix = np.zeros((nsuffix, nsuffix))
        Sa_unnormalized = block_diag(Cov_prefix, Sa_unnormalized, Cov_suffix)

        return Sa_unnormalized, Sa_inv_normalized, Sa_inv_sqrt_normalized

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError("Mismatched reflectances")
        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(
                self.bounds[i][0] + 0.001, min(self.bounds[i][1] - 0.001, r)
            )

        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance.

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
            We do not handle direct and diffuse photon path reflectance
            quantities differently for the multicomponent surface model.
            This is why we return the same quantity for both outputs.
        """

        rho_dir_dir = rho_dif_dir = self.calc_lamb(x_surface, geom)

        return rho_dir_dir, rho_dif_dir

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

    def drdn_drfl(self, L_tot, s_alb, rho_dif_dir):
        """Partial derivative of radiance with respect to
        surface reflectance"""

        return L_tot / ((1.0 - s_alb * rho_dif_dir) ** 2)

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

    def drdn_dLs(self, t_total_up):
        """Partial derivative of radiance with respect to
        surface emission"""

        return t_total_up

    def drdn_dsurface(
        self,
        rho_dif_dir,
        drfl_dsurface,
        dLs_dsurface,
        s_alb,
        t_total_up,
        L_tot,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
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

    def analytical_model(
        self,
        background,
        L_tot,
        geom,
        L_dir_dir=None,
        L_dir_dif=None,
        L_dif_dir=None,
        L_dif_dif=None,
    ):
        """
        Linearization of the surface reflectance terms to use in the
        AOE inner loop (see Susiluoto, 2025). We set the quadratic
        spherical albedo term to a constant background, which
        simplifies the linearization
        background = s * rho_bg
        """
        # If you ignore multi-scattering
        theta = L_tot + (L_tot * background / (1 - background))
        # theta = L_tot

        H = np.eye(self.n_wl, self.n_wl)
        H = theta[:, np.newaxis] * H

        return H

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ""

        return "Component: %i" % self.component(x_surface, geom)

    @staticmethod
    def euclidean_distance(lamb_ref, mus):
        return np.sum(np.power(lamb_ref[np.newaxis, :] - mus, 2), axis=1)

    @staticmethod
    def spectral_angle_distance(lamb_ref, mus):
        cos_theta = np.einsum("k,ik->i", lamb_ref, mus) / (
            np.linalg.norm(lamb_ref) * np.linalg.norm(mus, axis=1)
        )

        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def spectral_gradient_angle(self, lamb_ref, mus):
        def gradient(wl, val, sigma=2):
            val = gaussian_filter1d(val, sigma=sigma)
            return np.gradient(val, wl)

        grads = np.array([gradient(self.wl[self.idx_ref], mu) for mu in mus])

        return self.spectral_angle_distance(
            gradient(self.wl[self.idx_ref], lamb_ref), grads
        )
