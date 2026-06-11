#! /usr/bin/env python3
#
#  Copyright 2020 California Institute of Technology
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

from isofit.core.common import svd_inv_sqrt, eps
from isofit.surface.surface import Surface
from isofit.luts.reader import load_prebuilt_surface


class LUTSurface(Surface):
    """A model of the surface based on an N-dimensional lookup table
    indexed by one or more state vector elements.  We calculate the
    reflectance by multilinear interpolation.  This is good for
    surfaces like aquatic ecosystems or snow that can be
    described with just a few degrees of freedom.

    The lookup table must be precalculated based on the wavelengths
    of the instrument and can either be in MATLAB (.mat) format or NetCDF (.nc).

    For a MATLAB lookup table, it contains the following fields:
        - grids: an object array containing n lists of gridpoints
        - rho_dif_dir: an n+1 dimensional array containing the dif-dir reflectance for each gridpoint
        - rho_dir_dir: an n+1 dimensional array containing the dir-dir reflectance for each gridpoint [optional]
        - statevec_names: an array of n strings representing state vector element names
        - mean: an array of n prior mean values, one for each state vector element [optional]
        - sigma: an array of n prior standard deviations, one for each state vector element [optional]

    For a NetCDF lookup table, it contains the following fields:
        - Coordinates:
            * wl
            * Other LUT dimensions (e.g., solar_zenith, observer_zenith, relative_azimuth, grain size)
        - Data Variables:
            * rho_dif_dir: (LUT Axes..., n_wl)
            * rho_dir_dir: (LUT Axes..., n_wl) [optional]
            * statevec_names: (n_state) [optional]
            * mean: (n_state) [optional]
            * sigma: (n_state) [optional]

    Reflectance keys should either be rho_dif_dir (or both can be included).

    Any of the angles are optional, but if provided should be in degrees and named "solar_zenith", "observer_zenith", and "relative_azimuth".

    You can also choose to run a mixed pixel retrieval by adding data variables of length wl with key name "endmember_TYPE".
    Where you can fill in TYPE for given surface(s) you would like to mix. You may use any number of endmembers.

    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)
        config = full_config.forward_model.surface
        self.terrain_style = config.terrain_style
        self.max_slope = config.max_slope

        # Load dif-dir rfl data, optional dir-dir term, and other important parameters from the surface LUT
        self.itp_hd, self.itp_dd, lut_params = load_prebuilt_surface(
            config.surface_lut_file, terrain_style=self.terrain_style, config_only=False
        )

        for key in [
            "wl",
            "statevec_names",
            "statevec_idxs",
            "lut_names",
            "lut_grid",
            "solve_mixed_pixel",
            "idx_fractional_data",
            "idx_fractional_em",
            "endmember_matrix",
            "endmember_names",
            "sza_idx",
            "vza_idx",
            "raa_idx",
            "cos_i_idx",
        ]:
            setattr(self, key, lut_params[key])

        self.n_wl = len(self.wl)
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lut = np.arange(self.n_state)
        self.idx_lamb = np.arange(self.n_wl)
        self.idx_surface = np.arange(len(self.statevec_names))
        idx_em_rfls = []
        if self.solve_mixed_pixel:
            self.idx_em_rfls = [self.idx_surface[self.idx_fractional_data]]
            self.idx_em_rfls.extend(
                [
                    self.idx_surface[self.idx_fractional_em[f"FRACTIONAL_{n}"]]
                    for n in self.endmember_names
                ]
            )

        # Then, grab the priors and optimizaton parameters from the config
        self.init, self.bounds, self.scale, self.mean, self.sigma = [], [], [], [], []

        for name in self.statevec_names:
            state_config = getattr(config.statevector, name)
            self.init.append(state_config.get("init"))
            self.bounds.append(state_config.get("bounds"))
            self.scale.append(state_config.get("scale"))
            self.mean.append(state_config.get("prior_mean"))
            self.sigma.append(state_config.get("prior_sigma"))

        self.init = np.array(self.init)
        self.scale = np.array(self.scale)
        self.mean = np.array(self.mean)
        self.sigma = np.array(self.sigma)

        # Checking the statevec names prior to running
        for name in self.statevec_names:
            if name in ["solar_zenith", "observer_zenith", "relative_azimuth"]:
                raise ValueError(
                    f"Variable:{name} in the statevector is not supported."
                )
            if name.startswith("FRACTIONAL_"):
                continue
            if name not in self.lut_names:
                raise ValueError(
                    f"Statevector:{name} not found in LUT dimensions: {self.lut_names}"
                )

        # Ensure priors shape is correct
        if len(self.mean) != len(self.init) or len(self.sigma) != len(self.init):
            raise ValueError(
                f"Priors must match length of statevector (statevector length:{len(self.init)})."
            )

        # Cache some important computations
        # NOTE for now this assumes no off diagonal elements
        Cov = np.diag(self.sigma**2)
        Cov_normalized = Cov / np.mean(np.diag(Cov))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            Cov_normalized
        )

        # NOTE LUTSurface currently is not compatible with analytical line
        self.analytical_iv_idx = np.arange(self.n_state)

    def xa(self, x_surface, geom):
        """Mean of prior distribution."""

        mu = np.zeros(self.n_state)
        mu[self.idx_lut] = self.mean.copy()

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        variance = pow(self.sigma, 2)
        Sa_unnormalized = np.diag(variance)

        return Sa_unnormalized, self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized

    def Sb(self):
        """Uncertainty due to unmodeled variables."""
        return np.diagflat(np.power(self.bval, 2))

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""
        return self.init

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
        """
        point = self.get_point(x_surface, geom)
        rho_dir_dir = rho_dif_dir = self.itp_hd(point)

        if self.itp_dd is not None:
            rho_dir_dir = self.itp_dd(point)

        # Return here if this is not a mixed pixel
        if not self.solve_mixed_pixel:
            return rho_dir_dir, rho_dif_dir

        # Apply softmax for fractional components
        f = self.softmax(np.array(x_surface[self.idx_em_rfls]))

        # Apply linear mixture
        rho_dir_dir = rho_dir_dir * f[0] + np.dot(self.endmember_matrix, f[1:])
        rho_dif_dir = rho_dif_dir * f[0] + np.dot(self.endmember_matrix, f[1:])

        return rho_dir_dir, rho_dif_dir

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""
        _, rho_dif = self.calc_rfl(x_surface, geom)
        return rho_dif

    def get_point(self, x_surface, geom):
        """create point in grid prior to VectorInterpolator."""
        point = np.zeros(self.n_lut)

        for v, idx in zip(x_surface, self.statevec_idxs):
            point[idx] = v

        # Either take cosi from geom or from state, and clip to atm config.
        if self.cos_i_idx is not None:
            cos_i = x_surface[self.cos_i_idx]
        else:
            cos_i = geom.cos_i

        # solar zenith, view zenith, and relative azimuth are optional indicies
        if self.sza_idx is not None:
            point[self.sza_idx] = np.degrees(np.arccos(cos_i))

        if self.vza_idx is not None:
            point[self.vza_idx] = geom.observer_zenith

        if self.raa_idx is not None:
            point[self.raa_idx] = geom.relative_azimuth

        # Ensure the point is contained in the lut grid
        for i, grid_axis in enumerate(self.lut_grid):
            point[i] = max(grid_axis[0], min(point[i], grid_axis[-1]))

        return point

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface.  We calculate the
        reflectance with multilinear interpolation so the finite
        difference derivative is exact."""

        base = self.calc_lamb(x_surface, geom)
        dlamb = []

        for xi in range(self.n_state):
            x_new = x_surface.copy()
            x_new[xi] = x_new[xi] + eps
            perturbed = self.calc_lamb(x_new, geom)
            dlamb.append((perturbed - base) / eps)

        dlamb = np.array(dlamb).T

        return dlamb

    def drdn_drfl(self, L_tot, s_alb, rho_dif_dir):
        """Partial derivative of radiance with respect to
        surface reflectance"""

        return L_tot / ((1.0 - s_alb * rho_dif_dir) ** 2)

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to
        state vector, calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_state), dtype=float)

        return dLs

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

        drdn_dLs = t_total_up

        drdn_dsurface = np.zeros(drfl_dsurface.shape)
        drdn_drfl = self.drdn_drfl(L_tot, s_alb, rho_dif_dir)

        # Construct the output matrix:
        # Dimensions should be (len(RT.wl), len(x_surface))
        # which is correctly handled by the instrument resampling
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
        background - s * rho_bg

        NOTE FOR SURFACE_LUT:
        To avoid confusion this does not output anything.
        """
        pass

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ""

        return "Surface: " + " ".join(
            [f"{n}: {v:5.4f}" for n, v in zip(self.statevec_names, x_surface)]
        )
