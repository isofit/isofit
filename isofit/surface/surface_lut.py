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
import xarray as xr
from scipy.io import loadmat
from scipy.optimize import minimize

from isofit.core.common import VectorInterpolator, svd_inv_sqrt
from isofit.surface.surface import Surface


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
            * Other LUT dimensions (e.g., SZA, VZA, RAA, grain size)
        - Data Variables:
            * rho_dif_dir: (LUT Axes..., n_wl)
            * rho_dir_dir: (LUT Axes..., n_wl) [optional]
            * statevec_names: (n_state) [optional]
            * mean: (n_state) [optional]
            * sigma: (n_state) [optional]

    Reflectance keys should either be rho_dif_dir (or both can be included).

    Any of the angles are optional, but if provided should be in degrees and named "SZA", "VZA", and "RAA".

    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)
        config = full_config.forward_model.surface
        surface_n_wl = self.n_wl

        # Optimization parameters, if not set will come from the data, with a default scale of 1.0.
        # This could potentially live in surface config?
        self.bounds = None
        self.scale = None
        self.init = None

        # Check if model is stored as dictionaries in .mat format
        if config.surface_file.endswith(".mat"):
            model_dict = loadmat(config.surface_file)
            self.lut_grid = [grid[0] for grid in model_dict["grids"][0]]
            self.lut_names = [name.strip() for name in model_dict["lut_names"]]
            self.statevec_names = [sv.strip() for sv in model_dict["statevec_names"]]
            self.data_rho_dif = model_dict["rho_dif_dir"]
            self.data_rho_dir = model_dict.get("rho_dir_dir")
            self.wl = model_dict["wl"][0]
            self.mean = model_dict["mean"].ravel() if "mean" in model_dict else None
            self.sigma = model_dict["sigma"].ravel() if "sigma" in model_dict else None

        # Otherwise, assume xarray
        else:
            with xr.open_dataset(config.surface_file) as ds:
                self.lut_names = [str(n) for n in ds.coords.keys() if n != "wl"]
                self.lut_grid = [ds[n].values for n in self.lut_names]
                self.wl = ds["wl"].values
                self.data_rho_dif = ds["rho_dif_dir"].values
                self.data_rho_dir = (
                    ds["rho_dir_dir"].values if "rho_dir_dir" in ds else None
                )
                self.statevec_names = ds["statevec_names"].astype(str).values.tolist()
                self.mean = ds["mean"].values.ravel() if "mean" in ds else None
                self.sigma = ds["sigma"].values.ravel() if "sigma" in ds else None

        # Common dimensions
        self.n_wl = len(self.wl)
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lut = np.arange(self.n_state)
        self.idx_lamb = np.arange(self.n_wl)
        self.statevec_idxs = [self.lut_names.index(n) for n in self.statevec_names]

        # Ensure the LUT has been convolved to sensor wavelengths
        if self.n_wl != surface_n_wl:
            raise ValueError(
                f"LUTSurface grid (shape:{self.n_wl}) does not match input data (shape:{surface_n_wl})"
            )

        # Checking the statevec names prior to running
        for name in self.statevec_names:
            if name in ["SZA", "VZA", "RAA"]:
                raise ValueError(
                    f"Variable:{name} in the statevector is not supported."
                )
            if name not in self.lut_names:
                raise ValueError(
                    f"Statevector:{name} not found in LUT dimensions: {self.lut_names}"
                )

        # Set the optimization parameters based on data if none are provided
        if self.bounds is None:
            self.bounds = [[g.min(), g.max()] for g in self.lut_grid]
        if self.scale is None:
            self.scale = np.ones(self.n_state)
        if self.init is None:
            self.init = np.array([(b[0] + b[1]) / 2.0 for b in self.bounds])

        # If no priors given, assume uninformative, flat priors
        if self.mean is None:
            self.mean = self.init.copy()
        if self.sigma is None:
            self.sigma = np.ones(self.n_state) * 1e6

        # Ensure priors shape is correct
        if len(self.mean) != len(self.init) or len(self.sigma) != len(self.init):
            raise ValueError(
                f"Priors must match length of statevector (statevector length:{len(self.init)})."
            )

        # Cache some important computations
        Cov = np.diag(self.sigma**2)
        Cov_normalized = Cov / np.mean(np.diag(Cov))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            Cov_normalized
        )

        # Build the interpolator(s)
        self.itp_dif = None
        self.itp_dir = None
        self.itp_dif = VectorInterpolator(self.lut_grid, self.data_rho_dif)
        if self.data_rho_dir is not None:
            self.itp_dir = VectorInterpolator(self.lut_grid, self.data_rho_dir)

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(self.n_state)

        # Find any relevant geometry indices
        # This assumes LUT is in units of degrees for geometry
        self.sza_ind = self.lut_names.index("SZA") if "SZA" in self.lut_names else None
        self.vza_ind = self.lut_names.index("VZA") if "VZA" in self.lut_names else None
        self.raa_ind = self.lut_names.index("RAA") if "RAA" in self.lut_names else None

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
        init_point = self.get_point(self.init, geom)

        def fun(x):
            point = init_point.copy()

            for v, idx in zip(x, self.statevec_idxs):
                point[idx] = v

            return np.sum((self.itp_dif(point) - rfl_meas) ** 2)

        result = minimize(
            fun=fun,
            x0=self.init,
            method="L-BFGS-B",
            bounds=self.bounds,
        )

        return result.x

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

        rho_dir_dir = rho_dif_dir = self.itp_dif(point)

        if self.itp_dir is not None:
            rho_dir_dir = self.itp_dir(point)

        return rho_dir_dir, rho_dif_dir

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""
        return self.itp_dif(self.get_point(x_surface, geom))

    def get_point(self, x_surface, geom):
        """create point in grid prior to VectorInterpolator."""
        point = np.zeros(self.n_lut)

        for v, idx in zip(x_surface, self.statevec_idxs):
            point[idx] = v

        cos_i = geom.verify(geom.solar_zenith)["cos_i"]

        if self.sza_ind is not None and "cos_i" not in self.statevec_names:
            point[self.sza_ind] = np.degrees(np.arccos(cos_i))

        if self.vza_ind is not None:
            point[self.vza_ind] = geom.observer_zenith

        if self.raa_ind is not None:
            point[self.raa_ind] = geom.relative_azimuth

        # Ensure the point is contained in the lut grid
        for i, grid_axis in enumerate(self.lut_grid):
            point[i] = np.clip(point[i], grid_axis.min(), grid_axis.max())

        return point

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface.  We calculate the
        reflectance with multilinear interpolation so the finite
        difference derivative is exact."""

        eps = 1e-6
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
        This assumes that the only surface statevector terms are
        surface reflectance terms. Any additional surface state elements
        have to be explicitely handled in this function. How they are
        handled is dependent on the nature of the surface rfl model.
        The n-columns of H is equal to the number of statevector elements.
        Here, set to the number of wavelengths.
        """
        theta = L_tot + (L_tot * background)
        H = np.eye(self.n_wl, self.n_wl)
        H = theta[:, np.newaxis] * H

        return H

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        if len(x_surface) < 1:
            return ""

        return "Surface: " + " ".join(
            [f"{n}: {v:5.4f}" for n, v in zip(self.statevec_names, x_surface)]
        )
