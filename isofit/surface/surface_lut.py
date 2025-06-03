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
from scipy.io import loadmat

from isofit.core.common import VectorInterpolator
from isofit.surface.surface import Surface


class LUTSurface(Surface):
    """A model of the surface based on an N-dimensional lookup table
    indexed by one or more state vector elements.  We calculate the
    reflectance by multilinear interpolation.  This is good for
    surfaces like aquatic ecosystems or snow that can be
    described with just a few degrees of freedom.

    The lookup table must be precalculated based on the wavelengths
    of the instrument.  It is stored with other metadata in a matlab-
    format file. For an n-dimensional lookup table, it contains the
    following fields:
      - grids: an object array containing n lists of gridpoints
      - data: an n+1 dimensional array containing the reflectances
         for each gridpoint
      - bounds: a list of n [min,max] tuples representing the bounds
         for all state vector elements
      - statevec_names: an array of n strings representing state
         vector element names
      - mean: an array of n prior mean values, one for each state
         vector element
      - sigma: an array of n prior standard deviations, one for each
         state vector element
      - scale: an array of n scale values, one for each state vector
         element

    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        # Models are stored as dictionaries in .mat format
        model_dict = loadmat(config.surface_file)
        self.lut_grid = [grid[0] for grid in model_dict["grids"][0]]
        self.lut_names = [name.strip() for name in model_dict["lut_names"]]
        self.statevec_names = [sv.strip() for sv in model_dict["statevec_names"]]
        self.data = model_dict["data"]
        self.wl = model_dict["wl"][0]
        self.n_wl = len(self.wl)
        self.bounds = model_dict["bounds"]
        self.scale = model_dict["scale"][0]
        self.init = model_dict["init"][0]
        self.mean = model_dict["mean"][0]
        self.sigma = model_dict["sigma"][0]
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lut = np.arange(self.n_state)
        self.idx_lamb = np.empty(shape=0)

        # build the interpolator
        self.itp = VectorInterpolator(self.lut_grid, self.data)

        # Change this if you don't want to analytical solve for all the full statevector elements.
        self.analytical_iv_idx = np.arange(len(self.statevec_names))

    def xa(self, x_surface, geom):
        """Mean of prior distribution."""

        mu = np.zeros(self.n_state)
        mu[self.idx_lut] = self.mean.copy()

        return mu

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        variance = pow(self.sigma, 2)
        Cov = np.diag(variance)

        return Cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = self.mean.copy()

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
        """Lambertian reflectance.  Be sure to incorporate BRDF-related
        LUT dimensions such as solar and view zenith."""

        point = np.zeros(self.n_lut)

        for v, name in zip(x_surface, self.statevec_names):
            point[self.lut_names.index(name)] = v

        if "SOLZEN" in self.lut_names:
            solzen_ind = self.lut_names.index("SOLZEN")
            point[solzen_ind] = geom.solar_zenith

        if "VIEWZEN" in self.lut_names:
            viewzen_ind = self.lut_names.index("VIEWZEN")
            point[viewzen_ind] = geom.observer_zenith

        lamb = self.itp(point)

        return lamb

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
        L_down_dir,
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

        return "Surface: " + " ".join([("%5.4f" % x) for x in x_surface])
