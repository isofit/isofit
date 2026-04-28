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

from isofit.core.common import VectorInterpolator, svd_inv_sqrt, eps
from isofit.surface.surface import Surface, DefaultState


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
        self.terrain_style = full_config.forward_model.radiative_transfer.terrain_style

        self.bounds, self.scale, self.init, self.mean, self.sigma = [], [], [], [], []

        # Check if model is stored as dictionaries in .mat format
        if config.surface_lut_file.endswith(".mat"):
            data = loadmat(config.surface_lut_file)
            self.lut_grid = [grid[0].astype(np.float32) for grid in data["grids"][0]]
            self.lut_names = [name.strip() for name in data["lut_names"]]
            self.wl = data["wl"][0]
            data = {
                k: v.squeeze() if isinstance(v, np.ndarray) else v
                for k, v in data.items()
            }

        # Otherwise assume xarray
        else:
            with xr.open_dataset(config.surface_lut_file) as ds:
                data = {k: ds[k].values for k in ds.data_vars}
                for k in ds.coords:
                    data[k] = ds[k].values
                self.lut_names = [str(n) for n in ds.coords.keys() if n != "wl"]
                self.lut_grid = [
                    ds[n].values.astype(np.float32) for n in self.lut_names
                ]
                self.wl = ds["wl"].values

        # Load rfl data and optional dir-dir term
        self.data_rho_dif = data["rho_dif_dir"]
        self.data_rho_dir = data.get("rho_dir_dir")

        # Set dimensions based on lut (prior to endmembers)
        self.statevec_names = [n.strip() for n in data["statevec_names"]]
        self.statevec_idxs = [self.lut_names.index(n) for n in self.statevec_names]
        self.n_lut_states = len(self.statevec_idxs)

        # Grab endmember data if present and save indicies
        self.endmembers = {
            k.replace("endmember_", ""): v
            for k, v in data.items()
            if k.startswith("endmember_")
        }
        self.em_names = list(self.endmembers.keys())

        # Create each of the zfrac parts of the statevector
        # For reference this is related to fractional cover by e.g.:
        # z = np.array([zfrac_data , zfrac_em_1 ... , zfrac_em_n])
        # f = np.exp(z) / np.sum(np.exp(z))
        # f_em_n = f[n]
        if len(self.em_names) > 0:
            if "zfrac_data" not in self.statevec_names:
                self.statevec_names.append("zfrac_data")
            for em_name in self.em_names:
                self.statevec_names.append(f"zfrac_{em_name}")

        self.idx_z_data = next(
            (i for i, n in enumerate(self.statevec_names) if n == "zfrac_data"), None
        )
        self.idx_z_ems = {
            name: i
            for i, name in enumerate(self.statevec_names)
            if name.startswith("zfrac_")
            and name.replace("zfrac_", "") in self.endmembers
        }
        self.solve_mixed_pixel = self.idx_z_data is not None

        # Add cos_i to statevector if needed
        if self.terrain_style == "solved":
            self.statevec_names.append("cos_i")
        self.cos_i_idx = next(
            (i for i, n in enumerate(self.statevec_names) if n == "cos_i"), None
        )

        # Store important idxs and lengths
        self.n_wl = len(self.wl)
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lut = np.arange(self.n_state)
        self.idx_lamb = np.arange(self.n_wl)
        self.idx_surface = np.arange(len(self.statevec_names))

        # Checking the statevec names prior to running
        for name in self.statevec_names:
            if name in ["solar_zenith", "observer_zenith", "relative_azimuth"]:
                raise ValueError(
                    f"Variable:{name} in the statevector is not supported."
                )
            if name.startswith("zfrac_"):
                continue
            if name not in self.lut_names:
                raise ValueError(
                    f"Statevector:{name} not found in LUT dimensions: {self.lut_names}"
                )

        # Populate priors and optimization parameters
        for name in self.statevec_names:
            idx = self.lut_names.index(name) if name in self.lut_names else None
            state = self.get_default_state(name=name, config=config, index=idx)

            self.bounds.append(state.bounds)
            self.scale.append(state.scale)
            self.init.append(state.init)
            self.mean.append(state.prior_mean)
            self.sigma.append(state.prior_sigma)

            # Defend against partially filled in config, this may error later but safe to check here
            for field in state._fields:
                if getattr(state, field) is None:
                    raise AttributeError(
                        f"Variable '{name}' missing '{field}' in surface config."
                    )

        self.mean = np.array(self.mean)
        self.sigma = np.array(self.sigma)

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

        # Build the interpolator(s), once again the dir-dir is optional
        self.itp_dif = None
        self.itp_dir = None
        self.itp_dif = VectorInterpolator(
            self.lut_grid, self.data_rho_dif.astype(np.float32)
        )
        if self.data_rho_dir is not None:
            self.itp_dir = VectorInterpolator(
                self.lut_grid, self.data_rho_dir.astype(np.float32)
            )

        # NOTE LUTSurface currently is not compatible with analytical line
        self.analytical_iv_idx = np.arange(self.n_state)

        # Find any relevant geometry indices
        # This assumes LUT is in units of degrees for geometry
        self.sza_ind = self.lut_names.index("solar_zenith") if "solar_zenith" in self.lut_names else None
        self.vza_ind = self.lut_names.index("observer_zenith") if "observer_zenith" in self.lut_names else None
        self.raa_ind = self.lut_names.index("relative_azimuth") if "relative_azimuth" in self.lut_names else None

    def get_default_state(self, name, config, index=None):
        """Used to handle dynamic loading of state config which are unknown in the LUTSurface case."""

        # First priority goes to any that are defined in the config
        sv = getattr(config, "statevector", None)
        if sv and name in sv:
            return DefaultState(
                bounds=sv[name].get("bounds"),
                scale=sv[name].get("scale"),
                prior_mean=sv[name].get("prior_mean"),
                prior_sigma=sv[name].get("prior_sigma"),
                init=sv[name].get("init"),
            )

        # Fallback to setting it based on the dataset, with uninformative priors
        # zfrac and cos_i are based on predefined values in this case
        if name.startswith("zfrac_"):
            return DefaultState(
                bounds=[-5.0, 5.0], scale=1.0, prior_mean=0.0, prior_sigma=1e6, init=0.0
            )
        elif name == "cos_i":
            return DefaultState(
                bounds=[1e-6, 1.0], scale=1.0, prior_mean=0.5, prior_sigma=1e6, init=0.5
            )
        elif index is not None:
            lb = self.lut_grid[index].min().item()
            ub = self.lut_grid[index].max().item()
            middle_value = (lb + ub) / 2.0
            return DefaultState(
                bounds=[lb, ub],
                scale=1.0,
                prior_mean=middle_value,
                prior_sigma=1e6,
                init=middle_value,
            )

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
        rho_dir_dir = rho_dif_dir = self.itp_dif(point)

        if self.itp_dir is not None:
            rho_dir_dir = self.itp_dir(point)

        # Return here if this is not a mixed pixel
        if not self.solve_mixed_pixel:
            return rho_dir_dir, rho_dif_dir

        # Apply softmax for fractional components
        z = [x_surface[self.idx_z_data]]
        for name in self.em_names:
            z.append(x_surface[self.idx_z_ems[f"zfrac_{name}"]])
        f = np.exp(np.array(z)) / np.sum(np.exp(np.array(z)))

        # ... to first the LUT data
        rho_dir_dir = rho_dir_dir * f[0]
        rho_dif_dir = rho_dif_dir * f[0]

        # ... and then n-number of endmembers
        for i, name in enumerate(self.em_names):
            rho_dir_dir += self.endmembers[name] * f[i + 1]
            rho_dif_dir += self.endmembers[name] * f[i + 1]

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

        # Either take cosi from geom or from state, and clip to rte config.
        if self.cos_i_idx is not None:
            cos_i = x_surface[self.cos_i_idx]
        else:
            cos_i = geom.cos_i

        # SZA, VZA, and RAA are optional indicies
        if self.sza_ind is not None:
            point[self.sza_ind] = np.degrees(np.arccos(cos_i))

        if self.vza_ind is not None:
            point[self.vza_ind] = geom.observer_zenith

        if self.raa_ind is not None:
            point[self.raa_ind] = geom.relative_azimuth

        # Ensure the point is contained in the lut grid
        for i, grid_axis in enumerate(self.lut_grid):
            point[i] = max(grid_axis[0], min(point[i], grid_axis[-1]))

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
