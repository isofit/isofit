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

from isofit.core.common import svd_inv_sqrt, eps
from isofit.surface.surface import Surface
from isofit.core.units import micron_to_nm
from isofit.core.common import VectorInterpolator


KEYS = [
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
]


class LUTSurface(Surface):
    """A model of the surface based on an N-dimensional lookup table
    indexed by one or more state vector elements.  We calculate the
    reflectance by multilinear interpolation.  This is good for
    surfaces like aquatic ecosystems or snow that can be
    described with just a few degrees of freedom.

    The lookup table must be precalculated based on the wavelengths
    of the instrument from NetCDF (.nc) format.


    For the NetCDF lookup table, it contains the following fields:
        - Coordinates:
            * wl
            * Other LUT dimensions (e.g., solar_zenith, observer_zenith, relative_azimuth, grain_size)
        - Data Variables:
            * rho_dif_dir: (LUT Axes..., n_wl)
            * rho_dir_dir: (LUT Axes..., n_wl) [optional]
            * statevec_names: (n_state)

    Reflectance keys should either be rho_dif_dir or rho_dir_dir (or both can be included). At least rho_dif_dir is required.

    Any of the angles are optional, but if provided should be in degrees and named "solar_zenith", "observer_zenith", and "relative_azimuth".

    You can also choose to run a mixed pixel retrieval by adding data variables of length wl with key name "endmember_TYPE".
    Where you can fill in TYPE for given surface(s) you would like to mix. You may use any number of endmembers.
    For example, you could have data variables named: "endmember_CEANOTHUS", "endmember_CONIFER". The key thing is to have
    the "endmember_" before the type for the reader to find the data correctly.

    Below is an example output structure for the xarray case:

    ```python
        # Example structure for xarray dataset for LUTSurface
        shape = (len(sza_list), len(vza_list), len(raa_list), len(grain_list), len(WL_NM))
        ds = xr.Dataset(
            {
                "rho_dir_dir": (["solar_zenith", "observer_zenith", "relative_azimuth", "grain_radius", "wl"],
                                np.full(shape, np.nan, dtype=np.float32)),
                "rho_dif_dir": (["solar_zenith", "observer_zenith", "relative_azimuth", "grain_radius", "wl"],
                                np.full(shape, np.nan, dtype=np.float32)),
                "statevec_names": (["n_state"], statevec_names),
                "endmember_conifer": (["wl"], soil_spectrum)
            },
            coords={
                "solar_zenith": sza_list,
                "observer_zenith": vza_list,
                "relative_azimuth": raa_list,
                "grain_radius": grain_list,
                "wl": WL_NM,
            }
        )
    ```

    """

    def __init__(self, full_config: Config):
        """."""

        super().__init__(full_config)
        config = full_config.forward_model.surface
        self.terrain_style = config.terrain_style
        self.max_slope = config.max_slope

        # Load dif-dir rfl data, optional dir-dir term, and other important parameters from the surface LUT
        self.itp_hd, self.itp_dd, lut_params = load_prebuilt_surface(
            surface_lut_file=config.surface_lut_file,
            terrain_style=self.terrain_style,
            build_interpolators=True,
        )

        for key in KEYS:
            setattr(self, key, lut_params[key])

        # First, stash important lengths and indices from the LUT
        self.n_wl = len(self.wl)
        self.n_state = len(self.statevec_names)
        self.n_lut = len(self.lut_names)
        self.idx_lut = np.arange(self.n_state)
        self.idx_lamb = np.arange(self.n_wl)
        self.idx_surface = np.arange(len(self.statevec_names))
        self.idx_em_rfls = []
        if self.solve_mixed_pixel:
            self.idx_em_rfls = [self.idx_surface[self.idx_fractional_data]]
            self.idx_em_rfls.extend(
                [
                    self.idx_surface[self.idx_fractional_em[f"FRACTIONAL_{n}"]]
                    for n in self.endmember_names
                ]
            )

        # Then, assign the priors and optimizaton parameters from the surface config
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

        # Cache some important computations
        # NOTE for now this assumes no off diagonal elements
        Cov = np.diag(self.sigma**2)
        Cov_normalized = Cov / np.mean(np.diag(Cov))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            Cov_normalized
        )

        # NOTE LUTSurface currently is not compatible with analytical line
        self.analytical_iv_idx = np.arange(self.n_state)

        if self.use_background_rfl:
            self.drdn_drfl = self.drdn_drfl_heterogeneous_bkg
        else:
            self.drdn_drfl = self.drdn_drfl_homogeneous_bkg

    def update_heuristic_prior_means(self, x_surface, geom):
        """Don't update any of the priors. Return xa"""
        return self.xa(x_surface, geom)

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

        # Either take cosi from geom or from state
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
        "Used to maintain sum-to-1 condition and positive fractional covers"
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

    def drdn_drfl_heterogeneous_bkg(
        self, L_tot, s_alb, rho_dif_dir, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif
    ):
        """Partial derivative of radiance with respect to
        surface reflectance, treating dir-dif and dif-dif as constants."""
        return L_dir_dir + (L_dif_dir / (1.0 - s_alb * rho_dif_dir))

    def drdn_drfl_homogeneous_bkg(
        self, L_tot, s_alb, rho_dif_dir, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif
    ):
        """Partial derivative of radiance with respect to
        surface reflectance (dir-dir = dif-dir = dir-dif = dif-dif)."""
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
        drdn_drfl = self.drdn_drfl(
            L_tot,
            s_alb,
            rho_dif_dir,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
        )

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
        L_tot,
        geom,
        s_alb=None,
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


@staticmethod
def load_prebuilt_surface(
    surface_lut_file, terrain_style="flat", build_interpolators=True
):
    """
    Used under the hood for LUTSurface() (as well as config creation) to load required data into ISOFIT.
    A more thorough description of how to create the input data is provided in LUTSurface().

    Parameters
    ----------
    surface_lut_file: str
        Path to the prebuilt surface LUT
    terrain_style: str
        Terrain style used in ISOFIT ("flat", "dem", "solved")
    statevector_only: bool
        If set to true, this method does not create the interpolator objects

    Returns
    -------
    itp_hd, itp_dd, lut_params: tuple
        diffuse-direct interpolator, direct-direct interpolator, additional lut parameters
    """
    lut_params = {}

    # NOTE assumes xarray
    with xr.open_dataset(surface_lut_file) as ds:
        data = {k: ds[k].values for k in ds.data_vars}
        for k in ds.coords:
            data[k] = ds[k].values
        lut_names = [str(n) for n in ds.coords.keys() if n != "wl"]
        lut_grid = [ds[n].values.astype(np.float32) for n in lut_names]
        wl = ds["wl"].values

    # Ensure wavelength are in nanometers
    if wl[0] < 300.0:
        wl = micron_to_nm(wl)

    # Enforce statevector to be uppercase to match style of the others
    lut_names = [n.upper() for n in lut_names]

    # Set dimensions based on lut (prior to endmembers)
    statevec_names = [n.strip().upper() for n in data["statevec_names"]]
    statevec_idxs = [lut_names.index(n) for n in statevec_names]
    n_lut_states = len(statevec_idxs)

    # Grab endmember data if present and save indicies
    endmembers = {
        k.replace("endmember_", "").upper(): v
        for k, v in data.items()
        if k.lower().startswith("endmember_")
    }
    endmember_names = list(endmembers.keys())

    # Create matrix for linear mixture
    if len(endmember_names) > 0:
        endmember_matrix = np.column_stack(
            [endmembers[name] for name in endmember_names]
        )
    else:
        endmember_matrix = np.array([])

    # Create each of the fractional parts of the statevector
    if len(endmember_names) > 0:
        if "FRACTIONAL_DATA" not in statevec_names:
            statevec_names.append("FRACTIONAL_DATA")
        for em_name in endmember_names:
            statevec_names.append(f"FRACTIONAL_{em_name.upper()}")

    idx_fractional_data = next(
        (i for i, n in enumerate(statevec_names) if n == "FRACTIONAL_DATA"), None
    )
    idx_fractional_em = {
        name: i
        for i, name in enumerate(statevec_names)
        if name.startswith("FRACTIONAL_")
        and name.replace("FRACTIONAL_", "") in endmembers
    }
    solve_mixed_pixel = idx_fractional_data is not None

    # Add cos_i to statevector if needed
    if terrain_style == "solved" and "COS_I" and "COS_I" not in statevec_names:
        statevec_names.append("COS_I")

    cos_i_idx = next((i for i, n in enumerate(statevec_names) if n == "COS_I"), None)

    # Find any relevant geometry indices
    # This assumes LUT is in units of degrees for geometry
    sza_idx = lut_names.index("solar_zenith") if "solar_zenith" in lut_names else None
    vza_idx = (
        lut_names.index("observer_zenith") if "observer_zenith" in lut_names else None
    )
    raa_idx = (
        lut_names.index("relative_azimuth") if "relative_azimuth" in lut_names else None
    )

    # Set default priors and optimization params on load,
    # these can be overwritten during config setup
    lut_statevector_data = {
        "statevec_names": [],
        "bounds": [],
        "init": [[]],
        "prior_mean": [[]],
        "prior_sigma": [[]],
        "scale": [[]],
    }

    for name in statevec_names:

        idx = lut_names.index(name) if name in lut_names else None
        if idx is not None:
            lb = lut_grid[idx].min().item()
            ub = lut_grid[idx].max().item()
            init = (lb + ub) / 2.0

        # Check for special cases
        # Fractional covers (-5 to 5 for softmax)
        elif name.startswith("FRACTIONAL_"):
            init = 0.0
            lb, ub = -5.0, 5.0

        # TODO cos_i as a free parameter is not yet supported but could revist this
        elif name == "COS_I":
            lb, ub = 1e-6, 1.0
            init = (lb + ub) / 2.0

        lut_statevector_data["statevec_names"].append(name)
        lut_statevector_data["bounds"].append([lb, ub])
        lut_statevector_data["init"][0].append(init)
        lut_statevector_data["prior_mean"][0].append(init)
        lut_statevector_data["prior_sigma"][0].append(1e6)
        lut_statevector_data["scale"][0].append(1.0)

    lut_params["lut_statevector_data"] = {
        "statevec_names": lut_statevector_data["statevec_names"],
        "bounds": np.array(lut_statevector_data["bounds"]),
        "init": np.array(lut_statevector_data["init"]),
        "prior_mean": np.array(lut_statevector_data["prior_mean"]),
        "prior_sigma": np.array(lut_statevector_data["prior_sigma"]),
        "scale": np.array(lut_statevector_data["scale"]),
    }

    # Defend against geom in the statevec or user missing lut name that is in statevec
    for name in statevec_names:
        if name.lower() in ["solar_zenith", "observer_zenith", "relative_azimuth"]:
            raise ValueError(
                f"Variable:{name.lower()} in the statevector is not supported."
            )
        if name.startswith("FRACTIONAL_"):
            continue
        if name not in lut_names:
            raise ValueError(
                f"Statevector:{name} not found in LUT dimensions: {lut_names}"
            )

    itp_hd = None
    itp_dd = None

    if build_interpolators:
        data_hd = data["rho_dif_dir"]
        data_dd = data.get("rho_dir_dir")
        itp_hd = VectorInterpolator(lut_grid, data_hd.astype(np.float32))
        if data_dd is not None:
            itp_dd = VectorInterpolator(lut_grid, data_dd.astype(np.float32))

    lut_params.update({k: locals()[k] for k in KEYS})

    return itp_hd, itp_dd, lut_params
