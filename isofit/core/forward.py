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

import logging
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.linalg import block_diag

from isofit.atmosphere.atmosphere import Atmosphere
from isofit.core.common import eps
from isofit.core.geometry import Geometry
from isofit.core.instrument import Instrument
from isofit.core.multistate import match_statevector
from isofit.core.units import transm_to_rdn
from isofit.surface import Surface

Logger = logging.getLogger(__file__)


class ForwardModel:
    """ForwardModel contains all the information about how to calculate
    radiance measurements at a specific spectral calibration, given a
    state vector. It also manages the distributions of unretrieved,
    unknown parameters of the state vector (i.e. the S_b and K_b
    matrices of Rodgers et al.

    State vector elements always go in the following order:
      (1) Surface parameters
      (2) Atmospheric Radiative Transfer parameters
      (3) Instrument parameters

    The parameter bounds, scales, initial values, and names are all
    ordered in this way.  The variable self.statevec contains the name
    of each state vector element, in the proper ordering.

    The "b" vector corresponds to the K_b calculations in Rogers (2000);
    the variables bvec and bval represent the model unknowns' names and
    their  magnitudes, respectively.  Larger magnitudes correspond to
    a larger variance in the unknown values.  This acts as additional
    noise for the purpose of weighting the measurement information
    against the prior."""

    def __init__(self, full_config: Config, cache_atmosphere: Atmosphere = None):
        # load in the full config (in case of inter-module dependencies) and
        # then designate the current config
        self.full_config = full_config

        # Build the instrument model
        self.instrument = Instrument(self.full_config)
        self.n_meas = self.instrument.n_chan

        # Build the atmospheric radiative transfer model
        if cache_atmosphere:
            self.atmosphere = cache_atmosphere
        else:
            self.atmosphere = Atmosphere(self.full_config)

        # Build the surface model
        self.surface = Surface(full_config)

        # Check to see if using supported calibration surface model
        if self.surface.n_wl != len(self.atmosphere.wl) or not np.all(
            np.isclose(self.surface.wl, self.atmosphere.wl, atol=0.01)
        ):
            Logger.warning(
                "Surface and atmosphere wavelengths differ - if running at higher atmosphere"
                " spectral resolution or with variable wavelength position, this"
                " is expected. Otherwise, consider checking the surface model."
            )

        # Build combined vectors from surface, atmosphere, and instrument
        bounds, scale, init, statevec, bvec = ([] for i in range(5))
        for obj_with_statevec in [self.surface, self.atmosphere, self.instrument]:
            bounds.extend([deepcopy(x) for x in obj_with_statevec.bounds])
            scale.extend([deepcopy(x) for x in obj_with_statevec.scale])
            init.extend([deepcopy(x) for x in obj_with_statevec.init])
            statevec.extend([deepcopy(x) for x in obj_with_statevec.statevec_names])

            bvec.extend([deepcopy(x) for x in obj_with_statevec.bvec])

        self.bounds = tuple(np.array(bounds).T)
        self.scale = np.array(scale)
        self.init = np.array(init)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        self.bvec = np.array(bvec)
        self.nbvec = len(self.bvec)

        """Set up state vector indices - 
        MUST MATCH ORDER FROM ABOVE ASSIGNMENT

        Sometimes, it's convenient to have the index of the entire surface
        as one variable, and sometimes you want the sub-components
        
        Split surface state vector indices to cover cases where we retrieve
        additional non-reflectance surface parameters
        """
        self.full_idx = np.arange(len(self.statevec))
        self.full_miss = []

        # entire surface portion
        self.idx_surface = np.arange(len(self.surface.statevec_names), dtype=int)

        # surface reflectance portion
        self.idx_surf_rfl = self.idx_surface[: len(self.surface.idx_lamb)]

        # non-reflectance surface parameters
        self.idx_surf_nonrfl = self.idx_surface[len(self.surface.idx_lamb) :]

        # atmospheric radiative transfer portion
        self.idx_atmosphere = np.arange(
            len(self.atmosphere.statevec_names), dtype=int
        ) + len(self.idx_surface)

        # instrument portion
        self.idx_instrument = (
            np.arange(len(self.instrument.statevec_names), dtype=int)
            + len(self.idx_surface)
            + len(self.idx_atmosphere)
        )

        self.surface_b_inds = np.arange(len(self.surface.bvec), dtype=int)
        self.atmosphere_b_inds = np.arange(len(self.atmosphere.bvec), dtype=int) + len(
            self.surface_b_inds
        )
        self.instrument_b_inds = (
            np.arange(len(self.instrument.bvec), dtype=int)
            + len(self.surface_b_inds)
            + len(self.atmosphere_b_inds)
        )

        # Load model discrepancy correction
        if full_config.forward_model.model_discrepancy_file is not None:
            D = loadmat(full_config.forward_model.model_discrepancy_file)
            self.model_discrepancy = D["cov"]
        else:
            self.model_discrepancy = None

        # Special run modes:
        self.multipart_transmittance = full_config.forward_model.multipart_transmittance

    def out_of_bounds(self, x):
        """Check if state vector is within bounds."""

        x_atmosphere = x[self.idx_atmosphere]
        bound_lwr = self.bounds[0]
        bound_upr = self.bounds[1]
        return any(x_atmosphere >= (bound_upr[self.idx_atmosphere] - eps * 2.0)) or any(
            x_atmosphere <= (bound_lwr[self.idx_atmosphere] + eps * 2.0)
        )

    def xa(self, x, geom):
        """Calculate the prior mean of the state vector (the concatenation
        of state vectors for the surface, atmosphere, and instrument).

        NOTE: the surface prior mean depends on the current state;
        this is so we can calculate the local prior.
        """

        x_surface = x[self.idx_surface]
        xa_surface = self.surface.xa(x_surface, geom)
        xa_atmosphere = self.atmosphere.xa()
        xa_instrument = self.instrument.xa()
        return np.concatenate((xa_surface, xa_atmosphere, xa_instrument), axis=0)

    def Sa(self, x, geom):
        """Calculate the prior covariance of the state vector (the
        concatenation of state vectors for the surface and the atmosphere).

        NOTE: the surface prior depends on the current state; this
        is so we can calculate the local prior.
        """

        x_surface = x[self.idx_surface]
        Sa_surface, Sa_surf_inv_norm, Sa_surf_inv_sqrt_norm = self.surface.Sa(
            x_surface, geom
        )
        Sa_atmosphere = self.atmosphere.Sa()
        Sa_instrument = self.instrument.Sa()
        Sa_state = block_diag(
            Sa_surface[:, :], Sa_atmosphere[:, :], Sa_instrument[:, :]
        )

        # per block variance scaling for normalization
        scale_surface = np.sqrt(np.mean(np.diag(Sa_surface[:, :])))
        scale_atmosphere = np.sqrt(np.mean(np.diag(Sa_atmosphere[:, :])))
        scale_instrument = np.sqrt(np.mean(np.diag(Sa_instrument[:, :])))

        # Compute the Sa inv and Sa inv sqrt for measurement
        Sa_inv_state = block_diag(
            Sa_surf_inv_norm / scale_surface**2,
            self.atmosphere.Sa_inv_normalized / scale_atmosphere**2,
            self.instrument.Sa_inv_normalized / scale_instrument**2,
        )

        Sa_inv_sqrt_state = block_diag(
            Sa_surf_inv_sqrt_norm / scale_surface,
            self.atmosphere.Sa_inv_sqrt_normalized / scale_atmosphere,
            self.instrument.Sa_inv_sqrt_normalized / scale_instrument,
        )

        return Sa_state, Sa_inv_state, Sa_inv_sqrt_state

    def Sb(self, meas):
        """Accumulate the uncertainty due to unmodeled variables within
        respective forward model portions."""
        Sb_surface = self.surface.Sb()
        Sb_atmosphere = self.atmosphere.Sb()
        Sb_instrument = self.instrument.Sb(meas)

        return block_diag(Sb_surface, Sb_atmosphere, Sb_instrument)

    def eof_offset(self, x_instrument):
        """Empirical orthogonal function offset. FM wrapper in the style
        of xa, Sa, Seps in case we want to be able to extend this
        across surface, atmosphere, and instrument"""
        offset = self.instrument.eof_offset(x_instrument)

        return offset

    def calc_meas(self, x, geom, rfl=[]):
        """Calculate the model observation at instrument wavelengths."""
        # Unpack state vector - Copy to not change x fm-wide
        x_surface, x_atmosphere, x_instrument = self.unpack(np.copy(x))

        # if rfl passed, have to explicitly use those values
        if len(rfl):
            x_surface[self.idx_surf_rfl] = rfl

        # Call surface reflectance w.r.t. surface, upsample
        rho_dir_dir, rho_dif_dir = self.calc_rfl(x, geom)
        rho_dir_dir_hi = self.upsample(self.surface.wl, rho_dir_dir)
        rho_dif_dir_hi = self.upsample(self.surface.wl, rho_dif_dir)

        # Adjacency effects
        rho_dir_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dir_dir_hi
        )
        rho_dif_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dif_dir_hi
        )

        # Get atmosphere quantities
        (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        ) = self.calc_atmosphere_quantities(x_atmosphere, geom, rho_dif_dif_hi)

        # Call surface emission, upsample
        Ls_hi = self.upsample(self.surface.wl, self.calc_Ls(x_surface, geom))

        rdn = self.calc_rdn(
            x_atmosphere,
            rho_dir_dir=rho_dir_dir_hi,
            rho_dif_dir=rho_dif_dir_hi,
            rho_dir_dif=rho_dir_dif_hi,
            rho_dif_dif=rho_dif_dif_hi,
            Ls=Ls_hi,
            L_tot=L_tot,
            L_dir_dir=L_dir_dir,
            L_dif_dir=L_dif_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dif=L_dif_dif,
            r=r,
            geom=geom,
        )

        return self.instrument.sample(
            x_instrument, self.atmosphere.wl, rdn
        ) + self.eof_offset(x_instrument)

    def calc_atmosphere_quantities(
        self, x_atmosphere: np.ndarray, geom: Geometry, rho_dif_dif: np.ndarray = 0
    ):
        """Retrieves the atmosphere quantities including the LUT sample (r),
        and the radiances (L). This function handles the hand-off between
        the 1c and 6c model.

        In the 1c case, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = 0,
        and L_tot, L_down_dir, and L_down_dif are populated within the
        if statement.

        In the 6c case, we always use returns from get_L_coupled
        All quantities are on the sun-to-surface-to-sensor path.

        Args:
            x_atmosphere: atmosphere portion of the state vector.
            geom: Geometry object for the current observation.
            rho_dif_dif: Apparent surface reflectance for
                hemispherical-hemispherical photon paths.
                Included here to incorporate surface-atm coupling
                following Eq. 11 of Guanter et al., 2009.

        Returns:
            r: LUT sample dictionary of shared atmosphere quantities.
            L_tot: total downwelling radiance (uW/nm/sr/cm2).
            L_dir_dir: direct-to-direct radiance component; zero in 1c mode.
            L_dif_dir: diffuse-to-direct radiance component; zero in 1c mode.
            L_dir_dif: direct-to-diffuse radiance component; zero in 1c mode.
            L_dif_dif: diffuse-to-diffuse radiance component; zero in 1c mode.
        """

        # Propogate LUT
        r = self.atmosphere.get(x_atmosphere, geom)

        # Handle 1c L_tot. NOTE: transm_down_dif = total transm for 1c case.
        if self.atmosphere.multipart_transmittance:
            # Get directional radiances
            L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = self.get_L_coupled(
                r, geom, rho_dif_dif=rho_dif_dif
            )
        else:
            if self.atmosphere.atmosphere_mode == "rdn":
                L_tot = r["transm_down_dif"]
            else:
                L_tot = transm_to_rdn(
                    r["transm_down_dif"],
                    geom.coszen,
                    self.atmosphere.solar_irr,
                )
            L_dir_dir = 0
            L_dif_dir = 0
            L_dir_dif = 0
            L_dif_dif = 0

        return (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

    def get_L_coupled(self, r: dict, geom: Geometry, rho_dif_dif: np.ndarray = 0):
        """Get the interpolated radiance terms on the sun-to-surface-to-sensor path.
        These follow the physics as presented in Guanter (2006), Vermote et al. (1997), and Tanre et al. (1983).

        Note:   This function is only applicable to the 6c run case
                where r contains populated separated transmittances

        Args:
            r:      interpolated radiative transfer quantities from the LUT
            coszen: top-of-atmosphere solar zenith angle
            cos_i:  local solar zenith angle at the surface

        Returns:
            interpolated radiances along all optical paths:
            L_dir_dir => downward direct * upward direct
            L_dif_dir => downward diffuse * upward direct
            L_dir_dif => downward direct * upward diffuse
            L_dif_dif => downward diffuse * upward diffuse
        """

        # radiances along all optical paths
        L_coupled = []

        for key in self.atmosphere.coupling_terms:
            L_coupled.append(
                transm_to_rdn(
                    r[key], coszen=geom.coszen, solar_irr=self.atmosphere.solar_irr
                )
                if self.atmosphere.atmosphere_mode == "transm"
                else r[key]
            )

        # Topographic shadow mask (0=shadow, 1=sunlit pixel).
        # for now, this is always set to 1.0.
        b = 1.0

        # Assumption of the topography of the background
        cos_i_bg = geom.coszen
        skyview_factor_bg = 1.0

        # Assigning coupled terms, unscaling and rescaling downward direct radiance by local solar zenith angle.
        # Downward diffuse components are scaled by viewable sky fraction (i.e., "ungula" of viewable sky in solid geometry terms).
        L_dir_dir = L_coupled[0] / geom.coszen * geom.cos_i * b
        L_dif_dir = L_coupled[1]
        L_dir_dif = L_coupled[2] / geom.coszen * cos_i_bg
        L_dif_dif = L_coupled[3]

        # Note - we should really be doing the multiplication upstream before convolution - this is an approximation
        # Correct downward diffuse term for topographic assuming Hay's model (Hay 1979; Richter 1998; Guanter et al., 2009)
        t_down_dir = r["transm_down_dir"]
        L_dif_dir *= (b * t_down_dir * (geom.cos_i / geom.coszen)) + (
            (1 - b * t_down_dir) * geom.skyview_factor
        )
        L_dif_dif *= (t_down_dir * (cos_i_bg / geom.coszen)) + (
            (1 - t_down_dir) * skyview_factor_bg
        )

        # Apply equation 11
        # If no rho_dif_dif passed eq_11_term -> 1
        eq_11_term = 1 - (r["sphalb"] * rho_dif_dif)

        L_dif_dir /= eq_11_term
        L_dif_dif /= eq_11_term
        L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif

        return L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif

    def calc_rdn(
        self,
        x_atmosphere,
        rho_dir_dir,
        rho_dif_dir,
        rho_dir_dif,
        rho_dif_dif,
        Ls,
        L_tot,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
        r,
        geom,
    ):
        """
        Physics-based forward model to calculate at-sensor radiance.
        Includes topography, background reflectance, and glint.
        """
        # Atmospheric path radiance
        L_atm = self.atmosphere.get_L_atm(x_atmosphere, geom)

        # Atmospheric spherical albedo
        s_alb = r["sphalb"]
        atm_surface_scattering = s_alb * rho_dif_dif

        # Special case: 1-component model
        if not self.atmosphere.multipart_transmittance:
            # we assume rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
            rho_dif_dif = rho_dir_dir
            # eliminate spherical albedo and one reflectance term from numerator if using 1-component model
            atm_surface_scattering = 1

        # Thermal transmittance
        L_up = Ls * self.atmosphere.get_upward_transm(r=r, geom=geom)

        # Our radiance model follows the physics as presented in Guanter (2006), Vermote et al. (1997), and
        # Tanre et al. (1983). This particular formulation facilitates the consideration of topographic effects,
        # glint, or BRDF modeling in general. The contribution of the target to the signal at the top of the atmosphere
        # is decomposed as the sum of four terms:

        # 1. photons directly transmitted from the sun to the target and directly reflected back to the sensor
        #    rho_dir_dir => directional-directional surface reflectance of the target
        # 2. photons scattered by the atmosphere then reflected by the target and directly transmitted to the sensor
        #    rho_dif_dir => surface diffuse-directional reflectance
        # 3. photons directly transmitted to the target but scattered by the atmosphere on their way to the sensor
        #    rho_dir_dif => surface directional-diffuse reflectance
        # 4. photons having at least two interactions with the atmosphere and one with the target
        #    rho_dif_dif => surface diffuse-diffuse reflectance

        # These terms are also called coupling terms, as they are responsible for the coupling between atmospheric
        # radiative transfer and the surface reflectance properties.

        # The coupling terms are multiplied by four different combinations of direct and diffuse radiance terms:
        # 1. L_dir_dir => downward direct * upward direct
        # 2. L_dif_dir => downward diffuse * upward direct
        # 3. L_dir_dif => downward direct * upward diffuse
        # 4. L_dif_dif => downward diffuse * upward diffuse

        # When separated radiance terms and/or a BRDF model of the surface are not available,
        # the Lambertian assumption is made for the target reflectance:
        # rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
        # In this case, our radiance model reduces to:
        # L_atm + (L_tot * rho_dir_dir) / (1 - S * rho_dir_dir) + L_up,
        # with L_tot being the total radiance (downward * upward, direct + diffuse).

        # TOA radiance model
        ret = (
            L_atm
            + L_dir_dir * rho_dir_dir
            + L_dif_dir * rho_dif_dir
            + L_dir_dif * rho_dir_dif
            + L_dif_dif * rho_dif_dif
            + (L_tot * atm_surface_scattering * rho_dif_dif) / (1 - s_alb * rho_dif_dif)
            + L_up
        )

        return ret

    def calc_Ls(self, x, geom):
        """Calculate the surface emission."""

        return self.surface.calc_Ls(x[self.idx_surface], geom)

    def calc_rfl(self, x, geom):
        """Calculate the surface reflectance."""

        return self.surface.calc_rfl(x[self.idx_surface], geom)

    def calc_lamb(self, x, geom):
        """Calculate the Lambertian surface reflectance."""

        return self.surface.calc_lamb(x[self.idx_surface], geom)

    def Seps(self, x, meas, geom):
        """Calculate the total uncertainty of the observation, including
        up to three terms: (1) the instrument noise; (2) the uncertainty
        due to explicit unmodeled variables, i.e. the S_epsilon matrix of
        Rodgers et al.; and (3) an aggregate 'model discrepancy' term,
        Gamma."""

        if self.model_discrepancy is not None:
            Gamma = self.model_discrepancy
        else:
            Gamma = 0

        Sb = self.Sb(meas)
        Kb = self.Kb(x, geom)
        Sy = self.instrument.Sy(meas, geom)

        return Sy + Kb.dot(Sb).dot(Kb.T) + Gamma

    def K(self, x, geom):
        """Derivative of observation with respect to state vector. This is
        the concatenation of jacobians with respect to parameters of the
        surface and the atmosphere.
        """
        # Unpack state vector
        x_surface, x_atmosphere, x_instrument = self.unpack(x)

        # Call surface reflectance w.r.t. surface, upsample
        rho_dir_dir, rho_dif_dir = self.calc_rfl(x, geom)
        rho_dir_dir_hi = self.upsample(self.surface.wl, rho_dir_dir)
        rho_dif_dir_hi = self.upsample(self.surface.wl, rho_dif_dir)

        # Adjacency effects
        rho_dir_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dir_dir_hi
        )
        rho_dif_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dif_dir_hi
        )

        # Get atmosphere quantities
        (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        ) = self.calc_atmosphere_quantities(x_atmosphere, geom, rho_dif_dif_hi)

        # Call surface emission, upsample
        Ls_hi = self.upsample(self.surface.wl, self.calc_Ls(x_surface, geom))

        rdn = self.calc_rdn(
            x_atmosphere,
            rho_dir_dir=rho_dir_dir_hi,
            rho_dif_dir=rho_dif_dir_hi,
            rho_dir_dif=rho_dir_dif_hi,
            rho_dif_dif=rho_dif_dif_hi,
            Ls=Ls_hi,
            L_tot=L_tot,
            L_dir_dir=L_dir_dir,
            L_dif_dir=L_dif_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dif=L_dif_dif,
            r=r,
            geom=geom,
        )

        # Call derivative of rfl wrt surface state, upsample
        drfl_dsurface_hi = self.upsample(
            self.surface.wl,
            self.surface.drfl_dsurface(x_surface, geom).T,
        ).T

        # Call derivative of surface emission wrt surface state, upsample
        dLs_dsurface_hi = self.upsample(
            self.surface.wl, self.surface.dLs_dsurface(x_surface, geom).T
        ).T

        # To get the derivative w.r.t. atmosphere
        drdn_datmosphere = self.drdn_datmosphere(
            x_atmosphere,
            geom,
            rho_dir_dir=rho_dir_dir_hi,
            rho_dif_dir=rho_dif_dir_hi,
            rho_dir_dif=rho_dir_dif_hi,
            rho_dif_dif=rho_dif_dif_hi,
            Ls=Ls_hi,
            rdn=rdn,
        )

        # To get the derivative w.r.t. Surface
        # ToDo: Move this to ForwardModel as well?
        drdn_dsurface = self.surface.drdn_dsurface(
            rho_dif_dir=rho_dif_dir_hi,
            drfl_dsurface=drfl_dsurface_hi,
            dLs_dsurface=dLs_dsurface_hi,
            s_alb=r["sphalb"],
            t_total_up=self.atmosphere.get_upward_transm(r=r, geom=geom),
            L_tot=L_tot,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
        )

        # To get derivatives w.r.t. instrument, downsample to instrument wavelengths
        dmeas_dsurface = self.instrument.sample(
            x_instrument, self.atmosphere.wl, drdn_dsurface.T
        ).T
        dmeas_datmosphere = self.instrument.sample(
            x_instrument, self.atmosphere.wl, drdn_datmosphere.T
        ).T
        dmeas_dinstrument = self.instrument.dmeas_dinstrument(
            x_instrument, self.atmosphere.wl, rdn
        )

        # Put it all together
        K = np.zeros((self.n_meas, self.nstate), dtype=float)
        K[:, self.idx_surface] = dmeas_dsurface
        K[:, self.idx_atmosphere] = dmeas_datmosphere
        K[:, self.idx_instrument] = dmeas_dinstrument
        return K

    def Kb(self, x, geom):
        """Derivative of measurement with respect to unmodeled & unretrieved
        unknown variables, e.g. S_b. This is  the concatenation of Jacobians
        with respect to parameters of the surface, atmosphere,
        and instrument. Currently, we only treat uncertainties in the
        instrument and atmosphere model."""

        # Unpack state vector
        x_surface, x_atmosphere, x_instrument = self.unpack(x)

        # Call surface reflectance w.r.t. surface, upsample
        rho_dir_dir, rho_dif_dir = self.calc_rfl(x, geom)
        rho_dir_dir_hi = self.upsample(self.surface.wl, rho_dir_dir)
        rho_dif_dir_hi = self.upsample(self.surface.wl, rho_dif_dir)

        # Adjacency effects
        rho_dir_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dir_dir_hi
        )
        rho_dif_dif_hi = (
            self.upsample(self.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dif_dir_hi
        )

        # Get atmosphere quantities
        (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        ) = self.calc_atmosphere_quantities(x_atmosphere, geom, rho_dif_dif_hi)

        # Call surface emission, upsample
        Ls_hi = self.upsample(self.surface.wl, self.calc_Ls(x_surface, geom))

        rdn = self.calc_rdn(
            x_atmosphere,
            rho_dir_dir=rho_dir_dir_hi,
            rho_dif_dir=rho_dif_dir_hi,
            rho_dir_dif=rho_dir_dif_hi,
            rho_dif_dif=rho_dif_dif_hi,
            Ls=Ls_hi,
            L_tot=L_tot,
            L_dir_dir=L_dir_dir,
            L_dif_dir=L_dif_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dif=L_dif_dif,
            r=r,
            geom=geom,
        )

        drdn_datmosphereb = self.drdn_datmosphereb(
            x_atmosphere,
            geom=geom,
            rho_dir_dir=rho_dir_dir_hi,
            rho_dif_dir=rho_dif_dir_hi,
            rho_dir_dif=rho_dir_dif_hi,
            rho_dif_dif=rho_dif_dif_hi,
            Ls=Ls_hi,
            rdn=rdn,
        )

        # To get derivatives w.r.t. instrument, downsample to instrument wavelengths
        dmeas_datmosphereb = self.instrument.sample(
            x_instrument, self.atmosphere.wl, drdn_datmosphereb.T
        ).T
        dmeas_dinstrumentb = self.instrument.dmeas_dinstrumentb(
            x_instrument, self.atmosphere.wl, rdn
        )

        # Put it together
        Kb = np.zeros((self.n_meas, self.nbvec), dtype=float)
        Kb[:, self.atmosphere_b_inds] = dmeas_datmosphereb
        Kb[:, self.instrument_b_inds] = dmeas_dinstrumentb
        return Kb

    def drdn_datmosphere(
        self,
        x_atmosphere,
        geom,
        rho_dir_dir,
        rho_dif_dir,
        rho_dir_dif,
        rho_dif_dif,
        Ls,
        rdn,
    ):
        """Derivative of estimated radiance w.r.t. atmosphere statevector elements.
        We use a numerical approach to approximate datmosphere with a constant surface
        reflectance. This is a reasonable approx. for the multicomponent surface.

        When using the glint model however, this does not take into account
        the dependence of the surface reflectance on the atmosphere.
        """
        # perturb each element of the atmosphere state vector (finite difference)
        K_atmosphere = []
        x_atmospheres_perturb = x_atmosphere + np.eye(len(x_atmosphere)) * eps
        for x_atmosphere_perturb in list(x_atmospheres_perturb):
            (
                r,
                L_tot,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
            ) = self.calc_atmosphere_quantities(x_atmosphere_perturb, geom, rho_dif_dif)

            # Surface state is held constant?
            rdne = self.calc_rdn(
                x_atmosphere_perturb,
                rho_dir_dir,
                rho_dif_dir,
                rho_dir_dif,
                rho_dif_dif,
                Ls,
                L_tot,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
                r,
                geom,
            )
            K_atmosphere.append((rdne - rdn) / eps)

        K_atmosphere = np.array(K_atmosphere).T

        return K_atmosphere

    def drdn_datmosphereb(
        self,
        x_atmosphere,
        geom,
        rho_dir_dir,
        rho_dif_dir,
        rho_dir_dif,
        rho_dif_dif,
        Ls,
        rdn,
    ):
        """Derivative of estimated rdn w.r.t. H2O_ABSCO

        Currently, the K_b matrix only covers forward model derivatives
        due to H2O_ABSCO unknowns, so that subsequent errors might occur
        when water vapor is not part of the statevector
        (which is very unlikely though).
        """
        if len(self.bvec) == 0:
            Kb_atmosphere = np.zeros((0, len(self.atmosphere.wl)))

        # ToDo: might require modification in case more unknowns are added
        # The following statement captures the case that H2O is not part
        # of the statevector, but might need to be modified as soon as we
        # add more unknowns
        elif len(self.bvec) > 0 and "H2OSTR" not in self.atmosphere.statevec_names:
            Kb_atmosphere = np.zeros((1, len(self.atmosphere.wl)))
        else:
            # unknown parameters modeled as random variables per
            # Rodgers et al (2000) K_b matrix.  We calculate these derivatives
            # by finite differences
            Kb_atmosphere = []
            perturb = 1.0 + eps
            for unknown in self.bvec:
                if (
                    unknown == "H2O_ABSCO"
                    and "H2OSTR" in self.atmosphere.statevec_names
                ):
                    i = self.atmosphere.statevec_names.index("H2OSTR")
                    x_atmosphere_perturb = x_atmosphere.copy()
                    x_atmosphere_perturb[i] = x_atmosphere[i] * perturb
                    (
                        r,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                    ) = self.calc_atmosphere_quantities(
                        x_atmosphere_perturb, geom, rho_dif_dif
                    )

                    rdne = self.calc_rdn(
                        x_atmosphere_perturb,
                        rho_dir_dir,
                        rho_dif_dir,
                        rho_dir_dif,
                        rho_dif_dif,
                        Ls,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                        r,
                        geom,
                    )
                    Kb_atmosphere.append((rdne - rdn) / eps)

        Kb_atmosphere = np.array(Kb_atmosphere).T

        return Kb_atmosphere

    def summarize(self, x, geom):
        """State vector summary."""

        x_surface, x_atmosphere, x_instrument = self.unpack(x)
        return (
            self.surface.summarize(x_surface, geom)
            + " "
            + self.atmosphere.summarize(x_atmosphere, geom)
            + " "
            + self.instrument.summarize(x_instrument, geom)
        )

    def calibration(self, x):
        """Calculate measured wavelengths and fwhm."""

        x_inst = x[self.idx_instrument]
        return self.instrument.calibration(x_inst)

    def upsample(self, wl, q):
        """Linear interpolation to atmosphere wavelengths."""
        # Only interpolate if these aren't close
        close = len(wl) == len(self.atmosphere.wl) and np.allclose(
            wl, self.atmosphere.wl
        )

        # or if any dimension is the wrong size
        interp = (np.array(q.shape) != len(self.atmosphere.wl)).all()

        if not close or interp:
            if q.ndim > 1:
                return np.array(
                    [
                        interp1d(wl, qi, fill_value="extrapolate")(self.atmosphere.wl)
                        for qi in q
                    ]
                )
            else:
                p = interp1d(wl, q, fill_value="extrapolate")
                return p(self.atmosphere.wl)
        return q

    def unpack(self, x):
        """Unpack the state vector in appropriate index ordering."""

        x_surface = x[self.idx_surface]
        x_atmosphere = x[self.idx_atmosphere]
        x_instrument = x[self.idx_instrument]
        return x_surface, x_atmosphere, x_instrument

    def match_statevector(self, full_statevector):
        self.full_idx, self.full_miss = match_statevector(
            full_statevector, self.statevec
        )
