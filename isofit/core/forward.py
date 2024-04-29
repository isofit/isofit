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

import logging
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.linalg import block_diag

from isofit.configs import Config
from isofit.surface import (
    AdditiveGlintSurface,
    GlintModelSurface,
    LUTSurface,
    MultiComponentSurface,
    Surface,
    ThermalSurface,
)

from ..radiative_transfer.radiative_transfer import RadiativeTransfer
from .common import eps
from .instrument import Instrument

Logger = logging.getLogger(__file__)

### Classes ###


class ForwardModel:
    """ForwardModel contains all the information about how to calculate
    radiance measurements at a specific spectral calibration, given a
    state vector. It also manages the distributions of unretrieved,
    unknown parameters of the state vector (i.e. the S_b and K_b
    matrices of Rodgers et al.

    State vector elements always go in the following order:
      (1) Surface parameters
      (2) Radiative Transfer (RT) parameters
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

    def __init__(self, full_config: Config):
        # load in the full config (in case of inter-module dependencies) and
        # then designate the current config
        self.full_config = full_config
        self.config = full_config.forward_model

        # Build the instrument model
        self.instrument = Instrument(self.full_config)
        self.n_meas = self.instrument.n_chan

        # Build the radiative transfer model
        self.RT = RadiativeTransfer(self.full_config)

        # Build the surface model - this is a bit ugly with the conditionals, but the surface config should already be
        # forced to have appropriate properties, so at least safe
        # TODO: make surface a class with inheritance to make this cleaner
        self.surface = None
        if self.config.surface.surface_category == "surface":
            self.surface = Surface(self.full_config)
        elif self.config.surface.surface_category == "multicomponent_surface":
            self.surface = MultiComponentSurface(self.full_config)
        elif self.config.surface.surface_category == "additive_glint_surface":
            self.surface = AdditiveGlintSurface(self.full_config)
        elif self.config.surface.surface_category == "glint_model_surface":
            self.surface = GlintModelSurface(self.full_config)
            self.full_glint = (
                self.config.surface.full_glint
                if "full_glint" in self.config.surface.keys()
                else False
            )
        elif self.config.surface.surface_category == "thermal_surface":
            self.surface = ThermalSurface(self.full_config)
        elif self.config.surface.surface_category == "lut_surface":
            self.surface = LUTSurface(self.full_config)
        else:
            raise ValueError("Must specify a valid surface model")
            # No need to be more specific - should have been checked in config already

        if self.surface.n_wl != len(self.RT.wl) or not np.all(
            np.isclose(self.surface.wl, self.RT.wl, atol=0.01)
        ):
            Logger.warning(
                "Surface and RTM wavelengths differ - if running at higher RTM"
                " spectral resolution or with variable wavelength position, this"
                " is expected.  Otherwise, consider checking the surface model."
            )

        # Build combined vectors from surface, RT, and instrument
        bounds, scale, init, statevec, bvec, bval = ([] for i in range(6))
        for obj_with_statevec in [self.surface, self.RT, self.instrument]:
            bounds.extend([deepcopy(x) for x in obj_with_statevec.bounds])
            scale.extend([deepcopy(x) for x in obj_with_statevec.scale])
            init.extend([deepcopy(x) for x in obj_with_statevec.init])
            statevec.extend([deepcopy(x) for x in obj_with_statevec.statevec_names])

            bvec.extend([deepcopy(x) for x in obj_with_statevec.bvec])
            bval.extend([deepcopy(x) for x in obj_with_statevec.bval])

        self.bounds = tuple(np.array(bounds).T)
        self.scale = np.array(scale)
        self.init = np.array(init)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        self.bvec = np.array(bvec)
        self.nbvec = len(self.bvec)
        self.bval = np.array(bval)
        self.Sb = np.diagflat(np.power(self.bval, 2))

        # Set up indices for references - MUST MATCH ORDER FROM ABOVE ASSIGNMENT
        self.idx_surface = np.arange(len(self.surface.statevec_names), dtype=int)
        # Sometimes, it's convenient to have the index of the entire surface
        # as one variable, and sometimes you want the sub-components
        # Split surface state vector indices to cover cases where we retrieve
        # additional non-reflectance surface parameters
        self.idx_surf_rfl = self.idx_surface[
            : len(self.surface.idx_lamb)
        ]  # reflectance portion
        self.idx_surf_nonrfl = self.idx_surface[
            len(self.surface.idx_lamb) :
        ]  # all non-reflectance surface parameters
        self.idx_RT = np.arange(len(self.RT.statevec_names), dtype=int) + len(
            self.idx_surface
        )
        self.idx_instrument = (
            np.arange(len(self.instrument.statevec_names), dtype=int)
            + len(self.idx_surface)
            + len(self.idx_RT)
        )

        self.surface_b_inds = np.arange(len(self.surface.bvec), dtype=int)
        self.RT_b_inds = np.arange(len(self.RT.bvec), dtype=int) + len(
            self.surface_b_inds
        )
        self.instrument_b_inds = (
            np.arange(len(self.instrument.bvec), dtype=int)
            + len(self.surface_b_inds)
            + len(self.RT_b_inds)
        )

        # Load model discrepancy correction
        if self.config.model_discrepancy_file is not None:
            D = loadmat(self.config.model_discrepancy_file)
            self.model_discrepancy = D["cov"]
        else:
            self.model_discrepancy = None

    def out_of_bounds(self, x):
        """Check if state vector is within bounds."""

        x_RT = x[self.idx_RT]
        bound_lwr = self.bounds[0]
        bound_upr = self.bounds[1]
        return any(x_RT >= (bound_upr[self.idx_RT] - eps * 2.0)) or any(
            x_RT <= (bound_lwr[self.idx_RT] + eps * 2.0)
        )

    def xa(self, x, geom):
        """Calculate the prior mean of the state vector (the concatenation
        of state vectors for the surface, Radiative Transfer model, and
        instrument).

        NOTE: the surface prior mean depends on the current state;
        this is so we can calculate the local prior.
        """

        x_surface = x[self.idx_surface]
        xa_surface = self.surface.xa(x_surface, geom)
        xa_RT = self.RT.xa()
        xa_instrument = self.instrument.xa()
        return np.concatenate((xa_surface, xa_RT, xa_instrument), axis=0)

    def Sa(self, x, geom):
        """Calculate the prior covariance of the state vector (the
        concatenation of state vectors for the surface and radiative transfer
        model).

        NOTE: the surface prior depends on the current state; this
        is so we can calculate the local prior.
        """

        x_surface = x[self.idx_surface]
        Sa_surface = self.surface.Sa(x_surface, geom)[:, :]
        Sa_RT = self.RT.Sa()[:, :]
        Sa_instrument = self.instrument.Sa()[:, :]

        return block_diag(Sa_surface, Sa_RT, Sa_instrument)

    def calc_rdn(self, x, geom, rfl=None, Ls=None):
        """Calculate the high-resolution radiance, permitting overrides.
        Project to top-of-atmosphere and translate to radiance. The
        radiative transfer calculations may take place at higher resolution
        so we upsample surface terms.
        """
        x_surface, x_RT, x_instrument = self.unpack(x)
        if rfl is None:
            rfl = self.surface.calc_rfl(x_surface, geom)
        if Ls is None:
            Ls = self.surface.calc_Ls(x_surface, geom)

        rfl_hi = self.upsample(self.surface.wl, rfl)
        Ls_hi = self.upsample(self.surface.wl, Ls)
        return self.RT.calc_rdn(x_RT, x_surface, rfl_hi, Ls_hi, geom)

    def calc_meas(self, x, geom, rfl=None, Ls=None):
        """Calculate the model observation at instrument wavelengths."""

        x_surface, x_RT, x_instrument = self.unpack(x)
        rdn_hi = self.calc_rdn(x, geom, rfl, Ls)
        return self.instrument.sample(x_instrument, self.RT.wl, rdn_hi)

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

        Kb = self.Kb(x, geom)
        Sy = self.instrument.Sy(meas, geom)

        return Sy + Kb.dot(self.Sb).dot(Kb.T) + Gamma

    def K(self, x, geom):
        """Derivative of observation with respect to state vector. This is
        the concatenation of jacobians with respect to parameters of the
        surface and radiative transfer model."""

        # Unpack state vector
        x_surface, x_RT, x_instrument = self.unpack(x)

        # Get partials of reflectance WRT surface state variables, upsample
        rfl = self.surface.calc_rfl(x_surface, geom)
        drfl_dsurface = self.surface.drfl_dsurface(x_surface, geom)
        rfl_hi = self.upsample(self.surface.wl, rfl)
        drfl_dsurface_hi = self.upsample(self.surface.wl, drfl_dsurface.T).T

        # Get partials of emission WRT surface state variables, upsample
        Ls = self.surface.calc_Ls(x_surface, geom)
        dLs_dsurface = self.surface.dLs_dsurface(x_surface, geom)
        Ls_hi = self.upsample(self.surface.wl, Ls)
        dLs_dsurface_hi = self.upsample(self.surface.wl, dLs_dsurface.T).T

        # Derivatives of RTM radiance
        drdn_dRT, drdn_dsurface = self.RT.drdn_dRT(
            x_RT, x_surface, rfl_hi, drfl_dsurface_hi, Ls_hi, dLs_dsurface_hi, geom
        )

        # Derivatives of measurement, avoiding recalculation of rfl, Ls
        dmeas_dsurface = self.instrument.sample(
            x_instrument, self.RT.wl, drdn_dsurface.T
        ).T
        dmeas_dRT = self.instrument.sample(x_instrument, self.RT.wl, drdn_dRT.T).T
        rdn_hi = self.calc_rdn(x, geom, rfl=rfl, Ls=Ls)
        dmeas_dinstrument = self.instrument.dmeas_dinstrument(
            x_instrument, self.RT.wl, rdn_hi
        )

        # Put it all together
        K = np.zeros((self.n_meas, self.nstate), dtype=float)
        K[:, self.idx_surface] = dmeas_dsurface
        K[:, self.idx_RT] = dmeas_dRT
        K[:, self.idx_instrument] = dmeas_dinstrument
        return K

    def Kb(self, x, geom):
        """Derivative of measurement with respect to unmodeled & unretrieved
        unknown variables, e.g. S_b. This is  the concatenation of Jacobians
        with respect to parameters of the surface, radiative transfer model,
        and instrument.  Currently we only treat uncertainties in the
        instrument and RT model."""

        # Unpack state vector
        x_surface, x_RT, x_instrument = self.unpack(x)

        # Get partials of reflectance and upsample
        rfl = self.surface.calc_rfl(x_surface, geom)
        rfl_hi = self.upsample(self.surface.wl, rfl)
        Ls = self.surface.calc_Ls(x_surface, geom)
        Ls_hi = self.upsample(self.surface.wl, Ls)
        rdn_hi = self.calc_rdn(x, geom, rfl=rfl, Ls=Ls)

        drdn_dRTb = self.RT.drdn_dRTb(x_RT, x_surface, rfl_hi, Ls_hi, geom)
        dmeas_dRTb = self.instrument.sample(x_instrument, self.RT.wl, drdn_dRTb.T).T
        dmeas_dinstrumentb = self.instrument.dmeas_dinstrumentb(
            x_instrument, self.RT.wl, rdn_hi
        )

        # Put it together
        Kb = np.zeros((self.n_meas, self.nbvec), dtype=float)
        Kb[:, self.RT_b_inds] = dmeas_dRTb
        Kb[:, self.instrument_b_inds] = dmeas_dinstrumentb
        return Kb

    def summarize(self, x, geom):
        """State vector summary."""

        x_surface, x_RT, x_instrument = self.unpack(x)
        return (
            self.surface.summarize(x_surface, geom)
            + " "
            + self.RT.summarize(x_RT, geom)
            + " "
            + self.instrument.summarize(x_instrument, geom)
        )

    def calibration(self, x):
        """Calculate measured wavelengths and fwhm."""

        x_inst = x[self.idx_instrument]
        return self.instrument.calibration(x_inst)

    def upsample(self, wl, q):
        """Linear interpolation to RT wavelengths."""

        # In some cases, these differ only by a tiny amount,
        # so no need to waste time interpolating
        if (len(wl) == len(self.RT.wl)) and np.allclose(wl, self.RT.wl):
            return q

        if q.ndim > 1:
            q2 = []
            for qi in q:
                p = interp1d(wl, qi, fill_value="extrapolate")
                q2.append(p(self.RT.wl))
            return np.array(q2)
        else:
            p = interp1d(wl, q, fill_value="extrapolate")
            return p(self.RT.wl)

    def unpack(self, x):
        """Unpack the state vector in appropriate index ordering."""

        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        x_instrument = x[self.idx_instrument]
        return x_surface, x_RT, x_instrument
