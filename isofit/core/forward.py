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

import scipy as s
from copy import deepcopy
from scipy.linalg import det, norm, pinv, sqrtm, inv, block_diag
from importlib import import_module
from scipy.interpolate import interp1d

from .common import recursive_replace, eps
from .instrument import Instrument


### Variables ###

# Supported RT modules, filenames, and class names
RT_models = [('modtran_radiative_transfer', 'radiative_transfer.modtran', 'ModtranRT'),
             ('libradtran_radiative_transfer',
              'radiative_transfer.libradtran', 'LibRadTranRT'),
             ('planetary_radiative_transfer', 'planetary', 'PlanetaryRT'),
             ('uplooking_radiative_transfer', 'uplook', 'UplookRT'),
             ('sixs_radiative_transfer', 'radiative_transfer.six_s', 'SixSRT')]


# Supported surface modules, filenames, and class names
surface_models = [('surface', 'surface.surface', 'Surface'),
                  ('multicomponent_surface',
                   'surface.surface_multicomp', 'MultiComponentSurface'),
                  ('emissive_surface', 'surface.surface_emissive', 'MixBBSurface'),
                  ('cat_surface', 'surf_cat', 'CATSurface'),
                  ('glint_surface', 'surface.surface_glint', 'GlintSurface'),
                  ('iop_surface', 'surface.surface_iop', 'IOPSurface'),
                  ('poly_surface', 'surf_poly', 'PolySurface')]


### Classes ###

class ForwardModel:
    """ForwardModel contains all the information about how to calculate
     radiance measurements at a specific spectral calibration, given a 
     state vector. It also manages the distributions of unretrieved, 
     unknown parameters of the state vector (i.e. the S_b and K_b 
     matrices of Rodgers et al."""

    def __init__(self, config):

        self.instrument = Instrument(config['instrument'])
        self.n_meas = self.instrument.n_chan

        # Build the radiative transfer model
        self.RT = None
        for key, module, cname in RT_models:
            module = "isofit." + module
            if key in config:
                self.RT = getattr(import_module(module), cname)(config[key])
        if self.RT is None:
            raise ValueError('Must specify a valid radiative transfer model')

        # Build the surface model
        self.surface = None
        for key, module, cname in surface_models:
            module = "isofit." + module
            if key in config:
                self.surface = getattr(
                    import_module(module), cname)(config[key])
        if self.surface is None:
            raise ValueError('Must specify a valid surface model')

        # Set up passthrough option
        bounds, scale, init, statevec = [], [], [], []

        # Build state vector for each part of our forward model
        for name in ['surface', 'RT', 'instrument']:
            obj = getattr(self, name)
            inds = len(statevec) + s.arange(len(obj.statevec), dtype=int)
            setattr(self, 'idx_%s' % name, inds)
            for b in obj.bounds:
                bounds.append(deepcopy(b))
            for c in obj.scale:
                scale.append(deepcopy(c))
            for v in obj.init:
                init.append(deepcopy(v))
            for v in obj.statevec:
                statevec.append(deepcopy(v))

        self.bounds = tuple(s.array(bounds).T)
        self.scale = s.array(scale)
        self.init = s.array(init)
        self.statevec = statevec
        self.nstate = len(self.statevec)

        # Capture unmodeled variables
        bvec, bval = [], []
        for name in ['RT', 'instrument', 'surface']:
            obj = getattr(self, name)
            inds = len(bvec) + s.arange(len(obj.bvec), dtype=int)
            setattr(self, '%s_b_inds' % name, inds)
            for b in obj.bval:
                bval.append(deepcopy(b))
            for v in obj.bvec:
                bvec.append(deepcopy(v))

        self.bvec = s.array(bvec)
        self.nbvec = len(self.bvec)
        self.bval = s.array(bval)
        self.Sb = s.diagflat(pow(self.bval, 2))

    def out_of_bounds(self, x):
        """Check if state vector is within bounds."""

        x_RT = x[self.idx_RT]
        x_surface = x[self.idx_surface]
        bound_lwr = self.bounds[0]
        bound_upr = self.bounds[1]
        return any(x_RT >= (bound_upr[self.idx_RT] - eps*2.0)) or \
            any(x_RT <= (bound_lwr[self.idx_RT] + eps*2.0))

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
        return s.concatenate((xa_surface, xa_RT, xa_instrument), axis=0)

    def Sa(self, x, geom):
        """Calculate the prior covariance of the state vector (the concatenation
        of state vectors for the surface and radiative transfer model).

        NOTE: the surface prior depends on the current state; this
        is so we can calculate the local linearized answer.
        """

        x_surface = x[self.idx_surface]
        Sa_surface = self.surface.Sa(x_surface, geom)[:, :]
        Sa_RT = self.RT.Sa()[:, :]
        Sa_instrument = self.instrument.Sa()[:, :]
        return block_diag(Sa_surface, Sa_RT, Sa_instrument)

    def calc_rdn(self, x, geom, rfl=None, Ls=None):
        """Calculate the high-resolution radiance, permitting overrides.

        Project to top-of-atmosphere and translate to radiance."""

        x_surface, x_RT, x_instrument = self.unpack(x)
        if rfl is None:
            rfl = self.surface.calc_rfl(x_surface, geom)
        if Ls is None:
            Ls = self.surface.calc_Ls(x_surface, geom)
        rfl_hi = self.upsample(self.surface.wl, rfl)
        Ls_hi = self.upsample(self.surface.wl, Ls)
        return self.RT.calc_rdn(x_RT, rfl_hi, Ls_hi, geom)

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
        both the instrument noise and the uncertainty due to unmodeled
        variables. This is the S_epsilon matrix of Rodgers et al."""

        Kb = self.Kb(x, geom)
        Sy = self.instrument.Sy(meas, geom)
        return Sy + Kb.dot(self.Sb).dot(Kb.T)

    def K(self, x, geom):
        """Derivative of observation with respect to state vector. This is 
        the concatenation of jacobians with respect to parameters of the 
        surface and radiative transfer model."""

        # Unpack state vector
        x_surface, x_RT, x_instrument = self.unpack(x)

        # Get partials of reflectance WRT state, and upsample
        rfl = self.surface.calc_rfl(x_surface, geom)
        drfl_dsurface = self.surface.drfl_dsurface(x_surface, geom)
        rfl_hi = self.upsample(self.surface.wl, rfl)
        drfl_dsurface_hi = self.upsample(self.surface.wl, drfl_dsurface.T).T

        # Get partials of emission WRT state, and upsample
        Ls = self.surface.calc_Ls(x_surface, geom)
        dLs_dsurface = self.surface.dLs_dsurface(x_surface, geom)
        Ls_hi = self.upsample(self.surface.wl, Ls)
        dLs_dsurface_hi = self.upsample(self.surface.wl, dLs_dsurface.T).T

        # Derivatives of RTM radiance
        drdn_dRT, drdn_dsurface = self.RT.drdn_dRT(x_RT, x_surface, rfl_hi,
                                                   drfl_dsurface_hi, Ls_hi, dLs_dsurface_hi, geom)

        # Derivatives of measurement, avoiding recalculation of rfl, Ls
        dmeas_dsurface = self.instrument.sample(x_instrument, self.RT.wl,
                                                drdn_dsurface.T).T
        dmeas_dRT = self.instrument.sample(x_instrument, self.RT.wl,
                                           drdn_dRT.T).T
        rdn_hi = self.calc_rdn(x, geom, rfl=rfl, Ls=Ls)
        dmeas_dinstrument = self.instrument.dmeas_dinstrument(x_instrument,
                                                              self.RT.wl, rdn_hi)

        # Put it all together
        K = s.zeros((self.n_meas, self.nstate), dtype=float)
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

        drdn_dRTb = self.RT.drdn_dRTb(x_RT, rfl_hi, Ls_hi, geom)
        dmeas_dRTb = self.instrument.sample(x_instrument, self.RT.wl,
                                            drdn_dRTb.T).T
        dmeas_dinstrumentb = self.instrument.dmeas_dinstrumentb(
            x_instrument, self.RT.wl, rdn_hi)

        Kb = s.zeros((self.n_meas, self.nbvec), dtype=float)
        Kb[:, self.RT_b_inds] = dmeas_dRTb
        Kb[:, self.instrument_b_inds] = dmeas_dinstrumentb
        return Kb

    def summarize(self, x, geom):
        """State vector summary."""

        x_surface, x_RT, x_instrument = self.unpack(x)
        return self.surface.summarize(x_surface, geom) + \
            ' ' + self.RT.summarize(x_RT, geom) + \
            ' ' + self.instrument.summarize(x_instrument, geom)

    def calibration(self, x):
        """Calculate measured wavelengths and fwhm."""

        x_inst = x[self.idx_instrument]
        return self.instrument.calibration(x_inst)

    def upsample(self, wl, q):
        """Linear interpolation to RT wavelengths."""

        if q.ndim > 1:
            q2 = []
            for qi in q:
                p = interp1d(wl, qi, fill_value='extrapolate')
                q2.append(p(self.RT.wl))
            return s.array(q2)
        else:
            p = interp1d(wl, q, fill_value='extrapolate')
            return p(self.RT.wl)

    def unpack(self, x):
        """Unpack the state vector in appropriate index ordering."""

        x_surface = x[self.idx_surface]
        x_RT = x[self.idx_RT]
        x_instrument = x[self.idx_instrument]
        return x_surface, x_RT, x_instrument

    def reconfigure(self, config_surface, config_rt, config_instrument):
        """Reconfigure the components of the forward model. This could update
        components' initialization values and/or priors, or (for the case
        of a defined surface reflectance) the surface reflectance file itself."""

        self.surface.reconfigure(config_surface)
        self.RT.reconfigure(config_rt)
        self.instrument.reconfigure(config_instrument)
        self.init = s.concatenate((self.surface.init, self.RT.init,
                                   self.instrument.init))
