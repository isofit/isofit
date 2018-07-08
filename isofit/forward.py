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

from common import recursive_replace, eps
from copy import deepcopy
from scipy.linalg import block_diag, det, norm, pinv, sqrtm, inv
from rt_modtran import ModtranRT
from rt_libradtran import LibRadTranRT
from surf import Surface
from surf_multicomp import MultiComponentSurface
from surf_glint import GlintSurface
from surf_emissive import MixBBSurface
from surf_iop import IOPSurface
from instrument import Instrument


class ForwardModel:

    def __init__(self, config):
        '''ForwardModel contains all the information about how to calculate
         radiance measurements at a specific spectral calibration, given a 
         state vector.  It also manages the distributions of unretrieved, 
         unknown parameters of the state vector (i.e. the S_b and K_b 
         matrices of Rodgers et al.'''

        self.instrument = Instrument(config['instrument'])

        # Build the radiative transfer model
        if 'modtran_radiative_transfer' in config:
            self.RT = ModtranRT(config['modtran_radiative_transfer'],
                                self.instrument)
        elif 'libradtran_radiative_transfer' in config:
            self.RT = LibRadTranRT(config['libradtran_radiative_transfer'],
                                   self.instrument)
        else:
            raise ValueError('Must specify a valid radiative transfer model')

        # Build the surface model
        if 'surface' in config:
            self.surface = Surface(config['surface'], self.RT)
        elif 'multicomponent_surface' in config:
            self.surface = MultiComponentSurface(config['multicomponent_surface'],
                                                 self.RT)
        elif 'emissive_surface' in config:
            self.surface = MixBBSurface(config['emissive_surface'], self.RT)
        elif 'glint_surface' in config:
            self.surface = GlintSurface(config['glint_surface'], self.RT)
        elif 'iop_surface' in config:
            self.surface = IOPSurface(config['iop_surface'], self.RT)
        else:
            raise ValueError('Must specify a valid surface model')

        bounds, scale, init_val, statevec = [], [], [], []

        # Build state vector for each part of our forward model
        for name in ['surface', 'RT']:
            obj = getattr(self, name)
            inds = len(statevec) + s.arange(len(obj.statevec), dtype=int)
            setattr(self, '%s_inds' % name, inds)
            for b in obj.bounds:
                bounds.append(deepcopy(b))
            for c in obj.scale:
                scale.append(deepcopy(c))
            for v in obj.init_val:
                init_val.append(deepcopy(v))
            for v in obj.statevec:
                statevec.append(deepcopy(v))
        self.bounds = tuple(s.array(bounds).T)
        self.scale = s.array(scale)
        self.init_val = s.array(init_val)
        self.statevec = statevec
        self.wl = self.instrument.wl

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
        self.bval = s.array(bval)
        self.Sb = s.diagflat(pow(self.bval, 2))
        return

    def out_of_bounds(self, x):
        """Is state vector inside the bounds?"""
        x_RT = x[self.RT_inds]
        x_surface = x[self.surface_inds]
        bound_lwr = self.bounds[0]
        bound_upr = self.bounds[1]
        return any(x_RT >= (bound_upr[self.RT_inds] - eps*2.0)) or \
            any(x_RT <= (bound_lwr[self.RT_inds] + eps*2.0))

    def xa(self, x, geom):
        """Calculate the prior mean of the state vector (the concatenation
        of state vectors for the surface and Radiative Transfer model).
        Note that the surface prior mean depends on the current state - this
        is so we can calculate the local linearized answer."""

        x_surface = x[self.surface_inds]
        xa_surface = self.surface.xa(x_surface, geom)
        xa_RT = self.RT.xa()
        return s.concatenate((xa_surface, xa_RT), axis=0)

    def Sa(self, x, geom):
        """Calculate the prior covariance of the state vector (the concatenation
        of state vectors for the surface and Radiative Transfer model).
        Note that the surface prior depends on the current state - this
        is so we can calculate the local linearized answer."""

        x_surface = x[self.surface_inds]
        Sa_surface = self.surface.Sa(x_surface, geom)[:, :]
        Sa_RT = self.RT.Sa()[:, :]
        if Sa_surface.size > 0:
            return block_diag(Sa_surface, Sa_RT)
        else:
            return Sa_RT

    def invert_algebraic(self, x, rdn, geom):
        """Simple algebraic inversion of radiance based on the current 
        atmospheric state. Return the reflectance, and the atmospheric
        correction coefficients."""

        x_RT = x[self.RT_inds]
        x_surface = x[self.surface_inds]
        rhoatm, sphalb, transm, transup = self.RT.get(x_RT, geom)
        coeffs = rhoatm, sphalb, transm, self.RT.solar_irr, self.RT.coszen
        Ls = self.surface.calc_Ls(x_surface, geom)
        return self.RT.invert_algebraic(x_RT, rdn, Ls, geom), coeffs

    def init(self, rdn_meas, geom):
        """Find an initial guess at the state vector.  This currently uses
        traditional (non-iterative, heuristic) atmospheric correction."""

        # heuristic estimation of atmosphere using solar-reflected regime
        x_RT, rfl_est = self.RT.heuristic_atmosphere(rdn_meas, geom)
        if not isinstance(self.surface, MixBBSurface):
            Ls_est = None
        else:
            # modify reflectance and estimate surface emission
            rfl_est = self.surface.conditional_solrfl(rfl_est, geom)
            Ls_est = self.RT.estimate_Ls(x_RT, rfl_est, rdn_meas, geom)
        x_surface = self.surface.heuristic_surface(rfl_est, Ls_est, geom)
        return s.concatenate((x_surface.copy(), x_RT.copy()), axis=0)

    def calc_rdn(self, x, geom, rfl=None, Ls=None):
        """calculate the observed radiance, permitting overrides
        Project to top of atmosphere and translate to radiance"""

        x_surface = x[self.surface_inds]
        x_RT = x[self.RT_inds]
        if rfl is None:
            rfl = self.surface.calc_rfl(x_surface, geom)
        if Ls is None:
            Ls = self.surface.calc_Ls(x_surface, geom)
        return self.RT.calc_rdn(x_RT, rfl, Ls, geom)

    def calc_Ls(self, x, geom):
        """calculate the surface emission."""

        return self.surface.calc_Ls(x[self.surface_inds], geom)

    def calc_rfl(self, x, geom):
        """calculate the surface reflectance"""

        return self.surface.calc_rfl(x[self.surface_inds], geom)

    def calc_lrfl(self, x, geom):
        """calculate the Lambertiansurface reflectance"""

        return self.surface.calc_lrfl(x[self.surface_inds], geom)

    def Seps(self, meas, geom, init=None):
        """Calculate the total uncertainty of the observation, including
        both the instrument noise and the uncertainty due to unmodeled
        variables. This is the Sepsilon matrix of Rodgers et al."""

        Kb = self.Kb(meas, geom, init=None)
        Sy = self.instrument.Sy(meas, geom)
        return Sy + Kb.dot(self.Sb).dot(Kb.T)

    def K(self, x, geom):
        """Derivative of observation with respect to state vector. This is 
        the concatenation of jacobians with respect to parameters of the 
        surface and radiative transfer model."""

        x_surface = x[self.surface_inds]
        x_RT = x[self.RT_inds]
        rfl = self.surface.calc_rfl(x_surface, geom)
        Ls = self.surface.calc_Ls(x_surface, geom)
        drfl_dsurface = self.surface.drfl_dx(x_surface, geom)
        dLs_dsurface = self.surface.dLs_dx(x_surface, geom)
        K_RT, K_surface = self.RT.K_RT(x_RT, x_surface, rfl, drfl_dsurface,
                                       Ls, dLs_dsurface, geom)
        nmeas = K_RT.shape[0]
        K = s.zeros((nmeas, len(self.statevec)), dtype=float)
        K[:, self.surface_inds] = K_surface
        K[:, self.RT_inds] = K_RT
        return K

    def Kb(self, meas, geom, init=None):
        """Derivative of measurement with respect to unmodeled & unretrieved
        unknown variables, e.g. S_b. This is  the concatenation of jacobians 
        with respect to parameters of the surface, radiative transfer model, 
        and instrument. """

        if init is None:
            x = self.init(meas, geom)
        else:
            x = init.copy()
        x_surface = x[self.surface_inds]
        x_RT = x[self.RT_inds]
        rfl = self.surface.calc_rfl(x_surface, geom)
        Ls = self.surface.calc_Ls(x_surface, geom)
        Kb_RT = self.RT.Kb_RT(x_RT, rfl, Ls, geom)
        Kb_instrument = self.instrument.Kb_instrument(meas)
        Kb_surface = self.surface.Kb_surface(meas, geom)
        nmeas = len(meas)
        Kb = s.zeros((nmeas, len(self.bvec)), dtype=float)
        Kb[:, self.RT_b_inds] = Kb_RT
        Kb[:, self.instrument_b_inds] = Kb_instrument
        Kb[:, self.surface_b_inds] = Kb_surface
        return Kb

    def summarize(self, x, geom):
        """State vector summary"""
        x_RT = x[self.RT_inds]
        x_surface = x[self.surface_inds]
        return self.surface.summarize(x_surface, geom) + \
            ' ' + self.RT.summarize(x_RT, geom)

    def calc_Seps(self, rdn_meas, geom, init=None):
        """Calculate (zero-mean) measurement distribution in radiance terms.  
        This depends on the location in the state space. This distribution is 
        calculated over one or more subwindows of the spectrum. Return the 
        inverse covariance and its square root"""

        Seps = self.fm.Seps(rdn_meas, geom, init=init)
        Seps = s.array([Seps[i, self.winidx] for i in self.winidx])
        Seps_inv = s.real(chol_inv(Seps))
        Seps_inv_sqrt = s.real(sqrtm(Seps_inv))
        return Seps_inv, Seps_inv_sqrt
