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

import os
import scipy as s
from common import json_load_ascii, combos
from common import VectorInterpolator, VectorInterpolatorJIT
from common import recursive_replace, eps
from common import spectrumResample
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar as min1d



class PlanetaryRT:
    """A model of photon transport with no atmosphere."""

    def __init__(self, config):

        self.solar_irradiance_file = config['solar_irradiance_file']
        self.distance = float(config['sun_distance_AU'])
        self.solzen   = float(config['solar_zenith'])
        self.coszen   = s.cos(self.solzen)
        self.statevec = []
        self.bvec     = [] 
        self.bounds   = []
        self.scale    = []
        self.init_val = []
        self.unknowns = []
        self.bval     = []
        solar_irr = s.loadtxt(self.solar_irradiance_file, comments='#')
        self.solar_irr = spectrumResample(solar_irr[:,1], 
                solar_irr[:,0], self.wl, self.fwhm, fill=False)
        self.solar_irr = self.solar_irr / (self.distance**2)

    def xa(self):
        '''Mean of prior distribution, calculated at state x. This is the
           Mean of our LUT grid (why not).'''
        return s.array([])

    def Sa(self):
        '''Covariance of prior distribution. Our state vector covariance 
           is diagonal with very loose constraints.'''

        return s.zeros((0,0), dtype=float)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        '''Calculate radiance at aperture.
           rfl is the reflectance at surface. 
           Ls is the emissive radiance at surface.'''
        
        if Ls is None: 
            Ls = s.zeros(rfl.shape)
        rdn = rfl / s.pi * (self.solar_irr * self.coszen)
        rdn[s.logical_not(s.isfinite(rdn))] = 0
        rdn = rdn + Ls
        return rdn

    def estimate_Ls(self, x_RT, rfl, rdn, geom):
        """Estimate the surface emission for a given state vector and 
           reflectance/radiance pair"""

        Ls = rdn - rfl / s.pi * (self.solar_irr * self.coszen)
        return Ls

    def heuristic_atmosphere(self, rdn, geom):
        '''From a given radiance, estimate atmospheric state using band ratio
        heuristics.  Used to initialize gradient descent inversions.'''

        x_RT = self.init_val.copy()
        rfl_est = self.invert_algebraic(x_RT, rdn, None, geom)
        return x_RT, rfl_est

    def get(self, x_RT, geom):
        transm   = s.ones((len(self.wl),),  dtype=float)
        transup  = s.ones((len(self.wl),),  dtype=float)
        rhoatm   = s.zeros((len(self.wl),), dtype=float)
        sphalb   = s.zeros((len(self.wl),), dtype=float)
        return rhoatm, sphalb, transm, transup

    def invert_algebraic(self, x_RT, rdn, Ls, geom):
        '''Inverts radiance algebraically to get a reflectance.
           Ls is the surface emission, if present'''

        if Ls is None:
            rfl = rdn * s.pi / (self.solar_irr * self.coszen)
        else:
            rdn_solrefl = rdn - Ls
            rfl = rdn_solrefl * s.pi / (self.solar_irr * self.coszen)
        return rfl

    def K_RT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls, dLs_dsurface,
             geom):
        """Jacobian of radiance with respect to RT and surface state vectors"""

        K_RT = s.array([[]])

        # analytical jacobians for surface model state vector, via chain rule
        K_surface = []
        local_irr = self.solar_irr / (self.distance**2)
        for i in range(len(x_surface)):
            drdn_drfl = 1.0/s.pi*(local_irr * self.coszen)
            drdn_dLs = s.ones(drdn_drfl.shape, dtype=float)
            K_surface.append(drdn_drfl * drfl_dsurface[:, i] +
                             drdn_dLs * dLs_dsurface[:, i])
        K_surface = s.array(K_surface).T
        
        return K_RT, K_surface

    def Kb_RT(self, x_RT, rfl, Ls, geom):
        """Jacobian of radiance with respect to NOT RETRIEVED RT and surface 
           state. """

        return s.zeros((len(self.wl.shape),1),dtype=float)

    def summarize(self, x_RT, geom):
        '''Summary of state vector'''
        return ''
