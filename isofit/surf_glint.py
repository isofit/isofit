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
from common import recursive_replace, emissive_radiance, chol_inv, eps
from surf_multicomp import MultiComponentSurface


class GlintSurface(MultiComponentSurface):
    """A model of the surface based on a collection of multivariate 
       Gaussians, extended with a surface glint term. """

    def __init__(self, config, RT):

        MultiComponentSurface.__init__(self, config, RT)
        self.statevec.extend(['GLINT'])
        self.scale.extend([1.0])
        self.init_val.extend([0.005])
        self.bounds.extend([[0, 0.2]])
        self.glint_ind = len(self.statevec)-1

    def xa(self, x_surface, geom):
        '''Mean of prior distribution, calculated at state x.'''

        mu = MultiComponentSurface.xa(self, x_surface, geom)
        mu[self.glint_ind] = self.init_val[self.glint_ind]
        return mu

    def Sa(self, x_surface, geom):
        '''Covariance of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function.'''

        Cov = MultiComponentSurface.Sa(self, x_surface, geom)
        f = s.array([[(10.0 * self.scale[self.glint_ind])**2]])
        Cov[self.glint_ind, self.glint_ind] = f
        return Cov

    def heuristic_surface(self, rfl_meas, Ls, geom):
        '''Given a reflectance estimate and one or more emissive parameters, 
          fit a state vector.'''

        glint_band = s.argmin(abs(900-self.wl))
        glint = s.mean(rfl_meas[(glint_band-2):glint_band+2])
        water_band = s.argmin(abs(400-self.wl))
        water = s.mean(rfl_meas[(water_band-2):water_band+2])
        if glint > 0.05 or water < glint:
            glint = 0
        glint = max(self.bounds[self.glint_ind][0]+eps,
                    min(self.bounds[self.glint_ind][1]-eps, glint))
        lrfl_est = rfl_meas - glint
        x = MultiComponentSurface.heuristic_surface(self, lrfl_est, Ls, geom)
        x[self.glint_ind] = glint
        return x

    def calc_lrfl(self, x_surface, geom):
        '''Lambertian-equivalent reflectance'''

        return MultiComponentSurface.calc_lrfl(self, x_surface, geom)

    def dlrfl_dx(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface.'''

        return MultiComponentSurface.dlrfl_dx(self, x_surface, geom)

    def calc_rfl(self, x_surface, geom):
        '''Reflectance (includes specular glint)'''

        return self.calc_lrfl(x_surface, geom) + x_surface[self.glint_ind]

    def drfl_dx(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface.'''

        drfl = self.dlrfl_dx(x_surface, geom)
        drfl[:, self.glint_ind] = 1
        return drfl

    def summarize(self, x_surface, geom):
        '''Summary of state vector'''
        return MultiComponentSurface.summarize(self, x_surface, geom) + \
            ' Glint: %5.3f' % x_surface[self.glint_ind]
