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
from common import spectrumLoad, spectrumResample
from scipy.linalg import det, norm, pinv, sqrtm, inv, cholesky
from scipy.optimize import minimize
from surf_multicomp import MultiComponentSurface
from scipy.interpolate import interp1d


class CATSurface(MultiComponentSurface):
    """CAT = Continuum, Absorption, Temperature.
    A model of the surface based on Multicomponent continuum model, 
    a library of absorptions, and a Temperature"""

    def __init__(self, config):

        MultiComponentSurface.__init__(self, config)
        self.rfl_ind = s.arange(len(self.statevec), dtype=int)

        # Handle additional state vector elements
        self.statevec.extend(['SURF_TEMP_K'])
        self.init_val.extend([config['temperature_K']])
        self.scale.extend([100.0])
        self.bounds.extend([[0.0, 1000.0]])
        self.surf_temp_ind = len(self.statevec)-1
        self.surf_temp_sigma = 100.0
        self.library_file = config['absorption_library_file']
        lib_data = s.loadtxt(self.library_file)
        lib_wl, lib_absorptions = lib_data[:,0], lib_data[:,1:].T - 1.0
        self.absorptions = []
        for i, absrb in enumerate(lib_absorptions):
           p = interp1d(lib_wl, absrb, fill_value='extrapolate')  
           self.absorptions.append(p(RT.wl)) 
           self.statevec.extend(['ABSRB_%i'%i])
           if 'absorption_init' in config:
               self.init_val.extend([config['absorption_init'][i]])
           else:
               self.init_val.extend([0])
           self.scale.extend([1.0])
           self.bounds.extend([[-eps,9e99]])
        self.absrb_inds = s.arange(self.surf_temp_ind+1, 
                self.surf_temp_ind+len(lib_absorptions)+1, dtype=int)
        self.init_val = s.array(self.init_val)
        self.bounds = s.array(self.bounds)
        self.absorptions = s.array(self.absorptions)
        self.absorption_sigma = 9e99

        if 'reflectance_init_file' in config:
            init_rfl, init_wl = spectrumLoad(config['reflectance_init_file'])
            self.init_val[self.rfl_ind] = spectrumResample(init_rfl, init_wl,
                   RT.wl, RT.fwhm, fill=False)

    def xa(self, x_surface, geom):
        '''Mean of prior distribution, calculated at state x.  We find
        the covariance in a normalized space (normalizing by z) and then un-
        normalize the result for the calling function.'''

        xa = MultiComponentSurface.xa(self, x_surface, geom)
        xa[self.surf_temp_ind] = self.init_val[self.surf_temp_ind]
        for i in self.absrb_inds:
          xa[i] = self.init_val[i]
        return xa

    def Sa(self, x_surface, geom):
        '''Covariance of prior distribution, calculated at state x'''

        Sa   = MultiComponentSurface.Sa(self, x_surface, geom)
        Sa[self.surf_temp_ind, self.surf_temp_ind] = self.surf_temp_sigma
        for i in self.absrb_inds:
          Sa[i,i] = self.absorption_sigma
        return Sa

    def heuristic_surface(self, rfl_meas, Ls, geom):
        '''Given a reflectance estimate and one or more emissive parameters, 
          fit a state vector.'''

        def err(T):
            emissivity = s.ones(self.nwl, dtype=float)
            Ls_est, d = emissive_radiance(emissivity, T, self.wl)
            resid = Ls_est - Ls
            return sum(resid**2)

        x_surface = MultiComponentSurface.heuristic_surface(self, rfl_meas,
                                                                Ls, geom)
        T = minimize(err, s.array(self.init_val[self.surf_temp_ind])).x
        T = max(self.bounds[-2][0]+eps, min(T, self.bounds[-2][1]-eps))
        x_surface[self.surf_temp_ind] = T
        x_surface[self.absrb_inds] = 0
        return x_surface

    def conditional_solrfl(self, rfl_est, geom):
        '''Conditions the reflectance on solar-reflected channels.'''

        sol_inds = s.where(s.logical_and(self.wl > 450, self.wl < 2000))[0]
        x = s.zeros(len(self.statevec))
        x[self.lrfl_inds] = rfl_est

        mu = self.xa(x, geom)
        mu_sol = mu[sol_inds]
        C = self.Sa(x, geom)
        Cov_sol = s.array([C[i, sol_inds] for i in sol_inds])
       #mu = self.xa(x, geom)
       #lrfl = self.calc_lrfl(x_surface, geom)
       #ref_lrfl = lrfl[self.refidx]
       #mu = s.zeros(len(self.statevec))
       #ci = self.component(x_surface, geom)
       #mu_lrfl = self.components[ci][0]
       #mu_lrfl = mu_lrfl * self.norm(ref_lrfl)
       #mu[self.lrfl_inds] = mu_lrfl
       #return mu


       #c = self.components[self.component(x, geom)]
       #mu_sol = c[0][sol_inds]
       #Cov_sol = s.array([c[1][i, sol_inds] for i in s.where(sol_inds)[0]])
        Cinv = inv(Cov_sol)
        diff = rfl_est[sol_inds] - mu_sol
        full = mu + C[:, sol_inds].dot(Cinv.dot(diff))
        return full[:len(self.wl)]

    def calc_rfl(self, x_surface, geom):
        '''Reflectance'''

        return self.calc_lrfl(x_surface, geom)

    def drfl_dx(self, x_surface, geom):
        '''Partial derivative of reflectance with respect to state vector, 
        calculated at x_surface.'''

        return self.dlrfl_dx(x_surface, geom)

    def calc_lrfl(self, x_surface, geom):
        '''Lambertian Reflectance'''

        lrfl = MultiComponentSurface.calc_lrfl(self, x_surface, geom)
        lrfl = lrfl + lrfl * s.matmul(x_surface[self.absrb_inds],
                 self.absorptions)
        return lrfl

    def dlrfl_dx(self, x_surface, geom):
        '''Partial derivative of Lambertian reflectance with respect to state 
        vector, calculated at x_surface.'''

        lrfl = MultiComponentSurface.calc_lrfl(self, x_surface, geom)
        dlrfl = MultiComponentSurface.dlrfl_dx(self, x_surface, geom)
        dlrfl[:, self.surf_temp_ind] = 0
        for i, a in zip(self.absrb_inds, self.absorptions):
          dlrfl[:, i] = a * lrfl
        return dlrfl

    def calc_Ls(self, x_surface, geom):
        '''Emission of surface, as a radiance'''

        T = x_surface[self.surf_temp_ind]
        emissivity = s.ones(self.nwl) - self.calc_lrfl(x_surface, geom)
        Ls, dLs_dT = emissive_radiance(emissivity, T, self.wl)
        return Ls 

    def dLs_dx(self, x_surface, geom):
        '''Partial derivative of surface emission with respect to state vector, 
        calculated at x_surface.'''

        dLs_dx = MultiComponentSurface.dLs_dx(self, x_surface, geom)
        T = x_surface[self.surf_temp_ind]
        emissivity = s.ones(self.nwl) - self.calc_lrfl(x_surface, geom)
        demissivity_dx = -self.dlrfl_dx(x_surface, geom)
        Ls, dLs_dT = emissive_radiance(emissivity, T, self.wl)
        dLs_dx[:, self.lrfl_inds] = (Ls * demissivity_dx[:,self.lrfl_inds].T).T
        dLs_dx[:, self.surf_temp_ind] = dLs_dT
        dLs_dx[:, self.absrb_inds] = (Ls * demissivity_dx[:,self.absrb_inds].T).T
        return dLs_dx

    def summarize(self, x_surface, geom):
        '''Summary of state vector'''
        mcm = MultiComponentSurface.summarize(self, x_surface, geom)
        msg = ' Kelvins: %5.1f  Absorptions:'% x_surface[self.surf_temp_ind]
        for i in self.absrb_inds:
             msg = msg + ' %6.4f' % x_surface[i]
        msg = msg + '\n'
        return msg+mcm
