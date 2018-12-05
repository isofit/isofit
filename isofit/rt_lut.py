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
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar as min1d


class FileExistsError(Exception):
    def __init__(self, message):
        super(FileExistsError, self).__init__(message)


def spawn_rt(cmd):
    print(cmd)
    os.system(cmd)


class TabularRT:
    """A model of photon transport including the atmosphere."""

    def __init__(self, config, instrument):

        if 'auto_rebuild' in config:
          self.auto_rebuild = config['auto_rebuild']
        else:
          self.auto_rebuild = True
        self.lut_grid = config['lut_grid']
        self.lut_dir = config['lut_path']
        self.statevec = list(config['statevector'].keys())
        self.bvec = list(config['unknowns'].keys())
        self.n_point = len(self.lut_grid)
        self.n_state = len(self.statevec)

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init_val = [], [], []
        for key in self.statevec:
            element = config['statevector'][key]
            self.bounds.append(element['bounds'])
            self.scale.append(element['scale'])
            self.init_val.append(element['init'])
        self.bounds = s.array(self.bounds)
        self.scale = s.array(self.scale)
        self.init_val = s.array(self.init_val)
        self.bval = s.array([config['unknowns'][k] for k in self.bvec])

        if 'prior_sigma' in config['statevector']:
            self.prior_sigma = s.zeros((len(self.lut_grid),))
            for name, val in config['prior_sigma'].items():
                self.prior_sigma[self.statevec.index(name)] = val
        else:
            std_factor = 10.0
            self.prior_sigma = (s.diff(self.bounds) * std_factor).flatten()

    def xa(self):
        '''Mean of prior distribution, calculated at state x. This is the
           Mean of our LUT grid (why not).'''
        return self.init_val.copy()

    def Sa(self):
        '''Covariance of prior distribution. Our state vector covariance 
           is diagonal with very loose constraints.'''
        return s.diagflat(pow(self.prior_sigma, 2))

    def build_lut(self, instrument, rebuild=False):
        """ Each LUT is associated with a source directory.  We build a 
            lookup table by: 
              (1) defining the LUT dimensions, state vector names, and the grid 
                  of values; 
              (2) running modtran if needed, with each MODTRAN run defining a 
                  different point in the LUT; and 
              (3) loading the LUTs, one per key atmospheric coefficient vector,
                  into memory as VectorInterpolator objects."""

        # set up lookup table grid, and associated filename prefixes
        self.lut_dims, self.lut_grids, self.lut_names = [], [], []
        for key, val in self.lut_grid.items():
            self.lut_names.append(key)
            self.lut_grids.append(s.array(val))
            self.lut_dims.append(len(val))
            if val != sorted(val):
                raise ValueError(
                    'Lookup table grid must be in ascending order')

        # "points" contains all combinations of grid points
        # We will have one filename prefix per point
        self.points = combos(self.lut_grids)
        self.files = []
        for point in self.points:
            outf = '_'.join(['%s-%6.4f' % (n, x)
                             for n, x in zip(self.lut_names, point)])
            self.files.append(outf)

        rebuild_cmds = []
        for point, fn in zip(self.points, self.files):
            try:
                cmd = self.rebuild_cmd(point, fn)
                rebuild_cmds.append(cmd)
            except FileExistsError:
                pass

        if len(rebuild_cmds) > 0 and self.auto_rebuild:
            print("rebuilding")
            import multiprocessing
            cwd = os.getcwd()
            os.chdir(self.lut_dir)
            count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=count)
            r = pool.map_async(spawn_rt, rebuild_cmds)
            r.wait()
            os.chdir(cwd)

        # load the RT runs, one per grid point of the LUT
        # to do: use high-res output
        self.solar_irr = None
        for point, fn in zip(self.points, self.files):
            chnfile = self.lut_dir+'/'+fn+'.chn'
            wl, sol, solzen, rhoatm, transm, sphalb, transup = \
                self.load_rt(point, fn)

            if self.solar_irr is None:  # first file
                self.solar_irr = sol
                self.coszen = s.cos(solzen * s.pi / 180.0)
                dims_aug = self.lut_dims + [len(instrument.wl)]
                self.sphalb = s.zeros(dims_aug, dtype=float)
                self.transm = s.zeros(dims_aug, dtype=float)
                self.rhoatm = s.zeros(dims_aug, dtype=float)
                self.transup = s.zeros(dims_aug, dtype=float)
                self.wl = wl

            ind = [s.where(g == p)[0] for g, p in zip(self.lut_grids, point)]
            self.rhoatm[ind] = rhoatm
            self.sphalb[ind] = sphalb
            self.transm[ind] = transm
            self.transup[ind] = transup
        self.rhoatm_interp = VectorInterpolatorJIT(self.lut_grids, self.rhoatm)
        self.sphalb_interp = VectorInterpolatorJIT(self.lut_grids, self.sphalb)
        self.transm_interp = VectorInterpolatorJIT(self.lut_grids, self.transm)
        self.transup_interp = VectorInterpolatorJIT(
            self.lut_grids, self.transm)

    def lookup_lut(self, point):
        """Multi-linear interpolation in the LUT"""

        rhoatm = s.array(self.rhoatm_interp(point)).ravel()
        sphalb = s.array(self.sphalb_interp(point)).ravel()
        transm = s.array(self.transm_interp(point)).ravel()
        transup = s.array(self.transm_interp(point)).ravel()
        return rhoatm, sphalb, transm, transup

    def get(self, x_RT, geom):
        if self.n_point == self.n_state:
            return self.lookup_lut(x_RT)
        else:
            point = s.zeros((self.n_point,))
            for point_ind, name in enumerate(self.lut_grid):
                if name in self.statevec:
                    x_RT_ind = self.statevec.index(name)
                    point[point_ind] = x_RT[x_RT_ind]
                elif name == "OBSZEN":
                    point[point_ind] = geom.OBSZEN
                elif name == "TRUEAZ":
                    point[point_ind] = geom.TRUEAZ
                elif name == 'phi':
                    point[point_ind] = geom.phi
                elif name == 'umu':
                    point[point_ind] = geom.umu
                else:
                    # If a variable is defined in the lookup table but not 
                    # specified elsewhere, we will default to the minimum
                    point[point_ind] = min(self.lut_grid[name])
            for x_RT_ind, name in enumerate(self.statevec):
                point_ind = self.lut_names.index(name)
                point[point_ind] = x_RT[x_RT_ind]
            return self.lookup_lut(point)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        '''Calculate radiance at aperature for a radiative transfer state vector.
           rfl is the reflectance at surface. 
           Ls is the  emissive radiance at surface.'''

        if Ls is None:
            Ls = s.zeros(rfl.shape)

        rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
        rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
        rdn = rho/s.pi*(self.solar_irr*self.coszen) + (Ls * transup)
        return rdn

    def estimate_Ls(self, x_RT, rfl, rdn, geom):
        """Estimate the surface emission for a given state vector and 
           reflectance/radiance pair"""

        rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
        rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
        Ls = (rdn - rho/s.pi*(self.solar_irr*self.coszen)) / transup
        return Ls

    def heuristic_atmosphere(self, rdn, geom):
        '''From a given radiance, estimate atmospheric state using band ratio
        heuristics.  Used to initialize gradient descent inversions.'''

        x = self.init_val.copy()

        # Band ratio retrieval of H2O
        for h2oname in ['H2OSTR', 'h2o']:
            if (h2oname in self.lut_names) and \
                    any(self.wl > 850) and any(self.wl < 1050):

                ind_lut = self.lut_names.index(h2oname)
                ind_sv = self.statevec.index(h2oname)
                b865, b945, b1040 = [s.argmin(abs(self.wl-t))
                                     for t in (865, 945, 1040)]
                h2os, ratios = [], []  # will be corrected for transmission, path radiance

                for h2o in self.lut_grids[ind_lut]:

                    xnew = x.copy()
                    xnew[ind_sv] = h2o
                    rhoatm, sphalb, transm, transup = self.get(xnew, geom)

                    # assume no surface emission
                    r = (rdn*s.pi/(self.solar_irr*self.coszen) -
                         rhoatm) / (transm+1e-8)
                    ratios.append((r[b945]*2.0)/(r[b1040]+r[b865]))
                    h2os.append(h2o)

                p = interp1d(h2os, ratios)
                bounds = (h2os[0]+0.001, h2os[-1]-0.001)
                best = min1d(lambda h: abs(1-p(h)),
                             bounds=bounds, method='bounded')
                x[ind_sv] = best.x

        Ls_est = s.zeros(rdn.shape)
        rfl_est = self.invert_algebraic(x, rdn, Ls_est, geom)
        return x, rfl_est

    def invert_algebraic(self, x_RT, rdn, Ls, geom):
        '''Inverts radiance algebraically to get a reflectance.
           Ls is the surface emission, if present'''

        rhoatm, sphalb, transm, transup = self.get(x_RT, geom)

        if Ls is None:
            rho = rdn * s.pi / (self.solar_irr * self.coszen)
        else:
            rdn_solref = rdn - (transup * Ls)
            rho = rdn_solref * s.pi / (self.solar_irr * self.coszen)
        rfl = 1.0 / (transm / (rho - rhoatm) + sphalb)
        return rfl

    def K_RT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls, dLs_dsurface,
             geom):
        """Jacobian of radiance with respect to RT and surface state vectors"""

        # first the rdn at the current state vector
        rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
        rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
        rdn = rho/s.pi*(self.solar_irr*self.coszen) + (Ls * transup)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        for i in range(len(x_RT)):
            x_RT_perturb = x_RT.copy()
            x_RT_perturb[i] = x_RT[i] + eps
            rhoatme, sphalbe, transme, transupe = self.get(x_RT_perturb, geom)
            rhoe = rhoatme + transme * rfl / (1.0 - sphalbe * rfl)
            rdne = rhoe/s.pi*(self.solar_irr*self.coszen) + (Ls * transupe)
            K_RT.append((rdne-rdn) / eps)
        K_RT = s.array(K_RT).T

        # analytical jacobians for surface model state vector, via chain rule
        K_surface = []
        for i in range(len(x_surface)):
            drho_drfl = \
                (transm/(1-sphalb*rfl)-(sphalb*transm*rfl)/pow(1-sphalb*rfl, 2))
            drdn_drfl = drho_drfl/s.pi*(self.solar_irr*self.coszen)
            drdn_dLs = transup
            K_surface.append(drdn_drfl * drfl_dsurface[:, i] +
                             drdn_dLs * dLs_dsurface[:, i])
        K_surface = s.array(K_surface).T

        return K_RT, K_surface

    def Kb_RT(self, x_RT, rfl, Ls, geom):
        """Jacobian of radiance with respect to NOT RETRIEVED RT and surface 
           state.  Right now, this is just the sky view factor."""

        if len(self.bvec) == 0:
            Kb_RT = s.zeros((1, len(self.wl.shape)))

        else:
            # first the radiance at the current state vector
            rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
            rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
            rdn = rho/s.pi*(self.solar_irr*self.coszen) + (Ls * transup)

            # perturb the sky view
            Kb_RT = []
            perturb = (1.0+eps)
            for unknown in self.bvec:

                if unknown == 'Skyview':
                    rhoe = rhoatm + transm * rfl / (1.0 - sphalb * rfl *
                                                    perturb)
                    rdne = rhoe/s.pi*(self.solar_irr*self.coszen)
                    Kb_RT.append((rdne-rdn) / eps)

                elif unknown == 'H2O_ABSCO' and 'H2OSTR' in self.statevec:
                    # first the radiance at the current state vector
                    rhoatm, sphalb, transm, transup = self.get(x_RT, geom)
                    rho = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
                    rdn = rho/s.pi*(self.solar_irr*self.coszen) + (Ls *
                                                                   transup)
                    i = self.statevec.index('H2OSTR')
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    rhoatme, sphalbe, transme, transupe = self.get(
                        x_RT_perturb, geom)
                    rhoe = rhoatme + transme * rfl / (1.0 - sphalbe * rfl)
                    rdne = rhoe/s.pi*(self.solar_irr*self.coszen) + (Ls *
                                                                     transup)
                    Kb_RT.append((rdne-rdn) / eps)

        Kb_RT = s.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        '''Summary of state vector'''
        return 'Atmosphere: '+' '.join(['%5.3f' % xi for xi in x_RT])
