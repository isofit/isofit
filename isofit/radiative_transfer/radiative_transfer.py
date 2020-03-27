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
# Author: Jay E. Fahlen, jay.e.fahlen@jpl.nasa.gov
#

import scipy as s
import logging
from copy import deepcopy
from collections import OrderedDict

from ..core.common import json_load_ascii, eps
from ..radiative_transfer.modtran import modtran_bands_available, ModtranRT
from ..radiative_transfer.six_s import sixs_names, SixSRT
from ..radiative_transfer.libradtran import libradtran_names, LibRadTranRT


class RadiativeTransfer():
    """This class controls the radiative transfer component of the forward
    model. An ordered dict is maintained of individual RTMs (like MODTRAN,
    for example). The dict is looped over and the radiation and derivatives
    from each RTM and band are put together.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and TIR.
    This class maintains the master list of statevectors.
    """

    def __init__(self, config):
        """."""

        # Maintain order when looping for indexing convenience
        self.RTs = OrderedDict()

        self.statevec = list(config['statevector'].keys())
        self.statevec.sort()

        self.lut_grid = config['lut_grid']

        temp_RTs_list, temp_min_wavelen_list = [], []
        config_statevector = config['statevector']
        for key, local_config in config.items():

            if type(local_config) == dict and 'lut_names' in local_config:
                # Construct a dict with the LUT and state parameter
                # info needed for each individual RT
                temp_statevec, temp_lut_grid = {}, {}
                for local_lut_name in local_config['lut_names']:
                    temp_lut_grid[local_lut_name] = self.lut_grid[local_lut_name]
                local_config["lut_grid"] = temp_lut_grid

                # copy statevector into local config
                for local_sv_name in local_config['statevector_names']:
                    temp_statevec[local_sv_name] = config_statevector[local_sv_name]
                local_config["statevector"] = temp_statevec

            temp_RT = None
            if key in modtran_bands_available:
                temp_RT = ModtranRT(key, local_config, self.statevec)
            elif key in sixs_names:
                temp_RT = SixSRT(local_config, self.statevec)
            elif key in libradtran_names:
                temp_RT = LibRadTranRT(local_config, self.statevec)

            if temp_RT is not None:
                temp_RTs_list.append((key, temp_RT))
                temp_min_wavelen_list.append(temp_RT.wl[0])

        # Put the RT objects into self.RTs in wavelength order
        # This assumes that the input data wavelengths are all
        # ascending.
        sort_inds = s.argsort(s.array(temp_min_wavelen_list))
        for sort_ind in sort_inds:
            key, RT = temp_RTs_list[sort_ind]
            self.RTs[key] = RT

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv_key in self.statevec:    # Go in order
            sv = config_statevector[sv_key]
            self.bounds.append(sv['bounds'])
            self.scale.append(sv['scale'])
            self.init.append(sv['init'])
            self.prior_sigma.append(sv['prior_sigma'])
            self.prior_mean.append(sv['prior_mean'])

        self.bounds = s.array(self.bounds)
        self.scale = s.array(self.scale)
        self.init = s.array(self.init)
        self.prior_mean = s.array(self.prior_mean)
        self.prior_sigma = s.array(self.prior_sigma)

        self.wl = s.concatenate([RT.wl for RT in self.RTs.values()])

        self.bvec = list(config['unknowns'].keys())
        self.bval = s.array([config['unknowns'][k] for k in self.bvec])

        self.solar_irr = s.concatenate([RT.solar_irr for RT in self.RTs.values()])
        # These should all be the same so just grab one
        self.coszen = [RT.coszen for RT in self.RTs.values()][0]

    def xa(self):
        """Pull the priors from each of the individual RTs.
        """
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs.
        """
        return s.diagflat(pow(s.array(self.prior_sigma), 2))

    def get(self, x_RT, geom):

        ret = []
        for key, RT in self.RTs.items():
            ret.append(RT.get(x_RT, geom))

        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        r = self.get(x_RT, geom)
        L_atm = self.get_L_atm(x_RT, geom)
        L_down = self.get_L_down(x_RT, geom)

        L_up = self.get_L_up(x_RT, geom)
        L_up = L_up + Ls * r['transup']

        ret = L_atm + \
            L_down * rfl * r['transm'] / (1.0 - r['sphalb'] * rfl) + \
            L_up

        return ret

    def get_L_atm(self, x_RT, geom):
        L_atms = []
        for key, RT in self.RTs.items():
            L_atms.append(RT.get_L_atm(x_RT, geom))
        return s.hstack(L_atms)

    def get_L_down(self, x_RT, geom):
        L_downs = []
        for key, RT in self.RTs.items():
            L_downs.append(RT.get_L_down(x_RT, geom))
        return s.hstack(L_downs)

    def get_L_up(self, x_RT, geom):
        '''L_up is provided by the surface model, so just return
        0 here. The commented out code here is for future updates.'''
        #L_ups = []
        # for key, RT in self.RTs.items():
        #    L_ups.append(RT.get_L_up(x_RT, geom))
        # return s.hstack(L_ups)

        return 0.

    def drdn_dRT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls,
                 dLs_dsurface, geom):

        # first the rdn at the current state vector
        Ls = s.zeros(rfl.shape)  # TODO: Fix me!
        rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + s.eye(len(x_RT))*eps
        for x_RT_perturb in list(x_RTs_perturb):
            rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
            K_RT.append((rdne-rdn) / eps)
        K_RT = s.array(K_RT).T

        # Get K_surface
        r = self.get(x_RT, geom)
        L_down = self.get_L_down(x_RT, geom)
        drho_drfl = r['transm'] / (1 - r['sphalb']*rfl)**2
        drdn_drfl = drho_drfl * L_down
        drdn_dLs = r['transup']
        K_surface = drdn_drfl[:, s.newaxis] * drfl_dsurface + \
            drdn_dLs[:, s.newaxis] * dLs_dsurface

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):

        if len(self.bvec) == 0:
            Kb_RT = s.zeros((0, len(self.wl.shape)))

        else:
            # first the radiance at the current state vector
            r = self.get(x_RT, geom)
            rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

            # perturb the sky view
            Kb_RT = []
            perturb = (1.0+eps)
            for unknown in self.bvec:

                if unknown == 'Skyview':
                    #rdne = self.calc_rdn_vswir(r, rfl, perturb = perturb)
                    #Kb_RT.append((rdne-rdn) / eps)
                    raise NotImplementedError

                elif unknown == 'H2O_ABSCO' and 'H2OSTR' in self.statevec:
                    # first the radiance at the current state vector
                    i = self.statevec.index('H2OSTR')
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
                    Kb_RT.append((rdne-rdn) / eps)

        Kb_RT = s.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.RTs.values():
            ret.append(RT.summarize(x_RT, geom))
        ret = '\n'.join(ret)
        return ret

    def pack_arrays(self, list_of_r_dicts):
        """Take the list of dict outputs from each RTM (in order of RTs) and
        stack their internal arrays in the same order.
        """
        r_stacked = {}
        for key in list_of_r_dicts[0].keys():
            temp = [x[key] for x in list_of_r_dicts]
            r_stacked[key] = s.hstack(temp)
        return r_stacked
