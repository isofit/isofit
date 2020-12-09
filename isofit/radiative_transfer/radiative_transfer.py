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

import numpy as np
import logging

from ..core.common import eps
from ..radiative_transfer.modtran import ModtranRT
from ..radiative_transfer.six_s import SixSRT
from ..radiative_transfer.libradtran import LibRadTranRT
from ..radiative_transfer.sRTMnet import SimulatedModtranRT
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig


class RadiativeTransfer():
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation 
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and 
    TIR. This class maintains the master list of statevectors.
    """

    def __init__(self, full_config: Config):

        # Maintain order when looping for indexing convenience
        config = full_config.forward_model.radiative_transfer

        self.statevec_names = config.statevector.get_element_names()
        self.lut_grid = config.lut_grid

        # TODO: rework this so that we instead initialize an interpolator, that calls
        # RTEs as necessary based on LUT grid or other parameters..which may happen higher up
        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            rte_config: RadiativeTransferEngineConfig = config.radiative_transfer_engines[idx]

            if rte_config.engine_name == 'modtran':
                rte = ModtranRT(rte_config, full_config)
            elif rte_config.engine_name == 'libradtran':
                rte = LibRadTranRT(rte_config, full_config)
            elif rte_config.engine_name == '6s':
                rte = SixSRT(rte_config, full_config)
            elif rte_config.engine_name == 'simulated_modtran':
                rte = SimulatedModtranRT(rte_config, full_config)
            else:
                # Should never get here, checked in config
                raise AttributeError(
                    'Invalid radiative transfer engine name: {}'.format(rte_config.engine_name))

            self.rt_engines.append(rte)
        
        # The rest of the code relies on sorted order of the individual RT engines which cannot
        # be guaranteed by the dict JSON or YAML input
        self.rt_engines.sort(key=lambda x: x.wl[0])

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        self.wl = np.concatenate([RT.wl for RT in self.rt_engines])

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = np.concatenate([RT.solar_irr for RT in self.rt_engines])
        # These should all be the same so just grab one
        self.coszen = [RT.coszen for RT in self.rt_engines][0]

    def xa(self):
        """Pull the priors from each of the individual RTs.
        """
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs.
        """
        return np.diagflat(np.power(np.array(self.prior_sigma), 2))

    def get_shared_rtm_quantities(self, x_RT, geom):
        """Return only the set of RTM quantities (transup, sphalb, etc.) that are contained
        in all RT engines.
        """

        ret = []
        for RT in self.rt_engines:
            ret.append(RT.get(x_RT, geom))

        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        r = self.get_shared_rtm_quantities(x_RT, geom)
        L_atm = self.get_L_atm(x_RT, geom)
        L_up = Ls * r['transup']

        if geom.bg_rfl is not None:

            # adjacency effects are counted
            I = (self.solar_irr*self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r['t_down_dif'] + r['t_down_dir']

            ret = L_atm + \
              I / (1.0-r['sphalb'] * bg) * bg * t_down * r['t_up_dif'] + \
              I / (1.0-r['sphalb'] * bg) * rfl * t_down * r['t_up_dir'] + \
              L_up

        else:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)
             
            ret = L_atm + \
                L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl) + \
                L_up

        return ret

    def get_L_atm(self, x_RT, geom):
        L_atms = []
        for RT in self.rt_engines:
            L_atms.append(RT.get_L_atm(x_RT, geom))
        return np.hstack(L_atms)

    def get_L_down_transmitted(self, x_RT, geom):
        L_downs = []
        for RT in self.rt_engines:
            L_downs.append(RT.get_L_down_transmitted(x_RT, geom))
        return np.hstack(L_downs)

    def drdn_dRT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls,
                 dLs_dsurface, geom):

        # first the rdn at the current state vector
        rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + np.eye(len(x_RT))*eps
        for x_RT_perturb in list(x_RTs_perturb):
            rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
            K_RT.append((rdne-rdn) / eps)
        K_RT = np.array(K_RT).T

        # Get K_surface
        r = self.get_shared_rtm_quantities(x_RT, geom)

        if geom.bg_rfl is not None:

             # adjacency effects are counted
            I = (self.solar_irr*self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r['t_down_dif'] + r['t_down_dir']
            drdn_drfl = I / (1.0-r['sphalb'] * bg) * t_down * r['t_up_dir'] 

        else:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)
            
            # The reflected downwelling light is:
            # L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl), or
            # L_down_transmitted * rho_scaled_for_multiscattering
            # This term is the derivative of rho_scaled_for_multiscattering
            drho_scaled_for_multiscattering_drfl = 1. / (1 - r['sphalb']*rfl)**2
            
            drdn_drfl = L_down_transmitted * drho_scaled_for_multiscattering_drfl

        drdn_dLs = r['transup']
        K_surface = drdn_drfl[:, np.newaxis] * drfl_dsurface + \
            drdn_dLs[:, np.newaxis] * dLs_dsurface

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):

        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))

        else:
            # first the radiance at the current state vector
            r = self.get_shared_rtm_quantities(x_RT, geom)
            rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

            # unknown parameters modeled as random variables per
            # Rodgers et al (2000) K_b matrix.  We calculate these derivatives
            # by finite differences
            Kb_RT = []
            perturb = (1.0+eps)
            for unknown in self.bvec:
                if unknown == 'H2O_ABSCO' and 'H2OSTR' in self.statevec_names:
                    i = self.statevec_names.index('H2OSTR')
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
                    Kb_RT.append((rdne-rdn) / eps)

        Kb_RT = np.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.rt_engines:
            ret.append(RT.summarize(x_RT, geom))
        ret = '\n'.join(ret)
        return ret

    def pack_arrays(self, rtm_quantities_from_RT_engines):
        """Take the list of dict outputs from each RT engine and
        stack their internal arrays in the same order. Keep only
        those quantities that are common to all RT engines.
        """

        # Get the intersection of the sets of keys from each of the rtm_quantities_from_RT_engines
        shared_rtm_keys = set(rtm_quantities_from_RT_engines[0].keys())
        if len(rtm_quantities_from_RT_engines) > 1:
            for rtm_quantities_from_one_RT_engine in rtm_quantities_from_RT_engines[1:]:
                shared_rtm_keys.intersection_update(rtm_quantities_from_one_RT_engine.keys())

        # Concatenate the different band ranges
        rtm_quantities_concatenated_over_RT_bands = {}
        for key in shared_rtm_keys:
            temp = [x[key] for x in rtm_quantities_from_RT_engines]
            rtm_quantities_concatenated_over_RT_bands[key] = np.hstack(temp)

        return rtm_quantities_concatenated_over_RT_bands
