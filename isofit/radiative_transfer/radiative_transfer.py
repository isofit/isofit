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

from ..core.common import json_load_ascii
from ..radiative_transfer.modtran import modtran_bands_available, ModtranRT

class RadiativeTransfer():
    """This class controls the radiative transfer component of the forward
    model. An ordered dict is maintained of individual RTMs (like MODTRAN,
    for example). The dict is looped over and the radiation and derivatives
    from each RTM and band are put together.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and TIR.
    The element unique_statevec contains the unique state vector elements from
    each RTM component. For example, assume there are two RTMs: VISNIR and TIR.
    There will be three RT state vector elements: H20, AOT, and surface
    temperature. The VISNIR RTM's statevec will have H20 and AOT and the TIR
    RTM's statevec will have H20 and surface temperature.

    Maintaining the order of the inputs and outputs here is a bit tricky. The
    individual RTMs will maintain their own order for their state vector. This
    class will provide them in that order using the function make_statevec().
    The order of the output radiation (output from calc_rdn, for example) will
    be in whatever order the RTs ordered dict is filled, which is determined by
    the random order of the radiative_transfer elements in the json input. This
    means that there is no sorting with respect to wavelength of the different
    RTM bands.
    """

    def __init__(self, config):
        """."""

        # Maintain order when looping for indexing convenience
        self.RTs = OrderedDict()

        for key, local_config in config.items():
            if key in modtran_bands_available:
                self.RTs[key] = ModtranRT(key, local_config)
            else:
                raise NotImplementedError

        self.wl = s.squeeze(s.array([RT.wl for RT in self.RTs.values()]))

        # The state vector is made of up unique elements of the individual RTs.
        # The idea is that each RT LUT may need H20STR, for example, but we
        # only want a single H20STR value in the statevector as it is shared.
        all_statevec = [s for RT in self.RTs.values() for s in RT.statevec] # Flatten a list of lists
        self.unique_statevec = list(set(all_statevec)) # Get unique elements
        self.unique_statevec.sort()

        self.statevec = self.unique_statevec

        statevec_temp, bounds, scales, inits = [], [], [], []
        for RT in self.RTs.values():
            statevec_temp = statevec_temp + [statevec for statevec in RT.statevec]
            bounds = bounds + [bound for bound in RT.bounds]
            scales = scales + [scale for scale in RT.scale]
            inits = inits + [init for init in RT.init]
        
        self.bounds, self.scale, self.init = [], [], []
        for unique_statevec_component in self.unique_statevec:
            self.bounds.append(bounds[statevec_temp.index(unique_statevec_component)])
            self.scale.append(scales[statevec_temp.index(unique_statevec_component)])
            self.init.append(inits[statevec_temp.index(unique_statevec_component)])
        
        self.bvec = [bvec for bvec in RT.bvec for RT in self.RTs.values()]
        self.bval = s.hstack([RT.bval for RT in self.RTs.values()])
        # Not quite sure what to do with this right now. This check is to ensure
        # that future updates correctly handle additions to bvec and bval
        if len(self.bvec) != 1:
            raise NotImplementedError

        self.solar_irr = s.squeeze(s.array([RT.solar_irr for RT in self.RTs.values()]))
        self.coszen = s.squeeze(s.array([RT.coszen for RT in self.RTs.values()]))

    def xa(self):
        """Pull the priors from each of the individual RTs and order them
        in the same order as unique_statevec.
        TODO: check that the user correctly entered the priors
        TODO: the same for each radiative_transfer entry in the json input
        """
        prior_means = []

        statevec_temp, all_prior_means = [], []
        for RT in self.RTs.values():
            statevec_temp = statevec_temp + [statevec for statevec in RT.statevec]
            all_prior_means = all_prior_means + [prior_mean for prior_mean in RT.prior_mean]

        prior_mean = []
        for unique_statevec_component in self.unique_statevec:
            prior_mean.append(all_prior_means[statevec_temp.index(unique_statevec_component)])
        
        return s.array(prior_mean)

    def Sa(self):
        """Pull the priors from each of the individual RTs and order them
        in the same order as unique_statevec.
        TODO: check that the user correctly entered the priors
        TODO: the same for each radiative_transfer entry in the json input
        """

        prior_sigmas = []
        statevec_temp, all_prior_sigmas = [], []
        for RT in self.RTs.values():
            statevec_temp = statevec_temp + [statevec for statevec in RT.statevec]
            all_prior_sigmas = all_prior_sigmas + [prior_mean for prior_mean in RT.prior_sigma]

        prior_sigmas = []
        for unique_statevec_component in self.unique_statevec:
            prior_sigmas.append(all_prior_sigmas[statevec_temp.index(unique_statevec_component)])

        return s.diagflat(pow(s.array(prior_sigmas), 2))

    def get(self, x_RT, geom):

        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            ret.append(self.RTs[key].get(x_RT_dict[key], geom))

        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            #TODO: take rfl out of here and move it elsewhere
            ret.append(self.RTs[key].calc_rdn(x_RT_dict[key], rfl, Ls, geom))
        return s.hstack(ret)

    def drdn_dRT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls,
                 dLs_dsurface, geom):
        x_RT_dict = self.make_statevecs(x_RT)
        ret_K_RTs, ret_K_surfaces = [], []
        for key, RT in self.RTs.items():
            #TODO: take rfl out of here and move it elsewhere
            K_RT_temp, K_surface_temp = \
                self.RTs[key].drdn_dRT(x_RT_dict[key], x_surface, rfl, drfl_dsurface,
                                       Ls, dLs_dsurface, geom)
            ret_K_RTs.append(K_RT_temp)
            ret_K_surfaces.append(K_surface_temp)

        nxs = [x.shape[0] for x in ret_K_RTs]
        init_offsets = s.cumsum([0] + nxs)
        K_RT = s.zeros((sum(nxs), len(self.unique_statevec)))
        for RT, ret_K_RT, init_offset in zip(self.RTs.values(), ret_K_RTs, init_offsets[:-1]):
            tT = ret_K_RT.T
            for state_component, ret_K_RT_line in zip(RT.statevec, tT):
                st_ind = self.unique_statevec.index(state_component)
                sl = slice(init_offset, init_offset + len(ret_K_RT_line))
                K_RT[sl, st_ind] = ret_K_RT_line

        nxs = [x.shape[0] for x in ret_K_surfaces]
        nys = [x.shape[1] for x in ret_K_surfaces]
        init_x_offsets = s.cumsum([0] + nxs)
        init_y_offsets = s.cumsum([0] + nys)
        K_surface = s.zeros((sum(nxs), sum(nys)))
        for init_x_offset, init_y_offset, ret_K_surface in \
                zip(init_x_offsets[:-1], init_y_offsets[:-1], ret_K_surfaces):
            slx = slice(init_x_offset, init_x_offset + ret_K_surface.shape[0])
            sly = slice(init_y_offset, init_y_offset + ret_K_surface.shape[1])
            K_surface[slx, sly] = ret_K_surface

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):
        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            #TODO: take rfl out of here and move it elsewhere
            temp = self.RTs[key].drdn_dRTb(x_RT_dict[key], rfl, Ls, geom)
            ret.append(s.squeeze(temp))

        return s.hstack(ret)[:,s.newaxis]

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.RTs.values():
            ret.append(RT.summarize(x_RT, geom))
        ret = '\n'.join(ret)
        return ret

    def reconfigure(self, config_rt):
        for RT in self.RTs.values():
            RT.reconfigure(config_rt)
        return 

    def make_statevecs(self, x_RT):
        """Take the input state vector, whose elements are in the order
        of unique_statevec and create a dict of individual state vectors
        ready to be input to the individual RTMS in the RTs dict.
        """

        x_RT_dict = {}
        for key, RT in self.RTs.items():
            val = []
            # Go in order of unique_statevec
            #for state_component in self.unique_statevec:
                #if state_component in RT.statevec:
            for state_component in RT.statevec:
                val.append(x_RT[self.unique_statevec.index(state_component)])

            x_RT_dict[key] = s.squeeze(s.array(val))

        return x_RT_dict

    def pack_arrays(self, list_of_r_dicts):
        """Take the list of dict outputs from each RTM (in order of RTs) and
        stack their internal arrays in the same order.
        """
        r_stacked = {}
        for key in list_of_r_dicts[0].keys():
            temp = [x[key] for x in list_of_r_dicts]
            r_stacked[key] = s.hstack(temp)
        return r_stacked






