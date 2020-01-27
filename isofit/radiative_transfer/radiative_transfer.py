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
import pdb

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

        #TODO: fix for multiple bands
        self.wl = s.squeeze(s.array([RT.wl for RT in self.RTs.values()]))
        # The state vector is made of up unique elements of the individual RTs.
        # The idea is that each RT LUT may need H20STR, for example, but we
        # only want a single H20STR value in the statevector as it is shared.
        all_statevec = [s for RT in self.RTs.values() for s in RT.statevec] # Flatten a list of lists
        self.unique_statevec = list(set(all_statevec)) # Get unique elements
        self.unique_statevec.sort()
        self.len_statevecs = [RT.n_state for RT in self.RTs.values()]


        #self.statevec = self.RTs['modtran_visnir'].statevec
        self.statevec = self.unique_statevec

        #self.bounds = self.RTs['modtran_visnir'].bounds
        #self.scale = self.RTs['modtran_visnir'].scale
        #self.init = self.RTs['modtran_visnir'].init
        bounds = []
        scale = []
        init = []
        print(self.unique_statevec, self.RTs['modtran_visnir'].statevec)
        for key in self.unique_statevec:
            if key in self.RTs['modtran_visnir'].statevec:
                RT_temp = self.RTs['modtran_visnir']
                bounds.append(RT_temp.bounds[RT_temp.statevec.index(key)])
                scale.append(RT_temp.scale[RT_temp.statevec.index(key)])
                init.append(RT_temp.init[RT_temp.statevec.index(key)])
        self.bounds = bounds
        self.scale = scale
        self.init = init

        self.bvec = self.RTs['modtran_visnir'].bvec
        self.bval = self.RTs['modtran_visnir'].bval
        #self.lut_names = self.RTs['modtran_visnir'].lut_names
        #self.lut_names = self.unique_statevec

        #lut_grids = []
        #print(self.unique_statevec, self.RTs['modtran_visnir'].statevec)
        #for key in self.unique_statevec:
        #    if key in self.RTs['modtran_visnir'].statevec:
        #        RT_temp = self.RTs['modtran_visnir']
        #        lut_grids.append(RT_temp.lut_grids[RT_temp.statevec.index(key)])
        #self.lut_grids = lut_grids
        self.solar_irr = self.RTs['modtran_visnir'].solar_irr
        self.coszen = self.RTs['modtran_visnir'].coszen

        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))
        #self. = s.squeeze(s.array([RT. for RT in self.RTs.values()]))

    def xa(self):
        #xa = self.RTs['modtran_visnir'].xa()
        prior_means = []
        # We want to get the prior mean for each unique_statevec
        # element in its order. For state vector elements that are
        # shared between multiple RTMs, assume that the user
        # correctly entered the priors as being the same.
        # TODO: check that the user correctly entered the priors
        # TODO: the same for each radiative_transfer entry in the json input

        prior_means = []
        # List of tuples of all RTs: (statevec_name, prior_mean)
        # For example: 
        for key, RT in self.RTs.items():
            prior_means = prior_means + \
                          [(statevec, prior_mean) for statevec, prior_mean in zip(RT.statevec, RT.prior_mean)]
        
        prior_means = list(set(prior_means)) # Keep only unique entries

        # Unless the user entered different priors for the same state vector elements
        # in different 'radiative_tranfer' json entries, these should be the same size.
        if len(prior_means) != len(self.unique_statevec):
            raise AssertionError

        prior_means.sort()  # Sort them into the same order as self.unique_statevec
        prior_means = [prior_mean for _, prior_mean in prior_means] # Pull out prior value
        return s.array(prior_means)

    def Sa(self):

        prior_sigmas = []
        # List of tuples of all RTs: (statevec_name, prior_sigma)
        # For example: 
        for key, RT in self.RTs.items():
            prior_sigmas = prior_sigmas + \
                           [(statevec, prior_sigma) for statevec, prior_sigma in zip(RT.statevec, RT.prior_sigma)]
        
        prior_sigmas = list(set(prior_sigmas)) # Keep only unique entries

        # Unless the user entered different priors for the same state vector elements
        # in different 'radiative_tranfer' json entries, these should be the same size.
        if len(prior_sigmas) != len(self.unique_statevec):
            raise AssertionError

        prior_sigmas.sort()  # Sort them into the same order as self.unique_statevec
        prior_sigmas = [prior_sigmas for _, prior_sigmas in prior_sigmas] # Pull out prior value

        ret = s.diagflat(pow(s.array(prior_sigmas), 2))
        #temp = self.RTs['modtran_visnir'].Sa()
        return ret

    def get(self, x_RT, geom):

        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            ret.append(self.RTs[key].get(x_RT_dict[key], geom))
        #pdb.set_trace()
        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            #TODO: take rfl out of here and move it elsewhere
            ret.append(self.RTs[key].calc_rdn(x_RT_dict[key], rfl, Ls, geom))
        #pdb.set_trace()
        return s.hstack(ret)
        #return self.RTs['modtran_visnir'].calc_rdn(x_RT, rfl, Ls, geom)

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

        #pdb.set_trace()

        return K_RT, K_surface
        #return self.RTs['modtran_visnir'].drdn_dRT(x_RT, x_surface, rfl, drfl_dsurface,
        #                            Ls, dLs_dsurface, geom)

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):
        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            #TODO: take rfl out of here and move it elsewhere
            temp = self.RTs[key].drdn_dRTb(x_RT_dict[key], rfl, Ls, geom)
            ret.append(s.squeeze(temp))
        #pdb.set_trace()
        return s.hstack(ret)[:,s.newaxis]
        #return self.RTs['modtran_visnir'].drdn_dRTb(x_RT, rfl, Ls, geom)

    def summarize(self, x_RT, geom):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].summarize(x_RT, geom)

    def reconfigure(self, config_rt):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].reconfigure(config_rt)

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






