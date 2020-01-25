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
        self.len_statevecs = [RT.n_state for RT in self.RTs.values()]


        self.statevec = self.RTs['modtran_visnir'].statevec
        self.bounds = self.RTs['modtran_visnir'].bounds
        self.scale = self.RTs['modtran_visnir'].scale
        self.init = self.RTs['modtran_visnir'].init
        self.bvec = self.RTs['modtran_visnir'].bvec
        self.bval = self.RTs['modtran_visnir'].bval
        self.lut_names = self.RTs['modtran_visnir'].lut_names
        self.lut_grids = self.RTs['modtran_visnir'].lut_grids
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
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].xa()
        return self.RTs['modtran_visnir'].get(x_RT, geom)
    
    def Sa(self):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].Sa()
    
    def get(self, x_RT, geom):

        x_RT_dict = self.make_statevecs(x_RT)
        ret = []
        for key, RT in self.RTs.items():
            ret.append(self.RTs[key].get(x_RT_dict[key], geom))
        
        return self.pack_arrays(ret)
    
    def calc_rdn(self, x_RT, rfl, Ls, geom):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].calc_rdn(x_RT, rfl, Ls, geom)
    
    def drdn_dRT(self, x_RT, x_surface, rfl, drfl_dsurface, Ls, 
                 dLs_dsurface, geom):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].drdn_dRT(x_RT, x_surface, rfl, drfl_dsurface, 
                                    Ls, dLs_dsurface, geom)
    
    def drdn_dRTb(self, x_RT, rfl, Ls, geom):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].drdn_dRTb(x_RT, rfl, Ls, geom)
    
    def summarize(self, x_RT, geom):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].summarize(x_RT, geom)
    
    def reconfigure(self, config_rt):
        #TODO: fix for multiple bands
        return self.RTs['modtran_visnir'].reconfigure(config_rt)
    
    def make_statevecs(self, x_RT):
        x_RT_dict = {}
        for key, RT in self.RTs.items():
            val = []
            #for state_component in RT.statevec:
            for state_component in self.unique_statevec:
                if state_component in RT.statevec:
                    val.append(x_RT[self.unique_statevec.index(state_component)])

            x_RT_dict[key] = s.squeeze(s.array(val))
        
        return x_RT_dict
    
    def pack_arrays(self, r_dict):
        r_full = {}
        for key in r_dict[0].keys():
            temp = [x[key] for x in r_dict]
            r_full[key] = s.hstack(temp)
        return r_full






