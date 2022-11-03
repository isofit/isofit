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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import argparse
import os
from spectral.io import envi
import numpy as np
from glob import glob
import logging
import ray
import time

from isofit.utils import remap
from isofit.core.common import envi_header, svd_inv, svd_inv_sqrt
from isofit.configs import configs
from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_simple import invert_algebraic, invert_analytical
from isofit.core.geometry import Geometry
from isofit.core.fileio import write_bil_chunk

from isofit.utils.atm_interpolation import atm_interpolation
from collections import OrderedDict


def main(rawargs=None):
    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('rdn_file', type=str)
    parser.add_argument('loc_file', type=str)
    parser.add_argument('obs_file', type=str)
    parser.add_argument('isofit_dir', type=str)
    parser.add_argument('--n_atm_neighbors', type=int, default=20)
    parser.add_argument('--smoothing_sigma', type=int, default=2)
    parser.add_argument('--output_rfl_file', type=str, default=None)
    parser.add_argument('--output_unc_file', type=str, default=None)
    parser.add_argument('--atm_file', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='INFO')
    parser.add_argument('--logfile', type=str, default=None)
    args = parser.parse_args(rawargs)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                            filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    config_file = glob(os.path.join(args.isofit_dir, 'config','') + '*_modtran.json')[0]
    config = configs.create_new_config(config_file)
    config.forward_model.instrument.integrations = 1

    subs_state_file = config.output.estimated_state_file
    subs_loc_file = config.input.loc_file
    full_state_file = subs_state_file.replace('_subs_state', '_subs_state_mapped')
    lbl_file = subs_state_file.replace('_subs_state', '_lbl')
    if args.output_rfl_file is None:
        analytical_state_file = subs_state_file.replace('_subs_state', '_state_analytical')
    else:
        analytical_state_file = args.output_rfl_file

    if args.output_unc_file is None:
        analytical_state_unc_file = subs_state_file.replace('_subs_state', '_state_analytical_uncert')
    else:
        analytical_state_unc_file = args.output_unc_file

    if args.atm_file is None:
        atm_file = subs_state_file.replace('_subs_state', '_atm_interp')
    else:
        atm_file = args.atm_file

    if os.path.isfile(atm_file) is False:
       atm_interpolation(subs_state_file, subs_loc_file, lbl_file, args.loc_file, atm_file, 
                          nneighbors=args.n_atm_neighbors, gaussian_smoothing_sigma=args.smoothing_sigma) 

    fm = ForwardModel(config)
    iv = Inversion(config, fm)

    rdn_ds = envi.open(envi_header(args.rdn_file))
    rdn = rdn_ds.open_memmap(interleave='bip')
    rdns = rdn.shape
    loc = envi.open(envi_header(args.loc_file)).open_memmap(interleave='bip')#
    obs = envi.open(envi_header(args.obs_file)).open_memmap(interleave='bip')#
    atm = envi.open(envi_header(atm_file)).open_memmap(interleave='bip')

    hash_table = OrderedDict()
    hash_size = 500
    
    output_metadata = rdn_ds.metadata
    output_metadata['interleave'] = 'bil'
    output_metadata['description'] = 'L2A Analytyical per-pixel surface retrieval'
    if 'emit pge input files' in list(output_metadata.keys()):
        del output_metadata['emit pge input files']
    
    output_reflectance_img = envi.create_image(envi_header(analytical_state_file), ext='',
                                               metadata=output_metadata, force=True)
    del output_reflectance_img
    output_reflectance_unc_img = envi.create_image(envi_header(analytical_state_unc_file), ext='',
                                                   metadata=output_metadata, force=True)
    del output_reflectance_unc_img
    del atm, rdn

    rayargs = {'ignore_reinit_error': config.implementation.ray_ignore_reinit_error,
               "address": config.implementation.ip_head,
               '_temp_dir': config.implementation.ray_temp_dir,
               'include_dashboard': config.implementation.ray_include_dashboard,
               "_redis_password": config.implementation.redis_password}


    ray.init(**rayargs)
    fm_id = ray.put(fm)
    remote_worker = ray.remote(Worker)
    n_workers = 40
    workers = ray.util.ActorPool([remote_worker.remote(config, fm_id, atm_file, 
                                  analytical_state_file, analytical_state_unc_file, args.rdn_file, args.loc_file, args.obs_file,
                                  args.loglevel, args.logfile) for n in range(n_workers)])

    line_breaks = np.linspace(0,rdns[0], n_workers*3,dtype=int)
    line_breaks = [(line_breaks[n], line_breaks[n+1]) for n in range(len(line_breaks)-1)]
    
    start_time = time.time()
    res = list(workers.map_unordered(lambda a, b: a.run_lines.remote(b), line_breaks))

    total_time = time.time() - start_time
    logging.info(f'Analytical line inversions complete.  {round(total_time,2)}s total, {round(rdns[0]*rdns[1]/total_time,4)} spectra/s, '
                 f'{round(rdns[0]*rdns[1]/total_time/n_workers,4)} spectra/s/core')

class Worker(object):
    def __init__(self, config: configs.Config, fm: ForwardModel, RT_state_file: str, 
                 analytical_state_file: str, analytical_state_unc_file: str, 
                 rdn_file:str, loc_file:str, obs_file:str, loglevel: str, logfile: str):
        """
        Worker class to help run a subset of spectra.

        Args:
            forward_model: isofit forward_model
            loglevel: output logging level
            logfile: output logging file
            worker_id: worker ID for logging reference
            total_workers: the total number of workers running, for logging reference
        """

        logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=loglevel, filename=logfile,
                            datefmt='%Y-%m-%d,%H:%M:%S')
        self.config = config
        self.fm = fm
        self.iv = Inversion(self.config, self.fm)

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = 500
        self.RT_state_file = RT_state_file
        self.rdn_file = rdn_file
        self.loc_file = loc_file
        self.obs_file = obs_file
        self.analytical_state_file = analytical_state_file
        self.analytical_state_unc_file = analytical_state_unc_file

    def run_lines(self, startstop: tuple):


        rdn = envi.open(envi_header(self.rdn_file)).open_memmap(interleave='bip')
        loc = envi.open(envi_header(self.loc_file)).open_memmap(interleave='bip')
        obs = envi.open(envi_header(self.obs_file)).open_memmap(interleave='bip')
        rt_state = envi.open(envi_header(self.RT_state_file)).open_memmap(interleave='bip')

        start_line, stop_line = startstop
        output_state = np.zeros((stop_line-start_line, rt_state.shape[1], rdn.shape[2])) - 9999
        output_state_unc = np.zeros((stop_line-start_line, rt_state.shape[1], rdn.shape[2])) - 9999

        for r in range(start_line, stop_line):
            
            for c in range(output_state.shape[1]):
                meas = rdn[r,c,:]
                if np.all(meas < 0):
                    continue
                x_RT = rt_state[r,c,:]
                geom = Geometry(obs = obs[r,c,:], loc=loc[r,c,:])

                states, unc = invert_analytical(self.iv.fm,  self.iv.winidx, meas, geom, x_RT, 1, self.hash_table, self.hash_size)
                output_state[r-start_line,c,:] = states[-1][self.iv.fm.surface.idx_lamb]
                output_state_unc[r-start_line,c,:] = unc[self.iv.fm.surface.idx_lamb]

            logging.info(f'Analytical line writing line {r}')

            write_bil_chunk(output_state[r-start_line,...].T, self.analytical_state_file, r, (rdn.shape[0], rdn.shape[1], rdn.shape[2]))
            write_bil_chunk(output_state_unc[r-start_line,...].T, self.analytical_state_unc_file, r, (rdn.shape[0], rdn.shape[1], rdn.shape[2]))


if __name__ == "__main__":
    main()
