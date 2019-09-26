#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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

import argparse
from scipy import logical_and as aand
import os, sys
from os.path import join, exists, split, abspath
from shutil import copyfile
import scipy as s
from spectral.io import envi
from skimage.segmentation import slic
from scipy.linalg import eigh, norm
from spectral.io import envi 
import logging
import json

eps = 1e-6

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Representative subset")
    parser.add_argument('input_radiance',  type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('--h2o', action='store_true')
    parser.add_argument('--isofit_path', type=str)
    parser.add_argument('--sixs_path', type=str)
    parser.add_argument('--surface_path', type=str)
    parser.add_argument('--level', type=str, default="INFO")
    parser.add_argument('--flag', type=float, default=-9999)
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', level=args.level)

    if args.isofit_path:
        isofit_path = args.isofit_path
    else:
        isofit_path = os.getenv('ISOFIT_BASE')
    isofit_exe = join(isofit_path,'isofit','isofit.py')
    
    if args.sixs_path:
        sixs_path = args.sixs_path
    else:
        sixs_path = os.getenv('SIXS_DIR')

    if args.surface_path:
        surface_path = args.surface_path
    else:
        surface_path = os.getenv('ISOFIT_SURFACE_MODEL')

    wrk_path = args.working_directory
    rdn_path = args.input_radiance
    loc_path = args.input_loc
    obs_path = args.input_obs
    lut_sixs_path    = abspath(join(wrk_path,'lut_h2o/'))
    lut_modtran_path = abspath(join(wrk_path,'lut_full/'))
    config_path      = abspath(join(wrk_path,'config/'))
    data_path        = abspath(join(wrk_path,'data/'))
    input_path       = abspath(join(wrk_path,'input/'))
    output_path      = abspath(join(wrk_path,'output/'))

    # parse flightline ID (AVIRIS-NG assumptions)
    fid = split(rdn_path)[-1][:18]
    logging.info('Flightline ID: %s' % fid)
    month = int(fid[7:9])
    day = int(fid[9:11])

    # define staged file locations
    rdn_working_path     = abspath(join(input_path,  split(rdn_path)[-1]))
    obs_working_path     = abspath(join(input_path,  split(obs_path)[-1]))
    loc_working_path     = abspath(join(input_path,  split(loc_path)[-1]))
    rfl_working_path     = abspath(join(output_path, split(loc_path)[-1]))
    h2o_working_path     = abspath(join(output_path, fid+'_state_sixs'))
    surface_working_path = abspath(join(data_path,   'surface.mat'))
    wl_path              = abspath(join(data_path,   'wavelengths.txt'))
    sixs_config_path     = abspath(join(config_path, fid+'_sixs.json'))
    esd_path         = join(isofit_path, 'data', 'earth_sun_distance.txt')
    irradiance_path  = join(isofit_path, 'data', 'kurudz_0.1nm.dat')
    noise_path       = join(isofit_path, 'data', 'avirisng_noise.txt')
    aerosol_mdl_path = join(isofit_path, 'data', 'aerosol_model.txt')
    aerosol_tpl_path = join(isofit_path, 'data', 'aerosol_template.json')
                                                
    # create missing directories
    for dpath in [wrk_path, lut_sixs_path, lut_modtran_path, config_path,
            data_path, input_path, output_path]:
        if not exists(dpath):
            os.mkdir(dpath)

    # stage data files by copying into working directory
    for src, dst in [(rdn_path, rdn_working_path),
                (obs_path, obs_working_path),
                (loc_path, loc_working_path)]:
        logging.info('Staging %s to %s' % (src, dst))
        copyfile(src, dst)
        copyfile(src+'.hdr', dst+'.hdr')

    # Staging files without headers
    for src, dst in [(surface_path, surface_working_path)]:
        logging.info('Staging %s to %s' % (src, dst))
        copyfile(src, dst)

    # get radiance file, wavelengths
    rdn = envi.open(rdn_working_path+'.hdr')
    wl = s.array([float(w) for w in rdn.metadata['wavelength']])
    if 'fwhm' in rdn.metadata:
        fwhm = s.array([float(f) for f in rdn.metadata['fwhm']])
    else:
        fwhm = s.ones(wl.shape) * (wl[1]-wl[0])
    valid = (rdn.read_band(0) - args.flag) > eps
    
    # Convert to microns if needed
    if wl[0]>100: 
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # load bands to define the extrema of geometric information we
    # will need
    logging.info('Building configuration for 6SV')
    obs            = envi.open(obs_path + '.hdr')
    loc            = envi.open(loc_path + '.hdr')
    lats           = loc.read_band(1)
    lons           = loc.read_band(0)
    elevs          = loc.read_band(2)
    elevs_km       = elevs / 1000.0
    paths_km       = obs.read_band(0) / 1000.0
    obs_azimuths   = obs.read_band(1)
    obs_zeniths    = obs.read_band(2)
    solar_azimuth  = s.mean(obs.read_band(3)[valid])
    solar_zenith   = s.mean(obs.read_band(4)[valid])
    OBSZENs        = 180.0 - obs_zeniths  # MODTRAN convention
    RELAZs         = obs_azimuths - solar_azimuth + 180.0
    TRUEAZs        = RELAZs  # MODTRAN convention?
    GNDALT         = elevs
    latitude       = s.mean(lats[valid])
    longitude      = s.mean(lons[valid])
    longitudeE     = -s.mean(lons[valid])
    if longitude < 0:
        longitude = 360.0 - longitude
    obs_azimuth  = obs_azimuths[valid][0]
    obs_zenith   = obs_zeniths[valid][0]
    obs_zenith_rdn = (17.0/360.0 * 2.0 * s.pi)
    path_km      = paths_km[valid][0]
    elev_km      = elevs[valid][0]
    alt          = elev_km + s.cos(obs_zenith_rdn) * path_km
    logging.info('Altitude: %6.2f km' % alt)
    elevation_grid = s.arange(elevs_km[valid].min()-0.1, 
            elevs_km[valid].max()+0.1,0.2)
    relative_alt = abs(s.cos(obs_zenith_rdn) * path_km)

    # write wavelength file
    wl_data = s.concatenate([s.arange(len(wl))[:,s.newaxis], wl[:,s.newaxis],
                   fwhm[:,s.newaxis]], axis=1)
    s.savetxt(wl_path, wl_data, delimiter=' ')

    # make sixs configuration
    sixs_configuration = {
        "domain": {"start": 350, "end": 2520, "step": 0.1},
        "statevector": { "H2OSTR": { "bounds": [0.5, 6.0], "scale": 0.01,
            "prior_sigma": 100.0, "prior_mean": 1.0, "init": 1.75 }, },
        "lut_grid": { "H2OSTR": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 
            4.5, 5.0, 5.5, 6.0],
            "elev": list(elevation_grid)},
        "unknowns": {},
        "wavelength_file": wl_path,
        "earth_sun_distance_file" : esd_path, 
        "irradiance_file" : irradiance_path,
        "sixs_installation": sixs_path,
        "lut_path": lut_sixs_path,
        'solzen': solar_zenith,
        "solaz": solar_azimuth,
        "elev": elev_km,
        "alt": relative_alt,
        "viewzen": obs_zenith,
        "viewaz": obs_azimuth,
        "month": month,
        "day": day}

    print(data_path)
    print(wl_path)
    # make isofit configuration
    isofit_config_sixs = {'ISOFIT_base': isofit_path,
        'input':{'measured_radiance_file':rdn_working_path},
        'output':{'estimated_state_file':h2o_working_path},
        'forward_model': {
            'instrument': { 'wavelength_file': wl_path,
            'parametric_noise_file': noise_path,
            'integrations':1 },
        "multicomponent_surface": {"wavelength_file":wl_path,
            "surface_file":surface_working_path},
        "sixs_radiative_transfer": sixs_configuration},
        "inversion": {"windows": [[880.0,1000.0]], 'max_nfev': 10}}

    # write sixs configuration
    with open(sixs_config_path,'w') as fout:
        fout.write(json.dumps(isofit_config_sixs, indent=4, sort_keys=True))

    # Run sixs retrieval
    logging.info('Running ISOFIT to generate h2o first guesses')
    os.system('pythonw ' + isofit_exe + ' --level DEBUG ' + sixs_config_path)

if __name__ == "__main__":
    main()
