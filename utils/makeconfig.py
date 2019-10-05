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
import os
import sys
from os.path import join, exists, split, abspath
from shutil import copyfile
import scipy as s
from spectral.io import envi
from datetime import datetime
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
    parser.add_argument('--modtran_path', type=str)
    parser.add_argument('--sixs_path', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--rdn_factors_path', type=str)
    parser.add_argument('--surface_path', type=str)
    parser.add_argument('--surface_h2o_path', type=str)
    parser.add_argument('--level', type=str, default="INFO")
    parser.add_argument('--flag', type=float, default=-9999)
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', level=args.level)

    if args.isofit_path:
        isofit_path = args.isofit_path
    else:
        isofit_path = os.getenv('ISOFIT_BASE')
    isofit_exe = join(isofit_path, 'isofit', 'isofit.py')
    segment_exe = join(isofit_path, 'utils', 'segment.py')
    extract_exe = join(isofit_path, 'utils', 'extract.py')
    empline_exe = join(isofit_path, 'utils', 'empline.py')

    if args.sixs_path:
        sixs_path = args.sixs_path
    else:
        sixs_path = os.getenv('SIXS_DIR')

    if args.modtran_path:
        modtran_path = args.modtran_path
    else:
        modtran_path = os.getenv('MODTRAN_DIR')

    if args.surface_path:
        surface_path = args.surface_path
    else:
        surface_path = os.getenv('ISOFIT_SURFACE_MODEL')

    if args.surface_h2o_path:
        surface_h2o_path = args.surface_h2o_path
    else:
        surface_h2o_path = os.getenv('ISOFIT_SURFACE_H2O_MODEL')

    wrk_path = args.working_directory
    rdn_path = args.input_radiance
    loc_path = args.input_loc
    obs_path = args.input_obs
    lut_sixs_path = abspath(join(wrk_path, 'lut_h2o/'))
    lut_modtran_path = abspath(join(wrk_path, 'lut_full/'))
    config_path = abspath(join(wrk_path, 'config/'))
    data_path = abspath(join(wrk_path, 'data/'))
    input_path = abspath(join(wrk_path, 'input/'))
    output_path = abspath(join(wrk_path, 'output/'))

    # parse flightline ID (AVIRIS-NG assumptions)
    fid = split(rdn_path)[-1][:18]
    logging.info('Flightline ID: %s' % fid)
    month = int(fid[7:9])
    day = int(fid[9:11])
    hour = int(fid[12:14])
    minute = int(fid[14:16])
    gmtime = hour + float(minute)/60.0
    dt = datetime.strptime(fid[3:], '%Y%m%dt%H%M%S')
    dayofyear = dt.timetuple().tm_yday

    # define staged file locations
    rdn_fname            = fid+'_rdn'
    obs_fname            = fid+'_obs'
    loc_fname            = fid+'_loc'
    rfl_fname            = rdn_fname.replace('_rdn','_rfl')
    lbl_fname            = rdn_fname.replace('_rdn','_lbl')
    uncert_fname         = rdn_fname.replace('_rdn','_uncert')
    state_fname          = rdn_fname.replace('_rdn','_state')
    rdn_subs_fname       = rdn_fname.replace('_rdn','_subs_rdn')
    obs_subs_fname       = obs_fname.replace('_obs','_subs_obs')
    loc_subs_fname       = loc_fname.replace('_loc','_subs_loc')
    rfl_subs_fname       = rfl_fname.replace('_rfl','_subs_rfl')
    state_subs_fname     = rfl_fname.replace('_rfl','_subs_state')
    uncert_subs_fname    = rfl_fname.replace('_rfl','_subs_uncert')
    h2o_subs_fname       = loc_fname.replace('_loc','_subs_h2o')
    rdn_working_path     = abspath(join(input_path,  rdn_fname))
    obs_working_path     = abspath(join(input_path,  obs_fname))
    loc_working_path     = abspath(join(input_path,  loc_fname))
    rfl_working_path     = abspath(join(output_path, rfl_fname))
    uncert_working_path  = abspath(join(output_path, uncert_fname))
    lbl_working_path     = abspath(join(output_path, lbl_fname))
    rdn_subs_path        = abspath(join(input_path,  rdn_subs_fname))
    obs_subs_path        = abspath(join(input_path,  obs_subs_fname))
    loc_subs_path        = abspath(join(input_path,  loc_subs_fname))
    rfl_subs_path        = abspath(join(output_path, rfl_subs_fname))
    state_subs_path      = abspath(join(output_path, state_subs_fname))
    uncert_subs_path     = abspath(join(output_path, uncert_subs_fname))
    h2o_subs_path        = abspath(join(output_path, h2o_subs_fname))
    surface_working_path = abspath(join(data_path,   'surface.mat'))
    surface_h2o_working_path = abspath(join(data_path, 'surface_h2o.mat'))
    wl_path = abspath(join(data_path,   'wavelengths.txt'))
    modtran_tpl_path = abspath(join(config_path, fid+'_modtran_tpl.json'))
    modtran_config_path = abspath(join(config_path, fid+'_modtran.json'))
    sixs_config_path = abspath(join(config_path, fid+'_sixs.json'))
    esd_path = join(isofit_path, 'data', 'earth_sun_distance.txt')
    irradiance_path = join(isofit_path, 'data', 'kurudz_0.1nm.dat')
    noise_path = join(isofit_path, 'data', 'avirisng_noise.txt')
    aerosol_mdl_path = join(isofit_path, 'data', 'aerosol_model.txt')
    aerosol_tpl_path = join(isofit_path, 'data', 'aerosol_template.json')

    # create missing directories
    for dpath in [wrk_path, lut_sixs_path, lut_modtran_path, config_path,
                  data_path, input_path, output_path]:
        if not exists(dpath):
            os.mkdir(dpath)

    # stage data files by copying into working directory
                (obs_path, obs_working_path),
                (loc_path, loc_working_path)]:
        if not exists(dst):
            logging.info('Staging %s to %s' % (src, dst))
            copyfile(src, dst)
            copyfile(src+'.hdr', dst+'.hdr')

    # Staging files without headers
    for src, dst in [(surface_path, surface_working_path),
                     (surface_h2o_path, surface_h2o_working_path)]:
        if not exists(dst):
            logging.info('Staging %s to %s' % (src, dst))
            copyfile(src, dst)

    # Superpixel segmentation
    if not exists(lbl_working_path) or not exists(rdn_working_path):
        logging.info('Segmenting...')
        os.system('python ' + segment_exe + ' %s %s'%\
                (rdn_working_path, lbl_working_path))

    # Extract input data 
    for inp, outp in [(rdn_working_path, rdn_subs_path),
                      (obs_working_path, obs_subs_path),
                      (loc_working_path, loc_subs_path)]:
        if not exists(outp):
            logging.info('Extracting '+outp)
            os.system('python ' + extract_exe + ' %s %s %s'%\
                    (inp, lbl_working_path, outp))

    # get radiance file, wavelengths
    rdn = envi.open(rdn_subs_path+'.hdr')
    if args.wavelength_path:
        chn, wl, fwhm = s.loadtxt(args.wavelength_path).T
    else:
        wl = s.array([float(w) for w in rdn.metadata['wavelength']])
        if 'fwhm' in rdn.metadata:
            fwhm = s.array([float(f) for f in rdn.metadata['fwhm']])
        else:
            fwhm = s.ones(wl.shape) * (wl[1]-wl[0])

    # Convert to microns if needed
    if wl[0] > 100:
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # Recognize bad data flags
    valid = (rdn.read_band(0) - args.flag) > eps

    # load bands to define the extrema of geometric information we
    # will need
    logging.info('Building configuration for 6SV')
    obs = envi.open(obs_subs_path + '.hdr')
    loc = envi.open(loc_subs_path + '.hdr')
    lats = loc.read_band(1)
    lons = loc.read_band(0)
    elevs = loc.read_band(2)
    elevs_km = elevs / 1000.0
    paths_km = obs.read_band(0) / 1000.0
    obs_azimuths = obs.read_band(1)
    obs_zeniths = obs.read_band(2)
    solar_azimuth = s.mean(obs.read_band(3)[valid])
    solar_zenith = s.mean(obs.read_band(4)[valid])
    OBSZENs = 180.0 - obs_zeniths  # MODTRAN convention
    RELAZs = obs_azimuths - solar_azimuth + 180.0
    TRUEAZs = RELAZs  # MODTRAN convention?
    GNDALT = elevs
    latitude = s.mean(lats[valid])
    longitude = -s.mean(lons[valid])
    longitudeE = s.mean(lons[valid])
    obs_azimuth = obs_azimuths[valid][0]
    obs_zenith = obs_zeniths[valid][0]
    obs_zenith_rad = (obs_zenith/360.0 * 2.0 * s.pi)
    path_km = paths_km[valid][0]
    elev_km = elevs_km[valid][0]
    alt = elev_km + s.cos(obs_zenith_rad) * path_km
    relative_alt = abs(s.cos(obs_zenith_rad) * path_km)
    logging.info('Path: %f, Zenith: %f, Altitude: %6.2f km' %
                 (path_km, obs_zenith_rad, alt))

    # make view zenith and relaz grid - two points only for now
    OBSZEN_grid = s.array([OBSZENs[valid].min(), 0])

    # make elevation grid
    elev_grid_margin = 0.3
    elev_grid_step = 0.3
    elevation_grid = s.arange(elevs_km[valid].min(),
                              elevs_km[valid].max() + elev_grid_margin +
                              elev_grid_step,
                              elev_grid_step)

    # write wavelength file
    wl_data = s.concatenate([s.arange(len(wl))[:, s.newaxis], wl[:, s.newaxis],
                             fwhm[:, s.newaxis]], axis=1)
    s.savetxt(wl_path, wl_data, delimiter=' ')

    if not exists(h2o_subs_path + '.hdr') or not exists(h2o_subs_path):

        # make sixs configuration
        sixs_configuration = {
            "domain": {"start": 350, "end": 2520, "step": 0.1},
            "statevector": { "H2OSTR": { "bounds": [0.5, 6.0], "scale": 0.01,
                "prior_sigma": 100.0, "prior_mean": 1.0, "init": 1.75 }, },
            "lut_grid": { "H2OSTR": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 
                4.5, 5.0, 5.5, 6.0]},
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
        
        # make isofit configuration
        isofit_config_sixs = {'ISOFIT_base': isofit_path,
            'input':{'measured_radiance_file':rdn_subs_path,
                     'loc_file':loc_subs_path,
                     'obs_file':obs_subs_path},
            'output':{'estimated_state_file':h2o_subs_path},
            'forward_model': {
                'instrument': { 'wavelength_file': wl_path,
                'parametric_noise_file': noise_path,
                'integrations':1 },
            "multicomponent_surface": {"wavelength_file":wl_path,
                "surface_file":surface_h2o_working_path},
            "sixs_radiative_transfer": sixs_configuration},
            "inversion": {"windows": [[880.0,1000.0]], 'max_nfev': 10}}
        
        # write sixs configuration
        with open(sixs_config_path,'w') as fout:
            fout.write(json.dumps(isofit_config_sixs, indent=4, sort_keys=True))
        
        # Run sixs retrieval
        logging.info('Running ISOFIT to generate h2o first guesses')
        os.system('python ' + isofit_exe + ' --level DEBUG ' + sixs_config_path)

    # Extract h2o grid avoiding the zero label (periphery, bad data) 
    # and outliers
    h2o = envi.open(h2o_subs_path + '.hdr')
    h2o_est = h2o.read_band(-1)
    h2ostep = 0.2
    h2o_sorted = s.sort(h2o_est[1:].flat)
    nseg = len(h2o_sorted)
    h2o_lo = h2o_sorted[int(nseg*0.025)]
    h2o_hi = h2o_sorted[int(nseg*0.975)]
    h2o_median = s.median(h2o_sorted)
    h2o_grid = s.arange(h2o_lo, h2o_hi+h2ostep, h2ostep)
   
    if not exists(state_subs_path) or \
            not exists(uncert_subs_path) or \
            not exists(rfl_subs_path):

        atmosphere_type = 'ATM_MIDLAT_SUMMER'
        
        # make modtran configuration
        modtran_template = {"MODTRAN": [{
          "MODTRANINPUT": {
            "NAME": fid,
            "DESCRIPTION": "",
            "CASE": 0,
            "RTOPTIONS": {
              "MODTRN": "RT_CORRK_FAST",
              "LYMOLC": False,
              "T_BEST": False,
              "IEMSCT": "RT_SOLAR_AND_THERMAL",
              "IMULT": "RT_DISORT",
              "DISALB": False,
              "NSTR": 8,
              "SOLCON": 0.0
            },
            "ATMOSPHERE": {
              "MODEL": atmosphere_type,
              "M1": atmosphere_type,
              "M2": atmosphere_type,
              "M3": atmosphere_type,
              "M4": atmosphere_type,
              "M5": atmosphere_type,
              "M6": atmosphere_type,
              "CO2MX": 410.0,
              "H2OSTR": 1.0,
              "H2OUNIT": "g",
              "O3STR": 0.3,
              "O3UNIT": "a"
            },
            "AEROSOLS": { "IHAZE": "AER_RURAL" },
            "GEOMETRY": {
              "ITYPE": 3,
              "H1ALT": alt,
              "IDAY": dayofyear,
              "IPARM": 11,
              "PARM1": latitude,
              "PARM2": longitude,
              "GMTIME": gmtime
            },
            "SURFACE": {
              "SURFTYPE": "REFL_LAMBER_MODEL",
              "GNDALT": elev_km,
              "NSURF": 1,
              "SURFP": { "CSALB": "LAMB_CONST_0_PCT" }
            },
            "SPECTRAL": {
              "V1": 350.0,
              "V2": 2520.0,
              "DV": 0.1,
              "FWHM": 0.1,
              "YFLAG": "R",
              "XFLAG": "N",
              "FLAGS": "NT A   ",
              "BMNAME": "p1_2013"
            },
            "FILEOPTIONS": {
              "NOPRNT": 2,
              "CKPRNT": True
            }
          }
        }]}
        
        # write modtran_template 
        with open(modtran_tpl_path,'w') as fout:
            fout.write(json.dumps(modtran_template, indent=4, sort_keys=True))
        
        modtran_configuration = { 
            "wavelength_file":wl_path,
            "lut_path":lut_modtran_path,
            "aerosol_template_file_old":aerosol_tpl_path,
            "aerosol_model_file_old":aerosol_mdl_path,
            "modtran_template_file":modtran_tpl_path,
            "modtran_directory": modtran_path,
            "statevector": {
              "H2OSTR": {
                "bounds": [h2o_lo, h2o_hi],
                "scale": 0.01,
                "init": h2o_median,
                "prior_sigma": (h2o_hi - h2o_lo),
                "prior_mean": h2o_median
              },
              "VIS": {
                "bounds": [20,100],
                "scale": 1,
                "init": 20.5,
                "prior_sigma":1000,
                "prior_mean":20
              }
            },
            "lut_grid": {
              "H2OSTR": [float(q) for q in h2o_grid],
              "GNDALT": [float(q) for q in elevation_grid],
              "VIS": [20,50,100]
            },
            "unknowns": {
              "H2O_ABSCO": 0.0
            },
            "domain": { "start": 350, "end": 2520, "step": 0.1}
        }
        
        # make isofit configuration
        isofit_config_modtran = {'ISOFIT_base': isofit_path,
            'input':{'measured_radiance_file':rdn_subs_path,
                     'loc_file':loc_subs_path,
                     'obs_file':obs_subs_path},
            'output':{'estimated_state_file':state_subs_path,
                      'posterior_uncertainty_file':uncert_subs_path,
                      'estimated_reflectance_file':rfl_subs_path},
            'forward_model': {
                'instrument': { 'wavelength_file': wl_path,
                'parametric_noise_file': noise_path,
                'integrations':1 },
            "multicomponent_surface": {"wavelength_file":wl_path,
                "surface_file":surface_working_path},
            "modtran_radiative_transfer": modtran_configuration},
            "inversion": {"windows": 
                [[400.0,1300.0],
                [1450,1780.0],
                [2000.0,2450.0]]}}
        
        if args.rdn_factors_path:
            isofit_config_modtran['input']['radiommetry_correction_file'] = \
                args.rdn_factors_path
        
        # write modtran_template 
        with open(modtran_config_path,'w') as fout:
            fout.write(json.dumps(isofit_config_modtran, indent=4, sort_keys=True))
        
        # Run modtran retrieval
        logging.info('Running ISOFIT with full LUT')
        cmd = 'python ' +isofit_exe+ ' --level DEBUG ' + modtran_config_path
        print(cmd)
        os.system(cmd)

        # clean up unneeded storage
        for to_rm in ['*r_k','*t_k','*tp7','*wrn','*psc','*plt','*7sc','*acd']:
          cmd = 'rm '+ join(lut_modtran_path, to_rm)
          print(cmd)

    if not exists(rfl_working_path) or not exists(uncert_working_path):

        # Empirical line 
        logging.info('Empirical line inference')
        os.system(('python ' +empline_exe+ ' --level INFO --hash %s '+\
                '%s %s %s %s %s %s %s')%\
                (lbl_working_path, rdn_subs_path, rfl_subs_path, loc_subs_path,
                 rdn_working_path, loc_working_path, rfl_working_path,
                 uncert_working_path))

    logging.info('Done.')


if __name__ == "__main__":
    main()
