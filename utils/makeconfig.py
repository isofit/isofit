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
from os.path import join, exists
import scipy as s
from spectral.io import envi
from skimage.segmentation import slic
from scipy.linalg import eigh, norm
from spectral.io import envi 


sixs_radiative_transfer={
      "domain": {"start": 350, "end": 2520, "step": 0.1},
      "statevector": {
        "H2OSTR": {
          "bounds": [0.5, 1.0],
          "scale": 0.01,
          "prior_sigma": 100.0,
          "prior_mean": 1.0,
          "init": 0.75
        }
      },
      "lut_grid": { 
        "H2OSTR": [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
      },
      "unknowns": {}
    }


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Representative subset")
    parser.add_argument('input_radiance',  type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('inpug_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('--h2o', action='store_true')
    parser.add_argument('--isofit_path', type=str)
    parser.add_argument('--sixs_path', type=str)
    args = parser.parse_args()

    if hasattr(args, 'isofit_path'):
        isofit_path = args.isofit_path
    else:
        isofit_path = os.getenv('ISOFIT_BASE')
    if isofit_path is None:
        raise ValueError('please define ISOFIT_BASE in your environment')

    wkdir    = args.working_directory
    rdn_path = args.input_radiance
    loc_path = args.input_loc
    obs_path = args.input_obs

    for dpath in [wkdir, 
            join(wkdir,'lut_h2o'),
            join(wkdir,'lut_full'),
            join(wkdir,'config'),
            join(wkdir,'data'),
            join(wkdir,'input'),
            join(wkdir,'output')]:
        if exists(dpath):
            os.mkdir(dpath)

    # get radiance file, wavelengths
    rdn = envi.open_image(rdn_path)
    wl = s.array([float(w) for w in rdn.metadata['wavelength']])
    fwhm = s.array([float(f) for f in rdn.metadata['fwhm']])
    
    # Convert to microns if needed
    if wl[0]>100: 
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # draw geometric information from the center of the flightline
    ref_row = int(rdn.metadata['lines'])/2, 
    ref_col = int(rdn.metadata['samples'])/2
    obs = envi.open_image(obs_path).read_pixel(ref_row, ref_col)
    loc = envi.open_image(loc_file).read_pixel(ref_row, ref_col)

    # define geometric quantities
    path_length_km = obs[0] / 1000.0
    observer_azimuth = obs[1]  # 0 to 360 clockwise from N
    observer_zenith = obs[2]  # 0 to 90 from zenith
    solar_azimuth = obs[3]  # 0 to 360 clockwise from N
    solar_zenith = obs[4]  # 0 to 90 from zenith
    OBSZEN = 180.0 - abs(obs[2])  # MODTRAN convention?
    RELAZ = obs[1] - obs[3] + 180.0
    TRUEAZ = self.RELAZ  # MODTRAN convention?

    # define geographic location
    GNDALT = loc[2]
    surface_elevation_km = loc[2] / 1000.0
    latitude = loc[1]
    longitude = loc[0]
    longitudeE = -loc[0]
    longitude < 0:
        longitude = 360.0 - longitude

    # sensor altitude
    alt = surface_elevaation_km + s.cos(observer_zenith) * path_length_km

    # 
    wl_path = join(join(wkdir, 'data'), 'wavelengths.txt')
    wl_data = s.concatenate([s.arange(len(wl)[:,newaxis], wl[:,newaxis],
                   fwhm[:,newaxis])], axis=1)
    wl_data.savetxt(wl_file, delimiter=' ')

    # make sixs configuration
    sixs_config['wavelength_file'] = wl_path
    sixs_config['earth_sun_distance_file'] = \
        join(isofit_path,'data/earth_sun_distance.txt')
    sixs_config["irradiance_file"] = \
        join(isofit_path,'data/kurudz_0.1nm.dat')
    if hasattr(args, 'sixs_path'):
        sixs_config["sixs_path"] = args.sixs_path
    else:
        sixs_config["sixs_installation"] = os.getenv('SIXS_PATH')
    if sixs_config["sixs_installation"] is None:
        raise ValueError('please define SIXS_PATH in your environment')
    sixs_config['lut_path'] = lut_full
    sixs_config['solzen'] = geom.solzen    
      "solzen": 55.20,
      "solaz": 141.72,
    sixs_config["elev":  0.0,
      "alt": 20.0,
      "viewzen": 6.28,
      "viewaz": 231.6,
      "month": 10,
      "day": 26,

      "sixs_installation":       "${SIXS_DIR}",

    lbl_file   = args.labels
    out_file   = args.output
    nchunk     = args.chunksize
    flag       = args.flag
    dtm        = {'4': s.float32, '5': s.float64}
    
    # Open input data, get dimensions
    in_img = envi.open(in_file+'.hdr', in_file)
    meta   = in_img.metadata
        
    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    img_mm = in_img.open_memmap(interleave='source', writable=False)

    lbl_img = envi.open(lbl_file+'.hdr', lbl_file)
    labels  = lbl_img.read_band(0)
    nout    = len(s.unique(labels))

    # reindex from zero to n
   #lbl     = s.sort(s.unique(labels.flat))
   #idx     = s.arange(len(lbl))
   #nout    = len(lbl)
   #for i, L in enumerate(lbl):
   #    labels[labels==L] = i

    # Iterate through image "chunks," segmenting as we go
    next_label = 1
    extracted = s.zeros(nout)>1
    out = s.zeros((nout,nb))
    counts = s.zeros((nout))

    for lstart in s.arange(0,nl,nchunk):

        del img_mm
        img_mm = in_img.open_memmap(interleave='source', writable=False)

        # Which labels will we extract? ignore zero index
        lend = min(lstart+nchunk, nl)
        active = s.unique(labels[lstart:lend,:])
        active = active[active>=1]

        # Handle labels extending outside our chunk by expanding margins 
        active_area = s.zeros(labels.shape)
        lstart_adjust, lend_adjust = lstart, lend
        for i in active:
            active_area[labels == i] = True
        active_locs = s.where(active_area)
        lstart_adjust = min(active_locs[0])
        lend_adjust = max(active_locs[0])+1

        chunk_inp = s.array(img_mm[lstart_adjust:lend_adjust,:,:])
        if meta['interleave'] == 'bil':
            chunk_inp = chunk_inp.transpose((0,2,1))
        chunk_lbl = s.array(labels[lstart_adjust:lend_adjust,:])

        for i in active:
            idx = int(i)
            out[idx,:] = 0
            locs = s.where(chunk_lbl==i)
            for row, col in zip(locs[0],locs[1]):
                out[idx,:] = out[idx,:] + s.squeeze(chunk_inp[row,col,:])
            counts[idx] = len(locs[0])

    out = s.array((out.T / counts[s.newaxis,:]).T, dtype=s.float32)
    out[s.logical_not(s.isfinite(out))] = flag
    meta["lines"] = str(nout)
    meta["bands"] = str(nb)
    meta["samples"] = '1'
    meta["interleave"]="bil"
    out_img = envi.create_image(out_file+'.hdr',  metadata=meta, 
                ext='', force=True)
    out_mm = s.memmap(out_file, dtype=dtm[meta['data type']], mode='w+',
                shape=(nout,1,nb))
    if dtm[meta['data type']] == s.float32:
        out_mm[:,0,:] = s.array(out, s.float32)
    else:
        out_mm[:,0,:] = s.array(out, s.float64)
        


if __name__ == "__main__":
    main()
