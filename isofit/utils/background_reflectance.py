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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#
from glob import glob
import os
import numpy as np
from spectral.io import envi
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
import ray

from isofit.core.common import envi_header, load_wavelen
from isofit.inversion.inverse_simple import invert_simple, invert_algebraic
from isofit.core.fileio import IO, initialize_output
from isofit.core.forward import ForwardModel
from isofit.configs import configs
from isofit.core.geometry import Geometry

# TODO: delete this dataset at the end? its same shape as rdn

# made faster by just doing inverse-simple per chunk. spectra look okay I think for bkg rfl.


def approx_pixel_size(loc):
    """Approximate pixel size using haversine formula."""
    R = 6371000.0  # Earth radius in meters

    valid = np.logical_not(np.any(loc == -9999, axis=2))
    
    lat_mean = np.nanmean(loc[valid,0].flatten())
    lat_min = np.nanmin(loc[valid,0].flatten())
    lat_max = np.nanmax(loc[valid,0].flatten())

    lon_mean = np.nanmean(loc[valid,1].flatten())
    lon_min = np.nanmin(loc[valid,1].flatten())
    lon_max = np.nanmax(loc[valid,1].flatten())

    def haversine(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        a = np.clip(a, 0.0, 1.0)
        return 2 * R * np.arcsin(np.sqrt(a))

    # pixel size varying lon
    dx_total = haversine(np.radians(lat_mean), np.radians(lon_min),
                         np.radians(lat_mean), np.radians(lon_max))
    avg_dx = dx_total / max(loc.shape[1]-1,1)

    # pixel size varying lat
    dy_total = haversine(np.radians(lat_min), np.radians(lon_mean),
                         np.radians(lat_max), np.radians(lon_mean))
    avg_dy = dy_total / max(loc.shape[0]-1,1)

    return (avg_dx + avg_dy) / 2




def get_adjacency_range(sensor_alt_asl, ground_alt_asl, Rsat=1.0, min_range=0.2):
    """Estimate adjacency range based on Richter et al 2011"""

    # for spaceborne case
    if sensor_alt_asl>95.0:
        return Rsat
    
    else:
        # for airborne case
        z_rel = sensor_alt_asl - ground_alt_asl
        R1 = Rsat * (1 - np.exp(-z_rel / 8))
        R2 = Rsat * (1 - np.exp(-ground_alt_asl / 8))
        r = R1 - R2
        
        # assume min adj range
        if r<min_range:
            r=min_range

        return r


def background_reflectance(input_radiance, input_loc, input_obs, mean_altitude_km, mean_elevation_km, working_directory, bgrfl_path):
    """Produces background reflectance for each pixel."""

    # Load config
    file = glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    config = configs.create_new_config(file)

    # Open loc
    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape
    range_cols = range(cols)

    # Estimate pixel size and adjacency range (giving ~68.6 m for EMIT tile right now)
    pixel_size = approx_pixel_size(loc) / 1000.0  # km
    adj_range = get_adjacency_range(mean_altitude_km, mean_elevation_km, Rsat=1.0, min_range=0.2)

    # Output metadata
    wl_init, _ = load_wavelen(config.forward_model.instrument.wavelength_file)
    output_metadata = {
        "data type": 4,
        "file type": "ENVI Standard",
        "byte order": 0,
        "no data value": -9999,
        "wavelength units": "Nanometers",
        "wavelength": wl_init,
        "lines": rows,
        "samples": cols,
        "interleave": "bip",
    }
    rfl_output = initialize_output(
        output_metadata,
        bgrfl_path,
        (rows, cols, len(wl_init)),
        bands=f"{len(wl_init)}",
        description="Background reflectance for each pixel used in OE inversion",
    )

    # Set up forward model and esd
    fm = ForwardModel(config)
    esd = IO.load_esd()

    @ray.remote
    def invert_chunk(row_chunk, cols, rdn_file, loc_file, obs_file, fm, esd):
        # Open memmap for chunk
        rdn = np.array(envi.open(envi_header(rdn_file), rdn_file).open_memmap())
        loc = np.array(envi.open(envi_header(loc_file), loc_file).open_memmap())
        obs = np.array(envi.open(envi_header(obs_file), obs_file).open_memmap())
        rfl_chunk = np.zeros((len(row_chunk), len(cols), len(fm.surface.idx_lamb)), dtype=np.float32)

        # Pick center of chunk for water vapor inversion (inverse simple)
        center_r = row_chunk[len(row_chunk)//2]
        center_c = len(cols)//2
        spectra_center = rdn[center_r, center_c, :]
        geom_center = Geometry(obs=obs[center_r, center_c, :],
                            loc=loc[center_r, center_c, :],
                            esd=esd,
                            svf=1.0)
        x_center = invert_simple(fm, spectra_center, geom_center)
        _, _, x_instr = fm.unpack(fm.init.copy())

        # Use this to do invert_algebraic across whole chunk 
        for r, rr in enumerate(row_chunk): # NOte, rr is big r is chunk
            for c in cols:
                spectra = rdn[rr, c, :]
                if spectra[0] < -999 or np.isnan(spectra[0]):
                    rfl_chunk[r, c, :] = np.nan
                    continue

                geom = Geometry(obs=obs[rr, c, :],
                                loc=loc[rr, c, :],
                                esd=esd,
                                svf=1.0)
                
                rfl_est, _ = invert_algebraic(fm.surface,
                                                fm.RT,
                                                fm.instrument,
                                                x_center[fm.idx_surface],
                                                x_center[fm.idx_RT],
                                                x_instr,
                                                spectra,
                                                geom)
                rfl_chunk[r, c, :] = rfl_est

        return row_chunk, rfl_chunk


    # Run heuristic solve (light)
    chunk_size = 50
    row_chunks = [list(range(i, min(i + chunk_size, rows))) for i in range(0, rows, chunk_size)]

    fm_ray = ray.put(fm)
    esd_ray = ray.put(esd)

    futures = [
        invert_chunk.remote(chunk, range_cols, input_radiance, input_loc, input_obs, fm_ray, esd_ray)
        for chunk in row_chunks
    ]

    heuristic_rfl = np.zeros((rows, cols, len(fm.surface.idx_lamb)), dtype=np.float32)
    for row_chunk, rfl_chunk in ray.get(futures):
        for r, rr in enumerate(row_chunk):
            heuristic_rfl[rr, :, :] = rfl_chunk[r, :, :]

    # Fill NaNs with mean spectrum
    mean_spectrum = np.nanmean(heuristic_rfl, axis=(0, 1))
    heuristic_rfl = np.where(np.isnan(heuristic_rfl),
                            mean_spectrum[np.newaxis, np.newaxis, :],
                            heuristic_rfl)

    # Average over adjacency
    # TODO: for now, this is just a simple average. To return to this in the future.
    kernel_radius = int(np.ceil(np.max(adj_range) / pixel_size))
    bg_rfl = envi.open(envi_header(rfl_output), rfl_output).open_memmap(interleave="bip", writable=True)
    bg_rfl[:, :, :] = uniform_filter(heuristic_rfl, size=(kernel_radius, kernel_radius, 1), mode='nearest')

    # Apply SG filter to smooth any bad wv guess
    bg_rfl[:, :, :] = savgol_filter(bg_rfl, window_length=15, polyorder=2, axis=2)

    del bg_rfl
    del heuristic_rfl
    del loc

    return
