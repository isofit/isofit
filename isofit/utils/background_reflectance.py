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
import logging
from glob import glob
import os
import numpy as np
from spectral.io import envi
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import ray

from isofit.core.common import envi_header, load_wavelen
from isofit.inversion.inverse_simple import invert_simple, invert_algebraic
from isofit.core.fileio import IO, initialize_output
from isofit.core.forward import ForwardModel
from isofit.configs import configs
from isofit.core.geometry import Geometry

# TODO: delete this dataset at the end? its same shape as rdn


def approx_pixel_size(loc):
    """Average pixel size from nearby pixel assume planar locally."""
    R = 6371000.0

    lat = np.radians(loc[..., 0])
    lon = np.radians(loc[..., 1])

    dx = R * np.sqrt(
        (lat[:, 1:] - lat[:, :-1]) ** 2
        + (np.cos(lat[:, :-1]) * (lon[:, 1:] - lon[:, :-1])) ** 2
    )
    dy = R * np.sqrt(
        (lat[1:, :] - lat[:-1, :]) ** 2
        + (np.cos(lat[:-1, :]) * (lon[1:, :] - lon[:-1, :])) ** 2
    )

    valid_pixels = np.logical_not(np.any(loc == -9999, axis=2))
    pix_size = 0.5 * (
        np.nanmean(dx[valid_pixels[:, 1:]]) + np.nanmean(dy[valid_pixels[1:, :]])
    )

    return pix_size


def get_adjacency_range(sensor_alt_asl, ground_alt_asl, R_sat=1.0, min_range=0.2):
    """Estimate adjacency range based on Richter et al. (2011)."""
    if sensor_alt_asl > 95.0:
        r = R_sat
    else:  # for airborne case
        z_rel = sensor_alt_asl - ground_alt_asl
        R1 = R_sat * (1 - np.exp(-z_rel / 8))
        R2 = R_sat * (1 - np.exp(-ground_alt_asl / 8))
        r = R1 - R2

    if r < min_range:
        r = min_range

    return r


def background_reflectance(
    input_radiance,
    input_loc,
    input_obs,
    mean_altitude_km,
    mean_elevation_km,
    working_directory,
    bgrfl_path,
    logging_level,
    log_file,
):
    """Produces background reflectance for each pixel by running heuristic solutions on chunks,
    and then solves the rest of the pixels with algebraic solution. Limited window of spectra are
    retained and then smoothed using savgol filter. Justification is due to the relatively small effect
    on the returned radiance from background spectra."""

    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )

    config = configs.create_new_config(
        glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    )
    wl_init, _ = load_wavelen(config.forward_model.instrument.wavelength_file)
    
    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape

    # regions to keep from heuristic, remaining are interpolated and smoothed.
    BKG_WINDOWS = [
        [350.0, 900.0],
        [1000.0, 1050.0],
        [1150.0, 1250.0],
        [1600.0, 1700.0],
        [2100.0, 2300.0],
    ]

    # Estimate pixel size and adjacency range in km (pixel size is approx. based on loc file)
    pixel_size = approx_pixel_size(loc) / 1000.0  # km
    adj_range = get_adjacency_range(
        sensor_alt_asl=mean_altitude_km,
        ground_alt_asl=mean_elevation_km,
        R_sat=1.0,
        min_range=0.2,
    )
    logging.info(
        f"For background reflectance assuming pixel size of {np.round(pixel_size*1000,2)} m and adjacency range of {np.round(adj_range,2)} km."
    )

    # Output metadata
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
        rfl_chunk = np.zeros(
            (len(row_chunk), len(cols), len(fm.surface.idx_lamb)), dtype=np.float32
        )

        # Pick median spectra for water vapor inversion (inverse simple) of chunk.
        center_r = row_chunk[len(row_chunk) // 2]
        center_c = len(cols) // 2
        spectra_center = np.nanmedian(
            rdn[np.ix_(row_chunk, cols)].reshape(-1, rdn.shape[-1]), axis=0
        )
        geom_center = Geometry(
            obs=obs[center_r, center_c, :],
            loc=loc[center_r, center_c, :],
            esd=esd,
            svf=1.0,
        )
        x_center = invert_simple(fm, spectra_center, geom_center)
        _, _, x_instr = fm.unpack(fm.init.copy())

        # Use this to do invert_algebraic across whole chunk
        for r, rr in enumerate(row_chunk):
            for j, c in enumerate(cols):
                spectra = rdn[rr, c, :]
                if spectra[0] < -999 or np.isnan(spectra[0]):
                    rfl_chunk[r, j, :] = np.nan
                    continue

                geom = Geometry(obs=obs[rr, c, :], loc=loc[rr, c, :], esd=esd, svf=1.0)

                rfl_est, _ = invert_algebraic(
                    fm.surface,
                    fm.RT,
                    fm.instrument,
                    x_center[fm.idx_surface],
                    x_center[fm.idx_RT],
                    x_instr,
                    spectra,
                    geom,
                )

                rfl_chunk[r, j, :] = rfl_est

        return row_chunk, cols, rfl_chunk

    # Run heuristic solve for a limited set of chunks/tiles
    chunk_size = 50
    row_chunks = [
        list(range(r, min(r + chunk_size, rows))) for r in range(0, rows, chunk_size)
    ]
    col_chunks = [
        list(range(c, min(c + chunk_size, cols))) for c in range(0, cols, chunk_size)
    ]

    fm_ray = ray.put(fm)
    esd_ray = ray.put(esd)

    futures = [
        invert_chunk.remote(
            r_chunk, c_chunk, input_radiance, input_loc, input_obs, fm_ray, esd_ray
        )
        for r_chunk in row_chunks
        for c_chunk in col_chunks
    ]

    heuristic_rfl = np.zeros((rows, cols, len(fm.surface.idx_lamb)), dtype=np.float32)
    for row_chunk, col_chunk, rfl_chunk in ray.get(futures):
        for i, r in enumerate(row_chunk):
            for j, c in enumerate(col_chunk):
                heuristic_rfl[r, c, :] = rfl_chunk[i, j, :]

    # Fill NaNs with mean spectrum
    mean_spectrum = np.nanmean(heuristic_rfl, axis=(0, 1))
    heuristic_rfl = np.where(
        np.isnan(heuristic_rfl), mean_spectrum[np.newaxis, np.newaxis, :], heuristic_rfl
    )

    # Average over adjacency
    # TODO: for now, this is just a simple average. To return to this in the future.
    kernel_radius = int(np.ceil(np.max(adj_range) / pixel_size))
    bg_rfl = envi.open(envi_header(rfl_output), rfl_output).open_memmap(
        interleave="bip", writable=True
    )
    bg_rfl[:, :, :] = uniform_filter(
        heuristic_rfl, size=(kernel_radius, kernel_radius, 1), mode="nearest"
    )

    # Mask and interp over BKG_WINDOWS
    keep = np.zeros_like(wl_init, dtype=bool)
    for lo, hi in BKG_WINDOWS:
        keep |= (wl_init >= lo) & (wl_init <= hi)

    bg_rfl[:, :, :] = interp1d(
        wl_init[keep],
        bg_rfl[:, :, keep],
        axis=2,
        kind="linear",
        fill_value="extrapolate",
    )(wl_init)

    # Apply light SG filter to smooth any bad wv guess
    bg_rfl[:, :, :] = savgol_filter(bg_rfl, window_length=5, polyorder=1, axis=2)

    del bg_rfl
    del heuristic_rfl
    del loc
    del fm

    return
