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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import ray

from isofit.core.common import envi_header
from isofit.inversion.inverse_simple import invert_simple, invert_algebraic
from isofit.core.fileio import IO, initialize_output
from isofit.core.forward import ForwardModel
from isofit.configs import configs
from isofit.core.geometry import Geometry


def approx_pixel_size(loc, nodata=-9999):
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

    valid_pixels = np.logical_not(np.any(loc == nodata, axis=2))
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

    if r > R_sat:
        r = R_sat

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
    nodata=-9999,
):
    """Produces background reflectance for each pixel by running heuristic solutions on chunks,
    and then solves the rest of the pixels with algebraic solution. Limited window of spectra are
    retained to avoid shallow and deep water features. Justification is due to the relatively small effect
    on the returned radiance from background spectra."""

    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )

    config = configs.create_new_config(
        glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    )

    # load forward model
    fm = ForwardModel(config)
    esd = IO.load_esd()
    wl = fm.surface.wl

    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape

    # regions to keep from heuristic for bg_rfl, avoiding shallow and deep water features
    BKG_WINDOWS = [
        [480.0, 690.0],
        [740.0, 800.0],
        [850.0, 885.0],
        [1000.0, 1050.0],
        [1200.0, 1270.0],
        [1550.0, 1700.0],
        [2050.0, 2300.0],
    ]

    # Estimate pixel size and adjacency range in km (pixel size is approx. based on loc file)
    pixel_size = approx_pixel_size(loc=loc, nodata=nodata) / 1000.0  # km
    adj_range = get_adjacency_range(
        sensor_alt_asl=mean_altitude_km,
        ground_alt_asl=mean_elevation_km,
        R_sat=1.0,
        min_range=0.2,
    )
    logging.info(
        f"For background reflectance assuming pixel size of {np.round(pixel_size*1000,2)} m and adjacency range of {np.round(adj_range,2)} km."
    )

    # Identify water vapor state vector
    for h2oname in ["H2OSTR", "h2o"]:
        if h2oname in fm.RT.statevec_names:
            wv_idx = fm.RT.statevec_names.index(h2oname)
            break
    else:
        raise ValueError("Water vapor not found in RT names.")

    # Create bg_rfl output
    rfl_output = initialize_output(
        output_metadata={
            "data type": 4,
            "file type": "ENVI Standard",
            "byte order": 0,
            "no data value": nodata,
            "wavelength units": "Nanometers",
            "wavelength": wl,
            "lines": rows,
            "samples": cols,
            "interleave": "bip",
        },
        outpath=bgrfl_path,
        out_shape=(rows, cols, len(wl)),
        bands=f"{len(wl)}",
        description="Background reflectance for each pixel used in OE inversion",
    )

    @ray.remote
    def invert_wv(row_chunk, col_chunk, rdn_file, loc_file, obs_file, fm, esd):
        """Evaluated water vapor at center pixel of chunk."""
        rdn = envi.open(envi_header(rdn_file), rdn_file).open_memmap()
        loc = envi.open(envi_header(loc_file), loc_file).open_memmap()
        obs = envi.open(envi_header(obs_file), obs_file).open_memmap()

        spectra_chunk = rdn[np.ix_(row_chunk, col_chunk)].reshape(-1, rdn.shape[-1])
        spectra_chunk = np.where(spectra_chunk == nodata, np.nan, spectra_chunk)

        # Pick median spectra for water vapor inversion (inverse simple) of chunk.
        spectra_median = np.nanmedian(spectra_chunk, axis=0)
        center_r = row_chunk[len(row_chunk) // 2]
        center_c = col_chunk[len(col_chunk) // 2]

        x_center = invert_simple(
            fm,
            spectra_median,
            Geometry(
                obs=obs[center_r, center_c, :],
                loc=loc[center_r, center_c, :],
                esd=esd,
                svf=1.0,
                bg_rfl=None,
            ),
        )
        wv_value = x_center[fm.idx_RT][wv_idx]

        # check bounds on solution, if at edge return NaN
        if (
            wv_value <= fm.bounds[0][fm.idx_RT][wv_idx] + 0.1
            or wv_value >= fm.bounds[1][fm.idx_RT][wv_idx] - 0.1
        ):
            wv_value = np.nan

        return row_chunk, col_chunk, wv_value

    # Run heuristic solve for a limited set of chunks/tiles
    # 100 is ~ 6 km chunks for EMIT granules
    wv_chunk_size = 100
    row_chunks = [
        list(range(r, min(r + wv_chunk_size, rows)))
        for r in range(0, rows, wv_chunk_size)
    ]
    col_chunks = [
        list(range(c, min(c + wv_chunk_size, cols)))
        for c in range(0, cols, wv_chunk_size)
    ]

    futures = [
        invert_wv.remote(
            r_chunk,
            c_chunk,
            input_radiance,
            input_loc,
            input_obs,
            ray.put(fm),
            ray.put(esd),
        )
        for r_chunk in row_chunks
        for c_chunk in col_chunks
    ]

    # collect water vapor from inverse simple
    wv = np.full((rows, cols), np.nan, dtype=np.float32)
    for r, c, wv_value in ray.get(futures):
        wv[np.ix_(r, c)] = wv_value

    # impute missing data with mean, and apply filter
    wv[np.isnan(wv)] = np.nanmean(wv)
    wv = gaussian_filter(wv, sigma=5.0)

    @ray.remote
    def invert_rfl(row_chunk, rdn_file, loc_file, obs_file, fm, esd, wv):
        """Evaluate algebraic based on coarse water vapor solved from inverse_simple."""
        rdn = envi.open(envi_header(rdn_file), rdn_file).open_memmap()
        loc = envi.open(envi_header(loc_file), loc_file).open_memmap()
        obs = envi.open(envi_header(obs_file), obs_file).open_memmap()

        n_cols, n_bands = rdn.shape[1], rdn.shape[2]
        rfl_chunk = np.full((len(row_chunk), n_cols, n_bands), np.nan, dtype=np.float32)

        x_surface_init, x_RT_init, x_instr_init = fm.unpack(fm.init.copy())

        for i, row_idx in enumerate(row_chunk):
            rdn_r = np.where(rdn[row_idx, :, :] == nodata, np.nan, rdn[row_idx, :, :])
            loc_r = loc[row_idx, :, :]
            obs_r = obs[row_idx, :, :]

            for j in range(n_cols):
                spectra = rdn_r[j, :]
                if np.isnan(spectra).all() or spectra[0] < -999:
                    continue

                x_surface, x_RT, x_instr = (
                    x_surface_init.copy(),
                    x_RT_init.copy(),
                    x_instr_init.copy(),
                )
                x_RT[wv_idx] = wv[row_idx, j]

                try:
                    rfl_est, _ = invert_algebraic(
                        fm.surface,
                        fm.RT,
                        fm.instrument,
                        x_surface,
                        x_RT,
                        x_instr,
                        spectra,
                        geom=Geometry(
                            obs=obs_r[j, :],
                            loc=loc_r[j, :],
                            esd=esd,
                            svf=1.0,
                            bg_rfl=None,
                        ),
                    )
                    rfl_chunk[i, j, :] = rfl_est
                except Exception:
                    rfl_chunk[i, j, :] = np.nan

        return row_chunk, rfl_chunk

    # Run rfl inversions for every pixel
    heuristic_rfl = np.full((rows, cols, len(wl)), np.nan, dtype=np.float32)
    chunk_size = 50  # tuneable, only changes speed
    row_chunks = [
        range(r, min(r + chunk_size, rows)) for r in range(0, rows, chunk_size)
    ]

    futures_rfl = [
        invert_rfl.remote(
            r_chunk,
            input_radiance,
            input_loc,
            input_obs,
            ray.put(fm),
            ray.put(esd),
            ray.put(wv),
        )
        for r_chunk in row_chunks
    ]

    for row_chunk, rfl_chunk in ray.get(futures_rfl):
        heuristic_rfl[row_chunk, :, :] = rfl_chunk

    # Fill NaNs with mean spectrum
    mean_spectrum = np.nanmean(heuristic_rfl, axis=(0, 1))
    heuristic_rfl = np.where(
        np.isnan(heuristic_rfl), mean_spectrum[np.newaxis, np.newaxis, :], heuristic_rfl
    )

    # TODO: this could also be weighted by transmittance terms.
    # For now, this applies a uniform window average based on adjacency range.
    kernel_radius = int(np.ceil(np.max(adj_range) / pixel_size))
    bg_rfl = envi.open(envi_header(rfl_output), rfl_output).open_memmap(
        interleave="bip", writable=True
    )
    bg_rfl[:, :, :] = uniform_filter(
        heuristic_rfl, size=(kernel_radius, kernel_radius, 1), mode="nearest"
    )

    # Retaining only bands not in shallow or deep water features
    idx = np.zeros_like(wl, dtype=bool)
    for lo, hi in BKG_WINDOWS:
        idx |= (wl >= lo) & (wl <= hi)

    bg_rfl[:, :, :] = interp1d(
        wl[idx], bg_rfl[:, :, idx], axis=2, kind="linear", fill_value="extrapolate"
    )(wl)

    del bg_rfl, heuristic_rfl, loc, fm

    return
