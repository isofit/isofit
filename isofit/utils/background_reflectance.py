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
from scipy.ndimage import uniform_filter, gaussian_filter
import ray

from isofit.core.common import envi_header
from isofit.inversion.inverse_simple import invert_algebraic
from isofit.core.fileio import IO, initialize_output
from isofit.core.forward import ForwardModel
from isofit.configs import configs
from isofit.core.geometry import Geometry
from isofit.utils.atm_interpolation import atm_interpolation


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
    paths,
    mean_altitude_km,
    mean_elevation_km,
    working_directory,
    smoothing_sigma,
    logging_level,
    log_file,
    nodata=-9999,
):
    """Aggregates background reflectance term from the presolve."""

    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=logging_level, filename=log_file
    )

    # NOTE / TODO: relies on presolve to be true. if it is not, then no data created and bg_rfl remains None.
    if not os.path.exists(paths.h2o_subs_path):
        return

    config = configs.create_new_config(
        glob(os.path.join(working_directory, "config", "") + "*_isofit.json")[0]
    )

    # load forward model
    fm = ForwardModel(config)
    esd = IO.load_esd()
    wl = fm.surface.wl

    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape

    # Estimate pixel size and adjacency range in km (pixel size is approx. based on loc file)
    pixel_size = approx_pixel_size(loc=loc, nodata=nodata) / 1000.0  # km
    adj_range = get_adjacency_range(
        sensor_alt_asl=mean_altitude_km,
        ground_alt_asl=mean_elevation_km,
        R_sat=1.0,
        min_range=0.2,
    )
    logging.info(
        f"For background reflectance assuming pixel size of {pixel_size*1000:.2f} m "
        f"and adjacency range of {adj_range:.2f} km."
    )

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
        outpath=paths.bgrfl_working_path,
        out_shape=(rows, cols, len(wl)),
        bands=f"{len(wl)}",
        description="Background reflectance for each pixel used in OE inversion",
    )

    # Identify water vapor state vector
    for h2oname in ["H2OSTR", "h2o"]:
        if h2oname in fm.RT.statevec_names:
            wv_idx = fm.RT.statevec_names.index(h2oname)
            break
    else:
        raise ValueError("Water vapor not found in RT names.")

    @ray.remote
    def invert_rfl(row_chunk, rdn_file, loc_file, obs_file, fm, esd, wv):
        """Evaluate algebraic based on water vapor from presolve."""
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
            wv_r = wv[row_idx, :]

            for j in range(n_cols):
                spectra = rdn_r[j, :]
                if np.isnan(spectra).all() or spectra[0] < -999:
                    continue

                x_surface, x_RT, x_instr = (
                    x_surface_init.copy(),
                    x_RT_init.copy(),
                    x_instr_init.copy(),
                )
                x_RT[wv_idx] = wv_r[j]

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

    # load 1d water vapor data and get back to per pixel value
    wv = (
        envi.open(envi_header(paths.h2o_subs_path), paths.h2o_subs_path)
        .load()[:, :, -1]
        .squeeze()
    )
    labels = (
        envi.open(envi_header(paths.lbl_working_path), paths.lbl_working_path)
        .load()
        .squeeze()
        .astype(int)
    )

    full_wv = np.zeros_like(labels, dtype=float)
    for lbl_val in np.unique(labels):
        if lbl_val < wv.size:
            full_wv[labels == lbl_val] = wv[lbl_val]
        else:
            full_wv[labels == lbl_val] = np.nan

    full_wv = gaussian_filter(full_wv, sigma=np.max(smoothing_sigma))

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
            ray.put(full_wv),
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

    del bg_rfl, heuristic_rfl, loc, fm, wv, full_wv

    return
