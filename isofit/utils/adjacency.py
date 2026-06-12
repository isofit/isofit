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
import numpy as np
from spectral.io import envi
from scipy.ndimage import uniform_filter

from isofit.core.common import envi_header
from isofit.utils.algebraic_line import algebraic_line
from isofit.core.forward import ForwardModel
from isofit.core.units import m_to_km, km_to_m
from isofit.configs import configs
from isofit.core.multistate import update_config_for_surface
from isofit.core.fileio import initialize_output
from isofit.utils import extractions, reducers


def approx_pixel_size(loc, nodata_value=-9999):
    """Average, approximate pixel size assuming planar locally (units in m)."""
    R = 6371000.0

    valid_pixels = np.logical_not(np.any(loc == nodata_value, axis=2))

    # determine if lat/lon or N/E
    sample_coords = loc[np.where(np.all(loc != nodata_value, axis=-1))]
    is_lat_lon = np.nanmax(np.abs(sample_coords[0, :2])) <= 180

    if is_lat_lon:
        lon = np.radians(loc[..., 0])
        lat = np.radians(loc[..., 1])

        dx = R * np.sqrt(
            (lat[:, 1:] - lat[:, :-1]) ** 2
            + (np.cos(lat[:, :-1]) * (lon[:, 1:] - lon[:, :-1])) ** 2
        )
        dy = R * np.sqrt(
            (lat[1:, :] - lat[:-1, :]) ** 2
            + (np.cos(lat[:-1, :]) * (lon[1:, :] - lon[:-1, :])) ** 2
        )
    else:
        dx = np.sqrt(np.sum(np.diff(loc[..., :2], axis=1) ** 2, axis=-1))
        dy = np.sqrt(np.sum(np.diff(loc[..., :2], axis=0) ** 2, axis=-1))

    pix_size = 0.5 * (
        np.nanmean(dx[valid_pixels[:, 1:]]) + np.nanmean(dy[valid_pixels[1:, :]])
    )

    return pix_size


def get_adjacency_range(sensor_alt_asl, ground_alt_asl, R_sat=1.0, min_range=0.2):
    """Estimate adjacency range based on Richter et al. (2011)."""
    # for satellite case
    if sensor_alt_asl > 95.0:
        r = R_sat
    # for airborne case
    else:
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
    smoothing_sigma,
    n_cores,
    logging_level,
    log_file,
    chunksize,
    use_slic_rfls=True,
    use_superpixels=True,
    nodata_value=-9999,
):
    """Aggregates background reflectance term from the presolve."""
    R_SAT = 1.0  # assumed adjacency range of satellite [km]
    MIN_RANGE = 0.2  # assumed min range [km]

    conf = configs.create_new_config(paths.h2o_config_path)

    # If using multisurface, we grab the first surface from the list
    # TODO this is the only instance of this, but perhaps a config method
    # could help catch this for other future cases.
    if not conf.forward_model.surface.surface_category:
        conf = update_config_for_surface(
            conf, list(conf.forward_model.surface.Surfaces.keys())[0]
        )

    fm = ForwardModel(conf)
    wl = fm.surface.wl

    # Estimate pixel size and adjacency range in km (pixel size is approx. based on loc file)
    loc = envi.open(envi_header(input_loc), input_loc).open_memmap()
    rows, cols, _ = loc.shape
    pixel_size = m_to_km(approx_pixel_size(loc=loc, nodata_value=nodata_value))
    adj_range = get_adjacency_range(
        sensor_alt_asl=mean_altitude_km,
        ground_alt_asl=mean_elevation_km,
        R_sat=R_SAT,
        min_range=MIN_RANGE,
    )
    logging.info(
        f"For background reflectance assuming pixel size of {km_to_m(pixel_size):.2f} m "
        f"and adjacency range of {adj_range:.2f} km."
    )

    # Calls algebraic line using presolve config
    if not use_slic_rfls:
        algebraic_line(
            rdn_file=input_radiance,
            loc_file=input_loc,
            obs_file=input_obs,
            isofit_config=paths.h2o_config_path,
            segmentation_file=paths.lbl_working_path,
            isofit_dir=None,
            atm_file=paths.atm_presolve,
            atm_sigma=smoothing_sigma,
            output_rfl_file=paths.bgrfl_working_path,
            n_cores=n_cores,
            logging_level=logging_level,
            log_file=log_file,
        )
        bg_rfl = envi.open(
            envi_header(paths.bgrfl_working_path), paths.bgrfl_working_path
        ).open_memmap(interleave="bip", writable=True)

    # else, falls back to weighting the superpixel rfl directly
    else:
        rfl_output = initialize_output(
            output_metadata={
                "data type": 4,
                "file type": "ENVI Standard",
                "byte order": 0,
                "no data value": nodata_value,
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
        labels = (
            envi.open(envi_header(paths.lbl_working_path), paths.lbl_working_path)
            .load()
            .squeeze()
            .astype(int)
        )
        subs_state = envi.open(
            envi_header(paths.h2o_subs_path), paths.h2o_subs_path
        ).load()
        bg_rfl = envi.open(envi_header(rfl_output), rfl_output).open_memmap(
            interleave="bip", writable=True
        )
        rfl_samples = subs_state[:, 0, fm.idx_surf_rfl]
        bg_rfl[:, :, :] = np.squeeze(rfl_samples[labels, :])

    # For now, this applies a uniform window average based on adjacency range.
    kernel_diameter = int(np.ceil(2 * np.max(adj_range) / pixel_size + 1))
    bg_rfl[:, :, :] = uniform_filter(
        bg_rfl, size=(kernel_diameter, kernel_diameter, 1), mode="nearest"
    )

    del bg_rfl, loc

    # if using superpixels, we aggregate the bg rfl before OE
    if use_superpixels:
        extractions(
            inputfile=paths.bgrfl_working_path,
            labels=paths.lbl_working_path,
            output=paths.bgrfl_subs_path,
            chunksize=chunksize,
            flag=nodata_value,
            reducer=reducers.band_mean,
            n_cores=n_cores,
            loglevel=logging_level,
            logfile=log_file,
        )

    return
