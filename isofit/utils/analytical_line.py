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

import logging
import multiprocessing
import os
import time
from collections import OrderedDict
from glob import glob

import click
import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.configs import configs
from isofit.core.common import (
    envi_header,
    load_spectrum,
    load_wavelen,
    match_statevector,
)
from isofit.core.fileio import IO, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse_simple import invert_analytical
from isofit.utils.atm_interpolation import atm_interpolation
from isofit.utils.multistate import construct_full_state, index_image_by_class_and_sub


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: configs.Config,
        pixel_index: list,
        full_statevector: list,
        full_idx_surface: np.array,
        full_idx_RT: np.array,
        full_idx_surf_rfl: np.array,
        input_files: dict,
        output_files: dict,
        loglevel: str,
        logfile: str,
        subs_state_file: str = None,
    ):
        """
        Worker class to help run a subset of spectra.

        Args:
            fm: isofit forward_model
            loglevel: output logging level
            logfile: output logging file
        """
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
        self.config = config
        self.pixel_index = pixel_index

        self.esd = IO.load_esd(IO.earth_sun_distance_path)

        self.full_statevector = full_statevector
        self.full_idx_surface = full_idx_surface
        self.full_idx_RT = full_idx_RT
        self.full_idx_surf_rfl = full_idx_surf_rfl

        self.winidx = retrieve_winidx(self.config)

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = 500

        self.rdn_file = input_files["rdn_file"]
        self.loc_file = input_files["loc_file"]
        self.obs_file = input_files["obs_file"]
        self.RT_state_file = input_files["atm_file"]

        self.rfl_output = output_files["rfl_output"]
        self.unc_output = output_files["unc_output"]

        if config.input.radiometry_correction_file is not None:
            self.radiance_correction, wl = load_spectrum(
                config.input.radiometry_correction_file
            )
        else:
            self.radiance_correction = None

        # Open files at the worker level
        self.rdn = envi.open(envi_header(self.rdn_file)).open_memmap(interleave="bip")
        self.rdns = self.rdn.shape

        self.loc = envi.open(envi_header(self.loc_file)).open_memmap(interleave="bip")

        self.obs = envi.open(envi_header(self.obs_file)).open_memmap(interleave="bip")

        self.rt_state = envi.open(envi_header(self.RT_state_file)).open_memmap(
            interleave="bip"
        )

    def run_chunks(self, startstop: tuple, fill_value: float = -9999.0) -> None:
        """
        TODO: Description
        """
        # Unpack argument
        start_line, stop_line = startstop

        # Set up outputs
        output_state = (
            np.zeros(
                (
                    stop_line - start_line,
                    self.rt_state.shape[1],
                    len(self.full_idx_surface),
                )
            )
            + fill_value
        )
        output_state_unc = (
            np.zeros(
                (
                    stop_line - start_line,
                    self.rt_state.shape[1],
                    len(self.full_idx_surface),
                )
            )
            + fill_value
        )

        # Index chunk
        state_indexes = group_pixels_by_class(
            start_line, stop_line, self.rdns[1], self.pixel_index
        )
        for class_i, class_idx_pairs in enumerate(state_indexes):
            fm = ForwardModel(self.config, f"{class_i}")

            # Find bounds from the fm
            for r, c in class_idx_pairs:
                meas = self.rdn[r, c, :]

                if self.radiance_correction is not None:
                    meas *= self.radiance_correction

                if np.all(meas < 0):
                    continue

                x_RT = self.rt_state[
                    r, c, self.full_idx_RT - len(self.full_idx_surface)
                ]
                geom = Geometry(
                    obs=self.obs[r, c, :], loc=self.loc[r, c, :], esd=self.esd
                )

                states, unc = invert_analytical(
                    fm,
                    self.winidx,
                    meas,
                    geom,
                    x_RT,
                    1,
                    self.hash_table,
                    self.hash_size,
                )

                state_est = states[-1]
                full_state_est = match_statevector(
                    state_est, self.full_statevector, fm.statevec
                )
                output_state[r - start_line, c, :] = full_state_est[
                    self.full_idx_surface
                ]

                full_unc_est = match_statevector(
                    unc, self.full_statevector, fm.statevec
                )
                output_state_unc[r - start_line, c, :] = unc[self.full_idx_surface]

        # Only apply rfl check. Bounds vary between glint and rfl terms
        output_state = output_state[..., self.full_idx_surface]

        rfl_bounds = (np.min(fm.bounds, axis=0)[0], np.max(fm.bounds, axis=0)[1])
        logging.debug(
            "Reflectance output will be bounded to the surface" f"bounds: {rfl_bounds}"
        )

        mask = np.logical_and.reduce(
            [
                output_state < rfl_bounds[0],
                output_state > rfl_bounds[1],
                output_state != -9999,
                output_state != -0.01,
            ]
        )
        output_state[mask] = 0

        # Output surface rfl
        save_output_state = np.swapaxes(output_state, 1, 2)
        write_bil_chunk(
            save_output_state,
            self.rfl_output,
            start_line,
            (self.rdn.shape[0], self.rdn.shape[1], len(self.full_idx_surface)),
        )

        # Save surface state uncertainty
        save_output_state_unc = np.swapaxes(output_state_unc, 1, 2)
        write_bil_chunk(
            save_output_state_unc.T,
            self.unc_output,
            start_line,
            (self.rdn.shape[0], self.rdn.shape[1], len(self.full_idx_surface)),
        )


def retrieve_winidx(config):
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)
    windows = config.implementation.inversion.windows

    winidx = np.array((), dtype=int)
    for lo, hi in windows:
        idx = np.where(np.logical_and(wl_init > lo, wl_init < hi))[0]
        winidx = np.concatenate((winidx, idx), axis=0)

    return winidx


def construct_output(output_metadata, outpath, buffer_size=100, **kwargs):
    """
    Construct output file by updating metadata and creating object
    """
    for key, value in kwargs.items():
        output_metadata[key] = value
    if "emit pge input files" in list(output_metadata.keys()):
        del output_metadata["emit pge input files"]

    out_file = envi.create_image(
        envi_header(outpath), ext="", metadata=output_metadata, force=True
    )
    del out_file

    return outpath


def group_pixels_by_class(start_line, stop_line, n_cols, pixel_index):
    # Form the row-column pairs (pixels to run)
    index_pairs = np.vstack(
        [
            x.flatten(order="f")
            for x in np.meshgrid(range(start_line, stop_line), range(n_cols))
        ]
    ).T

    if not len(pixel_index):
        return [index_pairs]

    index_pairs_class = []
    for i, class_row_col in enumerate(pixel_index):
        if not len(class_row_col):
            continue

        class_row_col = np.delete(np.array(class_row_col), -1, axis=1)
        class_row_col = class_row_col[class_row_col[:, 0] < stop_line]
        class_row_col = class_row_col[class_row_col[:, 0] >= start_line]
        index_pairs_class.append(class_row_col)

    return index_pairs_class


def analytical_line(
    rdn_file: str,
    loc_file: str,
    obs_file: str,
    isofit_dir: str,
    isofit_config: str = None,
    segmentation_file: str = None,
    n_atm_neighbors: list = [20],
    n_cores: int = -1,
    smoothing_sigma: list = [2],
    output_rfl_file: str = None,
    output_unc_file: str = None,
    atm_file: str = None,
    loglevel: str = "INFO",
    logfile: str = None,
) -> None:
    """
    TODO: Description
    """

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    # Config handling
    if isofit_config is None:
        file = glob(os.path.join(isofit_dir, "config", "") + "*_isofit.json")[0]
    else:
        file = isofit_config

    config = configs.create_new_config(file)
    config.forward_model.instrument.integrations = 1

    # Set up input file paths
    subs_state_file = config.output.estimated_state_file
    subs_loc_file = config.input.loc_file

    # Rename files
    lbl_file = (
        segmentation_file
        if segmentation_file
        else (subs_state_file.replace("_subs_state", "_lbl"))
    )
    analytical_rfl_path = (
        output_rfl_file
        if output_rfl_file
        else (subs_state_file.replace("_subs_state", "_rfl_analytical"))
    )
    analytical_state_unc_path = (
        output_unc_file
        if output_unc_file
        else (subs_state_file.replace("_subs_state", "_state_analytical_uncert"))
    )
    atm_file = (
        atm_file
        if atm_file
        else (subs_state_file.replace("_subs_state", "_atm_interp"))
    )

    # Set up the multi-state pixel map by sub
    pixel_index = index_image_by_class_and_sub(config, lbl_file)

    # Initialize fm (if no lut, will create here)
    for i, indexes in enumerate(pixel_index):
        if not len(indexes):
            continue
        fm = ForwardModel(config, f"{i}")

    (
        full_statevector,
        full_idx_surface,
        full_idx_surf_rfl,
        full_idx_RT,
    ) = construct_full_state(config)

    # Perform the atmospheric interpolation
    if os.path.isfile(atm_file) is False:
        # This should match the necesary state elements based on the name
        atm_interpolation(
            reference_state_file=subs_state_file,
            reference_locations_file=subs_loc_file,
            input_locations_file=loc_file,
            segmentation_file=lbl_file,
            output_atm_file=atm_file,
            atm_band_names=[full_statevector[i] for i in full_idx_RT],
            nneighbors=n_atm_neighbors,
            gaussian_smoothing_sigma=smoothing_sigma,
            n_cores=n_cores,
        )

    # Get output shape
    rdn_ds = envi.open(envi_header(rdn_file))
    rdns = rdn_ds.shape
    output_metadata = rdn_ds.metadata
    del rdn_ds

    # Find the winidx
    winidx = retrieve_winidx(config)

    # Get string representation of bad band list
    outside_ret_windows = np.zeros(len(full_idx_surf_rfl), dtype=int)
    outside_ret_windows[winidx] = 1

    # Construct surf rfl output
    bbl = "{" + ",".join([f"{x}" for x in outside_ret_windows]) + "}"
    rfl_output = construct_output(
        output_metadata,
        analytical_rfl_path,
        bbl=bbl,
        interleave="bil",
        bands=f"{len(full_idx_surf_rfl)}",
        band_names=[("Channel %i" % i) for i in range(len(full_idx_surf_rfl))],
        wavelength_unts="Nanometers",
        description=("L2A Analytyical per-pixel surface retrieval"),
    )

    # Construct surf rfl uncertainty output
    bbl = "{" + ",".join([f"{x}" for x in outside_ret_windows])
    unc_output = construct_output(
        output_metadata,
        analytical_state_unc_path,
        bbl=bbl,
        interleave="bil",
        bands=f"{len(full_idx_surface)}",
        band_names=[full_statevector[i] for i in full_idx_surf_rfl],
        wavelength_unts="Nanometers",
        description=("L2A Analytyical per-pixel surface retrieval uncertainty"),
    )

    # Set up input files
    input_files = {
        "rdn_file": rdn_file,
        "loc_file": loc_file,
        "obs_file": obs_file,
        "atm_file": atm_file,
    }

    # Set up the output files
    output_files = {
        "rfl_output": rfl_output,
        "unc_output": unc_output,
    }

    # Ray initialization
    ray_dict = {
        "ignore_reinit_error": config.implementation.ray_ignore_reinit_error,
        "address": config.implementation.ip_head,
        "_temp_dir": config.implementation.ray_temp_dir,
        "include_dashboard": config.implementation.ray_include_dashboard,
        "_redis_password": config.implementation.redis_password,
        "num_cpus": n_cores,
    }
    ray.init(**ray_dict)

    """
    The looping over classes is very similar to isofit.run
    """
    n_workers = n_cores

    # Initialize workers
    wargs = [
        ray.put(obj)
        for obj in (
            config,
            pixel_index,
            full_statevector,
            full_idx_surface,
            full_idx_RT,
            full_idx_surf_rfl,
            input_files,
            output_files,
            loglevel,
            logfile,
        )
    ]
    workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

    n_iter = min(rdns[0], n_workers * config.implementation.task_inflation_factor)
    line_breaks = np.linspace(0, rdns[0], n_iter, dtype=int)
    line_breaks = [
        (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
    ]

    # run workers
    start_time = time.time()
    res = list(workers.map_unordered(lambda a, b: a.run_chunks.remote(b), line_breaks))
    total_time = time.time() - start_time
    print(total_time)

    logging.info(
        f"Analytical line inversions complete.  {round(total_time,2)}s total, "
        f"{round(rdns[0]*rdns[1]/total_time,4)} spectra/s, "
        f"{round(rdns[0]*rdns[1]/total_time/n_cores, 4)} spectra/s/core"
    )


@click.command(name="analytical_line")
@click.argument("rdn_file")
@click.argument("loc_file")
@click.argument("obs_file")
@click.argument("isofit_dir")
@click.option("--isofit_config", type=str, default=None)
@click.option("--segmentation_file", help="TODO", type=str, default=None)
@click.option("--n_atm_neighbors", help="TODO", type=int, default=20)
@click.option("--n_cores", help="TODO", type=int, default=-1)
@click.option("--smoothing_sigma", help="TODO", type=int, default=2)
@click.option("--output_rfl_file", help="TODO", type=str, default=None)
@click.option("--output_unc_file", help="TODO", type=str, default=None)
@click.option("--atm_file", help="TODO", type=str, default=None)
@click.option("--loglevel", help="TODO", type=str, default="INFO")
@click.option("--logfile", help="TODO", type=str, default=None)
def cli_analytical_line(**kwargs):
    """Execute the analytical line algorithm"""

    click.echo("Running analytical line")

    analytical_line(**kwargs)

    click.echo("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "analytical_line.py can no longer be called this way. "
        "Run as:\n isofit analytical_line [ARGS]"
    )
