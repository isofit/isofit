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
from isofit.core.fileio import write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion import Inversions
from isofit.inversion.inverse_simple import invert_analytical
from isofit.utils.atm_interpolation import atm_interpolation
from isofit.utils.multistate import (
    cache_forward_models,
    construct_full_state,
    index_image_by_class,
    match_class,
)


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: configs.Config,
        # fm: ForwardModel,
        fm_cache: dict,
        state_pixel_index: list,
        full_statevector: list,
        full_idx_surface: np.array,
        full_idx_RT: np.array,
        RT_state_file: str,
        analytical_state_file: str,
        analytical_state_unc_file: str,
        rdn_file: str,
        loc_file: str,
        obs_file: str,
        loglevel: str,
        logfile: str,
        subs_state_file: str = None,
        lbl_file: str = None,
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

        self.fm_cache = fm_cache
        self.state_pixel_index = state_pixel_index
        self.full_statevector = full_statevector
        self.full_idx_surface = full_idx_surface
        self.full_idx_RT = full_idx_RT

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = 500
        self.RT_state_file = RT_state_file
        self.rdn_file = rdn_file
        self.loc_file = loc_file
        self.obs_file = obs_file
        self.analytical_state_file = analytical_state_file
        self.analytical_state_unc_file = analytical_state_unc_file

        if subs_state_file is not None and lbl_file is not None:
            self.subs_state_file = subs_state_file
            self.lbl_file = lbl_file
        else:
            self.subs_state_file = None
            self.lbl_file = None

        if config.input.radiometry_correction_file is not None:
            self.radiance_correction, wl = load_spectrum(
                config.input.radiometry_correction_file
            )
        else:
            self.radiance_correction = None

        # Open files at the worker level
        self.rdn = envi.open(envi_header(self.rdn_file)).open_memmap(interleave="bip")

        self.loc = envi.open(envi_header(self.loc_file)).open_memmap(interleave="bip")

        self.obs = envi.open(envi_header(self.obs_file)).open_memmap(interleave="bip")

        self.rt_state = envi.open(envi_header(self.RT_state_file)).open_memmap(
            interleave="bip"
        )

    def run_lines(self, startstop: tuple) -> None:
        """
        TODO: Description
        """
        start_line, stop_line = startstop
        output_state = (
            np.zeros(
                (
                    stop_line - start_line,
                    self.rt_state.shape[1],
                    len(self.full_idx_surface),
                )
            )
            - 9999
        )

        output_state_unc = (
            np.zeros(
                (
                    stop_line - start_line,
                    self.rt_state.shape[1],
                    len(self.full_idx_surface),
                )
            )
            - 9999
        )

        for r in range(start_line, stop_line):
            for c in range(output_state.shape[1]):
                # class of pixel
                pixel_class = match_class(self.state_pixel_index, r, c)

                # get cached fm
                self.fm = self.fm_cache[pixel_class]

                # Construct inversion

                iv = Inversions.get(self.config.implementation.mode, None)
                if not iv:
                    logging.exception(
                        "Inversion implementation: "
                        f"{self.config.implementation.mode}, "
                        "did not match options"
                    )
                    raise KeyError
                self.iv = iv(self.config, self.fm)

                meas = self.rdn[r, c, :]
                if self.radiance_correction is not None:
                    meas *= self.radiance_correction
                if np.all(meas < 0):
                    continue

                # Atmospheric state elements
                x_RT = self.rt_state[
                    # r, c, self.full_idx_RT - len(self.full_idx_surface)
                    r,
                    c,
                    self.fm.state.idx_RT - len(self.fm.state.idx_surface),
                ]
                geom = Geometry(obs=self.obs[r, c, :], loc=self.loc[r, c, :])

                states, unc = invert_analytical(
                    self.fm,
                    self.iv.winidx,
                    meas,
                    geom,
                    x_RT,
                    1,
                    self.hash_table,
                    self.hash_size,
                )

                # Match pixel-specific to general statevector
                state_est = states[-1]
                full_state_est = match_statevector(
                    state_est, self.full_statevector, self.fm.state.statevec
                )

                output_state[r - start_line, c, :] = full_state_est[
                    self.full_idx_surface
                ]

                full_unc = match_statevector(
                    unc, self.full_statevector, self.fm.state.statevec
                )
                output_state_unc[r - start_line, c, :] = full_unc[self.full_idx_surface]

            logging.info(f"Analytical line writing line {r}")

            write_bil_chunk(
                output_state[r - start_line, ...].T,
                self.analytical_state_file,
                r,
                (self.rdn.shape[0], self.rdn.shape[1], len(self.full_idx_surface)),
            )
            write_bil_chunk(
                output_state_unc[r - start_line, ...].T,
                self.analytical_state_unc_file,
                r,
                (self.rdn.shape[0], self.rdn.shape[1], len(self.full_idx_surface)),
            )


def construct_outputs(
    rdn_file,
    full_idx_surface,
    full_idx_surf_rfl,
    winidx,
    analytical_state_file,
    analytical_state_unc_file,
):

    rdn_ds = envi.open(envi_header(rdn_file))
    rdns = rdn_ds.open_memmap(interleave="bip").shape
    output_metadata = rdn_ds.metadata
    output_metadata["interleave"] = "bil"
    output_metadata["description"] = "L2A Analytyical per-pixel surface retrieval"
    output_metadata["bands"] = f"{len(full_idx_surface)}"
    del rdn_ds

    outside_ret_windows = np.zeros(len(full_idx_surf_rfl), dtype=int)
    outside_ret_windows[winidx] = 1

    output_metadata["bbl"] = "{" + ",".join([f"{x}" for x in outside_ret_windows]) + "}"

    if "emit pge input files" in list(output_metadata.keys()):
        del output_metadata["emit pge input files"]

    img = envi.create_image(
        envi_header(analytical_state_file), ext="", metadata=output_metadata, force=True
    )
    del img

    img = envi.create_image(
        envi_header(analytical_state_unc_file),
        ext="",
        metadata=output_metadata,
        force=True,
    )
    del img

    return rdns


def retrieve_winidx(config):
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)
    windows = config.implementation.inversion.windows

    winidx = np.array((), dtype=int)
    for lo, hi in windows:
        idx = np.where(np.logical_and(wl_init > lo, wl_init < hi))[0]
        winidx = np.concatenate((winidx, idx), axis=0)

    return winidx


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
    subs_class_file = config.forward_model.surface.surface_class_file

    # Rename files
    lbl_file = (
        segmentation_file
        if segmentation_file
        else (subs_state_file.replace("_subs_state", "_lbl"))
    )
    analytical_state_file = (
        output_rfl_file
        if output_rfl_file
        else (subs_state_file.replace("_subs_state", "_state_analytical"))
    )
    analytical_state_unc_file = (
        output_unc_file
        if output_unc_file
        else (subs_state_file.replace("_subs_state", "_state_analytical_uncert"))
    )
    atm_file = (
        atm_file
        if atm_file
        else (subs_state_file.replace("_subs_state", "_atm_interp"))
    )

    # Set up the multi-state pixel map
    state_pixel_index = (
        index_image_by_class(config.forward_model.surface, subs=False)
        if config.forward_model.surface.multi_surface_flag
        else []
    )

    # fm = ForwardModel(config, subs=False)

    (
        full_statevector,
        full_idx_surface,
        full_idx_surf_rfl,
        full_idx_RT,
    ) = construct_full_state(config)

    # Find the winidx
    winidx = retrieve_winidx(config)

    # Perform the atmospheric interpolation
    if os.path.isfile(atm_file) is False:
        # This should match the necesary state elements based on the name
        atm_interpolation(
            reference_state_file=subs_state_file,
            reference_locations_file=subs_loc_file,
            input_locations_file=loc_file,
            segmentation_file=lbl_file,
            output_atm_file=atm_file,
            # atm_band_names=fm.RT.statevec_names,
            atm_band_names=[full_statevector[i] for i in full_idx_RT],
            nneighbors=n_atm_neighbors,
            gaussian_smoothing_sigma=smoothing_sigma,
            n_cores=n_cores,
        )

    # Construct output
    rdns = construct_outputs(
        rdn_file,
        full_idx_surface,
        full_idx_surf_rfl,
        winidx,
        analytical_state_file,
        analytical_state_unc_file,
    )

    # Divide into chunks to processes for each worker
    line_breaks = np.linspace(
        0, rdns[0], (n_cores * config.implementation.task_inflation_factor), dtype=int
    )

    line_breaks = [
        (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
    ]

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

    # Initialize workers
    wargs = [
        ray.put(obj)
        for obj in (
            config,
            fm,
            state_pixel_index,
            full_statevector,
            full_idx_surface,
            full_idx_RT,
            atm_file,
            analytical_state_file,
            analytical_state_unc_file,
            rdn_file,
            loc_file,
            obs_file,
            loglevel,
            logfile,
        )
    ]
    workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_cores)])

    # run workers
    start_time = time.time()
    res = list(workers.map_unordered(lambda a, b: a.run_lines.remote(b), line_breaks))
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
        "analytical_line.py can no longer be called this way.  Run as:\n isofit analytical_line [ARGS]"
    )
