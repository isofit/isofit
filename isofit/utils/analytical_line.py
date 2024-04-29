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
from isofit.core.common import envi_header, load_spectrum
from isofit.core.fileio import write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_simple import invert_analytical
from isofit.utils.atm_interpolation import atm_interpolation


def analytical_line(
    rdn_file: str,
    loc_file: str,
    obs_file: str,
    isofit_dir: str,
    isofit_config: str = None,
    segmentation_file: str = None,
    n_atm_neighbors: list = None,
    n_cores: int = -1,
    smoothing_sigma: list = None,
    output_rfl_file: str = None,
    output_unc_file: str = None,
    atm_file: str = None,
    loglevel: str = "INFO",
    logfile: str = None,
) -> None:
    """
    TODO: Description
    """
    if n_atm_neighbors is None:
        n_atm_neighbors = [20]
    if smoothing_sigma is None:
        smoothing_sigma = [2]

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    if isofit_config is None:
        file = glob(os.path.join(isofit_dir, "config", "") + "*_isofit.json")[0]
    else:
        file = isofit_config

    config = configs.create_new_config(file)
    config.forward_model.instrument.integrations = 1

    subs_state_file = config.output.estimated_state_file
    subs_loc_file = config.input.loc_file

    if (
        segmentation_file is None
        or config.forward_model.surface.surface_category == "glint_model_surface"
    ):
        lbl_file = subs_state_file.replace("_subs_state", "_lbl")
    else:
        lbl_file = segmentation_file

    if (
        output_rfl_file is None
        or config.forward_model.surface.surface_category == "glint_model_surface"
    ):
        analytical_state_file = subs_state_file.replace(
            "_subs_state", "_state_analytical"
        )
    else:
        analytical_state_file = output_rfl_file

    if (
        output_unc_file is None
        or config.forward_model.surface.surface_category == "glint_model_surface"
    ):
        analytical_state_unc_file = subs_state_file.replace(
            "_subs_state", "_state_analytical_uncert"
        )
    else:
        analytical_state_unc_file = output_unc_file

    if (
        atm_file is None
        or config.forward_model.surface.surface_category == "glint_model_surface"
    ):
        atm_file = subs_state_file.replace("_subs_state", "_atm_interp")
    else:
        atm_file = atm_file

    fm = ForwardModel(config)
    iv = Inversion(config, fm)

    if os.path.isfile(atm_file) is False:
        atm_interpolation(
            reference_state_file=subs_state_file,
            reference_locations_file=subs_loc_file,
            input_locations_file=loc_file,
            segmentation_file=lbl_file,
            output_atm_file=atm_file,
            atm_band_names=fm.RT.statevec_names,
            nneighbors=n_atm_neighbors,
            gaussian_smoothing_sigma=smoothing_sigma,
            n_cores=n_cores,
        )

    rdn_ds = envi.open(envi_header(rdn_file))
    rdn = rdn_ds.open_memmap(interleave="bip")
    rdns = rdn.shape

    output_metadata = rdn_ds.metadata
    output_metadata["interleave"] = "bil"
    output_metadata["description"] = "L2A Analytyical per-pixel surface retrieval"
    output_metadata["bands"] = str(len(fm.idx_surface))

    outside_ret_windows = np.zeros(len(fm.surface.idx_lamb), dtype=int)
    outside_ret_windows[iv.winidx] = 1
    output_metadata["bbl"] = "{" + ",".join([str(x) for x in outside_ret_windows]) + "}"

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
    del rdn, img

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    ray_dict = {
        "ignore_reinit_error": config.implementation.ray_ignore_reinit_error,
        "address": config.implementation.ip_head,
        "_temp_dir": config.implementation.ray_temp_dir,
        "include_dashboard": config.implementation.ray_include_dashboard,
        "_redis_password": config.implementation.redis_password,
        "num_cpus": n_cores,
    }
    ray.init(**ray_dict)

    n_workers = n_cores

    wargs = [
        ray.put(obj)
        for obj in (
            config,
            fm,
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
    workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

    line_breaks = np.linspace(
        0, rdns[0], n_workers * config.implementation.task_inflation_factor, dtype=int
    )
    line_breaks = [
        (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
    ]

    start_time = time.time()
    res = list(workers.map_unordered(lambda a, b: a.run_lines.remote(b), line_breaks))
    total_time = time.time() - start_time
    logging.info(
        f"Analytical line inversions complete.  {round(total_time,2)}s total, "
        f"{round(rdns[0]*rdns[1]/total_time,4)} spectra/s, "
        f"{round(rdns[0]*rdns[1]/total_time/n_workers,4)} spectra/s/core"
    )


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: configs.Config,
        fm: ForwardModel,
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
        self.fm = fm
        self.iv = Inversion(self.config, self.fm)

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

    def run_lines(self, startstop: tuple) -> None:
        """
        TODO: Description
        """
        rdn = envi.open(envi_header(self.rdn_file)).open_memmap(interleave="bip")
        loc = envi.open(envi_header(self.loc_file)).open_memmap(interleave="bip")
        obs = envi.open(envi_header(self.obs_file)).open_memmap(interleave="bip")
        rt_state = envi.open(envi_header(self.RT_state_file)).open_memmap(
            interleave="bip"
        )

        start_line, stop_line = startstop
        output_state = (
            np.zeros(
                (stop_line - start_line, rt_state.shape[1], len(self.fm.idx_surface))
            )
            - 9999
        )
        output_state_unc = (
            np.zeros(
                (stop_line - start_line, rt_state.shape[1], len(self.fm.idx_surface))
            )
            - 9999
        )

        for r in range(start_line, stop_line):
            for c in range(output_state.shape[1]):
                meas = rdn[r, c, :]
                if self.radiance_correction is not None:
                    meas *= self.radiance_correction
                if np.all(meas < 0):
                    continue
                x_RT = rt_state[r, c, self.fm.idx_RT - len(self.fm.idx_surface)]
                geom = Geometry(obs=obs[r, c, :], loc=loc[r, c, :])

                states, unc = invert_analytical(
                    self.iv.fm,
                    self.iv.winidx,
                    meas,
                    geom,
                    x_RT,
                    1,
                    self.hash_table,
                    self.hash_size,
                )

                output_state[r - start_line, c, :] = states[-1][self.fm.idx_surface]

                output_state_unc[r - start_line, c, :] = unc[self.fm.idx_surface]

            logging.info(f"Analytical line writing line {r}")

            write_bil_chunk(
                output_state[r - start_line, ...].T,
                self.analytical_state_file,
                r,
                (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surface)),
            )
            write_bil_chunk(
                output_state_unc[r - start_line, ...].T,
                self.analytical_state_unc_file,
                r,
                (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surface)),
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
