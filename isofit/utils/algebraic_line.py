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
from isofit.core.fileio import IO, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_simple import invert_algebraic
from isofit.utils.atm_interpolation import atm_interpolation


def algebraic_line(
    rdn_file: str,
    loc_file: str,
    obs_file: str,
    isofit_dir: str,
    isofit_config: str = None,
    segmentation_file: str = None,
    num_neighbors: list = None,
    atm_sigma: list = None,
    n_cores: int = -1,
    num_iter: int = 1,
    output_rfl_file: str = None,
    atm_file: str = None,
    logging_level: str = "INFO",
    log_file: str = None,
) -> None:
    """
    Run pixel-wise algebraic inversions, using provided atmosphere file.
    If no atmosphere file is provided, it will be created via interpolation
    from the state solution.

    Args:
        rdn_file: Path to the radiance file
        loc_file: Path to the location file
        obs_file: Path to the observation file
        isofit_dir: Path to the isofit directory
        isofit_config: Path to the isofit configuration file
        segmentation_file: Path to the segmentation file (optional)
        num_neighbors: Number of atmospheric neighbors to use for interpolation
        atm_sigma: Smoothing sigma for the atmospheric interpolation
        n_cores: Number of cores to use for parallel processing
        num_iter: Number of iterations for the inversion
        output_rfl_file: Path to the output reflectance file
        atm_file: Path to the atmosphere file
        logging_level: Logging level (default: INFO)
        log_file: Path to the log file (default: None)

    Returns:
        None
    """
    if num_neighbors is None:
        num_neighbors = [10]
    if atm_sigma is None:
        atm_sigma = [2]

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=logging_level,
        filename=log_file,
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

    if output_rfl_file is None:
        algebraic_rfl_file = subs_state_file.replace("_subs_state", "_rfl")
    else:
        algebraic_rfl_file = output_rfl_file

    if atm_file is None:
        atm_file = subs_state_file.replace("_subs_state", "_atm_interp")
    else:
        atm_file = atm_file

    if config.forward_model.surface.surface_category == "glint_model_surface":
        logging.warning("Glint is ignored in algebraic line retrieval")

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
            nneighbors=num_neighbors,
            gaussian_smoothing_sigma=atm_sigma,
            n_cores=n_cores,
        )

    rdn_ds = envi.open(envi_header(rdn_file))
    rdn = rdn_ds.open_memmap(interleave="bip")
    rdns = rdn.shape

    output_metadata = rdn_ds.metadata
    output_metadata["interleave"] = "bil"
    output_metadata["description"] = (
        "L2A Algebraic per-pixel surface reflectance retrieval"
    )
    output_metadata["bands"] = str(len(fm.idx_surf_rfl))
    output_metadata["band names"] = np.array(fm.surface.statevec_names)[fm.idx_surf_rfl]

    outside_ret_windows = np.zeros(len(fm.surface.idx_lamb), dtype=int)
    outside_ret_windows[iv.winidx] = 1
    output_metadata["bbl"] = "{" + ",".join([str(x) for x in outside_ret_windows]) + "}"

    if "emit pge input files" in list(output_metadata.keys()):
        del output_metadata["emit pge input files"]

    # Initialize rfl and unc file (always)
    img = envi.create_image(
        envi_header(algebraic_rfl_file), ext="", metadata=output_metadata, force=True
    )
    del img, rdn_ds, rdn

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
            algebraic_rfl_file,
            rdn_file,
            loc_file,
            obs_file,
            subs_state_file,
            lbl_file,
            num_iter,
            logging_level,
            log_file,
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
        f"Algebraic line inversions complete.  {round(total_time,2)}s total, "
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
        algebraic_rfl_file: str,
        rdn_file: str,
        loc_file: str,
        obs_file: str,
        subs_state_file: str,
        lbl_file: str,
        num_iter: int,
        logging_level: str,
        log_file: str,
    ):
        """
        Worker class to help run a subset of spectra.

        Args:
            fm: isofit forward_model
            logging_level: output logging level
            log_file: output logging file
        """
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=logging_level,
            filename=log_file,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )
        self.config = config
        self.fm = fm
        self.iv = Inversion(self.config, self.fm)

        self.rfl_bounds = np.min(fm.bounds, axis=0)[0], np.max(fm.bounds, axis=0)[1]
        logging.debug(
            f"Reflectance output will be bounded to the surface bounds: {self.rfl_bounds}"
        )

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = 500
        self.RT_state_file = RT_state_file
        self.rdn_file = rdn_file
        self.loc_file = loc_file
        self.obs_file = obs_file
        self.algebraic_rfl_file = algebraic_rfl_file

        # Can't see any reason to leave these as optional
        self.subs_state_file = subs_state_file
        self.lbl_file = lbl_file

        # If I only want to use some of the atm_interp bands
        # Empty if all
        self.atm_bands = []

        # How many iterations to use for invert_analytical
        self.num_iter = num_iter

        if config.input.radiometry_correction_file is not None:
            self.radiance_correction, wl = load_spectrum(
                config.input.radiometry_correction_file
            )
        else:
            self.radiance_correction = None

    def run_lines(self, startstop: tuple) -> None:
        """
        Run the algebraic line algorithm for a set of lines.
        Args:
            startstop: tuple of (start_line, stop_line) to process

        Returns:
            None
        """
        rdn = envi.open(envi_header(self.rdn_file)).open_memmap(interleave="bip")
        loc = envi.open(envi_header(self.loc_file)).open_memmap(interleave="bip")
        obs = envi.open(envi_header(self.obs_file)).open_memmap(interleave="bip")
        rt_state = envi.open(envi_header(self.RT_state_file)).open_memmap(
            interleave="bip"
        )
        lbl = envi.open(envi_header(self.lbl_file)).open_memmap(interleave="bip")

        start_line, stop_line = startstop

        esd = IO.load_esd()

        for r in range(start_line, stop_line):
            output_rfl = (
                np.zeros((1, rt_state.shape[1], len(self.fm.idx_surf_rfl))) - 9999
            )
            for c in range(output_rfl.shape[1]):
                meas = rdn[r, c, :]
                if self.radiance_correction is not None:
                    # sc - Creating copy to avoid the "output array read only" error
                    #      when applying correction factors
                    meas = np.copy(meas)
                    meas *= self.radiance_correction
                if np.all(meas < 0):
                    continue

                geom = Geometry(obs=obs[r, c, :], loc=loc[r, c, :], esd=esd)

                # "Atmospheric" state ALWAYS comes from all bands in the
                # atm_interpolated file
                x_RT = rt_state[r, c, :]

                x_surface, _, x_instrument = self.fm.unpack(self.fm.init.copy())
                rfl_est, coeffs = invert_algebraic(
                    self.fm.surface,
                    self.fm.RT,
                    self.fm.instrument,
                    x_surface,
                    x_RT,
                    x_instrument,
                    meas,
                    geom,
                )
                rfl_est = self.fm.surface.fit_params(rfl_est, geom)

                output_rfl[0, c, :] = rfl_est

            write_bil_chunk(
                output_rfl.T,
                self.algebraic_rfl_file,
                r,
                (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surf_rfl)),
            )
        del rdn, loc, obs, lbl


@click.command(name="algebraic_line")
@click.argument("rdn_file")
@click.argument("loc_file")
@click.argument("obs_file")
@click.argument("isofit_dir")
@click.option(
    "--isofit_config",
    type=str,
    help="isofit config file (inferred from isofit_dir if not specified)",
    default=None,
)
@click.option(
    "--segmentation_file",
    help="segmentation chunking file (inferred from isofit_dir if not specified)",
    type=str,
    default=None,
)
@click.option(
    "--num_neighbors",
    "-nn",
    type=int,
    multiple=True,
    default=[10],
    help="number of atmospheric neighbors for interpolation",
)
@click.option(
    "--atm_sigma",
    "-as",
    type=float,
    multiple=True,
    default=[2],
    help="smoothing sigma for atmospheric interpolation",
)
@click.option(
    "--n_cores",
    type=int,
    default=-1,
    help="number of cores to use for parallel processing (default: all available cores)",
)
@click.option(
    "--output_rfl_file",
    help="output reflectance file (inferred from isofit_dir if not specified)",
    type=str,
    default=None,
)
@click.option(
    "--atm_file",
    help="atmospheric interpolation file - created if not specified, used if provided",
    type=str,
    default=None,
)
@click.option("--logging_level", help="logging level", type=str, default="INFO")
@click.option("--log_file", help="logging file", type=str, default=None)
def cli(**kwargs):
    """Execute the algebraic line algorithm"""

    click.echo("Running algebraic line")

    algebraic_line(**kwargs)

    click.echo("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "algebraic_line.py can no longer be called this way.  Run as:\n isofit algebraic_line [ARGS]"
    )
