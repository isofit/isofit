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
from isofit.inversion.inverse_simple import (
    invert_algebraic,
    invert_analytical,
    invert_simple,
)
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
    num_iter: int = 1,
    smoothing_sigma: list = None,
    output_rfl_file: str = None,
    output_unc_file: str = None,
    atm_file: str = None,
    loglevel: str = "INFO",
    logfile: str = None,
    initializer: str = "algebraic",
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

    if output_rfl_file is None:
        analytical_rfl_file = subs_state_file.replace("_subs_state", "_rfl")
    else:
        analytical_rfl_file = output_rfl_file

    if output_unc_file is None:
        analytical_rfl_unc_file = subs_state_file.replace("_subs_state", "_rfl_uncert")
    else:
        analytical_rfl_unc_file = output_unc_file

    if config.forward_model.surface.surface_category == "glint_model_surface":
        analytical_non_rfl_surf_file = subs_state_file.replace(
            "_subs_state", "_surf_non_rfl"
        )
        analytical_non_rfl_surf_unc_file = subs_state_file.replace(
            "_subs_state", "_surf_non_rfl_uncert"
        )
    else:
        analytical_non_rfl_surf_file = None
        analytical_non_rfl_surf_unc_file = None

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
    output_metadata["description"] = (
        "L2A Analytyical per-pixel surface reflectance retrieval"
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
        envi_header(analytical_rfl_file), ext="", metadata=output_metadata, force=True
    )
    del img, rdn_ds

    img = envi.create_image(
        envi_header(analytical_rfl_unc_file),
        ext="",
        metadata=output_metadata,
        force=True,
    )
    del rdn, img

    # Initialize the surf non_rfl rile if needed
    if analytical_non_rfl_surf_file:
        output_metadata["description"] = (
            "L2A Analytyical per-pixel surface state retrieval (non-reflectance)"
        )
        output_metadata["bands"] = str(len(fm.idx_surf_nonrfl))
        output_metadata["band names"] = np.array(fm.surface.statevec_names)[
            fm.idx_surf_nonrfl
        ]
        for k in ["bbl", "wavelength", "wavelength units", "fwhm", "smoothing factors"]:
            if output_metadata.get(k):
                ret = output_metadata.pop(k)

        img = envi.create_image(
            envi_header(analytical_non_rfl_surf_file),
            ext="",
            metadata=output_metadata,
            force=True,
        )
        del img

        img = envi.create_image(
            envi_header(analytical_non_rfl_surf_unc_file),
            ext="",
            metadata=output_metadata,
            force=True,
        )
        del img

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
            analytical_rfl_file,
            analytical_rfl_unc_file,
            analytical_non_rfl_surf_file,
            analytical_non_rfl_surf_unc_file,
            rdn_file,
            loc_file,
            obs_file,
            subs_state_file,
            lbl_file,
            num_iter,
            loglevel,
            logfile,
            initializer,
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
        analytical_rfl_file: str,
        analytical_state_unc_file: str,
        analytical_non_rfl_surf_file: str,
        analytical_non_rfl_surf_unc_file: str,
        rdn_file: str,
        loc_file: str,
        obs_file: str,
        subs_state_file: str,
        lbl_file: str,
        num_iter: int,
        loglevel: str,
        logfile: str,
        initializer: str,
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
        self.analytical_rfl_file = analytical_rfl_file
        self.analytical_rfl_unc_file = analytical_state_unc_file
        self.analytical_non_rfl_surf_file = analytical_non_rfl_surf_file
        self.analytical_non_rfl_surf_unc_file = analytical_non_rfl_surf_unc_file

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

        self.initializer = initializer

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
        subs_state = envi.open(envi_header(self.subs_state_file)).open_memmap(
            interleave="bip"
        )
        lbl = envi.open(envi_header(self.lbl_file)).open_memmap(interleave="bip")

        start_line, stop_line = startstop
        output_rfl = np.zeros((1, rt_state.shape[1], len(self.fm.idx_surf_rfl))) - 9999
        output_rfl_unc = (
            np.zeros((1, rt_state.shape[1], len(self.fm.idx_surf_rfl))) - 9999
        )

        if self.analytical_non_rfl_surf_file:
            output_non_rfl = (
                np.zeros(
                    (
                        1,
                        rt_state.shape[1],
                        len(self.fm.idx_surf_nonrfl),
                    )
                )
                - 9999
            )
            output_non_rfl_unc = (
                np.zeros(
                    (
                        1,
                        rt_state.shape[1],
                        len(self.fm.idx_surf_nonrfl),
                    )
                )
                - 9999
            )

        esd = IO.load_esd()

        for r in range(start_line, stop_line):
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

                iv_idx = self.fm.surface.analytical_iv_idx
                sub_state = subs_state[int(lbl[r, c, 0]), 0, iv_idx]

                # Note: concatenation only works with the correct indexing.
                # TODO handle indexing in a safer way. See line 126
                sub_state = np.concatenate([sub_state, x_RT])

                # Build statevector to use for initialization.
                # Can be done three different ways.
                # SUPERPIXEL uses the superpixel value --> Fastests
                # ALGEBRAIC uses invert_algebraic for rfl,
                # and the superpixel for non_rfl surface elements
                # SIMPLE uses invert_simple for rfl and non_rfl surface elements
                if self.initializer == "superpixel":
                    x0 = sub_state

                elif self.initializer == "algebraic":
                    x_surface, _, x_instrument = self.fm.unpack(self.fm.init.copy())
                    rfl_est, Ls_est, coeffs = invert_algebraic(
                        self.fm.surface,
                        self.fm.RT,
                        self.fm.instrument,
                        x_surface,
                        x_RT,
                        x_instrument,
                        meas,
                        geom,
                    )
                    x0 = np.concatenate(
                        [
                            rfl_est,
                            sub_state[self.fm.idx_surf_nonrfl],
                            x_RT,
                            x_instrument,
                        ]
                    )

                elif self.initializer == "simple":
                    x0 = invert_simple(self.fm, meas, geom)
                    x0[self.fm.idx_RT] = x_RT

                else:
                    raise ValueError("No valid initializer given for AOE algorithm")

                states, unc, EXIT_CODE = invert_analytical(
                    self.iv.fm,
                    self.iv.winidx,
                    meas,
                    geom,
                    np.copy(x0),
                    sub_state,
                    self.num_iter,
                    self.hash_table,
                    self.hash_size,
                )

                if EXIT_CODE == -11:
                    logging.error(
                        f"Row, Col: {r, c} - Sa matrix is non-invertible. Statevector is likely NaNs."
                    )
                if EXIT_CODE == -15:
                    logging.error(
                        f"Row, Col: {r, c} - LinalgError. Eigenvalue calculation failed."
                    )

                output_rfl[0, c, :] = states[-1, self.fm.idx_surf_rfl]
                output_rfl_unc[0, c, :] = unc[self.fm.idx_surf_rfl]

                if self.analytical_non_rfl_surf_file:
                    output_non_rfl[0, c, :] = states[-1, self.fm.idx_surf_nonrfl]
                    output_non_rfl_unc[0, c, :] = unc[self.fm.idx_surf_nonrfl]

            # What do we want to do with the negative reflectances?
            # state = output_state[r - start_line, ...]
            # mask = np.logical_and.reduce(
            #     [
            #         state < self.rfl_bounds[0],
            #         state > self.rfl_bounds[1],
            #         state != -9999,
            #         state != -0.01,
            #     ]
            # )
            # state[mask] = 0

            logging.info(f"Analytical line writing line {r}")

            write_bil_chunk(
                output_rfl.T,
                self.analytical_rfl_file,
                r,
                (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surf_rfl)),
            )
            write_bil_chunk(
                output_rfl_unc.T,
                self.analytical_rfl_unc_file,
                r,
                (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surf_rfl)),
            )

            if self.analytical_non_rfl_surf_file:
                write_bil_chunk(
                    output_non_rfl.T,
                    self.analytical_non_rfl_surf_file,
                    r,
                    (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surf_nonrfl)),
                )
                write_bil_chunk(
                    output_non_rfl_unc.T,
                    self.analytical_non_rfl_surf_unc_file,
                    r,
                    (rdn.shape[0], rdn.shape[1], len(self.fm.idx_surf_nonrfl)),
                )
        del rdn, loc, obs, lbl


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
def cli(**kwargs):
    """Execute the analytical line algorithm"""

    click.echo("Running analytical line")

    analytical_line(**kwargs)

    click.echo("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "analytical_line.py can no longer be called this way.  Run as:\n isofit analytical_line [ARGS]"
    )
