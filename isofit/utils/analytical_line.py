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
from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import click
import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.configs import configs
from isofit.core.common import envi_header, load_spectrum, load_wavelen
from isofit.core.fileio import IO, initialize_output, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.core.multistate import (
    construct_full_state,
    index_spectra_by_surface,
    match_statevector,
    update_config_for_surface,
)
from isofit.inversion.inverse_simple import (
    invert_algebraic,
    invert_analytical,
    invert_simple,
)
from isofit.utils.atm_interpolation import atm_interpolation


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
    num_iter: int = 1,
    smoothing_sigma: list = [2],
    output_rfl_file: str = None,
    output_unc_file: str = None,
    atm_file: str = None,
    skyview_factor_file: str = None,
    loglevel: str = "INFO",
    logfile: str = None,
    initializer: str = "algebraic",
    segmentation_size: int = 40,
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
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)

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
        else (subs_state_file.replace("_subs_state", "_rfl"))
    )
    analytical_rfl_unc_path = (
        output_unc_file
        if output_unc_file
        else (subs_state_file.replace("_subs_state", "_uncert"))
    )

    # Files names for non-surface reflectance states
    analytical_non_rfl_surf_file = subs_state_file.replace(
        "_subs_state", "_surf_non_rfl"
    )
    analytical_non_rfl_surf_unc_file = subs_state_file.replace(
        "_subs_state", "_surf_non_rfl_uncert"
    )

    atm_file = (
        atm_file
        if atm_file
        else (subs_state_file.replace("_subs_state", "_atm_interp"))
    )

    # Get full statevector for image
    (
        full_statevector,
        full_idx_surface,
        full_idx_surf_rfl,
        _,
        full_idx_RT,
        full_idx_instrument,
    ) = construct_full_state(config)

    # Perform the atmospheric interpolation
    if os.path.isfile(atm_file) is False:
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

    # Get string representation of bad band list
    outside_ret_windows = np.zeros(len(full_idx_surf_rfl), dtype=int)
    outside_ret_windows[retrieve_winidx(config)] = 1

    # Get output shape
    rdn_ds = envi.open(envi_header(rdn_file))
    rdns = rdn_ds.shape
    rdn_meta = rdn_ds.metadata
    del rdn_ds

    # Construct surf rfl output
    output_metadata = {
        "data type": 4,
        "file type": "ENVI Standard",
        "byte order": 0,
        "no data value": -9999,
        "wavelength units": "Nanometers",
        "wavelength": wl_init,
        "fwhm": fwhm_init,
        "lines": rdn_meta["lines"],
        "samples": rdn_meta["samples"],
        "interleave": "bil",
    }
    if "map info" in rdn_meta:
        output_metadata["map info"] = (
            "{" + ", ".join(map(str, rdn_meta["map info"])) + "}"
        )

    output_metadata["band names"] = [
        full_statevector[i] for i in range(len(full_idx_surf_rfl))
    ]
    bbl = "{" + ",".join([f"{x}" for x in outside_ret_windows]) + "}"
    num_bands = len(full_idx_surf_rfl)
    engine_name = config.forward_model.radiative_transfer.radiative_transfer_engines[
        0
    ].engine_name
    isofit_version = config.implementation.isofit_version
    rfl_output = initialize_output(
        output_metadata,
        analytical_rfl_path,
        (rdns[0], num_bands, rdns[1]),
        bands=f"{num_bands}",
        bbl=bbl,
        description=(
            f"L2A Analytical per-pixel surface retrieval (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
        ),
    )

    # Construct surf rfl uncertainty output
    output_metadata["band names"] = [
        full_statevector[i] for i in range(len(full_idx_surf_rfl))
    ]
    unc_output = initialize_output(
        output_metadata,
        analytical_rfl_unc_path,
        (rdns[0], num_bands, rdns[1]),
        bands=f"{num_bands}",
        bbl=bbl,
        description=(
            f"L2A Analytical per-pixel surface retrieval uncertainty (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
        ),
    )

    # If there are more idx in surface than rfl, there are non_rfl surface states
    if len(full_idx_surface) > len(full_idx_surf_rfl):
        n_non_rfl_bands = len(full_idx_surface) - len(full_idx_surf_rfl)
        output_metadata["band names"] = [
            full_statevector[len(full_idx_surf_rfl) + i] for i in range(n_non_rfl_bands)
        ]
        non_rfl_output = initialize_output(
            output_metadata,
            analytical_non_rfl_surf_file,
            (rdns[0], n_non_rfl_bands, rdns[1]),
            bands=f"{n_non_rfl_bands}",
            description=(
                f"L2A Analytical per-pixel non_rfl surface retrieval  (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
            ),
        )

        non_rfl_unc_output = initialize_output(
            output_metadata,
            analytical_non_rfl_surf_unc_file,
            (rdns[0], n_non_rfl_bands, rdns[1]),
            bands=f"{n_non_rfl_bands}",
            description=(
                f"L2A Analytical per-pixel non_rfl surface retrieval uncertainty  (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
            ),
        )
    else:
        non_rfl_output = None
        non_rfl_unc_output = None

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
    n_workers = n_cores

    # Set up the memory-contiguous multi-state pixel map by sub
    index_pairs = np.empty((rdns[0] * rdns[1], 2), dtype=int)
    meshgrid = np.meshgrid(*(range(rdns[0]), range(rdns[1])))
    index_pairs[:, 0] = meshgrid[0].flatten(order="f")
    index_pairs[:, 1] = meshgrid[1].flatten(order="f")
    del meshgrid

    cache_RT = None
    input_config = deepcopy(config)
    surface_index = index_spectra_by_surface(
        input_config, index_pairs, force_full_res=True
    )
    for i, (surface_class_str, class_idx_pairs) in enumerate(surface_index.items()):
        # Handle multisurface
        config = update_config_for_surface(deepcopy(input_config), surface_class_str)

        fm = ForwardModel(config, cache_RT)

        # Initialize workers
        wargs = [ray.put(obj) for obj in (config, fm)]
        wargs += [
            surface_class_str,
            class_idx_pairs,
            full_statevector,
            full_idx_surface,
            full_idx_surf_rfl,
            full_idx_RT,
            rdn_file,
            loc_file,
            obs_file,
            atm_file,
            subs_state_file,
            lbl_file,
            rfl_output,
            unc_output,
            non_rfl_output,
            non_rfl_unc_output,
            num_iter,
            loglevel,
            logfile,
            initializer,
            skyview_factor_file,
        ]
        workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

        line_breaks = np.linspace(
            0,
            rdns[0],
            n_workers * config.implementation.task_inflation_factor,
            dtype=int,
        )

        line_breaks = [
            (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
        ]

        # run workers
        start_time = time.time()
        results = list(
            workers.map_unordered(lambda a, b: a.run_chunks.remote(b), line_breaks)
        )

        # Cache RT
        if not i:
            cache_RT = fm.RT

        del fm

    total_time = time.time() - start_time

    logging.info(
        f"Analytical line inversions complete.  {round(total_time,2)}s total, "
        f"{round(rdns[0]*rdns[1]/total_time,4)} spectra/s, "
        f"{round(rdns[0]*rdns[1]/total_time/n_cores, 4)} spectra/s/core"
    )


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: Config,
        fm: ForwardModel,
        surface_class_str: str,
        class_idx_pairs: np.array,
        full_statevector: list,
        full_idx_surface: np.array,
        full_idx_surf_rfl: np.array,
        full_idx_RT: np.array,
        rdn_file: str,
        loc_file: str,
        obs_file: str,
        atm_file: str,
        subs_state_file: str,
        lbl_file: str,
        rfl_output: str,
        unc_output: str,
        non_rfl_output: str,
        non_rfl_unc_output: str,
        num_iter: int,
        loglevel: str,
        logfile: str,
        initializer: str,
        skyview_factor_file: str,
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

        # Persist config
        self.config = config

        # Persist forward model
        self.fm = fm

        # Persist surface class (or all)
        self.surface_class_str = surface_class_str
        self.class_idx_pairs = class_idx_pairs

        # Will fail if env.data isn't set up
        self.esd = IO.load_esd()

        self.full_statevector = full_statevector
        self.full_idx_surface = full_idx_surface
        self.full_idx_surf_rfl = full_idx_surf_rfl
        self.full_idx_RT = full_idx_RT
        self.n_rfl_bands = len(full_idx_surf_rfl)
        self.n_non_rfl_bands = len(full_idx_surface) - len(full_idx_surf_rfl)

        self.winidx = retrieve_winidx(self.config)

        # input arrays
        self.rdn = envi.open(envi_header(rdn_file)).open_memmap(interleave="bip")
        self.loc = envi.open(envi_header(loc_file)).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(obs_file)).open_memmap(interleave="bip")
        self.rt_state = envi.open(envi_header(atm_file)).open_memmap(interleave="bip")
        self.subs_state = envi.open(envi_header(subs_state_file)).open_memmap(
            interleave="bip"
        )
        self.lbl = envi.open(envi_header(lbl_file)).open_memmap(interleave="bip")

        # Open skyview file for ALAlg, or create an array of 1s.
        if skyview_factor_file:
            self.svf = envi.open(envi_header(skyview_factor_file)).open_memmap(
                interleave="bip"
            )
        else:
            self.svf = []

        # Lines and samples
        self.n_lines = self.rdn.shape[0]
        self.n_samples = self.rdn.shape[1]

        # output paths
        self.rfl_outpath = rfl_output
        self.unc_outpath = unc_output
        self.non_rfl_outpath = non_rfl_output
        self.non_rfl_unc_outpath = non_rfl_unc_output

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = config.implementation.max_hash_table_size

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

    def run_chunks(self, line_breaks: tuple, fill_value: float = -9999.0) -> None:
        """
        TODO: Description
        """
        # Unpack arguments
        start_line, stop_line = line_breaks

        # Set up outputs
        output_rfl = (
            envi.open(envi_header(self.rfl_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        output_rfl_unc = (
            envi.open(envi_header(self.unc_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        if self.non_rfl_unc_outpath:
            output_non_rfl = (
                envi.open(envi_header(self.non_rfl_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

            output_non_rfl_unc = (
                envi.open(envi_header(self.non_rfl_unc_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

        # Find intersection between index_pairs and class_idx_pairs
        index_pairs = self.class_idx_pairs[
            np.where(
                (self.class_idx_pairs[:, 0] >= start_line)
                & (self.class_idx_pairs[:, 0] < stop_line)
            )
        ]

        for r, c, *_ in index_pairs:
            meas = self.rdn[r, c, :]

            if self.radiance_correction is not None:
                meas = meas.copy() * self.radiance_correction

            if np.all(meas < 0):
                continue

            geom = Geometry(
                obs=self.obs[r, c, :],
                loc=self.loc[r, c, :],
                esd=self.esd,
                svf=self.svf[r, c] if len(self.svf) else 1,
            )

            # "Atmospheric" state ALWAYS comes from all bands in the
            # atm_interpolated file
            x_RT = self.rt_state[r, c, :]

            # TODO depricate this iv_idx. Abstract the indexing a bit more
            # s.t. we can smooth any statevector element by specifying idx
            # iv_idx here is a relic from a version that
            # achieved this by using atm_band_names in atm_interpolation
            # need to improve that implementation
            iv_idx = self.fm.surface.analytical_iv_idx

            # Populate the "background" superpixel
            lbl_idx = int(self.lbl[r, c, 0])
            sub_state = np.zeros(self.fm.nstate)
            sub_state[self.fm.idx_surface] = self.subs_state[lbl_idx, 0, iv_idx]
            sub_state[self.fm.idx_RT] = x_RT
            sub_state[self.fm.idx_instrument] = self.subs_state[
                lbl_idx, 0, self.fm.idx_instrument
            ]
            # Enforce non-NaN
            sub_state[np.isnan(sub_state)] = self.fm.init[np.isnan(sub_state)]

            # Build statevector to use for initialization.
            # Can be done three different ways.
            # SUPERPIXEL uses the superpixel value --> Fastests
            # ALGEBRAIC uses invert_algebraic for rfl,
            # and the superpixel for non_rfl surface elements
            # SIMPLE uses invert_simple for rfl and non_rfl surface elements
            if self.initializer == "superpixel":
                x0 = sub_state
                x0[self.fm.idx_RT] = x_RT

            elif self.initializer == "algebraic":
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

                x0 = np.concatenate(
                    [
                        rfl_est,
                        x_RT,
                        x_instrument,
                    ]
                )

            elif self.initializer == "simple":
                x0 = invert_simple(self.fm, meas, geom)
                x0[self.fm.idx_RT] = x_RT

            else:
                raise ValueError("No valid initializer given for AOE algorithm")

            # NOTE: this line needs to be here to ensure geom.surf_cmp_init is populated
            geom.x_surf_init = x0[self.fm.idx_surface]

            states, unc = invert_analytical(
                self.fm,
                self.winidx,
                meas,
                geom,
                np.copy(x0),
                sub_state,
                self.num_iter,
                self.hash_table,
                self.hash_size,
            )
            state_est = states[-1]

            full_state_est = match_statevector(
                state_est, self.full_statevector, self.fm.statevec
            )
            output_rfl[r - start_line, c, :] = full_state_est[self.full_idx_surf_rfl]

            full_unc_est = match_statevector(
                unc, self.full_statevector, self.fm.statevec
            )
            output_rfl_unc[r - start_line, c, :] = full_unc_est[self.full_idx_surf_rfl]

            full_state_est[len(self.full_idx_surf_rfl) : self.n_non_rfl_bands]
            # Save the non_rfl portion
            if self.non_rfl_outpath:
                output_non_rfl[r - start_line, c, :] = full_state_est[
                    self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                ]
                output_non_rfl_unc[r - start_line, c, :] = full_unc_est[
                    self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                ]

        logging.info(
            f"Analytical line writing lines: {start_line} to {stop_line}. "
            f"Surface: {self.surface_class_str}"
        )

        # Output surface rfl
        write_bil_chunk(
            np.swapaxes(output_rfl, 1, 2),
            # output_rfl.T,
            self.rfl_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        # Save surface state uncertainty
        write_bil_chunk(
            np.swapaxes(output_rfl_unc, 1, 2),
            # output_rfl_unc.T,
            self.unc_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        if self.non_rfl_outpath:
            write_bil_chunk(
                np.swapaxes(output_non_rfl, 1, 2),
                self.non_rfl_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )
            write_bil_chunk(
                np.swapaxes(output_non_rfl_unc, 1, 2),
                self.non_rfl_unc_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
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
@click.option("--skyview_factor_file", help="TODO", type=str, default=None)
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
