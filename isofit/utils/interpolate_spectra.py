#! /usr/bin/env python3
#
#  Copyright 2025 California Institute of Technology
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
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov
import logging
import multiprocessing
import time

import click
import numpy as np
import scipy
from spectral.io import envi

from isofit import ray
from isofit.core.common import envi_header
from isofit.core.fileio import write_bil_chunk


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        infile,
        outfile,
        inplace,
        nodata_value,
        logfile,
        loglevel,
    ):

        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        self.infile = infile
        self.inplace = inplace

        if inplace:
            self.outfile = infile
        else:
            self.outfile = outfile

        self.nodata_value = nodata_value

        self.in_img = envi.open(envi_header(self.infile))
        self.nl, self.nb, self.ns = self.in_img.shape

        # Get wavelength grid
        self.wl = np.array(self.in_img.metadata["wavelength"]).astype(float)

    def interpolate_values(self, meas, replace_nan=False):
        if replace_nan:
            non_nan_locs = np.argwhere(~np.isnan(meas))
            all_replace = np.all(np.isnan(meas))
        else:
            non_nan_locs = np.argwhere(meas != self.nodata_value)
            all_replace = np.all(meas == self.nodata_value)

        if all_replace:
            return meas, 10

        elif len(non_nan_locs) != np.prod(meas.shape):
            interp = scipy.interpolate.interp1d(
                np.squeeze(self.wl[non_nan_locs]),
                np.squeeze(meas[non_nan_locs]),
                kind="linear",
                fill_value="extrapolate",
            )
            return interp(self.wl), 1
        else:
            return meas, 0

    def interpolate_chunk(self, startstop):
        lstart, lend = startstop

        logging.info(f"{lstart}: starting")
        img_mm = self.in_img.open_memmap(interleave="bip", writable=False)

        # Set up output
        output_state = (
            np.ones((lend - lstart, img_mm.shape[1], img_mm.shape[2]))
            * self.nodata_value
        )

        # Check if chunk is only null data
        use = np.logical_not(
            np.isclose(np.array(img_mm[lstart:lend, :, :]), self.nodata_value)
        )
        if np.sum(use) == 0:
            write_bil_chunk(
                output_state.T,
                self.outfile,
                lstart,
                (self.nl, self.nb, self.ns),
            )
            logging.debug(f"{lstart}: No non null data present, continuing")
            return 0

        # Iterate through the chunk
        for r in range(lstart, lend):
            for c in range(img_mm.shape[1]):
                meas = img_mm[r, c, :]
                # Replace no data flags
                meas_interp, exit_code_nodata = self.interpolate_values(meas)

                # Replace nans if there is both no data and nan
                if np.any(np.isnan(meas)):
                    meas_interp, exit_code_nan = self.interpolate_values(
                        meas, replace_nan=True
                    )
                else:
                    exit_code_nan = 0

                if exit_code_nodata + exit_code_nan >= 10:
                    logging.info(
                        f"Entire pixel is NaN or no data leaving as-is: {r, c}"
                    )
                elif exit_code_nodata + exit_code_nan >= 1:
                    logging.info(f"Interpolated NaN or no data in pixel: {r, c}")

                output_state[r - lstart, c, :] = meas_interp

            write_bil_chunk(
                output_state[r - lstart, ...].T,
                self.outfile,
                r,
                (self.nl, self.nb, self.ns),
            )

        return 0


def interpolate_spectra(
    infile: str,
    outfile: str = "",
    inplace: bool = False,
    nodata_value: float = -9999.0,
    n_cores: int = -1,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir: str = None,
    ray_ip_head=None,
    task_inflation_factor: int = 1,
    logfile: str = None,
    loglevel: str = "INFO",
):
    """\
    Interpolate wavelength bands that are either no data or Nan.
    The interpolation will only be applied to pixel-vectors that include partial NaNs.
    This is meant to be used if the number of wavelengths missing is minor, and has not
    been widely tested if a large number of wavelength vlues are missing.

    The interpolation will do two checks. One for "nodata values," the
    other for NaN values. Motivated by some sensor products which have rdn data 
    with both no data, and NaN values.

    \b
    Parameters
    ----------
    infile: str
        Input file that contains the wavelengths to be interpolated.
    inplace: bool
        Flag to tell algorithm to write to new file (False) or write to
        input file (True)
    outfile: str
        Output lcoation for the interpolated wavelengths
    nodata_value: float
        No data value to check against, and interpolate across
        Flexible typing in numpy boolean operations means this could be 
        float or int
    n_cores: int
        Number of cores to run. Substantial parallelism is available
        Defaults to maxing this out on the available system (-1)
    logfile: str
        File path to write logs to
    loglevel: str
        Logging level with which to run ISOFIT
    """
    # Get size of the image to interpolate
    ds = envi.open(envi_header(infile))
    ds_shape = ds.shape
    del ds

    # If you want to make a new file
    if not inplace:
        if not outfile:
            raise ValueError("Interpolation to write to new file, but no path given.")

        output_metadata = envi.open(envi_header(infile)).metadata
        img = envi.create_image(
            envi_header(outfile),
            ext="",
            metadata=output_metadata,
            force=True,
        )
        del img, output_metadata
    else:
        outfile = infile

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    # Define ray dict in the same way as extractions.py
    ray_dict = {
        "ignore_reinit_error": True,
        "address": ray_address,
        "include_dashboard": False,
        "_temp_dir": ray_temp_dir,
        "_redis_password": ray_redis_password,
    }
    if ray_ip_head is None and ray_redis_password is None:
        ray_dict["num_cpus"] = n_cores

    ray.init(**ray_dict)

    # Initialize workers
    n_workers = n_cores
    wargs = [
        ray.put(obj)
        for obj in (
            infile,
            outfile,
            inplace,
            nodata_value,
            logfile,
            loglevel,
        )
    ]
    workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

    # Assign chunks to each worker
    line_breaks = np.linspace(
        0, ds_shape[0], n_workers * task_inflation_factor, dtype=int
    )
    line_breaks = [
        (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
    ]

    start_time = time.time()
    res = list(
        workers.map_unordered(lambda a, b: a.interpolate_chunk.remote(b), line_breaks)
    )
    total_time = time.time() - start_time

    logging.info(
        f"Interpolations complete.  {round(total_time,2)}s total, "
        f"{round(ds_shape[0]*ds_shape[1]/total_time,4)} spectra/s, "
        f"{round(ds_shape[0]*ds_shape[1]/total_time/n_workers,4)} spectra/s/core"
    )


# Input arguments
@click.command(
    name="interpolate_spectra", help=interpolate_spectra.__doc__, no_args_is_help=True
)
@click.argument("infile")
@click.option("--outfile")
@click.option("--inplace", is_flag=True, default=False)
@click.option("--nodata_value", default=-9999)
@click.option("--n_cores", default=-1)
@click.option("--logfile")
@click.option("--loglevel")
def cli(**kwargs):

    interpolate_spectra(**kwargs)
    click.echo("Done")
