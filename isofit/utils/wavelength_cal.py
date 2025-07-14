#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick
#

import logging
import os
import subprocess
from datetime import datetime
from os.path import exists, join
from shutil import copyfile

import click
import numpy as np
import ray
from spectral.io import envi

import isofit.utils.template_construction as tmpl
from isofit.core import isofit, units
from isofit.core.common import envi_header
from isofit.utils import analytical_line as ALAlg
from isofit.utils import empirical_line as ELAlg
from isofit.utils import extractions, interpolate_spectra, segment
from isofit.utils.apply_oe import (
    EPS,
    INVERSION_WINDOWS,
    RTM_CLEANUP_LIST,
    SUPPORTED_SENSORS,
    UNCORRELATED_RADIOMETRIC_UNCERTAINTY,
)


def wavelength_cal(
    input_radiance,
    input_loc,
    input_obs,
    working_directory,
    sensor,
    surface_path,
    copy_input_files=False,
    modtran_path=None,
    wavelength_path=None,
    surface_category="multicomponent_surface",
    rdn_factors_path=None,
    atmosphere_type="ATM_MIDLAT_SUMMER",
    channelized_uncertainty_path=None,
    model_discrepancy_path=None,
    lut_config_file=None,
    multiple_restarts=False,
    logging_level="INFO",
    log_file=None,
    n_cores=1,
    presolve=False,
    ray_temp_dir="/tmp/ray",
    emulator_base=None,
    prebuilt_lut=None,
    inversion_windows=None,
):

    ##################### Front Matter #########################
    # Determine if we run in multipart-transmittance (4c) mode
    if emulator_base is not None:
        if emulator_base.endswith(".jld2"):
            multipart_transmittance = False
        else:
            if emulator_base.endswith(".npz"):
                emulator_aux_file = emulator_base
            else:
                emulator_aux_file = os.path.abspath(
                    os.path.splitext(emulator_base)[0] + "_aux.npz"
                )
            aux = np.load(emulator_aux_file)
            if (
                "transm_down_dir"
                and "transm_down_dif"
                and "transm_up_dir"
                and "transm_up_dif" in aux["rt_quantities"]
            ):
                multipart_transmittance = True
            else:
                multipart_transmittance = False
    else:
        # This is the MODTRAN case. Do we want to enable the 4c mode by default?
        multipart_transmittance = True

    if sensor not in SUPPORTED_SENSORS:
        if sensor[:3] != "NA-":
            errstr = (
                f'Argument error: sensor must be one of {SUPPORTED_SENSORS} or "NA-*"'
            )
            raise ValueError(errstr)

    if os.path.isdir(working_directory) is False:
        os.mkdir(working_directory)

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s || %(filename)s:%(funcName)s() | %(message)s",
        level=logging_level,
        filename=log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    ################## Staging Setup ##########################
    logging.info("Checking input data files...")
    rdn_dataset = envi.open(envi_header(input_radiance))
    rdn_size = (rdn_dataset.shape[0], rdn_dataset.shape[1])
    del rdn_dataset
    for infile_name, infile in zip(
        ["input_radiance", "input_loc", "input_obs"],
        [input_radiance, input_loc, input_obs],
    ):
        if os.path.isfile(infile) is False:
            err_str = (
                f"Input argument {infile_name} give as: {infile}.  File not found on"
                " system."
            )
            raise ValueError("argument " + err_str)
        if infile_name != "input_radiance":
            input_dataset = envi.open(envi_header(infile), infile)
            input_size = (input_dataset.shape[0], input_dataset.shape[1])
            if not (input_size[0] == rdn_size[0] and input_size[1] == rdn_size[1]):
                err_str = (
                    f"Input file: {infile_name} size is {input_size}, which does not"
                    f" match input_radiance size: {rdn_size}"
                )
                raise ValueError(err_str)
    logging.info("...Data file checks complete")

    lut_params = tmpl.LUTConfig(lut_config_file, emulator_base, False)

    # Based on the sensor type, get appropriate year/month/day info from initial condition.
    # We'll adjust for line length and UTC day overrun later
    global INVERSION_WINDOWS
    dt, sensor_inversion_window = tmpl.sensor_name_to_dt(sensor, paths.fid)
    if sensor_inversion_window is not None:
        INVERSION_WINDOWS = sensor_inversion_window

    # Collapse data row-wise

    logging.info("Setting up files and directories....")
    paths = tmpl.Pathnames(
        input_radiance,
        input_loc,
        input_obs,
        sensor,
        surface_path,
        working_directory,
        copy_input_files,
        modtran_path,
        rdn_factors_path,
        model_discrepancy_path,
        aerosol_climatology_path,
        channelized_uncertainty_path,
        ray_temp_dir,
        interpolate_inplace,
    )
    paths.make_directories()
    paths.stage_files()
    logging.info("...file/directory setup complete")


# Input arguments
@click.command(name="apply_oe", help=wavelength_cal.__doc__, no_args_is_help=True)
@click.argument("input_radiance")
@click.argument("input_loc")
@click.argument("input_obs")
@click.argument("working_directory")
@click.argument("sensor")
@click.option("--surface_path", "-sp", required=True, type=str)
@click.option("--copy_input_files", is_flag=True, default=False)
@click.option("--modtran_path")
@click.option("--wavelength_path")
@click.option("--surface_category", default="multicomponent_surface")
@click.option("--rdn_factors_path")
@click.option("--atmosphere_type", default="ATM_MIDLAT_SUMMER")
@click.option("--channelized_uncertainty_path")
@click.option("--model_discrepancy_path")
@click.option("--lut_config_file")
@click.option("--logging_level", default="INFO")
@click.option("--log_file")
@click.option("--n_cores", type=int, default=1)
@click.option("--presolve", is_flag=True, default=False)
@click.option("--ray_temp_dir", default="/tmp/ray")
@click.option("--emulator_base")
@click.option("--prebuilt_lut", type=str)
@click.option("--inversion_windows", type=float, nargs=2, multiple=True, default=None)
@click.option(
    "--debug-args",
    help="Prints the arguments list without executing the command",
    is_flag=True,
)
@click.option("--profile")
def cli(debug_args, profile, **kwargs):
    if debug_args:
        print("Arguments to be passed:")
        for key, value in kwitems():
            print(f"  {key} = {value!r}")
    else:
        if profile:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()

        wavelength_cal(**kwargs)

        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.dump_stats(profile)

    print("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "apply_oe.py can no longer be called this way.  Run as:\n isofit apply_oe [ARGS]"
    )
