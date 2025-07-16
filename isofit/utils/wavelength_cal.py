#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick
#

import json
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


def add_wavelength_elements(config_path, state_type="shift", spline_indices=None):
    config = json.load(open(config_path, "r"))
    
    # Set RTM WL range
    hi_res_wavlengths= os.path.join(f'{os.path.dirname(config_path)}','..','data','wavelengths_highres.txt')
    x = np.arange(360,2511,0.025) / 1000.
    w = np.ones(len(x))*0.025 / 1000.0
    n = np.ones(len(x))
    D = np.c_[n,x,w]
    np.savetxt(hi_res_wavlengths, D, fmt='%8.6f')
    config['forward_model']['radiative_transfer']['radiative_transfer_engines']['vswir']['wavelength_file'] =  hi_res_wavlengths

    config['forward_model']["instrument"]['calibration_fixed'] = False
    if state_type == "shift":
        config["forward_model"]["instrument"]["statevector"] = {
            "GROW_FWHM": {
                "bounds": [-5, 5],
                "init": 0,
                "prior_mean": 0,
                "prior_sigma": 100.0,
                "scale": 1,
            },
            "WL_SHIFT": {
                "bounds": [-5, 5],
                "init": 0,
                "prior_mean": 0,
                "prior_sigma": 100.0,
                "scale": 1,
            },
        }
    elif state_type == "spline":
        config["forward_model"]["instrument"]["statevector"] = {
            "GROW_FWHM": {
                "bounds": [-5, 5],
                "init": 0,
                "prior_mean": 0,
                "prior_sigma": 100.0,
                "scale": 1,
            }
        }
        for spline_index in spline_indices:
            config["forward_model"]["instrument"]["statevector"][
                f"WLSPL_{spline_index}"
            ] = {
                "bounds": [-5, 5],
                "init": 0,
                "prior_mean": 0,
                "prior_sigma": 100.0,
                "scale": 1,
            }
    logging.info(f'Writing config update with WL cal parameters to {config_path}')
    with open(config_path, "w") as fout:
        fout.write(json.dumps(config, cls=tmpl.SerialEncoder, indent=4, sort_keys=True))


def average_columns(
    input_radiance_file: str,
    input_loc_file: str,
    input_obs_file: str,
    output_radiance_file: str,
    output_loc_file: str,
    output_obs_file: str,
):
    """
    Averages the columns of input radiance, location, and observation files
    and saves the results to output files.

    Args:
        input_radiance_file (str): Path to the input radiance file.
        input_loc_file (str): Path to the input location file.
        input_obs_file (str): Path to the input observation file.
        output_radiance_file (str): Path to the output radiance file.
        output_loc_file (str): Path to the output location file.
        output_obs_file (str): Path to the output observation file.

    Returns:
        rows (int): The number of rows in the averaged output files.
    """
    rows = 0
    for infile, outfile in zip(
        [input_radiance_file, input_loc_file, input_obs_file],
        [output_radiance_file, output_loc_file, output_obs_file],
    ):
        in_ds = envi.open(envi_header(infile), infile)
        mm = in_ds.open_memmap(interleave="bip")
        rows = mm.shape[0]
        avg = np.mean(mm, axis=0)

        meta = in_ds.metadata.copy()
        meta["lines"] = 1

        out_ds = envi.create_image(
            envi_header(outfile),
            shape=(1, avg.shape[0], avg.shape[1]),
            ext='',
            dtype=in_ds.dtype,
            interleave="bil",
            force=True,
            metadata=meta,
        )

        out_ds.open_memmap(interleave="bip", writable=True)[:] = avg
        del in_ds, out_ds
    return rows


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
        None,
        channelized_uncertainty_path,
        ray_temp_dir,
        False,
    )
    paths.make_directories()
    paths.stage_files()
    logging.info("...file/directory setup complete")

    # Based on the sensor type, get appropriate year/month/day info from initial condition.
    # We'll adjust for line length and UTC day overrun later
    global INVERSION_WINDOWS
    dt, sensor_inversion_window = tmpl.sensor_name_to_dt(sensor, paths.fid)
    if sensor_inversion_window is not None:
        INVERSION_WINDOWS = sensor_inversion_window

    # Collapse data row-wise
    logging.info("Collapsing data row-wise...")
    n_rows = average_columns(
        paths.input_radiance_file,
        paths.input_loc_file,
        paths.input_obs_file,
        paths.rdn_subs_path,
        paths.loc_subs_path,
        paths.obs_subs_path,
    )

    dayofyear = dt.timetuple().tm_yday
    (
        h_m_s,
        day_increment,
        mean_path_km,
        mean_to_sensor_azimuth,
        mean_to_sensor_zenith,
        mean_to_sun_azimuth,
        mean_to_sun_zenith,
        mean_relative_azimuth,
        valid,
        to_sensor_zenith_lut_grid,
        to_sun_zenith_lut_grid,
        relative_azimuth_lut_grid,
    ) = tmpl.get_metadata_from_obs(paths.obs_subs_path, lut_params)

    # overwrite the time in case original obs has an error in that band
    if h_m_s[0] != dt.hour and h_m_s[0] >= 24:
        h_m_s[0] = dt.hour
        logging.info(
            "UTC hour did not match start time minute. Adjusting to that value."
        )
    if h_m_s[1] != dt.minute and h_m_s[1] >= 60:
        h_m_s[1] = dt.minute
        logging.info(
            "UTC minute did not match start time minute. Adjusting to that value."
        )

    if day_increment:
        dayofyear += 1

    gmtime = float(h_m_s[0] + h_m_s[1] / 60.0)

    # get radiance file, wavelengths, fwhm
    wl, fwhm = tmpl.get_wavelengths(
        paths.radiance_working_path,
        wavelength_path,
    )

    # write wavelength file
    wl_data = np.concatenate(
        [np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]],
        axis=1,
    )
    np.savetxt(paths.wavelength_path, wl_data, delimiter=" ")

    # check and rebuild surface model if needed
    paths.surface_path = tmpl.check_surface_model(
        surface_path=surface_path, wl=wl, paths=paths
    )

    # re-stage surface model if needed
    if paths.surface_path != surface_path:
        copyfile(paths.surface_path, paths.surface_working_path)

    (
        mean_latitude,
        mean_longitude,
        mean_elevation_km,
        elevation_lut_grid,
    ) = tmpl.get_metadata_from_loc(
        paths.loc_subs_path, lut_params, pressure_elevation=False
    )

    if emulator_base is not None:
        if elevation_lut_grid is not None and np.any(elevation_lut_grid < 0):
            to_rem = elevation_lut_grid[elevation_lut_grid < 0].copy()
            elevation_lut_grid[elevation_lut_grid < 0] = 0
            elevation_lut_grid = np.unique(elevation_lut_grid)
            if len(elevation_lut_grid) == 1:
                elevation_lut_grid = None
                mean_elevation_km = elevation_lut_grid[
                    0
                ]  # should be 0, but just in case
            logging.info(
                "Scene contains target lut grid elements < 0 km, and uses 6s (via"
                " sRTMnet).  6s does not support targets below sea level in km units. "
                f" Setting grid points {to_rem} to 0."
            )
        if mean_elevation_km < 0:
            mean_elevation_km = 0
            logging.info(
                f"Scene contains a mean target elevation < 0.  6s does not support"
                f" targets below sea level in km units.  Setting mean elevation to 0."
            )

    mean_altitude_km = (
        mean_elevation_km + np.cos(np.deg2rad(mean_to_sensor_zenith)) * mean_path_km
    )

    logging.info("Observation means:")
    logging.info(f"Path (km): {mean_path_km}")
    logging.info(f"To-sensor azimuth (deg): {mean_to_sensor_azimuth}")
    logging.info(f"To-sensor zenith (deg): {mean_to_sensor_zenith}")
    logging.info(f"To-sun azimuth (deg): {mean_to_sun_azimuth}")
    logging.info(f"To-sun zenith (deg): {mean_to_sun_zenith}")
    logging.info(f"Relative to-sun azimuth (deg): {mean_relative_azimuth}")
    logging.info(f"Altitude (km): {mean_altitude_km}")

    if emulator_base is not None and mean_altitude_km > 99:
        if not emulator_base.endswith(".jld2"):
            logging.info(
                "Adjusting altitude to 99 km for integration with 6S, because emulator is"
                " chosen."
            )
            mean_altitude_km = 99

    # We will use the model discrepancy with covariance OR uncorrelated
    # Calibration error, but not both.
    if model_discrepancy_path is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = UNCORRELATED_RADIOMETRIC_UNCERTAINTY

    h2o_lut_grid = lut_params.get_grid(
        lut_params.h2o_range[0],
        lut_params.h2o_range[1],
        lut_params.h2o_spacing,
        lut_params.h2o_spacing_min,
    )

    logging.info("Full (non-aerosol) LUTs:")
    logging.info(f"Elevation: {elevation_lut_grid}")
    logging.info(f"To-sensor zenith: {to_sensor_zenith_lut_grid}")
    logging.info(f"To-sun zenith: {to_sun_zenith_lut_grid}")
    logging.info(f"Relative to-sun azimuth: {relative_azimuth_lut_grid}")
    logging.info(f"H2O Vapor: {h2o_lut_grid}")

    if (
        not exists(paths.state_subs_path)
        or not exists(paths.uncert_subs_path)
        or not exists(paths.rfl_subs_path)
    ):
        tmpl.write_modtran_template(
            atmosphere_type=atmosphere_type,
            fid=paths.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            to_sun_zenith=mean_to_sun_zenith,
            relative_azimuth=mean_relative_azimuth,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=paths.modtran_template_path,
        )

        tmpl.build_main_config(
            paths=paths,
            lut_params=lut_params,
            h2o_lut_grid=h2o_lut_grid,
            elevation_lut_grid=(
                elevation_lut_grid
                if elevation_lut_grid is not None
                else [mean_elevation_km]
            ),
            to_sensor_zenith_lut_grid=(
                to_sensor_zenith_lut_grid
                if to_sensor_zenith_lut_grid is not None
                else [mean_to_sensor_zenith]
            ),
            to_sun_zenith_lut_grid=(
                to_sun_zenith_lut_grid
                if to_sun_zenith_lut_grid is not None
                else [mean_to_sun_zenith]
            ),
            relative_azimuth_lut_grid=(
                relative_azimuth_lut_grid
                if relative_azimuth_lut_grid is not None
                else [mean_relative_azimuth]
            ),
            mean_latitude=mean_latitude,
            mean_longitude=mean_longitude,
            dt=dt,
            use_superpixels=True,
            n_cores=n_cores,
            surface_category=surface_category,
            emulator_base=emulator_base,
            uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
            multiple_restarts=multiple_restarts,
            segmentation_size=n_rows,
            pressure_elevation=False,
            prebuilt_lut_path=prebuilt_lut,
            inversion_windows=INVERSION_WINDOWS,
            multipart_transmittance=multipart_transmittance,
        )

        add_wavelength_elements(paths.isofit_full_config_path)

        # Run retrieval
        logging.info("Running ISOFIT with full LUT")
        retrieval_full = isofit.Isofit(
            paths.isofit_full_config_path, level="INFO", logfile=log_file
        )
        retrieval_full.run()
        del retrieval_full

        # clean up unneeded storage
        if emulator_base is None:
            for to_rm in RTM_CLEANUP_LIST:
                cmd = "rm " + join(paths.full_lut_directory, to_rm)
                logging.info(cmd)
                subprocess.call(cmd, shell=True)


# Input arguments
@click.command(name="wavelength_cal", help=wavelength_cal.__doc__, no_args_is_help=True)
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
