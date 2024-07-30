#! /usr/bin/env python3
#
# Authors: David R Thompson and Philip G. Brodrick
#

import logging
import os
import subprocess
from datetime import datetime
from os.path import exists, join
from shutil import copyfile
from types import SimpleNamespace

import click
import numpy as np
import ray
from spectral.io import envi

import isofit.utils.template_construction as tmpl
from isofit.core import isofit
from isofit.core.common import envi_header
from isofit.utils import analytical_line, empirical_line, extractions, segment

EPS = 1e-6
CHUNKSIZE = 256

UNCORRELATED_RADIOMETRIC_UNCERTAINTY = 0.01
SUPPORTED_SENSORS = [
    "ang",
    "avcl",
    "neon",
    "prism",
    "emit",
    "enmap",
    "hyp",
    "prisma",
    "av3",
    "gao",
]
RTM_CLEANUP_LIST = ["*r_k", "*t_k", "*tp7", "*wrn", "*psc", "*plt", "*7sc", "*acd"]
INVERSION_WINDOWS = [[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]]


# Input arguments
@click.command(name="apply_oe")
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
@click.option("--aerosol_climatology_path")
@click.option("--rdn_factors_path")
@click.option("--atmosphere_type", default="ATM_MIDLAT_SUMMER")
@click.option("--channelized_uncertainty_path")
@click.option("--model_discrepancy_path")
@click.option("--lut_config_file")
@click.option("--multiple_restarts", is_flag=True, default=False)
@click.option("--logging_level", default="INFO")
@click.option("--log_file")
@click.option("--n_cores", type=int, default=1)
@click.option("--num_cpus", type=int, default=1)
@click.option("--memory_gb", type=int, default=-1)
@click.option("--presolve", is_flag=True, default=False)
@click.option("--empirical_line", is_flag=True, default=False)
@click.option("--analytical_line", is_flag=True, default=False)
@click.option("--ray_temp_dir", default="/tmp/ray")
@click.option("--emulator_base")
@click.option("--segmentation_size", default=40)
@click.option("--num_neighbors", "-nn", type=int, multiple=True)
@click.option("--atm_sigma", "-as", type=float, multiple=True, default=[2])
@click.option("--pressure_elevation", is_flag=True, default=False)
@click.option("--prebuilt_lut", type=str)
@click.option(
    "--debug-args",
    help="Prints the arguments list without executing the command",
    is_flag=True,
)
@click.option("--profile")
def cli_apply_oe(debug_args, profile, **kwargs):
    """Apply OE to a block of data"""

    if debug_args:
        click.echo("Arguments to be passed:")
        for key, value in kwargs.items():
            click.echo(f"  {key} = {value!r}")
    else:
        if profile:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()

        # SimpleNamespace converts a dict into dot-notational
        apply_oe(SimpleNamespace(**kwargs))

        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.dump_stats(profile)

    click.echo("Done")


def apply_oe(args):
    """This is a helper script to apply OE over a flightline using the MODTRAN radiative transfer engine.

    The goal is to run isofit in a fairly 'standard' way, accounting for the types of variation that might be
    considered typical.  For instance, we use the observation (obs) and location (loc) files to determine appropriate
    MODTRAN view-angle geometry look up tables, and provide a heuristic means of determing atmospheric water ranges.

    This code also proivdes the capicity for speedup through the empirical line solution.

    Args:
        input_radiance (str): radiance data cube [expected ENVI format]
        input_loc (str): location data cube, (Lon, Lat, Elevation) [expected ENVI format]
        input_obs (str): observation data cube, (path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth,
            to-sun zenith, phase, slope, aspect, cosine i, UTC time) [expected ENVI format]
        working_directory (str): directory to stage multiple outputs, will contain subdirectories
        sensor (str): the sensor used for acquisition, will be used to set noise and datetime settings.  choices are:
            [ang, avcl, neon, prism]
        surface_path (Required, str): Path to surface model or json dict of surface model configuration.
        copy_input_files (Optional, int): flag to choose to copy input_radiance, input_loc, and input_obs locally into
            the working_directory.  0 for no, 1 for yes.  Default 0
        modtran_path (Optional, str): Location of MODTRAN utility, alternately set with MODTRAN_DIR environment variable
        wavelength_path (Optional, str): Location to get wavelength information from, if not specified the radiance
            header will be used
        surface_category (Optional, str): The type of isofit surface priors to use.  Default is multicomponent_surface
        aerosol_climatology_path (Optional, str): Specific aerosol climatology information to use in MODTRAN,
            default None
        rdn_factors_path (Optional, str): Specify a radiometric correction factor, if desired. default None
        channelized_uncertainty_path (Optional, str): path to a channelized uncertainty file.  default None
        lut_config_file (Optional, str): Path to a look up table configuration file, which will override defaults
            chocies. default None
        logging_level (Optional, str): Logging level with which to run isofit.  Default INFO
        log_file (Optional, str): File path to write isofit logs to.  Default None
        n_cores (Optional, int): Number of cores to run isofit with.  Substantial parallelism is available, and full
            runs will be very slow in serial.  Suggested to max this out on the available system.  Default 1
        presolve (Optional, int): Flag to use a presolve mode to estimate the available atmospheric water range.  Runs
            a preliminary inversion over the image with a 1-D LUT of water vapor, and uses the resulting range (slightly
            expanded) to bound determine the full LUT.  Advisable to only use with small cubes or in concert with the
            empirical_line setting, or a significant speed penalty will be incurred.  Choices - 0 off, 1 on. Default 0
        empirical_line (Optional, int): Use an empirical line interpolation to run full inversions over only a subset
            of pixels, determined using a SLIC superpixel segmentation, and use a KDTREE Of local solutions to
            interpolate radiance->reflectance.  Generally a good option if not trying to analyze the atmospheric state
            at fine scale resolution.  Choices - 0 off, 1 on.  Default 0
        ray_temp_dir (Optional, str): Location of temporary directory for ray parallelization engine.  Default is
            '/tmp/ray'
        emulator_base (Optional, str): Location of emulator base path.  Point this at the model folder (or h5 file) of
            sRTMnet to use the emulator instead of MODTRAN.  An additional file with the same basename and the extention
            _aux.npz must accompany (e.g. /path/to/emulator.h5 /path/to/emulator_aux.npz)
        segmentation_size (Optional, int): Size of segments to construct for empirical line (if used).
        num_neighbors (Optional, int): Forced number of neighbors for empirical line extrapolation - overides default
            set from segmentation_size parameter.
        pressure_elevation (Optional, bool): If set, retrieves elevation.
        prebuilt_lut (Optional, str): Use this pres-constructed look up table for all retrievals.

            Reference:
            D.R. Thompson, A. Braverman,P.G. Brodrick, A. Candela, N. Carbon, R.N. Clark,D. Connelly, R.O. Green, R.F.
            Kokaly, L. Li, N. Mahowald, R.L. Miller, G.S. Okin, T.H.Painter, G.A. Swayze, M. Turmon, J. Susilouto, and
            D.S. Wettergreen. Quantifying Uncertainty for Remote Spectroscopy of Surface Composition. Remote Sensing of
            Environment, 2020. doi: https://doi.org/10.1016/j.rse.2020.111898.

            Emulator reference:
            P.G. Brodrick, D.R. Thompson, J.E. Fahlen, M.L. Eastwood, C.M. Sarture, S.R. Lundeen, W. Olson-Duvall,
            N. Carmon, and R.O. Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance
            retrievals. Remote Sensing of Environment, 261:112476, 2021.doi: 10.1016/j.rse.2021.112476.


    Returns:

    """

    use_superpixels = args.empirical_line or args.analytical_line

    ray.init(
        num_cpus=args.n_cores,
        _temp_dir=args.ray_temp_dir,
        include_dashboard=False,
        local_mode=args.n_cores == 1,
    )

    if args.sensor not in SUPPORTED_SENSORS:
        if args.sensor[:3] != "NA-":
            errstr = (
                f'Argument error: sensor must be one of {SUPPORTED_SENSORS} or "NA-*"'
            )
            raise ValueError(errstr)

    if args.num_neighbors is not None and len(args.num_neighbors) > 1:
        if not args.analytical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, --analytical_line must be True"
            )
        if args.empirical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, only --analytical_line is valid"
            )

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s || %(filename)s:%(funcName)s() | %(message)s",
        level=args.logging_level,
        filename=args.log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.info(args)

    logging.info("Checking input data files...")
    rdn_dataset = envi.open(envi_header(args.input_radiance))
    rdn_size = (rdn_dataset.shape[0], rdn_dataset.shape[1])
    del rdn_dataset
    for infile_name, infile in zip(
        ["input_radiance", "input_loc", "input_obs"],
        [args.input_radiance, args.input_loc, args.input_obs],
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

    lut_params = tmpl.LUTConfig(args.lut_config_file, args.emulator_base)

    logging.info("Setting up files and directories....")
    paths = tmpl.Pathnames(args)
    paths.make_directories()
    paths.stage_files()
    logging.info("...file/directory setup complete")

    # Based on the sensor type, get appropriate year/month/day info from initial condition.
    # We'll adjust for line length and UTC day overrun later
    global INVERSION_WINDOWS
    if args.sensor == "ang":
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
    elif args.sensor == "av3":
        # parse flightline ID (AVIRIS-3 assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
        INVERSION_WINDOWS = [[380.0, 1350.0], [1435, 1800.0], [1970.0, 2500.0]]
    elif args.sensor == "avcl":
        # parse flightline ID (AVIRIS-Classic assumptions)
        dt = datetime.strptime("20{}t000000".format(paths.fid[1:7]), "%Y%m%dt%H%M%S")
    elif args.sensor == "emit":
        # parse flightline ID (EMIT assumptions)
        dt = datetime.strptime(paths.fid[:19], "emit%Y%m%dt%H%M%S")
        INVERSION_WINDOWS = [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]]
    elif args.sensor == "enmap":
        # parse flightline ID (EnMAP assumptions)
        dt = datetime.strptime(paths.fid[:15], "%Y%m%dt%H%M%S")
    elif args.sensor == "hyp":
        # parse flightline ID (Hyperion assumptions)
        dt = datetime.strptime(paths.fid[10:17], "%Y%j")
    elif args.sensor == "neon":
        # parse flightline ID (NEON assumptions)
        dt = datetime.strptime(paths.fid, "NIS01_%Y%m%d_%H%M%S")
    elif args.sensor == "prism":
        # parse flightline ID (PRISM assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
    elif args.sensor == "prisma":
        # parse flightline ID (PRISMA assumptions)
        dt = datetime.strptime(paths.fid, "%Y%m%d%H%M%S")
    elif args.sensor == "gao":
        # parse flightline ID (GAO/CAO assumptions)
        dt = datetime.strptime(paths.fid[3:-5], "%Y%m%dt%H%M%S")
    elif args.sensor[:3] == "NA-":
        dt = datetime.strptime(args.sensor[3:], "%Y%m%d")
    else:
        raise ValueError(
            "Datetime object could not be obtained. Please check file name of input"
            " data."
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
    ) = tmpl.get_metadata_from_obs(paths.obs_working_path, lut_params)

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
    radiance_dataset = envi.open(envi_header(paths.radiance_working_path))
    wl_ds = np.array([float(w) for w in radiance_dataset.metadata["wavelength"]])
    if args.wavelength_path:
        if os.path.isfile(args.wavelength_path):
            chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
            if len(chn) != len(wl_ds) or not np.all(np.isclose(wl, wl_ds, atol=0.01)):
                raise ValueError(
                    "Number of channels or center wavelengths provided in wavelength file do not match"
                    " wavelengths in radiance cube. Please adjust your wavelength file."
                )
        else:
            pass
    else:
        logging.info(
            "No wavelength file provided. Obtaining wavelength grid from ENVI header of radiance cube."
        )
        wl = wl_ds
        if "fwhm" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["fwhm"]])
        elif "FWHM" in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata["FWHM"]])
        else:
            fwhm = np.ones(wl.shape) * (wl[1] - wl[0])

    # Close out radiance dataset to avoid potential confusion
    del radiance_dataset

    # Convert to microns if needed
    if wl[0] > 100:
        logging.info("Wavelength units of nm inferred...converting to microns")
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # write wavelength file
    wl_data = np.concatenate(
        [np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]],
        axis=1,
    )
    np.savetxt(paths.wavelength_path, wl_data, delimiter=" ")

    # check and rebuild surface model if needed
    paths.surface_path = tmpl.check_surface_model(
        surface_path=args.surface_path, wl=wl, paths=paths
    )

    # re-stage surface model if needed
    if paths.surface_path != args.surface_path:
        copyfile(paths.surface_path, paths.surface_working_path)

    (
        mean_latitude,
        mean_longitude,
        mean_elevation_km,
        elevation_lut_grid,
    ) = tmpl.get_metadata_from_loc(
        paths.loc_working_path, lut_params, pressure_elevation=args.pressure_elevation
    )

    if args.emulator_base is not None:
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
    logging.info(f"To-sun azimuth (deg): {mean_to_sun_zenith}")
    logging.info(f"To-sun zenith (deg): {mean_to_sun_zenith}")
    logging.info(f"Relative to-sun azimuth (deg): {mean_relative_azimuth}")
    logging.info(f"Altitude (km): {mean_altitude_km}")

    if args.emulator_base is not None and mean_altitude_km > 99:
        if not args.emulator_base.endswith(".jld2"):
            logging.info(
                "Adjusting altitude to 99 km for integration with 6S, because emulator is"
                " chosen."
            )
            mean_altitude_km = 99

    # We will use the model discrepancy with covariance OR uncorrelated
    # Calibration error, but not both.
    if args.model_discrepancy_path is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = UNCORRELATED_RADIOMETRIC_UNCERTAINTY

    # Superpixel segmentation
    if use_superpixels:
        if not exists(paths.lbl_working_path) or not exists(
            paths.radiance_working_path
        ):
            logging.info("Segmenting...")
            segment(
                spectra=(paths.radiance_working_path, paths.lbl_working_path),
                nodata_value=-9999,
                npca=5,
                segsize=args.segmentation_size,
                nchunk=CHUNKSIZE,
                n_cores=args.n_cores,
                loglevel=args.logging_level,
                logfile=args.log_file,
            )

        # Extract input data per segment
        for inp, outp in [
            (paths.radiance_working_path, paths.rdn_subs_path),
            (paths.obs_working_path, paths.obs_subs_path),
            (paths.loc_working_path, paths.loc_subs_path),
        ]:
            if not exists(outp):
                logging.info("Extracting " + outp)
                extractions(
                    inputfile=inp,
                    labels=paths.lbl_working_path,
                    output=outp,
                    chunksize=CHUNKSIZE,
                    flag=-9999,
                    n_cores=args.n_cores,
                    loglevel=args.logging_level,
                    logfile=args.log_file,
                )

    if args.presolve:
        # write modtran presolve template
        tmpl.write_modtran_template(
            atmosphere_type=args.atmosphere_type,
            fid=paths.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            to_sun_zenith=mean_to_sun_zenith,
            relative_azimuth=mean_relative_azimuth,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=paths.h2o_template_path,
            ihaze_type="AER_NONE",
        )

        if args.emulator_base is None and args.prebuilt_lut is None:
            max_water = tmpl.calc_modtran_max_water(paths)
        else:
            max_water = 6

        # run H2O grid as necessary
        if not exists(envi_header(paths.h2o_subs_path)) or not exists(
            paths.h2o_subs_path
        ):
            # Write the presolve connfiguration file
            h2o_grid = np.linspace(0.01, max_water - 0.01, 10).round(2)
            logging.info(f"Pre-solve H2O grid: {h2o_grid}")
            logging.info("Writing H2O pre-solve configuration file.")
            tmpl.build_presolve_config(
                paths=paths,
                h2o_lut_grid=h2o_grid,
                n_cores=args.n_cores,
                use_emp_line=use_superpixels,
                surface_category=args.surface_category,
                emulator_base=args.emulator_base,
                uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
                prebuilt_lut_path=args.prebuilt_lut,
                inversion_windows=INVERSION_WINDOWS,
            )

            # Run modtran retrieval
            logging.info("Run ISOFIT initial guess")
            retrieval_h2o = isofit.Isofit(
                paths.h2o_config_path,
                level="INFO",
                logfile=args.log_file,
            )
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if args.emulator_base is None:
                for to_rm in RTM_CLEANUP_LIST:
                    cmd = "rm " + join(paths.lut_h2o_directory, to_rm)
                    logging.info(cmd)
                    subprocess.call(cmd, shell=True)
        else:
            logging.info("Existing h2o-presolve solutions found, using those.")

        h2o = envi.open(envi_header(paths.h2o_subs_path))
        h2o_est = h2o.read_band(-1)[:].flatten()

        p05 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 2)
        p95 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 98)
        margin = (p95 - p05) * 0.5

        lut_params.h2o_range[0] = max(lut_params.h2o_min, p05 - margin)
        lut_params.h2o_range[1] = min(max_water, max(lut_params.h2o_min, p95 + margin))

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

    logging.info(paths.state_subs_path)
    if (
        not exists(paths.state_subs_path)
        or not exists(paths.uncert_subs_path)
        or not exists(paths.rfl_subs_path)
    ):
        tmpl.write_modtran_template(
            atmosphere_type=args.atmosphere_type,
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

        logging.info("Writing main configuration file.")
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
            use_emp_line=use_superpixels,
            n_cores=args.n_cores,
            surface_category=args.surface_category,
            emulator_base=args.emulator_base,
            uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
            multiple_restarts=args.multiple_restarts,
            segmentation_size=args.segmentation_size,
            pressure_elevation=args.pressure_elevation,
            prebuilt_lut_path=args.prebuilt_lut,
            inversion_windows=INVERSION_WINDOWS,
        )

        # Run retrieval
        logging.info("Running ISOFIT with full LUT")
        retrieval_full = isofit.Isofit(
            paths.isofit_full_config_path, level="INFO", logfile=args.log_file
        )
        retrieval_full.run()
        del retrieval_full

        # clean up unneeded storage
        if args.emulator_base is None:
            for to_rm in RTM_CLEANUP_LIST:
                cmd = "rm " + join(paths.full_lut_directory, to_rm)
                logging.info(cmd)
                subprocess.call(cmd, shell=True)

    if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
        # Determine the number of neighbors to use.  Provides backwards stability and works
        # well with defaults, but is arbitrary
        if not args.num_neighbors:
            nneighbors = [int(round(3950 / 9 - 35 / 36 * args.segmentation_size))]
        else:
            nneighbors = [n for n in args.num_neighbors]

        if args.empirical_line:
            # Empirical line
            logging.info("Empirical line inference")
            empirical_line(
                reference_radiance_file=paths.rdn_subs_path,
                reference_reflectance_file=paths.rfl_subs_path,
                reference_uncertainty_file=paths.uncert_subs_path,
                reference_locations_file=paths.loc_subs_path,
                segmentation_file=paths.lbl_working_path,
                input_radiance_file=paths.radiance_working_path,
                input_locations_file=paths.loc_working_path,
                output_reflectance_file=paths.rfl_working_path,
                output_uncertainty_file=paths.uncert_working_path,
                isofit_config=paths.isofit_full_config_path,
                nneighbors=nneighbors[0],
                n_cores=args.n_cores,
            )
        elif args.analytical_line:
            logging.info("Analytical line inference")
            analytical_line(
                paths.radiance_working_path,
                paths.loc_working_path,
                paths.obs_working_path,
                args.working_directory,
                output_rfl_file=paths.rfl_working_path,
                output_unc_file=paths.uncert_working_path,
                loglevel=args.logging_level,
                logfile=args.log_file,
                n_atm_neighbors=nneighbors,
                n_cores=args.n_cores,
                smoothing_sigma=args.atm_sigma,
            )

    logging.info("Done.")
    ray.shutdown()


if __name__ == "__main__":
    raise NotImplementedError(
        "apply_oe.py can no longer be called this way.  Run as:\n isofit apply_oe [ARGS]"
    )
