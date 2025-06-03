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

import click
import numpy as np
import ray
from spectral.io import envi

import isofit.utils.template_construction as tmpl
from isofit.core import isofit
from isofit.core.common import envi_header
from isofit.utils import analytical_line as ALAlg
from isofit.utils import empirical_line as ELAlg
from isofit.utils import extractions, interpolate_spectra, segment

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
    "oci",
    "tanager",
]
RTM_CLEANUP_LIST = ["*r_k", "*t_k", "*tp7", "*wrn", "*psc", "*plt", "*7sc", "*acd"]
INVERSION_WINDOWS = [[350.0, 1360.0], [1410, 1800.0], [1970.0, 2500.0]]


def apply_oe(
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
    aerosol_climatology_path=None,
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
    empirical_line=False,
    analytical_line=False,
    ray_temp_dir="/tmp/ray",
    emulator_base=None,
    segmentation_size=40,
    num_neighbors=[],
    atm_sigma=[2],
    pressure_elevation=False,
    prebuilt_lut=None,
    no_min_lut_spacing=False,
    inversion_windows=None,
    config_only=False,
    interpolate_bad_rdn=False,
    interpolate_inplace=False,
):
    """\
    Applies OE over a flightline using a radiative transfer engine. This executes
    ISOFIT in a generalized way, accounting for the types of variation that might be
    considered typical.

    Observation (obs) and location (loc) files are used to determine appropriate
    geometry lookup tables and provide a heuristic means of determining atmospheric
    water ranges.

    \b
    Parameters
    ----------
    input_radiance : str
        Radiance data cube. Expected to be ENVI format
    input_loc : str
        Location data cube of shape (Lon, Lat, Elevation). Expected to be ENVI format
    input_obs : str
        Observation data cube of shape:
            (path length, to-sensor azimuth, to-sensor zenith,
            to-sun azimuth, to-sun zenith, phase,
            slope, aspect, cosine i, UTC time)
        Expected to be ENVI format
    working_directory : str
        Directory to stage multiple outputs, will contain subdirectories
    sensor : str
        The sensor used for acquisition, will be used to set noise and datetime
        settings
    surface_path : str
        Path to surface model or json dict of surface model configuration
    copy_input_files : bool, default=False
        Flag to choose to copy input_radiance, input_loc, and input_obs locally into
        the working_directory
    modtran_path : str, default=None
        Location of MODTRAN utility. Alternately set with `MODTRAN_DIR` environment
        variable
    wavelength_path : str, default=None
        Location to get wavelength information from, if not specified the radiance
        header will be used
    surface_category : str, default="multicomponent_surface"
        The type of ISOFIT surface priors to use.  Default is multicomponent_surface
    aerosol_climatology_path : str, default=None
        Specific aerosol climatology information to use in MODTRAN
    rdn_factors_path : str, default=None
        Specify a radiometric correction factor, if desired
    atmosphere_type : str, default="ATM_MIDLAT_SUMMER"
        Atmospheric profile to be used for MODTRAN simulations.  Unused for other
        radiative transfer models.
    channelized_uncertainty_path : str, default=None
        Path to a channelized uncertainty file
    model_discrepancy_path : str, default=None
        Modifies S_eps in the OE formalism as the Gamma additive term, as:
        S_eps = Sy + Kb.dot(self.Sb).dot(Kb.T) + Gamma
    lut_config_file : str, default=None
        Path to a look up table configuration file, which will override defaults
        choices
    multiple_restarts : bool, default=False
        Use multiple initial starting poitns for each OE ptimization run, using
        the corners of the atmospheric variables as starting points.  This gives
        a more robust, albeit more expensive, solution.
    logging_level : str, default="INFO"
        Logging level with which to run ISOFIT
    log_file : str, default=None
        File path to write ISOFIT logs to
    n_cores : int, default=1
        Number of cores to run ISOFIT with. Substantial parallelism is available, and
        full runs will be very slow in serial. Suggested to max this out on the
        available system
    presolve : int, default=False
        Flag to use a presolve mode to estimate the available atmospheric water range.
        Runs a preliminary inversion over the image with a 1-D LUT of water vapor, and
        uses the resulting range (slightly expanded) to bound determine the full LUT.
        Advisable to only use with small cubes or in concert with the empirical_line
        setting, or a significant speed penalty will be incurred
    empirical_line : bool, default=False
        Use an empirical line interpolation to run full inversions over only a subset
        of pixels, determined using a SLIC superpixel segmentation, and use a KDTREE of
        local solutions to interpolate radiance->reflectance. Generally a good option
        if not trying to analyze the atmospheric state at fine scale resolution.
        Mutually exclusive with analytical_line
    analytical_line : bool, default=False
        Use an analytical solution to the fixed atmospheric state to solve for each
        pixel.  Starts by running a full OE retrieval on each SLIC superpixel, then
        interpolates the atmospheric state to each pixel, and closes with the
        analytical solution.
        Mutually exclusive with empirical_line
    ray_temp_dir : str, default="/tmp/ray"
        Location of temporary directory for ray parallelization engine
    emulator_base : str, default=None
        Location of emulator base path. Point this at the model folder (or h5 file) of
        sRTMnet to use the emulator instead of MODTRAN. An additional file with the
        same basename and the extention _aux.npz must accompany
        e.g. /path/to/emulator.h5 /path/to/emulator_aux.npz
    segmentation_size : int, default=40
        If empirical_line is enabled, sets the size of segments to construct
    num_neighbors : list[int], default=[]
        Forced number of neighbors for empirical line extrapolation - overides default
        set from segmentation_size parameter
    atm_sigma : list[int], default=[2]
        A list of smoothing factors to use during the atmospheric interpolation, one
        for each atmospheric parameter (or broadcast to all if only one is provided).
        Only used with the analytical line.
    pressure_elevation : bool, default=False
        Flag to retrieve elevation
    prebuilt_lut : str, default=None
        Use this pre-constructed look up table for all retrievals. Must be an
        ISOFIT-compatible RTE NetCDF
    no_min_lut_spacing : bool, default=False
        Don't allow the LUTConfig to remove a LUT dimension because of minimal spacing.
    inversion_windows : list[float], default=None
        Override the default inversion windows.  Will supercede any sensor specific
        defaults that are in place.
        Must be in 2-item tuples
    config_only : bool, default=False
        Generates the configuration then exits before execution. If presolve is
        enabled, that run will still occur.
    interpolate_bad_rdn : bool, default=False
        Flag to perform a per-pixel interpolation across no-data and NaN data bands.
        Does not interpolate vectors that are entire no-data or NaN, only partial.
        Currently only designed for wavelength interpolation on spectra.
        Does NOT do any spatial interpolation
    interpolate_inplace : bool, default=False
        Flag to tell interpolation to work on the file in place, or generate a
        new interpolated rdn file. The location of the new file will be in the
        "input" directory within the working directory.

    \b
    References
    ----------
    D.R. Thompson, A. Braverman,P.G. Brodrick, A. Candela, N. Carbon, R.N. Clark,D. Connelly, R.O. Green, R.F.
    Kokaly, L. Li, N. Mahowald, R.L. Miller, G.S. Okin, T.H.Painter, G.A. Swayze, M. Turmon, J. Susilouto, and
    D.S. Wettergreen. Quantifying Uncertainty for Remote Spectroscopy of Surface Composition. Remote Sensing of
    Environment, 2020. doi: https://doi.org/10.1016/j.rse.2020.111898.

    \b
    sRTMnet emulator:
    P.G. Brodrick, D.R. Thompson, J.E. Fahlen, M.L. Eastwood, C.M. Sarture, S.R. Lundeen, W. Olson-Duvall,
    N. Carmon, and R.O. Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance
    retrievals. Remote Sensing of Environment, 261:112476, 2021.doi: 10.1016/j.rse.2021.112476.
    """
    use_superpixels = empirical_line or analytical_line

    # Determine if we run in multipart-transmittance (4c) mode
    if emulator_base is not None:
        if emulator_base.endswith(".jld2"):
            multipart_transmittance = False
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

    if num_neighbors is not None and len(num_neighbors) > 1:
        if not analytical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, --analytical_line must be True"
            )
        if empirical_line:
            raise ValueError(
                "If num_neighbors has multiple elements, only --analytical_line is valid"
            )

    if os.path.isdir(working_directory) is False:
        os.mkdir(working_directory)

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s || %(filename)s:%(funcName)s() | %(message)s",
        level=logging_level,
        filename=log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

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

    lut_params = tmpl.LUTConfig(lut_config_file, emulator_base, no_min_lut_spacing)

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

    # Based on the sensor type, get appropriate year/month/day info from initial condition.
    # We'll adjust for line length and UTC day overrun later
    global INVERSION_WINDOWS
    if sensor == "ang":
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "av3":
        # parse flightline ID (AVIRIS-3 assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
        INVERSION_WINDOWS = [[380.0, 1350.0], [1435, 1800.0], [1970.0, 2500.0]]
    elif sensor == "av5":
        # parse flightline ID (AVIRIS-5 assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "avcl":
        # parse flightline ID (AVIRIS-Classic assumptions)
        dt = datetime.strptime("20{}t000000".format(paths.fid[1:7]), "%Y%m%dt%H%M%S")
    elif sensor == "emit":
        # parse flightline ID (EMIT assumptions)
        dt = datetime.strptime(paths.fid[:19], "emit%Y%m%dt%H%M%S")
        INVERSION_WINDOWS = [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]]
    elif sensor == "enmap":
        # parse flightline ID (EnMAP assumptions)
        dt = datetime.strptime(paths.fid[:15], "%Y%m%dt%H%M%S")
    elif sensor == "hyp":
        # parse flightline ID (Hyperion assumptions)
        dt = datetime.strptime(paths.fid[10:17], "%Y%j")
    elif sensor == "neon":
        # parse flightline ID (NEON assumptions)
        dt = datetime.strptime(paths.fid, "NIS01_%Y%m%d_%H%M%S")
    elif sensor == "prism":
        # parse flightline ID (PRISM assumptions)
        dt = datetime.strptime(paths.fid[3:], "%Y%m%dt%H%M%S")
    elif sensor == "prisma":
        # parse flightline ID (PRISMA assumptions)
        dt = datetime.strptime(paths.fid, "%Y%m%d%H%M%S")
    elif sensor == "gao":
        # parse flightline ID (GAO/CAO assumptions)
        dt = datetime.strptime(paths.fid[3:-5], "%Y%m%dt%H%M%S")
    elif sensor == "oci":
        # parse flightline ID (PACE OCI assumptions)
        dt = datetime.strptime(paths.fid[9:24], "%Y%m%dT%H%M%S")
    elif sensor == "tanager":
        # parse flightline ID (Tanager assumptions)
        dt = datetime.strptime(paths.fid[:15], "%Y%m%d_%H%M%S")
    elif sensor[:3] == "NA-":
        dt = datetime.strptime(sensor[3:], "%Y%m%d")
    else:
        raise ValueError(
            "Datetime object could not be obtained. Please check file name of input"
            " data."
        )

    if inversion_windows:
        assert all(
            [len(window) == 2 for window in inversion_windows]
        ), "Inversion windows must be in pairs"
        INVERSION_WINDOWS = inversion_windows
    logging.info(f"Using inversion windows: {INVERSION_WINDOWS}")

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
    if wavelength_path:
        if os.path.isfile(wavelength_path):
            chn, wl, fwhm = np.loadtxt(wavelength_path).T
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
        paths.loc_working_path, lut_params, pressure_elevation=pressure_elevation
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

    if mean_altitude_km < 0:
        raise ValueError(
            "Detected sensor altitude is negative, which is very unlikely and cannot be handled by ISOFIT."
            "Please check your input files and adjust."
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

    # Interpolate bad rdn data.
    if interpolate_bad_rdn:
        # if interpolate_inplace == True,
        # paths.radiance_working_path = paths.radiance_interp_path
        interpolate_spectra(
            paths.radiance_working_path,
            paths.radiance_interp_path,
            inplace=interpolate_inplace,
            logfile=log_file,
        )
        paths.radiance_working_path = paths.radiance_interp_path

    logging.debug("Radiance working path:")
    logging.debug(paths.radiance_working_path)
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
                segsize=segmentation_size,
                nchunk=CHUNKSIZE,
                n_cores=n_cores,
                loglevel=logging_level,
                logfile=log_file,
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
                    n_cores=n_cores,
                    loglevel=logging_level,
                    logfile=log_file,
                )

    if presolve:
        # write modtran presolve template
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
            output_file=paths.h2o_template_path,
            ihaze_type="AER_NONE",
        )

        if emulator_base is None and prebuilt_lut is None:
            max_water = tmpl.calc_modtran_max_water(paths)
        else:
            max_water = 6

        # run H2O grid as necessary
        if not exists(envi_header(paths.h2o_subs_path)) or not exists(
            paths.h2o_subs_path
        ):
            # Write the presolve connfiguration file
            h2o_grid = np.linspace(0.2, max_water - 0.01, 10).round(2)
            logging.info(f"Pre-solve H2O grid: {h2o_grid}")
            logging.info("Writing H2O pre-solve configuration file.")
            tmpl.build_presolve_config(
                paths=paths,
                h2o_lut_grid=h2o_grid,
                n_cores=n_cores,
                use_emp_line=use_superpixels,
                surface_category=surface_category,
                emulator_base=emulator_base,
                uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
                prebuilt_lut_path=prebuilt_lut,
                inversion_windows=INVERSION_WINDOWS,
                multipart_transmittance=multipart_transmittance,
            )

            # Run modtran retrieval
            logging.info("Run ISOFIT initial guess")
            retrieval_h2o = isofit.Isofit(
                paths.h2o_config_path,
                level="INFO",
                logfile=log_file,
            )
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if emulator_base is None:
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
            n_cores=n_cores,
            surface_category=surface_category,
            emulator_base=emulator_base,
            uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
            multiple_restarts=multiple_restarts,
            segmentation_size=segmentation_size,
            pressure_elevation=pressure_elevation,
            prebuilt_lut_path=prebuilt_lut,
            inversion_windows=INVERSION_WINDOWS,
            multipart_transmittance=multipart_transmittance,
        )

        if config_only:
            logging.info("`config_only` enabled, exiting early")
            return

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

    if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
        # Determine the number of neighbors to use.  Provides backwards stability and works
        # well with defaults, but is arbitrary
        if not num_neighbors:
            nneighbors = [int(round(3950 / 9 - 35 / 36 * segmentation_size))]
        else:
            nneighbors = [n for n in num_neighbors]

        if empirical_line:
            # Empirical line
            logging.info("Empirical line inference")
            ELAlg(
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
                n_cores=n_cores,
            )
        elif analytical_line:
            logging.info("Analytical line inference")
            ALAlg(
                paths.radiance_working_path,
                paths.loc_working_path,
                paths.obs_working_path,
                working_directory,
                output_rfl_file=paths.rfl_working_path,
                output_unc_file=paths.uncert_working_path,
                loglevel=logging_level,
                logfile=log_file,
                n_atm_neighbors=nneighbors,
                n_cores=n_cores,
                smoothing_sigma=atm_sigma,
            )

    logging.info("Done.")
    ray.shutdown()


# Input arguments
@click.command(name="apply_oe", help=apply_oe.__doc__, no_args_is_help=True)
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
@click.option("--no_min_lut_spacing", is_flag=True, default=False)
@click.option("--inversion_windows", type=float, nargs=2, multiple=True, default=None)
@click.option("--config_only", is_flag=True, default=False)
@click.option("--interpolate_bad_rdn", is_flag=True, default=False)
@click.option("--interpolate_inplace", is_flag=True, default=False)
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

        apply_oe(**kwargs)

        if profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.dump_stats(profile)

    print("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "apply_oe.py can no longer be called this way.  Run as:\n isofit apply_oe [ARGS]"
    )
