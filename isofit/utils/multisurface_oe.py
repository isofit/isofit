#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#

import logging
import multiprocessing
import os
from collections import OrderedDict
from datetime import datetime
from os.path import exists, join
from shutil import copyfile
from types import SimpleNamespace

import click
import numpy as np
import yaml
from spectral.io import envi

import isofit.utils.template_construction as tc
from isofit.core import isofit
from isofit.core.common import envi_header
from isofit.utils import (
    analytical_line,
    empirical_line,
    extractions,
    segment,
    surface_model,
)


def multisurface_oe(args):
    """
    TODO
    """
    infiles = {
        "input_radiance": args.input_radiance,
        "input_loc": args.input_loc,
        "input_obs": args.input_obs,
        "config_file": args.config_file,
    }

    # Check files exist
    for infile_name, infile in infiles.items():
        if os.path.isfile(infile) is False:
            err_str = (
                f"Input argument {infile_name} give as: {infile}.  File not found on"
                " system."
            )
            raise ValueError("argument " + err_str)

    # Check file sizes match
    rdn_dataset = envi.open(envi_header(args.input_radiance), args.input_radiance)
    rdn_size = rdn_dataset.shape[:2]

    for infile_name in ["input_loc", "input_obs"]:
        input_dataset = envi.open(
            envi_header(infiles[infile_name]), infiles[infile_name]
        )
        input_size = input_dataset.shape[:2]
        if not (input_size[0] == rdn_size[0] and input_size[1] == rdn_size[1]):
            err_str = (
                f"Input file: {infile_name} size is {input_size}, "
                f"which does not match input_radiance size: {rdn_size}"
            )
            raise ValueError(err_str)

    # load options from config file
    with open(args.config_file, "r") as f:
        config = OrderedDict(yaml.safe_load(f))

    opt = config["general_options"]
    gip = config["processors"]["general_inversion_parameters"]
    tsip = config["processors"]["type_specific_inversion_parameters"]
    surface_macro_config = config["surface"]

    use_superpixels = opt["empirical_line"] or opt["analytical_line"]

    # set up logger
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=args.logging_level,
        filename=args.log_file,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.info(args)

    # Determine FID based on sensor name
    # Based on the sensor type, get appropriate year/month/day info for initial condition.
    # We"ll adjust for line length and UTC day overrun later
    if opt["sensor"] == "ang":
        fid = os.path.split(args.input_radiance)[-1][:18]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
    elif opt["sensor"] == "avcl":
        fid = os.path.split(args.input_radiance)[-1][:16]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (AVIRIS-CL assumptions)
        dt = datetime.strptime("20{}t000000".format(fid[1:7]), "%Y%m%dt%H%M%S")
    elif opt["sensor"] == "prism":
        fid = os.path.split(args.input_radiance)[-1][:18]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (PRISM assumptions)
        dt = datetime.strptime(fid[3:], "%Y%m%dt%H%M%S")
    elif opt["sensor"] == "neon":
        fid = os.path.split(args.input_radiance)[-1][:21]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (NEON assumptions)
        dt = datetime.strptime(fid, "NIS01_%Y%m%d_%H%M%S")
    elif opt["sensor"] == "emit":
        fid = os.path.split(args.input_radiance)[-1][:19]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (EMIT assumptions)
        dt = datetime.strptime(fid[:19], "emit%Y%m%dt%H%M%S")
        gip["options"]["inversion_windows"] = [
            [380.0, 1270.0],
            [1410, 1800.0],
            [1970.0, 2500.0],
        ]
    elif opt["sensor"][:3] == "NA-":
        fid = os.path.splitext(os.path.basename(args.input_radiance))[0]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (PRISMA assumptions)
        dt = datetime.strptime(args.sensor[3:], "%Y%m%d")
    elif opt["sensor"] == "hyp":
        fid = os.path.split(args.input_radiance)[-1][:22]
        logging.info("Flightline ID: %s" % fid)
        # parse flightline ID (Hyperion assumptions)
        dt = datetime.strptime(fid[10:17], "%Y%j")
    else:
        raise ValueError(
            "Neither flight line ID nor datetime object could be obtained. "
            "Please provide valid sensor name in config file "
            "(choose from 'ang', 'avcl', 'prism', 'neon', 'emit', 'NA-*', 'hyp')."
        )

    # get path names
    paths = tc.Pathnames(opt=opt, gip=gip, args=args, fid=fid)

    # build subdirectories for surface-specific in- and output files
    surface_types = ["base"]
    paths.add_surface_subs_files(surface_type=surface_types[0])

    if len(tsip.items()) > 0:
        for st in tsip.keys():
            surface_types.append(st)
            paths.add_surface_subs_files(surface_type=st)

    paths.make_directories(surface_types=surface_types)

    # get wavelengths and fwhm
    try:
        chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
        if len(chn) != rdn_dataset.shape[2]:
            logging.warning(
                "Number of channels in provided wavelength file does not match"
                " wavelengths in radiance cube. Adopting center wavelengths from ENVI"
                " header."
            )
            raise ValueError
    except ValueError:
        wl = np.array(rdn_dataset.metadata["wavelength"], dtype=float)
        fwhm = np.array(rdn_dataset.metadata["fwhm"], dtype=float)

    del rdn_dataset

    # Convert to microns if needed
    if wl[0] > 100:
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    wl_data = np.concatenate(
        [np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]],
        axis=1,
    )
    np.savetxt(paths.wavelength_path, wl_data, delimiter=" ")

    # get LUT parameters
    lut_params = tc.LUTConfig(
        gip=gip, tsip=tsip, lut_config_file=gip["filepaths"]["lut_config_path"]
    )

    if gip["filepaths"]["emulator_base"] is not None:
        lut_params.aot_550_range = lut_params.aerosol_2_range
        lut_params.aot_550_spacing = lut_params.aerosol_2_spacing
        lut_params.aot_550_spacing_min = lut_params.aerosol_2_spacing_min
        lut_params.aerosol_2_spacing = 0

    # get surface model, rebuild if needed
    if gip["filepaths"]["surface_path"]:
        pass
    else:
        logging.info(
            "No surface model defined. Build new one including each given 'source'"
            " (i.e., spectral library)."
        )
        tc.build_surface_config(
            macro_config=surface_macro_config,
            flight_id=fid,
            output_path=paths.data_directory,
            wvl_file=paths.wavelength_path,
        )
        config_path = os.path.join(paths.data_directory, fid + "_surface.json")
        # isofit file should live at isofit/isofit/core/isofit.py
        isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(isofit.__file__)))

        for source in surface_macro_config["sources"]:
            for file in [
                source["input_spectrum_files"][0],
                source["input_spectrum_files"][0] + ".hdr",
            ]:
                copyfile(
                    os.path.abspath(
                        os.path.join(isofit_path, "data", "reflectance", file)
                    ),
                    os.path.abspath(os.path.join(paths.data_directory, file)),
                )

        surface_model(config_path=config_path)

    paths.stage_files()

    dayofyear = dt.timetuple().tm_yday

    (
        h_m_s,
        day_increment,
        mean_path_km,
        mean_to_sensor_azimuth,
        mean_to_sensor_zenith,
        valid,
        to_sensor_azimuth_lut_grid,
        to_sensor_zenith_lut_grid,
    ) = tc.get_metadata_from_obs(obs_file=paths.obs_working_path, lut_params=lut_params)

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

    (
        mean_latitude,
        mean_longitude,
        mean_elevation_km,
        elevation_lut_grid,
    ) = tc.get_metadata_from_loc(
        loc_file=paths.loc_working_path, gip=gip, tsip=tsip, lut_params=lut_params
    )

    if gip["filepaths"]["emulator_base"] is not None:
        if elevation_lut_grid is not None and np.any(elevation_lut_grid < 0):
            to_rem = elevation_lut_grid[elevation_lut_grid < 0].copy()
            elevation_lut_grid[elevation_lut_grid < 0] = 0
            elevation_lut_grid = np.unique(elevation_lut_grid)
            logging.info(
                "Scene contains target lut grid elements < 0 km, and uses 6s (via"
                " sRTMnet). 6s does not support targets below sea level in km units."
                f" Setting grid points {to_rem} to 0."
            )

        if mean_elevation_km < 0:
            mean_elevation_km = 0
            logging.info(
                "Scene contains a mean target elevation < 0. 6s does not support"
                " targets below sea level in km units. Setting mean elevation to 0."
            )

    # Need a 180 - here, as this is already in MODTRAN convention
    mean_altitude_km = (
        mean_elevation_km
        + np.cos(np.deg2rad(180 - mean_to_sensor_zenith)) * mean_path_km
    )

    logging.info("Observation means:")
    logging.info(f"Path (km): {mean_path_km}")
    logging.info(f"To-sensor Zenith (deg): {mean_to_sensor_zenith}")
    logging.info(f"To-sensor Azimuth (deg): {mean_to_sensor_azimuth}")
    logging.info(f"Altitude (km): {mean_altitude_km}")

    if gip["filepaths"]["emulator_base"] is not None and mean_altitude_km > 99:
        logging.info(
            "Adjusting altitude to 99 km for integration with 6S, because emulator is"
            " chosen."
        )
        mean_altitude_km = 99

    # We will use the model discrepancy with covariance OR uncorrelated calibration error, but not both.
    if gip["filepaths"]["model_discrepancy_path"] is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = (
            0.01
            if not gip["options"]["uncorrelated_radiometric_uncertainty"]
            else gip["options"]["uncorrelated_radiometric_uncertainty"]
        )

    # Chunk scene => superpixel segmentation
    if use_superpixels:
        if not exists(paths.lbl_working_path) or not exists(
            paths.radiance_working_path
        ):
            logging.info("Segmenting...")
            if not opt["segmentation_size"]:
                logging.info(
                    "Segmentation size not  given in config. Setting to default value"
                    " of 400."
                )

            segment(
                spectra=(paths.radiance_working_path, paths.lbl_working_path),
                nodata_value=-9999,
                npca=5 if not opt["n_pca"] else opt["n_pca"],
                segsize=400
                if not opt["segmentation_size"]
                else opt["segmentation_size"],
                nchunk=256 if not opt["chunksize"] else opt["chunksize"],
                n_cores=multiprocessing.cpu_count()
                if not opt["n_cores"]
                else opt["n_cores"],
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
                    chunksize=256 if not opt["chunksize"] else opt["chunksize"],
                    flag=-9999,
                    n_cores=multiprocessing.cpu_count()
                    if not opt["n_cores"]
                    else opt["n_cores"],
                    loglevel=args.logging_level,
                    logfile=args.log_file,
                )

        rdnfile = paths.rdn_subs_path
        locfile = paths.loc_subs_path
        obsfile = paths.obs_subs_path
    else:
        rdnfile = paths.radiance_working_path
        locfile = paths.loc_working_path
        obsfile = paths.obs_working_path

    # Run surface type classification
    detected_surface_types = []

    if len(tsip.items()) > 0:
        logging.info("Classifying surface...")
        available_surface_types = ["base", "cloud", "water"]

        surface_type_labels = tc.define_surface_types(
            tsip=tsip,
            rdnfile=rdnfile,
            obsfile=obsfile,
            out_class_path=paths.class_subs_path,
            wl=wl,
            fwhm=fwhm,
        )

        un_surface_type_labels = np.unique(surface_type_labels)
        un_surface_type_labels = un_surface_type_labels[
            un_surface_type_labels != -1
        ].astype(int)

        for ustl in un_surface_type_labels:
            logging.info(f"Found surface type: {available_surface_types[ustl]}")
            detected_surface_types.append(available_surface_types[ustl])

        surface_types = (
            envi.open(envi_header(paths.class_subs_path))
            .open_memmap(interleave="bip")
            .copy()
        )

        # Break up input files based on surface type
        for _st, surface_type in enumerate(available_surface_types):
            if surface_type in detected_surface_types:
                paths.add_surface_subs_files(surface_type=surface_type)
                tc.copy_file_subset(
                    surface_types == _st,
                    [
                        (rdnfile, paths.surface_subs_files[surface_type]["rdn"]),
                        (locfile, paths.surface_subs_files[surface_type]["loc"]),
                        (obsfile, paths.surface_subs_files[surface_type]["obs"]),
                    ],
                )
    else:
        surface_type_labels = None

    if opt["presolve_wv"]:
        # write modtran presolve template
        tc.write_modtran_template(
            gip=gip,
            fid=paths.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            latitude=mean_latitude,
            longitude=mean_longitude,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=paths.h2o_template_path,
            ihaze_type="AER_NONE",
        )

        if gip["filepaths"]["emulator_base"] is None:
            max_water = tc.calc_modtran_max_water(paths=paths)
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
            tc.build_presolve_config(
                opt=opt,
                gip=gip,
                paths=paths,
                h2o_lut_grid=h2o_grid,
                use_superpixels=use_superpixels,
                uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
            )

            # Run modtran retrieval
            logging.info("Run ISOFIT initial guess")
            retrieval_h2o = isofit.Isofit(
                config_file=paths.h2o_config_path, level="INFO", logfile=args.log_file
            )
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if gip["filepaths"]["emulator_base"] is None:
                for to_rm in [
                    "*r_k",
                    "*t_k",
                    "*tp7",
                    "*wrn",
                    "*psc",
                    "*plt",
                    "*7sc",
                    "*acd",
                ]:
                    cmd = "rm " + join(paths.lut_h2o_directory, to_rm)
                    logging.info(cmd)
                    os.system(cmd)
        else:
            logging.info("Existing h2o-presolve solutions found, using those.")

        h2o = envi.open(envi_header(paths.h2o_subs_path))
        h2o_est = h2o.read_band(-1)[:].flatten()

        p05 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 2)
        p95 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 98)
        margin = (p95 - p05) * 0.5

        lut_params.h2o_range[0] = max(lut_params.h2o_min, p05 - margin)
        lut_params.h2o_range[1] = min(max_water, max(lut_params.h2o_min, p95 + margin))

    h2o_lut_grid = tc.get_grid(
        minval=lut_params.h2o_range[0],
        maxval=lut_params.h2o_range[1],
        spacing=lut_params.h2o_spacing,
        min_spacing=lut_params.h2o_spacing_min,
    )

    logging.info("Full (non-aerosol) LUTs:")
    logging.info(f"Elevation: {elevation_lut_grid}")
    logging.info(f"To-sensor azimuth: {to_sensor_azimuth_lut_grid}")
    logging.info(f"To-sensor zenith: {to_sensor_zenith_lut_grid}")
    logging.info(f"H2O Vapor: {h2o_lut_grid}")

    if len(detected_surface_types) == 0:
        detected_surface_types.append("base")

    logging.info(f"Surface Types: {detected_surface_types}")
    if not exists(paths.uncert_subs_path) or not exists(paths.rfl_subs_path):
        tc.write_modtran_template(
            gip=gip,
            fid=paths.fid,
            altitude_km=mean_altitude_km,
            dayofyear=dayofyear,
            latitude=mean_latitude,
            longitude=mean_longitude,
            to_sensor_azimuth=mean_to_sensor_azimuth,
            to_sensor_zenith=mean_to_sensor_zenith,
            gmtime=gmtime,
            elevation_km=mean_elevation_km,
            output_file=paths.modtran_template_path,
        )

        logging.info("Writing main configuration file.")
        for st in detected_surface_types:
            tc.build_main_config(
                opt=opt,
                gip=gip,
                tsip=tsip,
                paths=paths,
                lut_params=lut_params,
                h2o_lut_grid=h2o_lut_grid,
                elevation_lut_grid=elevation_lut_grid,
                to_sensor_azimuth_lut_grid=to_sensor_azimuth_lut_grid,
                to_sensor_zenith_lut_grid=to_sensor_zenith_lut_grid,
                mean_latitude=mean_latitude,
                mean_longitude=mean_longitude,
                dt=dt,
                use_superpixels=use_superpixels,
                uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
                surface_type=st,
            )

        # Run modtran retrieval
        for st in detected_surface_types:
            if st == "base" and len(detected_surface_types) == 1:
                logging.info("Running ISOFIT with full LUT - Universal Surface")
                retrieval_full = isofit.Isofit(
                    config_file=paths.surface_config_paths[st],
                    level="INFO",
                    logfile=args.log_file,
                )
            else:
                if os.path.isfile(paths.surface_subs_files[st]["rdn"]):
                    logging.info(f"Running ISOFIT with full LUT - Surface: {st}")
                    retrieval_full = isofit.Isofit(
                        config_file=paths.surface_config_paths[st],
                        level="INFO",
                        logfile=args.log_file,
                    )
                else:
                    continue

            retrieval_full.run()
            del retrieval_full

        # clean up unneeded storage
        if gip["filepaths"]["emulator_base"] is None:
            for to_rm in [
                "*r_k",
                "*t_k",
                "*tp7",
                "*wrn",
                "*psc",
                "*plt",
                "*7sc",
                "*acd",
            ]:
                cmd = "rm " + join(paths.lut_modtran_directory, to_rm)
                logging.info(cmd)
                os.system(cmd)

    if len(detected_surface_types) > 1:
        tc.reassemble_cube(matching_indices=surface_type_labels, paths=paths)
        stl_path = paths.class_subs_path
    else:
        stl_path = None

    if use_superpixels:
        if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
            if not opt["num_neighbors"] and opt["segmentation_size"]:
                if opt["segmentation_size"] > 441:
                    logging.info(
                        f"Segmentation size of {opt['segmentation_size']} too large"
                        " (max. allowed size: 441). Setting number of neighbors to"
                        " minimum value of 10."
                    )
                    nneighbors = 10
                else:
                    logging.info(
                        "Number of neighbors not given in config. Calculating based on"
                        " segmentation size."
                    )
                    nneighbors = int(
                        round(3950 / 9 - 35 / 36 * opt["segmentation_size"])
                    )
            elif not opt["num_neighbors"] and not opt["segmentation_size"]:
                logging.info(
                    "Neither number of neighbors nor segmentation size given in config."
                    " Setting number of neighbors to minimum value of 10."
                )
                nneighbors = 10
            else:
                nneighbors = opt["num_neighbors"]

            if opt["empirical_line"]:
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
                    isofit_config=paths.surface_config_paths["base"],
                    nneighbors=nneighbors,
                    n_cores=multiprocessing.cpu_count()
                    if not opt["n_cores"]
                    else opt["n_cores"],
                    reference_class_file=stl_path,
                )
            elif opt["analytical_line"]:
                logging.info("Analytical line inference")
                analytical_line.main(
                    [
                        paths.radiance_working_path,
                        paths.loc_working_path,
                        paths.obs_working_path,
                        args.working_directory,
                        "--isofit_config",
                        paths.surface_config_paths["base"],
                        "--segmentation_file",
                        paths.lbl_working_path,
                        "--n_atm_neighbors",
                        str(nneighbors),
                        "--n_cores",
                        str(multiprocessing.cpu_count())
                        if not opt["n_cores"]
                        else str(opt["n_cores"]),
                        "--smoothing_sigma",
                        "2",
                        "--output_rfl_file",
                        paths.rfl_working_path,
                        "--output_unc_file",
                        paths.uncert_working_path,
                        "--loglevel",
                        args.logging_level,
                        "--logfile",
                        args.log_file,
                    ]
                )

    logging.info("Done.")


@click.command(name="multisurface_oe")
@click.argument("input_radiance")
@click.argument("input_loc")
@click.argument("input_obs")
@click.argument("working_directory")
@click.argument("config_file")
@click.option("--wavelength_path")
@click.option("--log_file")
@click.option("--logging_level", default="INFO")
@click.option(
    "--pressure_elevation", type=int, default=0
)  # ("--pressure_elevation", is_flag=True)
@click.option("--debug-args", is_flag=True)
def _cli(debug_args, **kwargs):
    """\
    Apply ISOFIT to a block of data with mixed surface
    """
    click.echo("Running multisurface_oe")
    if debug_args:
        click.echo("Arguments to be passed:")
        for key, value in kwargs.items():
            click.echo(f"  {key} = {value!r}")
    else:
        # SimpleNamespace converts a dict into dot-notational
        multisurface_oe(SimpleNamespace(**kwargs))

    click.echo("Done")


if __name__ == "__main__":
    _cli()
else:
    from isofit import cli

    cli.add_command(_cli)
