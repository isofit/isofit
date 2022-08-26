#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#

import argparse
import os
from os.path import join, exists
from spectral.io import envi
import logging
from geoarray import GeoArray
import numpy as np
import yaml
from collections import OrderedDict
from datetime import datetime

from isofit.utils.template_construction import LUTConfig, Pathnames, build_presolve_config, calc_modtran_max_water,\
    define_surface_types, copy_file_subset, get_metadata_from_obs, get_metadata_from_loc, write_modtran_template,\
    get_grid, build_main_config, reassemble_cube

from isofit.core import isofit
from isofit.core.common import envi_header
from isofit.utils import segment, extractions, empirical_line


def main(rawargs=None):

    parser = argparse.ArgumentParser(description="Apply ISOFIT to a block of data with mixed surface.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('config_file', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default="INFO")
    args = parser.parse_args(rawargs)

    infiles = {'input_radiance': args.input_radiance, 'input_loc': args.input_loc, 'input_obs': args.input_obs,
               'config_file': args.config_file}

    # Check files exist
    for infile_name, infile in infiles.items():
        if os.path.isfile(infile) is False:
            err_str = f'Input argument {infile_name} give as: {infile}.  File not found on system.'
            raise ValueError('argument ' + err_str)

    # Check file sizes match
    rdn_dataset = GeoArray(args.input_radiance)
    rdn_size = rdn_dataset.shape[:2]

    for infile_name in ['input_loc', 'input_obs']:
        input_dataset = GeoArray(infiles[infile_name])
        input_size = input_dataset.shape[:2]
        if not (input_size[0] == rdn_size[0] and input_size[1] == rdn_size[1]):
            err_str = f'Input file: {infile_name} size is {input_size}, ' \
                      f'which does not match input_radiance size: {rdn_size}'
            raise ValueError(err_str)

    # load options from config file
    with open(args.config_file, 'r') as f:
        config = OrderedDict(yaml.safe_load(f))

    opt = config['general_options']
    gip = config['processors']['general_inversion_parameters']
    tsip = config['processors']['type_specific_inversion_parameters']

    # check if name of sensor is given and valid
    if opt["sensor"] not in ['ang', 'avcl', 'neon', 'prism', 'emit', 'hyp']:
        if opt["sensor"][:3] != 'NA-':
            raise ValueError('argument sensor: invalid choice: "NA-test" (choose from "ang", "avcl", "neon", "prism", '
                             '"emit", "NA-*")')

    # set up logger
    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.logging_level,
                        filename=args.log_file, datefmt='%Y-%m-%d,%H:%M:%S')

    # get LUT parameters
    lut_params = LUTConfig(gip=gip, lut_config_file=gip["filepaths"]["lut_config_path"])
    if gip["filepaths"]["emulator_base"] is not None:
        lut_params.aot_550_range = lut_params.aerosol_2_range
        lut_params.aot_550_spacing = lut_params.aerosol_2_spacing
        lut_params.aot_550_spacing_min = lut_params.aerosol_2_spacing_min
        lut_params.aerosol_2_spacing = 0

    # get path names
    paths = Pathnames(opt=opt, gip=gip, args=args)

    if len(tsip.items()) > 0:
        for st in tsip.keys():
            paths.add_surface_subs_files(surface_type=st)

    paths.make_directories()
    paths.stage_files()

    # Based on the sensor type, get appropriate year/month/day info for initial condition.
    # We'll adjust for line length and UTC day overrun later
    if opt["sensor"] == 'ang':
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(paths.fid[3:], '%Y%m%dt%H%M%S')
    elif opt["sensor"] == 'avcl':
        # parse flightline ID (AVIRIS-CL assumptions)
        dt = datetime.strptime('20{}t000000'.format(paths.fid[1:7]), '%Y%m%dt%H%M%S')
    elif opt["sensor"] == 'neon':
        dt = datetime.strptime(paths.fid, 'NIS01_%Y%m%d_%H%M%S')
    elif opt["sensor"] == 'prism':
        dt = datetime.strptime(paths.fid[3:], '%Y%m%dt%H%M%S')
    elif opt["sensor"] == 'emit':
        dt = datetime.strptime(paths.fid[:19], 'emit%Y%m%dt%H%M%S')
    elif opt["sensor"][:3] == 'NA-':
        dt = datetime.strptime(args.sensor[3:], '%Y%m%d')
    elif opt["sensor"] == 'hyp':
        dt = datetime.strptime(paths.fid[10:17], '%Y%j')
    else:
        raise ValueError('No datetime object could be obtained. Please check file name of input data.')

    dayofyear = dt.timetuple().tm_yday

    h_m_s, day_increment, mean_path_km, mean_to_sensor_azimuth, mean_to_sensor_zenith, valid, \
    to_sensor_azimuth_lut_grid, to_sensor_zenith_lut_grid = get_metadata_from_obs(obs_file=paths.obs_working_path,
                                                                                  lut_params=lut_params)

    # overwrite the time in case original obs has an error in that band
    if h_m_s[0] != dt.hour:
        h_m_s[0] = dt.hour
    if h_m_s[1] != dt.minute:
        h_m_s[1] = dt.minute

    if day_increment:
        dayofyear += 1

    gmtime = float(h_m_s[0] + h_m_s[1] / 60.)

    # get wavelengths and fwhm
    if args.wavelength_path:
        chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
        del rdn_dataset
    else:
        wl = np.array(rdn_dataset.metadata.band_meta["wavelength"])
        fwhm = np.array(rdn_dataset.metadata.band_meta["fwhm"])
        del rdn_dataset

    # Convert to microns if needed
    if wl[0] > 100:
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    wl_data = np.concatenate([np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis], fwhm[:, np.newaxis]], axis=1)
    np.savetxt(paths.wavelength_path, wl_data, delimiter=' ')

    mean_latitude, mean_longitude, mean_elevation_km, elevation_lut_grid = get_metadata_from_loc(
        loc_file=paths.loc_working_path, lut_params=lut_params)

    if gip["filepaths"]["emulator_base"] is not None:
        if elevation_lut_grid is not None and np.any(elevation_lut_grid < 0):
            to_rem = elevation_lut_grid[elevation_lut_grid < 0].copy()
            elevation_lut_grid[elevation_lut_grid < 0] = 0
            elevation_lut_grid = np.unique(elevation_lut_grid)
            logging.info("Scene contains target lut grid elements < 0 km, and uses 6s (via sRTMnet). 6s does not "
                         f"support targets below sea level in km units. Setting grid points {to_rem} to 0.")
        if mean_elevation_km < 0:
            mean_elevation_km = 0
            logging.info("Scene contains a mean target elevation < 0. 6s does not support targets below sea level in "
                         "km units. Setting mean elevation to 0.")

    # Need a 180 - here, as this is already in MODTRAN convention
    mean_altitude_km = mean_elevation_km + np.cos(np.deg2rad(180 - mean_to_sensor_zenith)) * mean_path_km

    logging.info('Observation means:')
    logging.info(f'Path (km): {mean_path_km}')
    logging.info(f'To-sensor Zenith (deg): {mean_to_sensor_zenith}')
    logging.info(f'To-sensor Azimuth (deg): {mean_to_sensor_azimuth}')
    logging.info(f'Altitude (km): {mean_altitude_km}')

    if gip["filepaths"]["emulator_base"] is not None and mean_altitude_km > 99:
        logging.info('Adjusting altitude to 99 km for integration with 6S, because emulator is chosen.')
        mean_altitude_km = 99

    # We will use the model discrepancy with covariance OR uncorrelated calibration error, but not both.
    if gip["filepaths"]["model_discrepancy_path"] is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = gip["options"]["uncorrelated_radiometric_uncertainty"]

    # Chunk scene => superpixel segmentation
    if not exists(paths.lbl_working_path) or not exists(paths.radiance_working_path):
        logging.info('Segmenting...')
        segment(spectra=(paths.radiance_working_path, paths.lbl_working_path), nodata_value=-9999, npca=5,
                segsize=opt['segmentation_size'], nchunk=opt['chunksize'], n_cores=opt["n_cores"],
                loglevel=args.logging_level, logfile=args.log_file)

    # Extract input data per segment
    for inp, outp in [(paths.radiance_working_path, paths.rdn_subs_path), (paths.obs_working_path, paths.obs_subs_path),
                      (paths.loc_working_path, paths.loc_subs_path)]:
        if not exists(outp):
            logging.info('Extracting ' + outp)
            extractions(inputfile=inp, labels=paths.lbl_working_path, output=outp, chunksize=opt["chunksize"],
                        flag=-9999, n_cores=opt["n_cores"], loglevel=args.logging_level, logfile=args.log_file)

    # Run surface type classification
    if len(tsip.items()) > 0:
        surface_type_labels = define_surface_types(tsip=tsip, rdnfile=paths.rdn_subs_path, locfile=paths.loc_subs_path,
                                                   dt=dt, out_class_path=paths.class_subs_path, wl=wl, fwhm=fwhm)
        un_surface_type_labels = np.unique(surface_type_labels)
        un_surface_type_labels = un_surface_type_labels[un_surface_type_labels != -1].astype(int)
        detected_surface_types = []

        for ustl in un_surface_type_labels:
            logging.info(f'Found surface type: {["base", "cloud", "water"][ustl]}')
            detected_surface_types.append(["base", "cloud", "water"][ustl])

        surface_types = envi.open(envi_header(paths.class_subs_path)).open_memmap(interleave='bip').copy()

        # Break up input files based on surface type
        for _st, surface_type in enumerate(list(tsip.keys())):
            if surface_type in detected_surface_types:
                paths.add_surface_subs_files(surface_type=surface_type)
                copy_file_subset(surface_types == _st + 1, [(paths.rdn_subs_path,
                                                             paths.surface_subs_files[surface_type]['rdn']),
                                                            (paths.loc_subs_path,
                                                             paths.surface_subs_files[surface_type]['loc']),
                                                            (paths.obs_subs_path,
                                                             paths.surface_subs_files[surface_type]['obs'])])

    if opt['presolve_wv']:
        # write modtran presolve template
        write_modtran_template(atmosphere_type=gip["radiative_transfer_parameters"]["atmosphere_type"], fid=paths.fid,
                               altitude_km=mean_altitude_km, dayofyear=dayofyear, latitude=mean_latitude,
                               longitude=mean_longitude, to_sensor_azimuth=mean_to_sensor_azimuth,
                               to_sensor_zenith=mean_to_sensor_zenith, gmtime=gmtime, elevation_km=mean_elevation_km,
                               output_file=paths.h2o_template_path, ihaze_type='AER_NONE')

        if gip["filepaths"]['emulator_base'] is None:
            max_water = calc_modtran_max_water(paths=paths)
        else:
            max_water = 6

        # run H2O grid as necessary
        if not exists(envi_header(paths.h2o_subs_path)) or not exists(paths.h2o_subs_path):
            # Write the presolve connfiguration file
            h2o_grid = np.linspace(0.01, max_water - 0.01, 10).round(2)
            logging.info(f'Pre-solve H2O grid: {h2o_grid}')
            logging.info('Writing H2O pre-solve configuration file.')
            build_presolve_config(opt=opt, gip=gip, paths=paths, h2o_lut_grid=h2o_grid,
                                  uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty)

            # Run modtran retrieval
            logging.info('Run ISOFIT initial guess')
            retrieval_h2o = isofit.Isofit(config_file=paths.h2o_config_path, level='INFO', logfile=args.log_file)
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if gip["filepaths"]["emulator_base"] is None:
                for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
                    cmd = 'rm ' + join(paths.lut_h2o_directory, to_rm)
                    logging.info(cmd)
                    os.system(cmd)
        else:
            logging.info('Existing h2o-presolve solutions found, using those.')

        h2o = envi.open(envi_header(paths.h2o_subs_path))
        h2o_est = h2o.read_band(-1)[:].flatten()

        p05 = np.min(h2o_est[h2o_est > lut_params.h2o_min])
        p95 = np.max(h2o_est[h2o_est > lut_params.h2o_min])
        margin = (p95-p05) * 0.25

        lut_params.h2o_range[0] = max(lut_params.h2o_min, p05 - margin)
        lut_params.h2o_range[1] = min(max_water, max(lut_params.h2o_min, p95 + margin))

    h2o_lut_grid = get_grid(minval=lut_params.h2o_range[0], maxval=lut_params.h2o_range[1],
                            spacing=lut_params.h2o_spacing, min_spacing=lut_params.h2o_spacing_min)

    logging.info('Full (non-aerosol) LUTs:')
    logging.info(f'Elevation: {elevation_lut_grid}')
    logging.info(f'To-sensor azimuth: {to_sensor_azimuth_lut_grid}')
    logging.info(f'To-sensor zenith: {to_sensor_zenith_lut_grid}')
    logging.info(f'H2O Vapor: {h2o_lut_grid}')

    surface_types = list(paths.surface_config_paths.keys())
    surface_types.append(None)

    logging.info(f"Surface Types: {surface_types}")
    if not exists(paths.uncert_subs_path) or not exists(paths.rfl_subs_path):
        write_modtran_template(atmosphere_type=gip["radiative_transfer_parameters"]["atmosphere_type"],
                               fid=paths.fid, altitude_km=mean_altitude_km, dayofyear=dayofyear,
                               latitude=mean_latitude, longitude=mean_longitude,
                               to_sensor_azimuth=mean_to_sensor_azimuth, to_sensor_zenith=mean_to_sensor_zenith,
                               gmtime=gmtime, elevation_km=mean_elevation_km,
                               output_file=paths.modtran_template_path)

        logging.info('Writing main configuration file.')
        for st in surface_types:
            build_main_config(opt=opt, gip=gip, tsip=tsip, paths=paths, lut_params=lut_params,
                              h2o_lut_grid=h2o_lut_grid, elevation_lut_grid=elevation_lut_grid,
                              to_sensor_azimuth_lut_grid=to_sensor_azimuth_lut_grid,
                              to_sensor_zenith_lut_grid=to_sensor_zenith_lut_grid, mean_latitude=mean_latitude,
                              mean_longitude=mean_longitude, dt=dt,
                              uncorrelated_radiometric_uncertainty=uncorrelated_radiometric_uncertainty,
                              surface_type=st)

        # Run modtran retrieval
        for _st, st in enumerate(surface_types):
            if st is None and _st == 0:
                logging.info('Running ISOFIT with full LUT - Universal Surface')
                retrieval_full = isofit.Isofit(config_file=paths.modtran_config_path, level='INFO',
                                               logfile=args.log_file)
            elif st is None:
                continue
            else:
                if os.path.isfile(paths.surface_subs_files[st]['rdn']):
                    logging.info(f'Running ISOFIT with full LUT - Surface: {st}')
                    retrieval_full = isofit.Isofit(config_file=paths.surface_config_paths[st], level='INFO',
                                                   logfile=args.log_file)
                else:
                    continue
            retrieval_full.run()
            del retrieval_full

        # clean up unneeded storage
        if gip["filepaths"]["emulator_base"] is None:
            for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
                cmd = 'rm ' + join(paths.lut_modtran_directory, to_rm)
                logging.info(cmd)
                os.system(cmd)

    if surface_types[0] is not None:
        reassemble_cube(matching_indices=surface_type_labels, paths=paths)
        stl_path = paths.class_subs_path
    else:
        stl_path = None

    if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
        # Empirical line
        logging.info('Empirical line inference')
        # Determine the number of neighbors to use. Provides backwards stability and works
        # well with defaults, but is arbitrary
        nneighbors = int(round(3950 / 9 - 35 / 36 * opt["segmentation_size"]))
        empirical_line(reference_radiance_file=paths.rdn_subs_path, reference_reflectance_file=paths.rfl_subs_path,
                       reference_uncertainty_file=paths.uncert_subs_path,
                       reference_locations_file=paths.loc_subs_path, segmentation_file=paths.lbl_working_path,
                       input_radiance_file=paths.radiance_working_path, input_locations_file=paths.loc_working_path,
                       output_reflectance_file=paths.rfl_working_path,
                       output_uncertainty_file=paths.uncert_working_path, isofit_config=paths.modtran_config_path,
                       nneighbors=nneighbors, reference_class_file=stl_path)

    logging.info('Done.')


if __name__ == "__main__":
    main()
