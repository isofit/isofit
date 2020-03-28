#! /usr/bin/env python3
#
# Authors: David R Thompson and Philip G. Brodrick
#

import argparse
import os
from os.path import join, exists, split, abspath
from shutil import copyfile
from datetime import datetime
from spectral.io import envi
import logging
import json
import gdal
import numpy as np
from sklearn import mixture
import scipy.linalg

from isofit.utils import segment, extractions, empirical_line
from isofit.core import isofit

EPS = 1e-6
CHUNKSIZE = 256
SEGMENTATION_SIZE = 400
NUM_INTEGRATIONS = 400

NUM_ELEV_LUT_ELEMENTS = 1
NUM_H2O_LUT_ELEMENTS = 3
NUM_TO_SENSOR_AZIMUTH_LUT_ELEMENTS = 1
NUM_TO_SENSOR_ZENITH_LUT_ELEMENTS = 1

# Setting any of these to '1' will also remove that aerosol from the statevector
NUM_AEROSOL_1_LUT_ELEMENTS = 1
NUM_AEROSOL_2_LUT_ELEMENTS = 3
NUM_AEROSOL_3_LUT_ELEMENTS = 3

AEROSOL_1_LUT_RANGE = [0.001, 0.5]
AEROSOL_2_LUT_RANGE = [0.001, 0.5]
AEROSOL_3_LUT_RANGE = [0.001, 0.5]

H2O_MIN = 0.2

# Minimum degree-spacing for zenith angle.  Overule the number of lut elements if 
# step size is smaller than this value
ZENITH_MIN_SPACING = 2

UNCORRELATED_RADIOMETRIC_UNCERTAINTY = 0.02

INVERSION_WINDOWS = [[400.0, 1300.0], [1450, 1780.0], [2050.0, 2450.0]]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('sensor', type=str, choices=['ang', 'avcl'])
    parser.add_argument('--copy_input_files', type=int, choices=[0,1], default=0)
    parser.add_argument('--h2o', action='store_true')
    parser.add_argument('--modtran_path', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--aerosol_climatology_path', type=str, default=None)
    parser.add_argument('--rdn_factors_path', type=str)
    parser.add_argument('--surface_path', type=str)
    parser.add_argument('--channelized_uncertainty_path', type=str)
    parser.add_argument('--logging_level', type=str, default="INFO")
    parser.add_argument('--log_file', type=str, default=None)

    args = parser.parse_args()

    if args.copy_input_files == 1:
        args.copy_input_files = True
    else:
        args.copy_input_files = False

    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.logging_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.logging_level, filename=args.log_file)

    paths = Pathnames(args)
    paths.make_directories()
    paths.stage_files()

    # Based on the sensor type, get appropriate year/month/day info fro intial condition.
    # We'll adjust for line length and UTC day overrun later
    if args.sensor == 'ang':
        # parse flightline ID (AVIRIS-NG assumptions)
        dt = datetime.strptime(paths.fid[3:], '%Y%m%dt%H%M%S')
        dayofyear = dt.timetuple().tm_yday
    elif args.sensor == 'avcl':
        # parse flightline ID (AVIRIS-CL assumptions)
        dt = datetime.strptime('20{}t000000'.format(paths.fid[1:7]), '%Y%m%dt%H%M%S')
        dayofyear = dt.timetuple().tm_yday


    h_m_s, day_increment, mean_path_km, mean_to_sensor_azimuth, mean_to_sensor_zenith, valid, \
    to_sensor_azimuth_lut_grid, to_sensor_zenith_lut_grid = get_metadata_from_obs(paths.obs_working_path)

    if day_increment:
        dayofyear += 1

    gmtime = float(h_m_s[0] + h_m_s[1] / 60.)

    # Superpixel segmentation
    if not exists(paths.lbl_working_path) or not exists(paths.radiance_working_path):
        logging.info('Segmenting...')
        segment(spectra=(paths.radiance_working_path, paths.lbl_working_path),
                flag=-9999, npca=5, segsize=SEGMENTATION_SIZE, nchunk=CHUNKSIZE)

    # Extract input data per segment
    for inp, outp in [(paths.radiance_working_path, paths.rdn_subs_path),
                      (paths.obs_working_path, paths.obs_subs_path),
                      (paths.loc_working_path, paths.loc_subs_path)]:
        if not exists(outp):
            logging.info('Extracting ' + outp)
            extractions(inputfile=inp, labels=paths.lbl_working_path,
                        output=outp, chunksize=CHUNKSIZE, flag=-9999)

    # get radiance file, wavelengths
    if args.wavelength_path:
        chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
    else:
        radiance_dataset = envi.open(paths.rdn_subs_path + '.hdr')
        wl = np.array([float(w) for w in radiance_dataset.metadata['wavelength']])
        if 'fwhm' in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata['fwhm']])
        else:
            fwhm = np.ones(wl.shape) * (wl[1] - wl[0])

    # Convert to microns if needed
    if wl[0] > 100:
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # write wavelength file
    wl_data = np.concatenate([np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis],
                              fwhm[:, np.newaxis]], axis=1)
    np.savetxt(paths.wavelength_path, wl_data, delimiter=' ')

    mean_latitude, mean_longitude, mean_elevation_km, elevation_lut_grid = get_metadata_from_loc(paths.loc_working_path)

    # Need a 180 - here, as this is already in MODTRAN convention
    mean_altitude_km = mean_elevation_km + np.cos(np.deg2rad(180 - mean_to_sensor_zenith)) * mean_path_km

    logging.info('Path (km): %f, 180 - To-sensor Zenith (deg): %f, To-sensor Azimuth (deg) : %f, Altitude: %6.2f km' %
                 (mean_path_km, mean_to_sensor_zenith, mean_to_sensor_azimuth, mean_altitude_km))


    if not exists(paths.h2o_subs_path + '.hdr') or not exists(paths.h2o_subs_path):

        write_modtran_template(atmosphere_type='ATM_MIDLAT_SUMMER', fid=paths.fid, altitude_km=mean_altitude_km,
                               dayofyear=dayofyear, latitude=mean_latitude, longitude=mean_longitude,
                               to_sensor_azimuth=mean_to_sensor_azimuth, to_sensor_zenith=mean_to_sensor_zenith,
                               gmtime=gmtime, elevation_km=mean_elevation_km,
                               output_file=paths.h2o_template_path, ihaze_type='AER_NONE')

        # Write the presolve connfiguration file
        logging.info('Writing H2O pre-solve configuration file.')
        build_presolve_config(paths, np.linspace(0.5, 5, 10))

        # Run modtran retrieval
        logging.info('Run ISOFIT initial guess')
        retrieval_h2o = isofit.Isofit(paths.h2o_config_path, level='DEBUG')
        retrieval_h2o.run()

        # clean up unneeded storage
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'rm ' + join(paths.lut_h2o_directory, to_rm)
            logging.info(cmd)
            os.system(cmd)

    # Extract h2o grid avoiding the zero label (periphery, bad data)
    # and outliers
    h2o = envi.open(paths.h2o_subs_path + '.hdr')
    h2o_est = h2o.read_band(-1)[:].flatten()

    h2o_lut_grid = np.linspace(np.percentile(
        h2o_est[h2o_est > H2O_MIN], 5), np.percentile(h2o_est[h2o_est > H2O_MIN], 95), NUM_H2O_LUT_ELEMENTS)

    logging.info('Full (non-aerosol) LUTs:\nElevation: {}\nTo-sensor azimuth: {}\nTo-sensor zenith: {}\nh2o-vis: {}:'.format(elevation_lut_grid, to_sensor_azimuth_lut_grid, to_sensor_zenith_lut_grid, h2o_lut_grid))

    logging.info(paths.state_subs_path)
    if not exists(paths.state_subs_path) or \
            not exists(paths.uncert_subs_path) or \
            not exists(paths.rfl_subs_path):

        write_modtran_template(atmosphere_type='ATM_MIDLAT_SUMMER', fid=paths.fid, altitude_km=mean_altitude_km,
                               dayofyear=dayofyear, latitude=mean_latitude, longitude=mean_longitude,
                               to_sensor_azimuth=mean_to_sensor_azimuth, to_sensor_zenith=mean_to_sensor_zenith,
                               gmtime=gmtime, elevation_km=mean_elevation_km, output_file=paths.modtran_template_path)

        logging.info('Writing main configuration file.')
        build_main_config(paths, h2o_lut_grid, elevation_lut_grid, to_sensor_azimuth_lut_grid,
                          to_sensor_zenith_lut_grid, mean_latitude, mean_longitude, dt)

        # Run modtran retrieval
        logging.info('Running ISOFIT with full LUT')
        retrieval_full = isofit.Isofit(paths.modtran_config_path, level='DEBUG')
        retrieval_full.run()

        # clean up unneeded storage
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'rm ' + join(paths.lut_modtran_directory, to_rm)
            logging.info(cmd)
            os.system(cmd)

    if not exists(paths.rfl_working_path) or not exists(paths.uncert_working_path):
        # Empirical line
        logging.info('Empirical line inference')
        empirical_line(reference_radiance=paths.rdn_subs_path,
                       reference_reflectance=paths.rfl_subs_path,
                       reference_uncertainty=paths.uncert_subs_path,
                       reference_locations=paths.loc_subs_path,
                       hashfile=paths.lbl_working_path,
                       input_radiance=paths.radiance_working_path,
                       input_locations=paths.loc_working_path,
                       output_reflectance=paths.rfl_working_path,
                       output_uncertainty=paths.uncert_working_path,
                       isofit_config=paths.modtran_config_path)

    logging.info('Done.')

class Pathnames():

    def __init__(self, args):

        # Determine FID based on sensor name
        if args.sensor == 'ang':
            self.fid = split(args.input_radiance)[-1][:18]
            logging.info('Flightline ID: %s' % self.fid)
        elif args.sensor == 'avcl':
            self.fid = split(args.input_radiance)[-1][:16]
            logging.info('Flightline ID: %s' % self.fid)

        # Names from inputs
        self.aerosol_climatology = args.aerosol_climatology_path
        self.input_radiance_file = args.input_radiance
        self.input_loc_file = args.input_loc
        self.input_obs_file = args.input_obs
        self.working_directory = abspath(args.working_directory)

        self.lut_modtran_directory = abspath(join(self.working_directory, 'lut_full/'))

        if args.surface_path:
            self.surface_path = args.surface_path
        else:
            self.surface_path = os.getenv('ISOFIT_SURFACE_MODEL')

        # set up some sub-directories
        self.lut_h2o_directory = abspath(join(self.working_directory, 'lut_h2o/'))
        self.config_directory = abspath(join(self.working_directory, 'config/'))
        self.data_directory = abspath(join(self.working_directory, 'data/'))
        self.input_data_directory = abspath(join(self.working_directory, 'input/'))
        self.output_directory = abspath(join(self.working_directory, 'output/'))


        # define all output names
        rdn_fname = self.fid + '_rdn'
        self.rfl_working_path = abspath(join(self.output_directory, rdn_fname.replace('_rdn', '_rfl')))
        self.uncert_working_path = abspath(join(self.output_directory, rdn_fname.replace('_rdn', '_uncert')))
        self.lbl_working_path = abspath(join(self.output_directory, rdn_fname.replace('_rdn', '_lbl')))
        self.surface_working_path = abspath(join(self.data_directory, 'surface.mat'))

        if args.copy_input_files is True:
            self.radiance_working_path = abspath(join(self.input_data_directory, rdn_fname))
            self.obs_working_path = abspath(join(self.input_data_directory, self.fid + '_obs'))
            self.loc_working_path = abspath(join(self.input_data_directory, self.fid + '_loc'))
        else:
            self.radiance_working_path = self.input_radiance_file
            self.obs_working_path = self.input_obs_file
            self.loc_working_path = self.input_loc_file

        if args.channelized_uncertainty_path:
            self.input_channelized_uncertainty_path = args.channelized_uncertainty_path
        else:
            self.input_channelized_uncertainty_path = os.getenv('ISOFIT_CHANNELIZED_UNCERTAINTY')

        self.channelized_uncertainty_working_path = abspath(join(self.data_directory, 'channelized_uncertainty.txt'))

        self.rdn_subs_path = abspath(join(self.input_data_directory, self.fid + '_subs_rdn'))
        self.obs_subs_path = abspath(join(self.input_data_directory, self.fid + '_subs_obs'))
        self.loc_subs_path = abspath(join(self.input_data_directory, self.fid + '_subs_loc'))
        self.rfl_subs_path = abspath(join(self.output_directory, self.fid + '_subs_rfl'))
        self.state_subs_path = abspath(join(self.output_directory, self.fid + '_subs_state'))
        self.uncert_subs_path = abspath(join(self.output_directory, self.fid + '_subs_uncert'))
        self.h2o_subs_path = abspath(join(self.output_directory, self.fid + '_subs_h2o'))

        self.wavelength_path = abspath(join(self.data_directory, 'wavelengths.txt'))

        self.modtran_template_path = abspath(join(self.config_directory, self.fid + '_modtran_tpl.json'))
        self.h2o_template_path = abspath(join(self.config_directory, self.fid + '_h2o_tpl.json'))

        self.modtran_config_path = abspath(join(self.config_directory, self.fid + '_modtran.json'))
        self.h2o_config_path = abspath(join(self.config_directory, self.fid + '_h2o.json'))

        if args.modtran_path:
            self.modtran_path = args.modtran_path
        else:
            self.modtran_path = os.getenv('MODTRAN_DIR')

        # isofit file should live at isofit/isofit/core/isofit.py
        self.isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(isofit.__file__)))

        if args.sensor == 'ang':
            self.noise_path = join(self.isofit_path, 'data', 'avirisng_noise.txt')
        elif args.sensor == 'avcl':
            self.noise_path = join(self.isofit_path, 'data', 'avirisc_noise.txt')
        else:
            logging.info('no noise path found, check sensor type')
            quit()

        self.aerosol_tpl_path = join(self.isofit_path, 'data', 'aerosol_template.json')
        self.rdn_factors_path = args.rdn_factors_path

    def make_directories(self):
        # create missing directories
        for dpath in [self.working_directory, self.lut_h2o_directory, self.lut_modtran_directory, self.config_directory,
                      self.data_directory, self.input_data_directory, self.output_directory]:
            if not exists(dpath):
                os.mkdir(dpath)

    def stage_files(self):
        # stage data files by copying into working directory
        files_to_stage = [(self.input_radiance_file, self.radiance_working_path, True),
                          (self.input_obs_file, self.obs_working_path, True),
                          (self.input_loc_file, self.loc_working_path, True),
                          (self.surface_path, self.surface_working_path, False)]

        if (self.input_channelized_uncertainty_path is not None):
            files_to_stage.append((self.input_channelized_uncertainty_path, self.channelized_uncertainty_working_path, False))
        else:
            self.channelized_uncertainty_working_path = None
            logging.info('No valid channelized uncertainty file found, proceeding without uncertainty')


        for src, dst, hasheader in files_to_stage:
            if not exists(dst):
                logging.info('Staging %s to %s' % (src, dst))
                copyfile(src, dst)
                if hasheader:
                    copyfile(src + '.hdr', dst + '.hdr')


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)


def load_climatology(config_path: str, latitude: float, longitude: float, acquisition_datetime: datetime, isofit_path: str):
    """ Load climatology data, based on location and configuration
    Args:
        config_path: path to the base configuration directory for isofit
        latitude: latitude to set for the segment (mean of acquisition suggested)
        longitude: latitude to set for the segment (mean of acquisition suggested)
        acquisition_datetime: datetime to use for the segment( mean of acquisition suggested)
        isofit_path: base path to isofit installation (needed for data path references)

    :Returns
        aerosol_state_vector: A dictionary that defines the aerosol state vectors for isofit
        aerosol_lut_grid: A dictionary of the aerosol lookup table (lut) grid to be explored
        aerosol_model_path: A path to the location of the aerosol model to use with MODTRAN.
    """

    aerosol_model_path = join(isofit_path, 'data', 'aerosol_model.txt')
    aerosol_state_vector = {}
    aerosol_lut_grid = {}
    aerosol_lut_ranges = [AEROSOL_1_LUT_RANGE, AEROSOL_2_LUT_RANGE, AEROSOL_3_LUT_RANGE]
    num_aerosol_lut_elements = [NUM_AEROSOL_1_LUT_ELEMENTS, NUM_AEROSOL_2_LUT_ELEMENTS, NUM_AEROSOL_3_LUT_ELEMENTS]
    for _a, alr in enumerate(aerosol_lut_ranges):
        if num_aerosol_lut_elements[_a] != 1:
            aerosol_state_vector['AERFRAC_{}'.format(_a)] = {
                "bounds": [float(alr[0]), float(alr[1])],
                "scale": 1,
                "init": float((alr[1] - alr[0]) / 10. + alr[0]),
                "prior_sigma": 10.0,
                "prior_mean": float((alr[1] - alr[0]) / 10. + alr[0])}

            aerosol_lut = np.linspace(alr[0], alr[1], num_aerosol_lut_elements[_a])
            aerosol_lut_grid['AERFRAC_{}'.format(_a)] = [float(q) for q in aerosol_lut]

    logging.info('Loading Climatology')
    # If a configuration path has been provided, use it to get relevant info
    if config_path is not None:
        month = acquisition_datetime.timetuple().tm_mon
        year = acquisition_datetime.timetuple().tm_year
        with open(config_path, 'r') as fin:
            for case in json.load(fin)['cases']:
                match = True
                logging.info('matching', latitude, longitude, month, year)
                for criterion, interval in case['criteria'].items():
                    logging.info(criterion, interval, '...')
                    if criterion == 'latitude':
                        if latitude < interval[0] or latitude > interval[1]:
                            match = False
                    if criterion == 'longitude':
                        if longitude < interval[0] or longitude > interval[1]:
                            match = False
                    if criterion == 'month':
                        if month < interval[0] or month > interval[1]:
                            match = False
                    if criterion == 'year':
                        if year < interval[0] or year > interval[1]:
                            match = False

                if match:
                    aerosol_state_vector = case['aerosol_state_vector']
                    aerosol_lut_grid = case['aerosol_lut_grid']
                    aerosol_model_path = case['aerosol_mdl_path']
                    break

    logging.info('Climatology Loaded.  Aerosol State Vector:\n{}\nAerosol LUT Grid:\n{}\nAerosol model path:{}'.format(
        aerosol_state_vector, aerosol_lut_grid, aerosol_model_path))
    return aerosol_state_vector, aerosol_lut_grid, aerosol_model_path


def find_angular_seeds(angle_data_input: np.array, num_points: int, units: str = 'd'):
    """ Find either angular data 'center points' (num_points = 1), or a lut set that spans
    angle variation in a systematic fashion.
    Args:
        angle_data_input: set of angle data to use to find center points
        num_points: the number of points to find - if 1, this will return a single point (the 'centerpoint'), if > 1,
                    this will return a numpy array spanning the specified number of poitns
        units: specifies if data are in degrees (default) or radians

    :Returns:
    """

    # Convert everything to radians so we don't have to track throughout
    if units == 'r':
        angle_data = np.rad2deg(angle_data_input)
    else:
        angle_data = angle_data_input.copy()

    spatial_data = np.hstack([np.cos(np.deg2rad(angle_data)).reshape(-1, 1),
                              np.sin(np.deg2rad(angle_data)).reshape(-1, 1)])

    # find which quadrants have data
    quadrants = np.zeros((2,2))
    if np.any(np.logical_and(spatial_data[:,0] > 0, spatial_data[:,1] > 0 )):
        quadrants[1,0] = 1
    if np.any(np.logical_and(spatial_data[:,0] > 0, spatial_data[:,1] < 0 )):
        quadrants[1,1] += 1
    if np.any(np.logical_and(spatial_data[:,0] < 0, spatial_data[:,1] > 0 )):
        quadrants[0,0] += 1
    if np.any(np.logical_and(spatial_data[:,0] < 0, spatial_data[:,1] < 0 )):
        quadrants[0,1] += 1

    # Handle the case where angles are < 180 degrees apart
    if np.sum(quadrants) < 3 and num_points != 1:
        if (np.sum(quadrants[1,:]) == 2):
            # If angles cross the 0-degree line:
            angle_spread = np.linspace(np.min(angle_data+180), np.max(angle_data+180), num_points) - 180
            return angle_spread
        else:
            # Otherwise, just space things out:
            return np.linspace(np.min(angle_data), np.max(angle_data), num_points)
    else:
        # If we're greater than 180 degree spread, there's no universal answer. Try GMM.

        if num_points == 2:
            logging.warning('2 angle interpolation selected when angle divergence > 180. '
                            'At least 3 points are recommended')

        gmm = mixture.GaussianMixture(n_components=num_points, covariance_type='full')
        gmm.fit(spatial_data)
        central_angles = np.degrees(np.arctan2(gmm.means_[:,1], gmm.means_[:,0]))
        if (num_points == 1):
            return central_angles[0]

        ca_quadrants = np.zeros((2, 2))
        if np.any(np.logical_and(gmm.means_[:, 0] > 0, gmm.means_[:, 1] > 0)):
            ca_quadrants[1, 0] = 1
        elif np.any(np.logical_and(gmm.means_[:, 0] > 0, gmm.means_[:, 1] < 0)):
            ca_quadrants[1, 1] += 1
        elif np.any(np.logical_and(gmm.means_[:, 0] < 0, gmm.means_[:, 1] > 0)):
            ca_quadrants[0, 0] += 1
        elif np.any(np.logical_and(gmm.means_[:, 0] < 0, gmm.means_[:, 1] < 0)):
            ca_quadrants[0, 1] += 1

        if np.sum(ca_quadrants) < np.sum(quadrants):
            logging.warning('GMM angles {} span {} quadrants, while data spans {} quadrants'.format(central_angles,
                            np.sum(ca_quadrants), np.sum(quadrants)))

        return central_angles


def get_metadata_from_obs(obs_file: str, trim_lines: int = 5,
                          max_flight_duration_h: int = 8, nodata_value: float = -9999):
    """ Get metadata needed for complete runs from the observation file
    (bands: path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth,
    to-sun zenith, phase, slope, aspect, cosine i, UTC time).
    Args:
        obs_file: file name to pull data from
        trim_lines: number of lines to ignore at beginning and end of file (good if lines contain values that are
                    erroneous but not nodata
        max_flight_duration_h: maximum length of the current acquisition, used to check if we've lapped a UTC day
        nodata_value: value to ignore from location file

    :Returns:
        h_m_s: list of the mean-time hour, minute, and second within the line
        increment_day: indicator of whether the UTC day has been changed since the beginning of the line time
        mean_path_km: mean distance between sensor and ground in km for good data
        mean_to_sensor_azimuth: mean to-sensor-azimuth for good data
        mean_to_sensor_zenith_rad: mean to-sensor-zenith in radians for good data
        valid: boolean array indicating which pixels were NOT nodata
        to_sensor_azimuth_lut_grid: the to-sensor azimuth angle look up table for good data
        to_sensor_zenith_lut_grid: the to-sensor zenith look up table for good data
    """
    obs_dataset = gdal.Open(obs_file, gdal.GA_ReadOnly)

    # Initialize values to populate
    valid = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize), dtype=bool)

    path_km = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    to_sensor_azimuth = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    to_sensor_zenith = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))
    time = np.zeros((obs_dataset.RasterYSize, obs_dataset.RasterXSize))

    for line in range(obs_dataset.RasterYSize):

        # Read line in
        obs_line = obs_dataset.ReadAsArray(0, line, obs_dataset.RasterXSize, 1)

        # Populate valid
        valid[line,:] = np.logical_not(np.any(obs_line == nodata_value,axis=0))

        path_km[line,:] = obs_line[0, ...] / 1000.
        to_sensor_azimuth[line,:] = obs_line[1, ...]
        to_sensor_zenith[line,:] = obs_line[2, ...]
        time[line,:] = obs_line[9, ...]

    if trim_lines != 0:
        actual_valid = valid.copy()
        valid[:trim_lines,:] = False
        valid[-trim_lines:,:] = False

    mean_path_km = np.mean(path_km[valid])
    del path_km

    mean_to_sensor_azimuth = find_angular_seeds(to_sensor_azimuth[valid], 1) % 360
    mean_to_sensor_zenith = 180 - find_angular_seeds(to_sensor_zenith[valid], 1)

    #geom_margin = EPS * 2.0
    if NUM_TO_SENSOR_ZENITH_LUT_ELEMENTS == 1:
        to_sensor_zenith_lut_grid = None
    else:
        to_sensor_zenith_lut_grid = np.sort(180 - find_angular_seeds(to_sensor_zenith[valid], NUM_TO_SENSOR_ZENITH_LUT_ELEMENTS))
        if (to_sensor_zenith_lut_grid[1] - to_sensor_zenith_lut_grid[0] < ZENITH_MIN_SPACING):
            to_sensor_zenith_lut_grid = None

    if NUM_TO_SENSOR_AZIMUTH_LUT_ELEMENTS == 1:
        to_sensor_azimuth_lut_grid = None
    else:
        to_sensor_azimuth_lut_grid = np.sort(np.array([x % 360 for x in find_angular_seeds(to_sensor_azimuth[valid], NUM_TO_SENSOR_AZIMUTH_LUT_ELEMENTS)]))

    del to_sensor_azimuth
    del to_sensor_zenith

    # Make time calculations
    mean_time = np.mean(time[valid])
    min_time = np.min(time[valid])
    max_time = np.max(time[valid])

    increment_day = False
    # UTC day crossover corner case
    if (max_time > 24 - max_flight_duration_h and
            min_time < max_flight_duration_h):
        time[np.logical_and(time < max_flight_duration_h,valid)] += 24
        mean_time = np.mean(time[valid])

        # This means the majority of the line was really in the next UTC day,
        # increment the line accordingly
        if (mean_time > 24):
            mean_time -= 24
            increment_day = True

    # Calculate hour, minute, second
    h_m_s = [np.floor(mean_time)]
    h_m_s.append(np.floor((mean_time - h_m_s[-1]) * 60))
    h_m_s.append(np.floor((mean_time - h_m_s[-2] - h_m_s[-1] / 60.) * 3600))

    if trim_lines != 0:
        valid = actual_valid

    return h_m_s, increment_day, mean_path_km, mean_to_sensor_azimuth, mean_to_sensor_zenith, valid, \
           to_sensor_azimuth_lut_grid, to_sensor_zenith_lut_grid


def get_metadata_from_loc(loc_file: str, trim_lines: int = 5, nodata_value: float = -9999):
    """ Get metadata needed for complete runs from the location file (bands long, lat, elev).
    Args:
        loc_file: file name to pull data from
        trim_lines: number of lines to ignore at beginning and end of file (good if lines contain values that are
                    erroneous but not nodata
        nodata_value: value to ignore from location file

    :Returns:
        mean_latitude: mean latitude of good values from the location file
        mean_longitude: mean latitude of good values from the location file
        mean_elevation_km: mean ground estimate of good values from the location file
        elevation_lut_grid: the elevation look up table, based on globals and values from location file
    """

    loc_dataset = gdal.Open(loc_file, gdal.GA_ReadOnly)

    loc_data = np.zeros((loc_dataset.RasterCount, loc_dataset.RasterYSize, loc_dataset.RasterXSize))
    for line in range(loc_dataset.RasterYSize):
        # Read line in
        loc_data[:,line:line+1,:] = loc_dataset.ReadAsArray(0, line, loc_dataset.RasterXSize, 1)

    valid = np.logical_not(np.any(loc_data == nodata_value,axis=0))
    if trim_lines != 0:
        valid[:trim_lines, :] = False
        valid[-trim_lines:, :] = False

    # Grab zensor position and orientation information
    mean_latitude = find_angular_seeds(loc_data[1,valid].flatten(),1)
    mean_longitude = find_angular_seeds(-1 * loc_data[0,valid].flatten(),1)

    mean_elevation_km = np.mean(loc_data[2,valid]) / 1000.0

    # make elevation grid
    if NUM_ELEV_LUT_ELEMENTS == 1:
        elevation_lut_grid = None
    else:
        min_elev = np.min(loc_data[2, valid])/1000.
        max_elev = np.max(loc_data[2, valid])/1000.
        elevation_lut_grid = np.linspace(max(min_elev, EPS),
                                         max_elev,
                                         NUM_ELEV_LUT_ELEMENTS)

    return mean_latitude, mean_longitude, mean_elevation_km, elevation_lut_grid



def build_presolve_config(paths: Pathnames, h2o_lut_grid: np.array):
    """ Write an isofit config file for a presolve, with limited info.
    Args:
        paths: object containing references to all relevant file locations
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve

    :Returns:
        None
    """

    h2o_configuration = {
            "modtran_vswir": {
                "wavelength_file": paths.wavelength_path,
                "lut_path": paths.lut_h2o_directory,
                "modtran_template_file": paths.h2o_template_path,
                "modtran_directory": paths.modtran_path,
                "lut_names": ["H2OSTR"],
                "statevector_names": ["H2OSTR"],
            },
            "statevector": {
                "H2OSTR": {
                    "bounds": [float(np.min(h2o_lut_grid)), float(np.max(h2o_lut_grid))],
                    "scale": 0.01,
                    "init": np.percentile(h2o_lut_grid,25),
                    "prior_sigma": 100.0,
                    "prior_mean": 1.5}
            },
            "lut_grid": {
                "H2OSTR": [float(x) for x in h2o_lut_grid],
            },
            "unknowns": {
                "H2O_ABSCO": 0.0
            },
            "domain": {"start": 340, "end": 2520, "step": 0.1}
    }

    # make isofit configuration
    isofit_config_h2o = {'ISOFIT_base': paths.isofit_path,
                         'input': {'measured_radiance_file': paths.rdn_subs_path,
                                   'loc_file': paths.loc_subs_path,
                                   'obs_file': paths.obs_subs_path},
                         'output': {'estimated_state_file': paths.h2o_subs_path},
                         'forward_model': {
                             'instrument': {'wavelength_file': paths.wavelength_path,
                                            'parametric_noise_file': paths.noise_path,
                                            'integrations': NUM_INTEGRATIONS,
                                            'unknowns': {
                                                'uncorrelated_radiometric_uncertainty': UNCORRELATED_RADIOMETRIC_UNCERTAINTY}},
                                                    'multicomponent_surface': {'wavelength_file': paths.wavelength_path,
                                                                               'surface_file': paths.surface_working_path,
                                                                               'select_on_init': True},
                             'radiative_transfer': h2o_configuration},
                         'inversion': {'windows': INVERSION_WINDOWS}}

    if paths.channelized_uncertainty_working_path is not None:
        isofit_config_h2o['forward_model']['unknowns'][
            'channelized_radiometric_uncertainty_file'] = paths.channelized_uncertainty_working_path

    if paths.rdn_factors_path:
        isofit_config_h2o['input']['radiometry_correction_file'] = paths.rdn_factors_path

    # write modtran_template
    with open(paths.h2o_config_path, 'w') as fout:
        fout.write(json.dumps(isofit_config_h2o, cls=SerialEncoder, indent=4, sort_keys=True))


def build_main_config(paths: Pathnames, h2o_lut_grid: np.array = None, elevation_lut_grid: np.array = None,
                      to_sensor_azimuth_lut_grid: np.array = None, to_sensor_zenith_lut_grid: np.array = None,
                      mean_latitude: float = None, mean_longitude: float = None, dt: datetime = None):
    """ Write an isofit config file for the main solve, using the specified pathnames and all given info
    Args:
        paths: object containing references to all relevant file locations
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid: the ground elevation look up table grid isofit should use for this solve
        to_sensor_azimuth_lut_grid: the to-sensor azimuth angle look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid: the to-sensor zenith angle look up table grid isofit should use for this solve
        mean_latitude: the latitude isofit should use for this solve
        mean_longitude: the longitude isofit should use for this solve
        dt: the datetime object corresponding to this flightline to use for this solve

    :Returns:
        None
    """
    modtran_configuration = {
            "modtran_vswir": {
                "wavelength_file": paths.wavelength_path,
                "lut_path": paths.lut_modtran_directory,
                "aerosol_template_file": paths.aerosol_tpl_path,
                "modtran_template_file": paths.modtran_template_path,
                "modtran_directory": paths.modtran_path,
                #lut_names - populated below
                #statevector_names - populated below
            },
            "statevector": {
                "H2OSTR": {
                    "bounds": [h2o_lut_grid[0], h2o_lut_grid[-1]],
                    "scale": 0.01,
                    "init": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
                    "prior_sigma": 100.0,
                    "prior_mean": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
                }
            },
            "lut_grid": {},
            "unknowns": {
                "H2O_ABSCO": 0.0
            },
            "domain": {"start": 340, "end": 2520, "step": 0.1}
    }
    if h2o_lut_grid is not None:
        modtran_configuration['lut_grid']['H2OSTR'] = [max(0.0, float(q)) for q in h2o_lut_grid]
    if elevation_lut_grid is not None:
        modtran_configuration['lut_grid']['GNDALT'] = [max(0.0, float(q)) for q in elevation_lut_grid]
    if to_sensor_azimuth_lut_grid is not None:
        modtran_configuration['lut_grid']['TRUEAZ'] = [float(q) for q in to_sensor_azimuth_lut_grid]
    if to_sensor_zenith_lut_grid is not None:
        modtran_configuration['lut_grid']['OBSZEN'] = [float(q) for q in to_sensor_zenith_lut_grid] # modtran convension

    # add aerosol elements from climatology
    aerosol_state_vector, aerosol_lut_grid, aerosol_model_path = \
        load_climatology(paths.aerosol_climatology, mean_latitude, mean_longitude, dt,
                         paths.isofit_path)
    modtran_configuration['statevector'].update(aerosol_state_vector)
    modtran_configuration['lut_grid'].update(aerosol_lut_grid)
    modtran_configuration['modtran_vswir']['aerosol_model_file'] = aerosol_model_path

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    modtran_configuration['modtran_vswir']['statevector_names'] = list(modtran_configuration['statevector'].keys())
    modtran_configuration['modtran_vswir']['lut_names'] = list(modtran_configuration['lut_grid'].keys())

    # make isofit configuration
    isofit_config_modtran = {'ISOFIT_base': paths.isofit_path,
                             'input': {'measured_radiance_file': paths.rdn_subs_path,
                                       'loc_file': paths.loc_subs_path,
                                       'obs_file': paths.obs_subs_path},
                             'output': {'estimated_state_file': paths.state_subs_path,
                                        'posterior_uncertainty_file': paths.uncert_subs_path,
                                        'estimated_reflectance_file': paths.rfl_subs_path},
                             'forward_model': {
                                 'instrument': {'wavelength_file': paths.wavelength_path,
                                                'parametric_noise_file': paths.noise_path,
                                                'integrations': NUM_INTEGRATIONS,
                                                'unknowns': {
                                                    'uncorrelated_radiometric_uncertainty': UNCORRELATED_RADIOMETRIC_UNCERTAINTY}},
                                 "multicomponent_surface": {"wavelength_file": paths.wavelength_path,
                                                            "surface_file": paths.surface_working_path,
                                                            "select_on_init": True},
                                 "radiative_transfer": modtran_configuration},
                             "inversion": {"windows": INVERSION_WINDOWS}}

    if paths.channelized_uncertainty_working_path is not None:
        isofit_config_modtran['forward_model']['unknowns'][
            'channelized_radiometric_uncertainty_file'] = paths.channelized_uncertainty_working_path

    if paths.rdn_factors_path:
        isofit_config_modtran['input']['radiometry_correction_file'] = \
            paths.rdn_factors_path

    # write modtran_template
    with open(paths.modtran_config_path, 'w') as fout:
        fout.write(json.dumps(isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True))


def write_modtran_template(atmosphere_type: str, fid: str, altitude_km: float, dayofyear: int,
                           latitude: float, longitude: float, to_sensor_azimuth: float, to_sensor_zenith: float,
                           gmtime: float, elevation_km: float, output_file: str, ihaze_type: str = 'AER_RURAL'):
    """ Write a MODTRAN template file for use by isofit look up tables
    Args:
        atmosphere_type: label for the type of atmospheric profile to use in modtran
        fid: flight line id (name)
        altitude_km: altitude of the sensor in km
        dayofyear: the current day of the given year
        latitude: acquisition latitude
        longitude: acquisition longitude
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees (AVIRIS convention)
        to_sensor_zenith: azimuth view angle to the sensor, in degrees (MODTRAN convention: 180 - AVIRIS convention)
        gmtime: greenwich mean time
        elevation_km: elevation of the land surface in km
        output_file: location to write the modtran template file to

    :Returns:
        None
    """
    # make modtran configuration
    h2o_template = {"MODTRAN": [{
        "MODTRANINPUT": {
            "NAME": fid,
            "DESCRIPTION": "",
            "CASE": 0,
            "RTOPTIONS": {
                "MODTRN": "RT_CORRK_FAST",
                "LYMOLC": False,
                "T_BEST": False,
                "IEMSCT": "RT_SOLAR_AND_THERMAL",
                "IMULT": "RT_DISORT",
                "DISALB": False,
                "NSTR": 8,
                "SOLCON": 0.0
            },
            "ATMOSPHERE": {
                "MODEL": atmosphere_type,
                "M1": atmosphere_type,
                "M2": atmosphere_type,
                "M3": atmosphere_type,
                "M4": atmosphere_type,
                "M5": atmosphere_type,
                "M6": atmosphere_type,
                "CO2MX": 410.0,
                "H2OSTR": 1.0,
                "H2OUNIT": "g",
                "O3STR": 0.3,
                "O3UNIT": "a"
            },
            "AEROSOLS": {"IHAZE": ihaze_type},
            "GEOMETRY": {
                "ITYPE": 3,
                "H1ALT": altitude_km,
                "IDAY": dayofyear,
                "IPARM": 11,
                "PARM1": latitude,
                "PARM2": longitude,
                "TRUEAZ": to_sensor_azimuth,
                "OBSZEN": to_sensor_zenith,
                "GMTIME": gmtime
            },
            "SURFACE": {
                "SURFTYPE": "REFL_LAMBER_MODEL",
                "GNDALT": elevation_km,
                "NSURF": 1,
                "SURFP": {"CSALB": "LAMB_CONST_0_PCT"}
            },
            "SPECTRAL": {
                "V1": 340.0,
                "V2": 2520.0,
                "DV": 0.1,
                "FWHM": 0.1,
                "YFLAG": "R",
                "XFLAG": "N",
                "FLAGS": "NT A   ",
                "BMNAME": "p1_2013"
            },
            "FILEOPTIONS": {
                "NOPRNT": 2,
                "CKPRNT": True
            }
        }
    }]}

    # write modtran_template
    with open(output_file, 'w') as fout:
        fout.write(json.dumps(h2o_template, cls=SerialEncoder, indent=4, sort_keys=True))



if __name__ == "__main__":
    main()
