#! /usr/bin/env python3
#
# Authors: Philip G. Brodrick and Niklas Bohn
#

import logging
import os
import json
import numpy as np
import argparse
from isofit.core.common import envi_header
from shutil import copyfile
from datetime import datetime
import subprocess

INVERSION_WINDOWS = [[380.0, 1340.0], [1450, 1800.0], [1970.0, 2500.0]]


class Pathnames:
    """
    Class to determine and hold the large number of relative and absolute paths that are needed for isofit and MODTRAN
    configuration files.

    Args:
        args: an argparse Namespace object with all inputs
    """

    def __init__(self, args: argparse.Namespace):

        # Determine FID based on sensor name
        if args.sensor == 'ang':
            self.fid = os.path.split(args.input_radiance)[-1][:18]
            logging.info('Flightline ID: %s' % self.fid)
        elif args.sensor == 'prism':
            self.fid = os.path.split(args.input_radiance)[-1][:18]
            logging.info('Flightline ID: %s' % self.fid)
        elif args.sensor == 'avcl':
            self.fid = os.path.split(args.input_radiance)[-1][:16]
            logging.info('Flightline ID: %s' % self.fid)
        elif args.sensor == 'neon':
            self.fid = os.path.split(args.input_radiance)[-1][:21]
        elif args.sensor == 'emit':
            self.fid = os.path.split(args.input_radiance)[-1][:19]
        elif args.sensor[:3] == 'NA-':
            self.fid = os.path.os.path.splitext(os.path.basename(args.input_radiance))[0]
        elif args.sensor == 'hyp': 
            self.fid = os.path.split(args.input_radiance)[-1][:22]

        # Names from inputs
        self.aerosol_climatology = args.aerosol_climatology_path
        self.input_radiance_file = args.input_radiance
        self.input_loc_file = args.input_loc
        self.input_obs_file = args.input_obs
        self.working_directory = os.path.abspath(args.working_directory)

        self.lut_modtran_directory = os.path.abspath(os.path.join(self.working_directory, 'lut_full',''))

        if args.surface_path:
            self.surface_path = args.surface_path
        else:
            self.surface_path = os.getenv('ISOFIT_SURFACE_MODEL')
        if self.surface_path is None:
            logging.info('No surface model defined')

        # set up some sub-directories
        self.lut_h2o_directory = os.path.abspath(os.path.join(self.working_directory, 'lut_h2o',''))
        self.config_directory = os.path.abspath(os.path.join(self.working_directory, 'config',''))
        self.data_directory = os.path.abspath(os.path.join(self.working_directory, 'data',''))
        self.input_data_directory = os.path.abspath(os.path.join(self.working_directory, 'input',''))
        self.output_directory = os.path.abspath(os.path.join(self.working_directory, 'output',''))


        # define all output names
        rdn_fname = self.fid + '_rdn'
        self.rfl_working_path = os.path.abspath(os.path.join(self.output_directory, rdn_fname.replace('_rdn', '_rfl')))
        self.uncert_working_path = os.path.abspath(os.path.join(self.output_directory, rdn_fname.replace('_rdn', '_uncert')))
        self.lbl_working_path = os.path.abspath(os.path.join(self.output_directory, rdn_fname.replace('_rdn', '_lbl')))
        self.state_working_path = os.path.abspath(os.path.join(self.output_directory, rdn_fname.replace('_rdn', '_state')))
        self.surface_working_path = os.path.abspath(os.path.join(self.data_directory, 'surface.mat'))

        if args.copy_input_files is True:
            self.radiance_working_path = os.path.abspath(os.path.join(self.input_data_directory, rdn_fname))
            self.obs_working_path = os.path.abspath(os.path.join(self.input_data_directory, self.fid + '_obs'))
            self.loc_working_path = os.path.abspath(os.path.join(self.input_data_directory, self.fid + '_loc'))
        else:
            self.radiance_working_path = os.path.abspath(self.input_radiance_file)
            self.obs_working_path = os.path.abspath(self.input_obs_file)
            self.loc_working_path = os.path.abspath(self.input_loc_file)

        if args.channelized_uncertainty_path:
            self.input_channelized_uncertainty_path = args.channelized_uncertainty_path
        else:
            self.input_channelized_uncertainty_path = os.getenv('ISOFIT_CHANNELIZED_UNCERTAINTY')

        self.channelized_uncertainty_working_path = os.path.abspath(os.path.join(self.data_directory, 'channelized_uncertainty.txt'))

        if args.model_discrepancy_path:
            self.input_model_discrepancy_path = args.model_discrepancy_path
        else:
            self.input_model_discrepancy_path = None

        self.model_discrepancy_working_path = os.path.abspath(os.path.join(self.data_directory, 'model_discrepancy.mat'))

        self.rdn_subs_path = os.path.abspath(os.path.join(self.input_data_directory, self.fid + '_subs_rdn'))
        self.obs_subs_path = os.path.abspath(os.path.join(self.input_data_directory, self.fid + '_subs_obs'))
        self.loc_subs_path = os.path.abspath(os.path.join(self.input_data_directory, self.fid + '_subs_loc'))
        self.rfl_subs_path = os.path.abspath(os.path.join(self.output_directory, self.fid + '_subs_rfl'))
        self.atm_coeff_path = os.path.abspath(os.path.join(self.output_directory, self.fid + '_subs_atm'))
        self.state_subs_path = os.path.abspath(os.path.join(self.output_directory, self.fid + '_subs_state'))
        self.uncert_subs_path = os.path.abspath(os.path.join(self.output_directory, self.fid + '_subs_uncert'))
        self.h2o_subs_path = os.path.abspath(os.path.join(self.output_directory, self.fid + '_subs_h2o'))
        self.surface_subs_files = {}

        self.wavelength_path = os.path.abspath(os.path.join(self.data_directory, 'wavelengths.txt'))

        self.modtran_template_path = os.path.abspath(os.path.join(self.config_directory, self.fid + '_modtran_tpl.json'))
        self.h2o_template_path = os.path.abspath(os.path.join(self.config_directory, self.fid + '_h2o_tpl.json'))

        self.modtran_config_path = os.path.abspath(os.path.join(self.config_directory, self.fid + '_modtran.json'))
        self.h2o_config_path = os.path.abspath(os.path.join(self.config_directory, self.fid + '_h2o.json'))

        if args.modtran_path:
            self.modtran_path = args.modtran_path
        else:
            self.modtran_path = os.getenv('MODTRAN_DIR')

        self.sixs_path = os.getenv('SIXS_DIR')

        if os.getenv('ISOFIT_DIR'):
            self.isofit_path = os.getenv('ISOFIT_DIR')
        else:
             # isofit file should live at isofit/isofit/core/isofit.py
            self.isofit_path = os.path.dirname(os.path.dirname(os.path.dirname(isofit.__file__)))

        if args.sensor == 'ang':
            self.noise_path = os.path.join(self.isofit_path, 'data', 'avirisng_noise.txt')
        elif args.sensor == 'avcl':
            self.noise_path = os.path.join(self.isofit_path, 'data', 'avirisc_noise.txt')
        else:
            self.noise_path = None
            logging.info('no noise path found, proceeding without')
            #quit()

        self.earth_sun_distance_path = os.path.abspath(os.path.join(self.isofit_path,'data','earth_sun_distance.txt'))
        self.irradiance_file = os.path.abspath(os.path.join(self.isofit_path,'examples','20151026_SantaMonica','data','prism_optimized_irr.dat'))

        self.aerosol_tpl_path = os.path.join(self.isofit_path, 'data', 'aerosol_template.json')
        self.rdn_factors_path = None
        if args.rdn_factors_path is not None:
            self.rdn_factors_path = os.path.abspath(args.rdn_factors_path)


        self.ray_temp_dir = args.ray_temp_dir

    def make_directories(self):
        """ Build required subdirectories inside working_directory
        """
        for dpath in [self.working_directory, self.lut_h2o_directory, self.lut_modtran_directory, self.config_directory,
                      self.data_directory, self.input_data_directory, self.output_directory]:
            if not os.path.exists(dpath):
                os.mkdir(dpath)

    def stage_files(self):
        """ Stage data files by copying into working directory
        """
        files_to_stage = [(self.input_radiance_file, self.radiance_working_path, True),
                          (self.input_obs_file, self.obs_working_path, True),
                          (self.input_loc_file, self.loc_working_path, True),
                          (self.surface_path, self.surface_working_path, False),
                          (self.input_channelized_uncertainty_path, 
                                self.channelized_uncertainty_working_path, False),
                          (self.input_model_discrepancy_path,
                                self.model_discrepancy_working_path, False)]

        for src, dst, hasheader in files_to_stage:
            if src is None:
                continue
            if not os.path.exists(dst):
                logging.info('Staging %s to %s' % (src, dst))
                copyfile(src, dst)
                if hasheader:
                    copyfile(envi_header(src), envi_header(dst))
    
    def add_surface_subs_files(self, surface_type):
        self.surface_subs_files[surface_type] = {
            'rdn': self.rdn_subs_path + '_' + surface_type,
            'loc': self.loc_subs_path + '_' + surface_type,
            'obs': self.obs_subs_path + '_' + surface_type,
            'rfl': self.rfl_subs_path + '_' + surface_type,
            'state': self.state_subs_path + '_' + surface_type,
            'uncert': self.uncert_subs_path + '_' + surface_type,
            'h2o': self.h2o_subs_path + '_' + surface_type
        }




def build_presolve_config(paths: Pathnames, h2o_lut_grid: np.array, n_cores: int=-1,
        use_emp_line:bool = False, surface_category="multicomponent_surface",
        emulator_base: str = None, uncorrelated_radiometric_uncertainty: float = 0.0,
        segmentation_size: int = 400):
    """ Write an isofit config file for a presolve, with limited info.

    Args:
        paths: object containing references to all relevant file locations
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        n_cores: number of cores to use in processing
        use_emp_line: flag whether or not to set up for the empirical line estimation
        surface_category: type of surface to use
        emulator_base: the basename of the emulator, if used
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        segmentation_size: image segmentation size if empirical line is used
    """

    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_emp_line:
        spectra_per_inversion = segmentation_size
    else: 
        spectra_per_inversion = 1 


    if emulator_base is None:
        engine_name = 'modtran'
    else:
        engine_name = 'sRTMnet'

    radiative_transfer_config = {
            "radiative_transfer_engines": {
                "vswir": {
                    "engine_name": engine_name,
                    "lut_path": paths.lut_h2o_directory,
                    "template_file": paths.h2o_template_path,
                    "lut_names": ["H2OSTR"],
                    "statevector_names": ["H2OSTR"],
                }
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
            }
    }

    if emulator_base is not None:
        radiative_transfer_config['radiative_transfer_engines']['vswir']['emulator_file'] = os.path.abspath(emulator_base)
        radiative_transfer_config['radiative_transfer_engines']['vswir']['emulator_aux_file'] = os.path.abspath(os.path.os.path.splitext(emulator_base)[0] + '_aux.npz')
        radiative_transfer_config['radiative_transfer_engines']['vswir']['interpolator_base_path'] = os.path.abspath(os.path.os.path.join(paths.lut_h2o_directory,os.path.basename(emulator_base) + '_vi'))
        radiative_transfer_config['radiative_transfer_engines']['vswir']['earth_sun_distance_file'] = paths.earth_sun_distance_path
        radiative_transfer_config['radiative_transfer_engines']['vswir']['irradiance_file'] = paths.irradiance_file
        radiative_transfer_config['radiative_transfer_engines']['vswir']["engine_base_dir"] = paths.sixs_path
  
    else:
        radiative_transfer_config['radiative_transfer_engines']['vswir']["engine_base_dir"] = paths.modtran_path

    # make isofit configuration
    isofit_config_h2o = {'ISOFIT_base': paths.isofit_path,
                         'output': {'estimated_state_file': paths.h2o_subs_path},
                         'input': {},
                         'forward_model': {
                             'instrument': {'wavelength_file': paths.wavelength_path,
                                            'integrations': spectra_per_inversion,
                                            'unknowns': {
                                                'uncorrelated_radiometric_uncertainty': uncorrelated_radiometric_uncertainty}},
                                                    'surface': {"surface_category": surface_category,
                                                                'surface_file': paths.surface_working_path,
                                                                'select_on_init': True},
                             'radiative_transfer': radiative_transfer_config},
                         "implementation": {
                            "ray_temp_dir": paths.ray_temp_dir,
                            'inversion': {
                              'windows': INVERSION_WINDOWS},
                            "n_cores": n_cores}
                         }

    if paths.input_channelized_uncertainty_path is not None:
        isofit_config_h2o['forward_model']['instrument']['unknowns'][
            'channelized_radiometric_uncertainty_file'] = paths.channelized_uncertainty_working_path

    if paths.input_model_discrepancy_path is not None:
        isofit_config_h2o['forward_model']['model_discrepancy_file'] = \
            paths.model_discrepancy_working_path

    if paths.noise_path is not None:
        isofit_config_h2o['forward_model']['instrument']['parametric_noise_file'] = paths.noise_path
    else:
        isofit_config_h2o['forward_model']['instrument']['SNR'] = 1000

    if paths.rdn_factors_path:
        isofit_config_h2o['input']['radiometry_correction_file'] = paths.rdn_factors_path

    if use_emp_line:
        isofit_config_h2o['input']['measured_radiance_file'] = paths.rdn_subs_path
        isofit_config_h2o['input']['loc_file'] = paths.loc_subs_path
        isofit_config_h2o['input']['obs_file'] = paths.obs_subs_path
    else:
        isofit_config_h2o['input']['measured_radiance_file'] = paths.radiance_working_path
        isofit_config_h2o['input']['loc_file'] = paths.loc_working_path
        isofit_config_h2o['input']['obs_file'] = paths.obs_working_path



    # write modtran_template
    with open(paths.h2o_config_path, 'w') as fout:
        fout.write(json.dumps(isofit_config_h2o, cls=SerialEncoder, indent=4, sort_keys=True))


def build_main_config(paths: Pathnames, lut_params: LUTConfig, h2o_lut_grid: np.array = None,
                      elevation_lut_grid: np.array = None, to_sensor_azimuth_lut_grid: np.array = None,
                      to_sensor_zenith_lut_grid: np.array = None, mean_latitude: float = None,
                      mean_longitude: float = None, dt: datetime = None, use_emp_line: bool = True, 
                      n_cores: int = -1, surface_category='multicomponent_surface',
                      emulator_base: str = None, uncorrelated_radiometric_uncertainty: float = 0.0,
                      multiple_restarts: bool = False, segmentation_size=400):
    """ Write an isofit config file for the main solve, using the specified pathnames and all given info

    Args:
        paths: object containing references to all relevant file locations
        lut_params: configuration parameters for the lut grid
        h2o_lut_grid: the water vapor look up table grid isofit should use for this solve
        elevation_lut_grid: the ground elevation look up table grid isofit should use for this solve
        to_sensor_azimuth_lut_grid: the to-sensor azimuth angle look up table grid isofit should use for this solve
        to_sensor_zenith_lut_grid: the to-sensor zenith angle look up table grid isofit should use for this solve
        mean_latitude: the latitude isofit should use for this solve
        mean_longitude: the longitude isofit should use for this solve
        dt: the datetime object corresponding to this flightline to use for this solve
        use_emp_line: flag whether or not to set up for the empirical line estimation
        n_cores: the number of cores to use during processing
        surface_category: type of surface to use
        emulator_base: the basename of the emulator, if used
        uncorrelated_radiometric_uncertainty: uncorrelated radiometric uncertainty parameter for isofit
        segmentation_size: image segmentation size if empirical line is used
    """

    # Determine number of spectra included in each retrieval.  If we are
    # operating on segments, this will average down instrument noise
    if use_emp_line:
        spectra_per_inversion = segmentation_size
    else: 
        spectra_per_inversion = 1 

    if emulator_base is None:
        engine_name = 'modtran'
    else:
        engine_name = 'sRTMnet'
    radiative_transfer_config = {

            "radiative_transfer_engines": {
                "vswir": {
                    "engine_name": engine_name,
                    "lut_path": paths.lut_modtran_directory,
                    "aerosol_template_file": paths.aerosol_tpl_path,
                    "template_file": paths.modtran_template_path,
                    #lut_names - populated below
                    #statevector_names - populated below
                }
            },
            "statevector": {},
            "lut_grid": {},
            "unknowns": {
                "H2O_ABSCO": 0.0
            }
    }

    if h2o_lut_grid is not None:
        radiative_transfer_config['statevector']['H2OSTR'] = {
            "bounds": [h2o_lut_grid[0], h2o_lut_grid[-1]],
            "scale": 1,
            "init": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
            "prior_sigma": 100.0,
            "prior_mean": (h2o_lut_grid[1] + h2o_lut_grid[-1]) / 2.0,
        }

    if emulator_base is not None:
        radiative_transfer_config['radiative_transfer_engines']['vswir']['emulator_file'] = os.path.abspath(emulator_base)
        radiative_transfer_config['radiative_transfer_engines']['vswir']['emulator_aux_file'] = os.path.abspath(os.path.os.path.splitext(emulator_base)[0] + '_aux.npz')
        radiative_transfer_config['radiative_transfer_engines']['vswir']['interpolator_base_path'] = os.path.abspath(os.path.os.path.join(paths.lut_modtran_directory,os.path.basename(os.path.os.path.splitext(emulator_base)[0]) + '_vi'))
        radiative_transfer_config['radiative_transfer_engines']['vswir']['earth_sun_distance_file'] = paths.earth_sun_distance_path
        radiative_transfer_config['radiative_transfer_engines']['vswir']['irradiance_file'] = paths.irradiance_file
        radiative_transfer_config['radiative_transfer_engines']['vswir']["engine_base_dir"] = paths.sixs_path
  
    else:
        radiative_transfer_config['radiative_transfer_engines']['vswir']["engine_base_dir"] = paths.modtran_path


    if h2o_lut_grid is not None:
        radiative_transfer_config['lut_grid']['H2OSTR'] = h2o_lut_grid.tolist()
    if elevation_lut_grid is not None:
        radiative_transfer_config['lut_grid']['GNDALT'] = elevation_lut_grid.tolist()
    if to_sensor_azimuth_lut_grid is not None:
        radiative_transfer_config['lut_grid']['TRUEAZ'] = to_sensor_azimuth_lut_grid.tolist()
    if to_sensor_zenith_lut_grid is not None:
        radiative_transfer_config['lut_grid']['OBSZEN'] = to_sensor_zenith_lut_grid.tolist() # modtran convension

    # add aerosol elements from climatology
    aerosol_state_vector, aerosol_lut_grid, aerosol_model_path = \
        load_climatology(paths.aerosol_climatology, mean_latitude, mean_longitude, dt,
                         paths.isofit_path, lut_params=lut_params)
    radiative_transfer_config['statevector'].update(aerosol_state_vector)
    radiative_transfer_config['lut_grid'].update(aerosol_lut_grid)
    radiative_transfer_config['radiative_transfer_engines']['vswir']['aerosol_model_file'] = aerosol_model_path

    # MODTRAN should know about our whole LUT grid and all of our statevectors, so copy them in
    radiative_transfer_config['radiative_transfer_engines']['vswir']['statevector_names'] = list(radiative_transfer_config['statevector'].keys())
    radiative_transfer_config['radiative_transfer_engines']['vswir']['lut_names'] = list(radiative_transfer_config['lut_grid'].keys())

    # make isofit configuration
    isofit_config_modtran = {'ISOFIT_base': paths.isofit_path,
                             'input': {},
                             'output': {},
                             'forward_model': {
                                 'instrument': {'wavelength_file': paths.wavelength_path,
                                                'integrations': spectra_per_inversion,
                                                'unknowns': {
                                                    'uncorrelated_radiometric_uncertainty': uncorrelated_radiometric_uncertainty}},
                                 "surface": {"surface_file": paths.surface_working_path,
                                             "surface_category": surface_category,
                                             "select_on_init": True},
                                 "radiative_transfer": radiative_transfer_config},
                             "implementation": {
                                "ray_temp_dir": paths.ray_temp_dir,
                                "inversion": {"windows": INVERSION_WINDOWS},
                                "n_cores": n_cores}
                             }

    if use_emp_line:
        isofit_config_modtran['input']['measured_radiance_file'] = paths.rdn_subs_path
        isofit_config_modtran['input']['loc_file'] = paths.loc_subs_path
        isofit_config_modtran['input']['obs_file'] = paths.obs_subs_path
        isofit_config_modtran['output']['estimated_state_file'] = paths.state_subs_path
        isofit_config_modtran['output']['posterior_uncertainty_file'] = paths.uncert_subs_path
        isofit_config_modtran['output']['estimated_reflectance_file'] = paths.rfl_subs_path
        isofit_config_modtran['output']['atmospheric_coefficients_file'] = paths.atm_coeff_path
    else:
        isofit_config_modtran['input']['measured_radiance_file'] = paths.radiance_working_path
        isofit_config_modtran['input']['loc_file'] = paths.loc_working_path
        isofit_config_modtran['input']['obs_file'] = paths.obs_working_path
        isofit_config_modtran['output']['posterior_uncertainty_file'] = paths.uncert_working_path
        isofit_config_modtran['output']['estimated_reflectance_file'] = paths.rfl_working_path
        isofit_config_modtran['output']['estimated_state_file'] = paths.state_working_path

    if multiple_restarts:
        eps = 1e-2
        grid = {}
        if h2o_lut_grid is not None:
            h2o_delta = float(h2o_lut_grid[-1]) - float(h2o_lut_grid[0])
            grid['H2OSTR'] = [round(h2o_lut_grid[0]+h2o_delta*0.02,4), 
                              round(h2o_lut_grid[-1]-h2o_delta*0.02,4)]

        # We will initialize using different AODs for the first aerosol in the LUT
        if len(aerosol_lut_grid)>0:
            key = list(aerosol_lut_grid.keys())[0]
            aer_delta = aerosol_lut_grid[key][-1] - aerosol_lut_grid[key][0]
            grid[key] = [round(aerosol_lut_grid[key][0]+aer_delta*0.02,4), 
                         round(aerosol_lut_grid[key][-1]-aer_delta*0.02,4)]
        isofit_config_modtran['implementation']['inversion']['integration_grid'] = grid
        isofit_config_modtran['implementation']['inversion']['inversion_grid_as_preseed'] = True

    if paths.input_channelized_uncertainty_path is not None:
        isofit_config_modtran['forward_model']['instrument']['unknowns'][
            'channelized_radiometric_uncertainty_file'] = paths.channelized_uncertainty_working_path

    if paths.input_model_discrepancy_path is not None:
        isofit_config_modtran['forward_model']['model_discrepancy_file'] = \
            paths.model_discrepancy_working_path

    if paths.noise_path is not None:
        isofit_config_modtran['forward_model']['instrument']['parametric_noise_file'] = paths.noise_path
    else:
        isofit_config_modtran['forward_model']['instrument']['SNR'] = 1000

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


class LUTConfig:
    """ A look up table class, containing default grid options.  All properties may be overridden with the optional
        input configuration file path

    Args:
        lut_config_file: configuration file to override default values
    """

    def __init__(self, lut_config_file: str = None):
        if lut_config_file is not None:
            with open(lut_config_file, 'r') as f:
                lut_config = json.load(f)

        # For each element, set the look up table spacing (lut_spacing) as the
        # anticipated spacing value, or 0 to use a single point (not LUT).
        # Set the 'lut_spacing_min' as the minimum distance allowed - if separation
        # does not meet this threshold based on the available data, on a single
        # point will be used.

        # Units of kilometers
        self.elevation_spacing = 0.5
        self.elevation_spacing_min = 0.2

        # Units of g / m2
        self.h2o_spacing = 0.25
        self.h2o_spacing_min = 0.03

        # Special parameter to specify the minimum allowable water vapor value in g / m2
        self.h2o_min = 0.2

        # Set defaults, will override based on settings
        # Units of g / m2
        self.h2o_range = [0.05, 5]

        # Units of degrees
        self.to_sensor_azimuth_spacing = 60
        self.to_sensor_azimuth_spacing_min = 60

        # Units of degrees
        self.to_sensor_zenith_spacing = 10
        self.to_sensor_zenith_spacing_min = 2

        # Units of AOD
        self.aerosol_0_spacing = 0
        self.aerosol_0_spacing_min = 0

        # Units of AOD
        self.aerosol_1_spacing = 0
        self.aerosol_1_spacing_min = 0

        # Units of AOD
        self.aerosol_2_spacing = 0.25
        self.aerosol_2_spacing_min = 0

        # Units of AOD
        self.aerosol_0_range = [0.001, 0.5]
        self.aerosol_1_range = [0.001, 0.5]
        self.aerosol_2_range = [0.001, 0.5]
        self.aot_550_range = [0.001, 0.5]

        self.aot_550_spacing = 0
        self.aot_550_spacing_min = 0

        # overwrite anything that comes in from the config file
        if lut_config_file is not None:
            for key in lut_config:
                if key in self.__dict__:
                    setattr(self, key, lut_config[key])

    def get_grid(self, minval: float, maxval: float, spacing: float, min_spacing: float):
        if spacing == 0:
            logging.debug('Grid spacing set at 0, using no grid.')
            return None
        num_gridpoints = int(np.ceil((maxval-minval)/spacing)) + 1

        grid = np.linspace(minval, maxval, num_gridpoints)

        if min_spacing > 0.0001:
            grid = np.round(grid, 4)
        if len(grid) == 1:
            logging.debug(f'Grid spacing is 0, which is less than {min_spacing}.  No grid used')
            return None
        elif np.abs(grid[1] - grid[0]) < min_spacing:
            logging.debug(f'Grid spacing is {grid[1]-grid[0]}, which is less than {min_spacing}.  No grid used')
            return None
        else:
            return grid

    def get_angular_grid(self, angle_data_input: np.array, spacing: float, min_spacing: float, units : str = 'd'):
        """ Find either angular data 'center points' (num_points = 1), or a lut set that spans
        angle variation in a systematic fashion.

        Args:
            angle_data_input: set of angle data to use to find center points
            spacing: the desired angular spacing between points, or mean if -1
            min_spacing: the minimum angular spacing between points allowed (if less, no grid)
            units: specifies if data are in degrees (default) or radians

        :Returns:
            angular data center point or lut set spanning space

        """
        if spacing == 0:
            logging.debug('Grid spacing set at 0, using no grid.')
            return None

        # Convert everything to radians so we don't have to track throughout
        if units == 'r':
            angle_data = np.rad2deg(angle_data_input)
        else:
            angle_data = angle_data_input.copy()

        spatial_data = np.hstack([np.cos(np.deg2rad(angle_data)).reshape(-1, 1),
                                  np.sin(np.deg2rad(angle_data)).reshape(-1, 1)])

        # find which quadrants have data
        quadrants = np.zeros((2, 2))
        if np.any(np.logical_and(spatial_data[:, 0] > 0, spatial_data[:, 1] > 0)):
            quadrants[1, 0] = 1
        if np.any(np.logical_and(spatial_data[:, 0] > 0, spatial_data[:, 1] < 0)):
            quadrants[1, 1] += 1
        if np.any(np.logical_and(spatial_data[:, 0] < 0, spatial_data[:, 1] > 0)):
            quadrants[0, 0] += 1
        if np.any(np.logical_and(spatial_data[:, 0] < 0, spatial_data[:, 1] < 0)):
            quadrants[0, 1] += 1

        # Handle the case where angles are < 180 degrees apart
        if np.sum(quadrants) < 3 and spacing != -1:
            if np.sum(quadrants[1, :]) == 2:
                # If angles cross the 0-degree line:
                angle_spread = self.get_grid(np.min(angle_data + 180), np.max(angle_data + 180), spacing, min_spacing)
                if angle_spread is None:
                    return None
                else:
                    return angle_spread - 180
            else:
                # Otherwise, just space things out:
                return self.get_grid(np.min(angle_data), np.max(angle_data), spacing, min_spacing)
        else:
            if spacing >= 180:
                logging.warning(f'Requested angle spacing is {spacing}, but obs angle divergence is > 180.  '
                                'Tighter  spacing recommended')

            # If we're greater than 180 degree spread, there's no universal answer. Try GMM.
            if spacing == -1:
                num_points = 1
            else:
                # This very well might overly space the grid, but we don't / can't know in general
                num_points = int(np.ceil(360 / spacing))

            # We initialize the GMM with a static seed for repeatability across runs
            gmm = mixture.GaussianMixture(n_components=num_points, covariance_type='full',
                                          random_state=1)
            if spatial_data.shape[0]  == 1:
                spatial_data = np.vstack([spatial_data, spatial_data])

            # Protect memory against huge images
            if spatial_data.shape[0] > 1e6:
                 use = np.linspace(0,spatial_data.shape[0]-1,int(1e6),dtype=int)
                 spatial_data = spatial_data[use,:]

            gmm.fit(spatial_data)
            central_angles = np.degrees(np.arctan2(gmm.means_[:, 1], gmm.means_[:, 0]))
            if num_points == 1:
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
                logging.warning(f'GMM angles {central_angles} span {np.sum(ca_quadrants)} quadrants, '
                                f'while data spans {np.sum(ca_quadrants)} quadrants')

            return central_angles

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

def load_climatology(config_path: str, latitude: float, longitude: float, acquisition_datetime: datetime,
                     isofit_path: str, lut_params: LUTConfig):
    """ Load climatology data, based on location and configuration

    Args:
        config_path: path to the base configuration directory for isofit
        latitude: latitude to set for the segment (mean of acquisition suggested)
        longitude: latitude to set for the segment (mean of acquisition suggested)
        acquisition_datetime: datetime to use for the segment( mean of acquisition suggested)
        isofit_path: base path to isofit installation (needed for data path references)
        lut_params: parameters to use to define lut grid

    :Returns
        tuple containing:
            aerosol_state_vector - A dictionary that defines the aerosol state vectors for isofit
            aerosol_lut_grid - A dictionary of the aerosol lookup table (lut) grid to be explored
            aerosol_model_path - A path to the location of the aerosol model to use with MODTRAN.

    """

    aerosol_model_path = os.path.join(isofit_path, 'data', 'aerosol_model.txt')
    aerosol_state_vector = {}
    aerosol_lut_grid = {}
    aerosol_lut_ranges = [lut_params.aerosol_0_range, lut_params.aerosol_1_range, lut_params.aerosol_2_range]
    aerosol_lut_spacing = [lut_params.aerosol_0_spacing, lut_params.aerosol_1_spacing, lut_params.aerosol_2_spacing]
    aerosol_lut_spacing_mins = [lut_params.aerosol_0_spacing_min, lut_params.aerosol_1_spacing_min, lut_params.aerosol_2_spacing_min]
    for _a, alr in enumerate(aerosol_lut_ranges):
        aerosol_lut = lut_params.get_grid(alr[0], alr[1], aerosol_lut_spacing[_a], aerosol_lut_spacing_mins[_a])
        if aerosol_lut is not None:
            aerosol_state_vector['AERFRAC_{}'.format(_a)] = {
                "bounds": [float(alr[0]), float(alr[1])],
                "scale": 1,
                "init": float((alr[1] - alr[0]) / 10. + alr[0]),
                "prior_sigma": 10.0,
                "prior_mean": float((alr[1] - alr[0]) / 10. + alr[0])}

            aerosol_lut_grid['AERFRAC_{}'.format(_a)] = aerosol_lut.tolist()

    aot_550_lut = lut_params.get_grid(lut_params.aot_550_range[0], lut_params.aot_550_range[1],
                                      lut_params.aot_550_spacing, lut_params.aot_550_spacing_min)
    if aot_550_lut is not None:
        aerosol_lut_grid['AOT550'] = aot_550_lut.tolist()
        alr = [aerosol_lut_grid['AOT550'][0], aerosol_lut_grid['AOT550'][-1]]
        aerosol_state_vector['AOT550'] = {
                        "bounds": [float(alr[0]), float(alr[1])],
                        "scale": 1,
                        "init": float((alr[1] - alr[0]) / 10. + alr[0]),
                        "prior_sigma": 10.0,
                        "prior_mean": float((alr[1] - alr[0]) / 10. + alr[0])}

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

def calc_modtran_max_water(paths: Pathnames) -> float:
    """MODTRAN may put a ceiling on "legal" H2O concentrations.  This function calculates that ceiling.  The intended
     use is to make sure the LUT does not contain useless gridpoints above it.

    Args:
        paths: object containing references to all relevant file locations

    Returns:
        max_water - maximum MODTRAN H2OSTR value for provided obs conditions
    """

    max_water = None
    # TODO: this is effectively redundant from the radiative_transfer->modtran. Either devise a way
    # to port in from there, or put in utils to reduce redundancy.
    xdir = {
        'linux': 'linux',
        'darwin': 'macos',
        'windows': 'windows'
    }
    name = 'H2O_bound_test'
    filebase = os.path.join(paths.lut_h2o_directory, name)
    with open(paths.h2o_template_path, 'r') as f:
        bound_test_config = json.load(f)

    bound_test_config['MODTRAN'][0]['MODTRANINPUT']['NAME'] = name
    bound_test_config['MODTRAN'][0]['MODTRANINPUT']['ATMOSPHERE']['H2OSTR'] = 50
    with open(filebase + '.json', 'w') as fout:
        fout.write(json.dumps(bound_test_config, cls=SerialEncoder, indent=4, sort_keys=True))

    cmd = os.path.join(paths.modtran_path, 'bin', xdir[platform], 'mod6c_cons ' + filebase + '.json')
    try:
        subprocess.call(cmd, shell=True, timeout=10, cwd=paths.lut_h2o_directory)
    except:
        pass

    with open(filebase + '.tp6', errors='ignore') as tp6file:
        for count, line in enumerate(tp6file):
            if 'The water column is being set to the maximum' in line:
                max_water = line.split(',')[1].strip()
                max_water = float(max_water.split(' ')[0])
                break

    if max_water is None:
        logging.error('Could not find MODTRAN H2O upper bound in file {}'.format(filebase + '.tp6'))
        raise KeyError('Could not find MODTRAN H2O upper bound')

    return max_water
