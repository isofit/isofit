



import argparse
import os
from os.path import join, exists, split, abspath
from shutil import copyfile
from datetime import datetime
from spectral.io import envi
import logging
import json
from osgeo import gdal
import numpy as np
from sklearn import mixture
import subprocess
from sys import platform
from typing import List
import yaml
from collections import OrderedDict

from template_construction import build_main_config, build_presolve_config, Pathnames, calc_modtran_max_water


from isofit.utils import segment, extractions, empirical_line
from isofit.core import isofit, common
from isofit.core.common import envi_header
from isofit.utils import segment, extractions, empirical_line



def main(rawargs=None):


    parser = argparse.ArgumentParser(description="Apply OE to a block of data with mixed surface.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('config_file', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--modtran_path', type=str)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args(rawargs)


    # Check files exist
    infiles = {'input_radiance': args.input_radiance, 'input_loc': args.input_loc, 
               'input_obs': args.input_obs, 'config_file': args.config_file }
    for infile_name, infile in infiles.itmes():
        if os.path.isfile(infile) is False:
            err_str = f'Input argument {infile_name} give as: {infile}.  File not found on system.'
            raise ValueError('argument ' + err_str)

    # Check file sizes match
    rdn_dataset = gdal.Open(args.input_radiance, gdal.GA_ReadOnly)
    rdn_size = (rdn_dataset.RasterXSize, rdn_dataset.RasterYSize)
    del rdn_dataset
    for infile_name, infile in infiles.itmes():
        if infile_name != 'input_radiance':
            input_dataset = gdal.Open(infile, gdal.GA_ReadOnly)
            input_size = (input_dataset.RasterXSize, input_dataset.RasterYSize)
            if not (input_size[0] == rdn_size[0] and input_size[1] == rdn_size[1]):
                err_str = f'Input file: {infile_name} size is {input_size}, which does not match input_radiance size: {rdn_size}'
                raise ValueError(err_str)

    with open(args.config_file, 'r') as f:
        config = OrderedDict(yaml.safe_load(f))
    gip = config['general_inversion_parameters']
    wf = config['workflow']
    tsip = config['type_specific_inversion_parameters']

    # Calc mask

    # Chunk scene
    if not exists(paths.lbl_working_path) or not exists(paths.radiance_working_path):
        logging.info('Segmenting...')
        segment(spectra=(paths.radiance_working_path, paths.lbl_working_path),
                nodata_value=-9999, npca=5, segsize=gip['segmentation_size'], nchunk=wf['chunksize'],
                n_cores=args.n_cores, loglevel=args.logging_level, logfile=args.log_file)

    # Extract input data per segment
    for inp, outp in [(paths.radiance_working_path, paths.rdn_subs_path),
                      (paths.obs_working_path, paths.obs_subs_path),
                      (paths.loc_working_path, paths.loc_subs_path)]:
        if not exists(outp):
            logging.info('Extracting ' + outp)
            extractions(inputfile=inp, labels=paths.lbl_working_path, output=outp,
                        chunksize=wf["chunksize"], flag=-9999, n_cores=wf["n_cores"],
                        loglevel=args.logging_level, logfile=args.log_file)



    if len(tsip.items()) > 0:

        #TODO: class_subs_path is new
        define_surface_types(paths.rdn_subs_path, paths.loc_subs_path, paths.obs_subs_path, paths.class_subs_path, list(tsip.keys()))

        surface_types = envi.open(envi_header(paths.class_subs_path)).open_memmap(interleave='bip').copy()
        un_surface_types = np.unique(surface_types)

        # Break up input files based on surface type
        for _st, surface_type in enumerate(list(tsip.keys())):
            paths.add_surface_subs_files(surface_type)
            copy_file_subset(surface_types == _st, [(paths.rdn_subs_path, paths.surface_subs_files[surface_type]['rdn']), 
                                                    (paths.loc_subs_path, paths.surface_subs_files[surface_type]['loc']),
                                                    (paths.obs_subs_path, paths.surface_subs_files[surface_type]['obs'])] )
            

    # We will use the model discrepancy with covariance OR uncorrelated 
    # Calibration error, but not both.
    if gip['model_discrepancy_path'] is not None:
        uncorrelated_radiometric_uncertainty = 0
    else:
        uncorrelated_radiometric_uncertainty = gip['uncorrelated_radiometric_uncertainty'] 


    if wf['presolve_wv']:
        if not exists(envi_header(paths.h2o_subs_path)) or not exists(paths.h2o_subs_path):
            if gip['emulator_base'] is None:
                max_water = calc_modtran_max_water(paths)
            else:
                max_water = 6

            h2o_grid = np.linspace(0.01, max_water - 0.01, 10).round(2)

            logging.info(f'Pre-solve H2O grid: {h2o_grid}')
            logging.info('Writing H2O pre-solve configuration file.')
            build_presolve_config(paths, h2o_grid, args.n_cores, args.empirical_line == 1, args.surface_category,
                args.emulator_base, uncorrelated_radiometric_uncertainty)


            # Run retrieval
            logging.info('Run ISOFIT presolve')
            retrieval_h2o = isofit.Isofit(paths.h2o_config_path, level='INFO', logfile=args.log_file)
            retrieval_h2o.run()
            del retrieval_h2o

            # clean up unneeded storage
            if args.emulator_base is None:
                for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
                    cmd = 'rm ' + join(paths.lut_h2o_directory, to_rm)
                    logging.info(cmd)
                    os.system(cmd) 

        else:
            logging.info('Existing h2o-presolve solutions found, using those.')



        h2o = envi.open(envi_header(paths.h2o_subs_path))
        h2o_est = h2o.read_band(-1)[:].flatten()

        p05 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 5)
        p95 = np.percentile(h2o_est[h2o_est > lut_params.h2o_min], 95)
        margin = (p95-p05) * 0.25

        lut_params.h2o_range[0] = max(lut_params.h2o_min, p05 - margin)
        lut_params.h2o_range[1] = min(max_water, max(lut_params.h2o_min, p95 + margin))

        h2o_lut_grid = lut_params.get_grid(lut_params.h2o_range[0], lut_params.h2o_range[1], lut_params.h2o_spacing, lut_params.h2o_spacing_min)








def copy_file_subset(matching_indices:np.array, pathnames:List):
    """Copy over subsets of given files to new locations 

    Args:
        matching_indices (np.array): indices to select from (y dimension) from source dataset
        pathnames (List): list of tuples (input_filename, output_filename) to read/write to/from
    """
    for inp, outp in pathnames:
        input_ds = envi.open(envi_header(inp), inp)
        header = input_ds.metadata.copy()
        header['lines'] = np.sum(matching_indices)
        output_ds = envi.create_image(envi_header(outp), header, ext='', force=True)
        output_mm = output_ds.open_memmap(interleave='bip', writable=True)
        input_mm = input_ds.open_memmap(interleave='bip', writable=True)
        output_mm[...] = input_mm[matching_indices,...].copy()
    



if __name__ == "__main__":
    main()

