#! /usr/bin/env python3
#
#  Copyright 2020 California Institute of Technology
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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

from scipy.linalg import inv
from isofit.core.instrument import Instrument
from spectral.io import envi
from scipy.spatial import KDTree
import numpy as np
import logging
import time
import matplotlib
import pylab as plt
from isofit.configs import configs
import multiprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA

plt.switch_backend("Agg")


def _write_bil_chunk(dat: np.array, outfile: str, line: int, shape: tuple, dtype: str = 'float32') -> None:
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()


def _run_chunk(start_line: int, stop_line: int, reference_radiance_file: str, reference_atm_file: str,
               reference_locations_file: str, input_radiance_file: str,
               input_locations_file: str, segmentation_file: str, isofit_config: dict, output_reflectance_file: str,
               output_uncertainty_file: str, radiance_factors: np.array, nneighbors: int,
               nodata_value: float) -> None:
    """
    Args:
        start_line: line to start empirical line run at
        stop_line:  line to stop empirical line run at
        reference_radiance_file: source file for radiance (interpolation built from this)
        reference_atm_file:  source file for atmosphere coefficients (interpolation built from this)
        reference_locations_file:  source file for file locations (lon, lat, elev), (interpolation built from this)
        input_radiance_file: input radiance file (interpolate over this)
        input_locations_file: input location file (interpolate over this)
        segmentation_file: input file noting the per-pixel segmentation used
        isofit_config: dictionary-stype isofit configuration
        output_reflectance_file: location to write output reflectance to
        output_uncertainty_file: location to write output uncertainty to
        radiance_factors: radiance adjustment factors
        nneighbors: number of neighbors to use for interpolation
        nodata_value: nodata value of input and output

    Returns:
        None

    """

    # Load reference images
    reference_radiance_img = envi.open(reference_radiance_file + '.hdr', reference_radiance_file)
    reference_atm_img = envi.open(reference_atm_file + '.hdr', reference_atm_file)
    reference_locations_img = envi.open(reference_locations_file + '.hdr', reference_locations_file)

    n_reference_lines, n_radiance_bands, n_reference_columns = [int(reference_radiance_img.metadata[n])
                                                                for n in ('lines', 'bands', 'samples')]

    # Load input images
    input_radiance_img = envi.open(input_radiance_file + '.hdr', input_radiance_file)
    n_input_lines, n_input_bands, n_input_samples = [int(input_radiance_img.metadata[n])
                                                     for n in ('lines', 'bands', 'samples')]
    wl = np.array([float(w) for w in input_radiance_img.metadata['wavelength']])

    input_locations_img = envi.open(input_locations_file + '.hdr', input_locations_file)
    n_location_bands = int(input_locations_img.metadata['bands'])

    # Load output images
    output_reflectance_img = envi.open(output_reflectance_file + '.hdr', output_reflectance_file)
    output_uncertainty_img = envi.open(output_uncertainty_file + '.hdr', output_uncertainty_file)
    n_output_reflectance_bands = int(output_reflectance_img.metadata['bands'])
    n_output_uncertainty_bands = int(output_uncertainty_img.metadata['bands'])

    # Load reference data
    reference_locations_mm = reference_locations_img.open_memmap(interleave='source', writable=False)
    reference_locations = np.array(reference_locations_mm[:, :, :]).reshape((n_reference_lines, n_location_bands))

    reference_radiance_mm = reference_radiance_img.open_memmap(interleave='source', writable=False)
    reference_radiance = np.array(reference_radiance_mm[:, :, :]).reshape((n_reference_lines, n_radiance_bands))

    reference_atm_mm = reference_atm_img.open_memmap(interleave='source', writable=False)
    reference_atm = np.array(reference_atm_mm[:, :, :]).reshape((n_reference_lines, n_radiance_bands*5))
    rhoatm = reference_atm[:,:n_radiance_bands]
    sphalb = reference_atm[:,n_radiance_bands:(n_radiance_bands*2)] 
    transm = reference_atm[:,(n_radiance_bands*2):(n_radiance_bands*3)] 
    solirr = reference_atm[:,(n_radiance_bands*3):(n_radiance_bands*4)] 
    coszen = reference_atm[:,(n_radiance_bands*4):(n_radiance_bands*5)]  

    # Load segmentation data
    if segmentation_file:
        segmentation_img = envi.open(segmentation_file + '.hdr', segmentation_file)
        segmentation_img = segmentation_img.read_band(0)
    else:
        segmentation_img = None

    # Prepare instrument model, if available
    if isofit_config is not None:
        config = configs.create_new_config(isofit_config)
        instrument = Instrument(config)
        logging.info('Loading instrument')
    else:
        instrument = None

    # Load radiance factors
    if radiance_factors is None:
        radiance_adjustment = np.ones(n_radiance_bands, )
    else:
        radiance_adjustment = np.loadtxt(radiance_factors)

    # PCA coefficients
    rdn_pca = PCA(n_components=2)
    reference_pca = rdn_pca.fit_transform(reference_radiance * radiance_adjustment)

    # Create the tree to find nearest neighbor segments.
    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the
    # Third element, and the first two are latitude/longitude coordinates
    # The fourth and fifth elements are "spectral distance" determined by the
    # top principal component coefficients
    loc_scaling = np.array([1e5, 1e5, 10, 100, 100])
    scaled_ref_loc = np.concatenate((reference_locations,reference_pca),axis=1) * loc_scaling
    tree = KDTree(scaled_ref_loc)

    # Fit GP parameters on transmissivity of an H2O feature, in the 
    # first 400 datapoints
    use = np.arange(min(len(rhoatm),400))
    h2oband = np.argmin(abs(wl-940))
    scale = (500,500,500,500,500)
    bounds = ((100,2000),(100,2000),(100,2000),(100,2000),(100,2000))
    kernel =  RBF(length_scale=scale, length_scale_bounds=bounds) +\
                  WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 0.1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
    gp = gp.fit(scaled_ref_loc[use,:], transm[use,h2oband])
    kernel = gp.kernel_
    

    # Iterate through image.  Each segment has its own GP, stored in a 
    # hash table indexed by location in the segmentation map
    hash_table = {}

    for row in np.arange(start_line, stop_line):

        # Load inline input data
        input_radiance_mm = input_radiance_img.open_memmap(
            interleave='source', writable=False)
        input_radiance = np.array(input_radiance_mm[row, :, :])
        if input_radiance_img.metadata['interleave'] == 'bil':
            input_radiance = input_radiance.transpose((1, 0))
        input_radiance = input_radiance * radiance_adjustment

        input_locations_mm = input_locations_img.open_memmap(
            interleave='source', writable=False)
        input_locations = np.array(input_locations_mm[row, :, :])
        if input_locations_img.metadata['interleave'] == 'bil':
            input_locations = input_locations.transpose((1, 0))

        output_reflectance_row = np.zeros(input_radiance.shape) + nodata_value
        output_uncertainty_row = np.zeros(input_radiance.shape) + nodata_value

        nspectra, start = 0, time.time()
        for col in np.arange(n_input_samples):

            # Get radiance, pca coordinates, physical location for this datum
            my_rdn = input_radiance[col, :]
            my_pca = rdn_pca.transform(my_rdn[np.newaxis,:])
            my_loc = np.r_[input_locations[col, :], my_pca[0,:]] * loc_scaling

            if np.all(np.isclose(my_rdn, nodata_value)):
                output_reflectance_row[col, :] = nodata_value
                output_uncertainty_row[col, :] = nodata_value
                continue

            # Retrieve or build the GP
            gp_rhoatm, gp_sphalb, gp_transm, irr = None, None, None, None
            hash_idx = segmentation_img[row, col]
            if hash_idx in hash_table:
                gp_rhoatm, gp_sphalb, gp_transm, irr = hash_table[hash_idx]
            else:

                # There is no GP for this segment, so we build one from 
                # the atmospheric coefficients from closest neighbors
                dists, nn = tree.query(my_loc, nneighbors)
                neighbor_rhoatm = rhoatm[nn, :]
                neighbor_transm = transm[nn, :]
                neighbor_sphalb = sphalb[nn, :]
                neighbor_coszen = coszen[nn, :]
                neighbor_solirr = solirr[nn, :]
                neighbor_locs = scaled_ref_loc[nn, :]

                # Create a new GP using the optimized parameters as a fixed kernel
                gp_rhoatm = GaussianProcessRegressor(kernel=kernel, alpha=0.0, 
                      normalize_y=True, optimizer=None)
                gp_rhoatm.fit(neighbor_locs, neighbor_rhoatm)
                gp_sphalb = GaussianProcessRegressor(kernel=kernel, alpha=0.0, 
                      normalize_y=True, optimizer=None)
                gp_sphalb.fit(neighbor_locs, neighbor_sphalb)
                gp_transm = GaussianProcessRegressor(kernel=kernel, alpha=0.0, 
                      normalize_y=True, optimizer=None)
                gp_transm.fit(neighbor_locs, neighbor_transm)
                irr = solirr[1,:]*coszen[1,:]
                irr[irr<1e-8] = 1e-8

                hash_table[hash_idx] = (gp_rhoatm, gp_sphalb, gp_transm, irr)

            my_rhoatm = gp_rhoatm.predict(my_loc[np.newaxis,:])
            my_sphalb = gp_sphalb.predict(my_loc[np.newaxis,:])
            my_transm = gp_transm.predict(my_loc[np.newaxis,:])
            my_rho = (my_rdn * np.pi) / irr
            my_rfl = 1.0 / (my_transm / (my_rho - my_rhoatm) + my_sphalb)
            output_reflectance_row[col, :] = my_rfl

            # Calculate uncertainties.  Sy approximation rather than Seps for
            # speed, for now... but we do take into account instrument
            # radiometric uncertainties
            #output_uncertainty_row[col, :] = np.zeros()
            #if instrument is None:
            #else:
            #    Sy = instrument.Sy(x, geom=None)
            #    calunc = instrument.bval[:instrument.n_chan]
            #    output_uncertainty_row[col, :] = np.sqrt(
            #        np.diag(Sy) + pow(calunc * x, 2)) * bhat[:, 1]
            # if loglevel == 'DEBUG':
            #    plot_example(xv, yv, bhat)

            nspectra = nspectra + 1

        elapsed = float(time.time() - start)
        logging.info('row {}/{}, ({}/{} local), {} spectra per second'.format(row, n_input_lines, int(row - start_line),
                                                                              int(stop_line - start_line),
                                                                              round(float(nspectra) / elapsed, 2)))

        del input_locations_mm
        del input_radiance_mm

        output_reflectance_row = output_reflectance_row.transpose((1, 0))
        output_uncertainty_row = output_uncertainty_row.transpose((1, 0))
        shp = output_reflectance_row.shape
        output_reflectance_row = output_reflectance_row.reshape((1, shp[0], shp[1]))
        shp = output_uncertainty_row.shape
        output_uncertainty_row = output_uncertainty_row.reshape((1, shp[0], shp[1]))

        _write_bil_chunk(output_reflectance_row, output_reflectance_file, row,
                         (n_input_lines, n_output_reflectance_bands, n_input_samples))
        _write_bil_chunk(output_uncertainty_row, output_uncertainty_file, row,
                         (n_input_lines, n_output_uncertainty_bands, n_input_samples))



def interpolate_atmosphere(reference_radiance_file: str, reference_atm_file: str, 
                   reference_locations_file: str, segmentation_file: str, input_radiance_file: str,
                   input_locations_file: str, output_reflectance_file: str, output_uncertainty_file: str,
                   nneighbors: int = 15, nodata_value: float = -9999.0, level: str = 'INFO',
                   radiance_factors: np.array = None, isofit_config: dict = None, n_cores: int = -1) -> None:
    """
    Perform a Gaussian process interpolation of atmospheric parameters.  It relies on precalculated
    atmospheric coefficients at a subset of spatial locations stored in a file.  The file has 
    each coefficient defined for every radiance channel, appearing in the order: (1) atmospheric
    path reflectance; (2) spherical sky albedo; (3) total diffuse and direct transmittance of the 
    two-part downwelling and upwelling path; (4) extraterrestrial solar irradiance; (5) cosine of solar
    zenith angle.
    Args:
        reference_radiance_file: source file for radiance (interpolation built from this)
        reference_atm_file:  source file for atmospheric coefficients (interpolation from this)
        reference_locations_file:  source file for file locations (lon, lat, elev), (interpolation from this)
        segmentation_file: input file noting the per-pixel segmentation used
        input_radiance_file: input radiance file (interpolate over this)
        input_locations_file: input location file (interpolate over this)
        output_reflectance_file: location to write output reflectance
        output_uncertainty_file: location to write output uncertainty

        nneighbors: number of neighbors to use for interpolation
        nodata_value: nodata value of input and output
        level: logging level
        radiance_factors: radiance adjustment factors
        isofit_config: dictionary-stype isofit configuration
        n_cores: number of cores to run on
    Returns:
        None
    """

    loglevel = level

    logging.basicConfig(format='%(message)s', level=loglevel)

    # Open input data to check that band formatting is correct
    # Load reference set radiance
    reference_radiance_img = envi.open(reference_radiance_file + '.hdr', reference_radiance_file)
    n_reference_lines, n_radiance_bands, n_reference_columns = [int(reference_radiance_img.metadata[n])
                                                                for n in ('lines', 'bands', 'samples')]
    if n_reference_columns != 1:
        raise IndexError("Reference data should be a single-column list")

    # Load reference set atmospheric coefficients
    reference_atm_img = envi.open(reference_atm_file + '.hdr', reference_atm_file)
    nrefa, nba, srefa = [int(reference_atm_img.metadata[n]) for n in ('lines', 'bands', 'samples')]
    if nrefa != n_reference_lines  or srefa != n_reference_columns:
        raise IndexError("Reference file dimension mismatch (atmosphere)")
    if nba  != (n_radiance_bands * 5):
        raise IndexError("Reference atmosphere file has incorrect dimensioning")

    # Load reference set locations
    reference_locations_img = envi.open(reference_locations_file + '.hdr', reference_locations_file)
    nrefl, lb, ls = [int(reference_locations_img.metadata[n]) for n in ('lines', 'bands', 'samples')]
    if nrefl != n_reference_lines or lb != 3:
        raise IndexError("Reference file dimension mismatch (locations)")

    input_radiance_img = envi.open(input_radiance_file + '.hdr', input_radiance_file)
    n_input_lines, n_input_bands, n_input_samples = [int(input_radiance_img.metadata[n])
                                                     for n in ('lines', 'bands', 'samples')]
    if n_radiance_bands != n_input_bands:
        msg = 'Number of channels mismatch: input (%i) vs. reference (%i)'
        raise IndexError(msg % (n_input_bands, n_radiance_bands))

    input_locations_img = envi.open(input_locations_file + '.hdr', input_locations_file)
    nll, nlb, nls = [int(input_locations_img.metadata[n])
                     for n in ('lines', 'bands', 'samples')]
    if nll != n_input_lines or nlb != 3 or nls != n_input_samples:
        raise IndexError('Input location dimension mismatch')

    # Create output files
    output_metadata = input_radiance_img.metadata
    output_metadata['interleave'] = 'bil'
    output_reflectance_img = envi.create_image(output_reflectance_file + '.hdr', ext='',
                                               metadata=output_metadata, force=True)

    output_uncertainty_img = envi.create_image(output_uncertainty_file + '.hdr', ext='',
                                               metadata=output_metadata, force=True)

    # Now cleanup inputs and outputs, we'll write dynamically above
    del output_reflectance_img, output_uncertainty_img
    del reference_atm_img, reference_locations_img, input_radiance_img, input_locations_img

    # Determine the number of cores to use
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()
    n_cores = min(n_cores, n_input_lines)

    # Break data into sections
    line_sections = np.linspace(0, n_input_lines, num=n_cores + 1, dtype=int)

    # Set up our pool
    pool = multiprocessing.Pool(processes=n_cores)
    start_time = time.time()
    logging.info('Beginning atmospheric interpolation inversions using {} cores'.format(n_cores))

    # Run the pool (or run serially)
    results = []
    for l in range(len(line_sections) - 1):
        args = (line_sections[l], line_sections[l + 1], reference_radiance_file, reference_atm_file,
                reference_locations_file, input_radiance_file,
                input_locations_file, segmentation_file, isofit_config, output_reflectance_file,
                output_uncertainty_file, radiance_factors, nneighbors, nodata_value,)
        if n_cores != 1:
            results.append(pool.apply_async(_run_chunk, args))
        else:
            _run_chunk(*args)

    pool.close()
    pool.join()

    total_time = time.time() - start_time
    logging.info('Parallel empirical line inversions complete.  {} s total, {} spectra/s, {} spectra/s/core'.format(
        total_time, line_sections[-1] * n_input_samples / total_time,
                    line_sections[-1] * n_input_samples / total_time / n_cores))
