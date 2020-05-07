#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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
plt.switch_backend("Agg")

writelock = multiprocessing.Lock()

def run_chunk(start_line, stop_line, reference_radiance_file, reference_reflectance_file, reference_uncertainty_file,
              reference_locations_file, input_radiance_file, input_locations_file, segmentation_file, isofit_config,
              output_reflectance_file, output_uncertainty_file, radiance_factors, eps, nneighbors, nodata_value,
              out_of_core=True):


    # Load reference images
    reference_radiance_img = envi.open(reference_radiance_file + '.hdr', reference_radiance_file)
    reference_reflectance_img = envi.open(reference_reflectance_file + '.hdr', reference_reflectance_file)
    reference_uncertainty_img = envi.open(reference_uncertainty_file + '.hdr', reference_uncertainty_file)
    reference_locations_img = envi.open(reference_locations_file + '.hdr', reference_locations_file)

    n_reference_lines, n_radiance_bands, n_reference_columns = [int(reference_radiance_img.metadata[n])
                                                                for n in ('lines', 'bands', 'samples')]
    n_reference_uncertainty_bands = int(reference_uncertainty_img.metadata['bands'])

    # Load input images
    input_radiance_img = envi.open(input_radiance_file + '.hdr', input_radiance_file)
    n_input_lines, n_input_bands, n_input_samples = [int(input_radiance_img.metadata[n])
                                        for n in ('lines', 'bands', 'samples')]

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

    reference_reflectance_mm = reference_reflectance_img.open_memmap(interleave='source', writable=False)
    reference_reflectance = np.array(reference_reflectance_mm[:, :, :]).reshape((n_reference_lines, n_radiance_bands))

    reference_uncertainty_mm = reference_uncertainty_img.open_memmap(interleave='source', writable=False)
    reference_uncertainty = np.array(reference_uncertainty_mm[:, :, :]).reshape((n_reference_lines,
                                                                                 n_reference_uncertainty_bands))
    reference_uncertainty = reference_uncertainty[:, :n_radiance_bands].reshape((n_reference_lines, n_radiance_bands))

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

    # Load Tree
    loc_scaling = np.array([1e5, 1e5, 0.1])
    scaled_ref_loc = reference_locations * loc_scaling
    tree = KDTree(scaled_ref_loc)
    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the
    # Third element, and the first two are latitude/longitude coordinates

    # Iterate through image
    hash_table = {}

    if out_of_core:
        output_reflectance_mm = output_reflectance_img.open_memmap(interleave='source',
                                             writable=True)
        output_uncertainty_mm = output_uncertainty_img.open_memmap(interleave='source',
                                             writable=True)
    else:
        output_reflectance = np.zeros((stop_line-start_line, n_output_reflectance_bands, n_input_samples)) + nodata_value
        output_uncertainty = np.zeros((stop_line-start_line, n_output_uncertainty_bands, n_input_samples)) + nodata_value



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

            x = input_radiance[col, :]
            if np.all(abs(x-nodata_value) < eps):
                output_reflectance_row[col, :] = nodata_value
                output_uncertainty_row[col, :] = nodata_value
                continue

            bhat = None
            if segmentation_img is not None:
                hash_idx = segmentation_img[row, col]
                if hash_idx in hash_table:
                    bhat, bmarg, bcov = hash_table[hash_idx]
                else:
                    loc = reference_locations[np.array(
                        hash_idx, dtype=int), :] * loc_scaling
            else:
                loc = input_locations[col, :] * loc_scaling

            if bhat is None:
                dists, nn = tree.query(loc, nneighbors)
                xv = reference_radiance[nn, :]
                yv = reference_reflectance[nn, :]
                uv = reference_uncertainty[nn, :]
                bhat = np.zeros((n_radiance_bands, 2))
                bmarg = np.zeros((n_radiance_bands, 2))
                bcov = np.zeros((n_radiance_bands, 2, 2))

                for i in np.arange(n_radiance_bands):
                    use = yv[:, i] > 0
                    n = sum(use)
                    X = np.concatenate((np.ones((n, 1)), xv[use, i:i+1]), axis=1)
                    W = np.diag(np.ones(n))  # /uv[use, i])
                    y = yv[use, i:i+1]
                    bhat[i, :] = (inv(X.T @ W @ X) @ X.T @ W @ y).T
                    bcov[i, :, :] = inv(X.T @ W @ X)
                    bmarg[i, :] = np.diag(bcov[i, :, :])

            if (segmentation_img is not None) and not (hash_idx in hash_table):
                hash_table[hash_idx] = bhat, bmarg, bcov

            A = np.array((np.ones(n_radiance_bands), x))
            output_reflectance_row[col, :] = (np.multiply(bhat.T, A).sum(axis=0))

            # Calculate uncertainties.  Sy approximation rather than Seps for
            # speed, for now... but we do take into account instrument
            # radiometric uncertainties
            if instrument is None:
                output_uncertainty_row[col, :] = np.sqrt(np.multiply(bmarg.T, A).sum(axis=0))
            else:
                Sy = instrument.Sy(x, geom=None)
                calunc = instrument.bval[:instrument.n_chan]
                output_uncertainty_row[col, :] = np.sqrt(
                    np.diag(Sy)+pow(calunc*x, 2))*bhat[:, 1]
            #if loglevel == 'DEBUG':
            #    plot_example(xv, yv, bhat)

            nspectra = nspectra+1

        elapsed = float(time.time()-start)
        logging.info('row %i/%i, %5.1f spectra per second' %
                     (row, n_input_lines, float(nspectra)/elapsed))

        del input_locations_mm
        del input_radiance_mm

        if input_radiance_img.metadata['interleave'] == 'bil':
            output_reflectance_row = output_reflectance_row.transpose((1, 0))
        if input_radiance_img.metadata['interleave'] == 'bil':
            output_uncertainty_row = output_uncertainty_row.transpose((1, 0))

        if out_of_core:
            output_reflectance_mm[row, :, :] = output_reflectance_row
            output_uncertainty_mm[row, :, :] = output_uncertainty_row

            del output_reflectance_mm
            del output_uncertainty_mm

            output_reflectance_mm = output_reflectance_img.open_memmap(interleave='source',
                                                                       writable=True)
            output_uncertainty_mm = output_uncertainty_img.open_memmap(interleave='source',
                                                                       writable=True)
        else:
            output_reflectance[row - start_line, :, :] = output_reflectance_row
            output_uncertainty[row - start_line, :, :] = output_uncertainty_row

    if out_of_core:
        return None, None, None, None
    else:
        return start_line, stop_line, output_reflectance, output_uncertainty




def plot_example(xv, yv, b):
    """Plot for debugging purposes."""

    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams['font.sans-serif'] = "Times"
    matplotlib.rcParams["legend.edgecolor"] = "None"
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.bottom"] = True
    matplotlib.rcParams["axes.spines.left"] = True
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['axes.grid.axis'] = 'both'
    matplotlib.rcParams['axes.grid.which'] = 'major'
    matplotlib.rcParams['legend.edgecolor'] = '1.0'
    plt.plot(xv[:, 113], yv[:, 113], 'ko')
    plt.plot(xv[:, 113], xv[:, 113]*b[113, 1] + b[113, 0], 'nneighbors')
    #plt.plot(x[113], x[113]*b[113, 1] + b[113, 0], 'ro')
    plt.grid(True)
    plt.xlabel('Radiance, $\mu{W }nm^{-1} sr^{-1} cm^{-2}$')
    plt.ylabel('Reflectance')
    plt.show(block=True)
    plt.savefig('empirical_line.pdf')


def empirical_line(reference_radiance_file, reference_reflectance_file, reference_uncertainty_file,
                   reference_locations_file, segmentation_file,
                   input_radiance_file, input_locations_file, output_reflectance_file, output_uncertainty_file,
                   nneighbors=15, nodata_value=-9999.0, level='INFO',
                   radiance_factors=None, isofit_config=None, n_cores=-1):
    """..."""


    eps = 1e-6
    loglevel = level

    logging.basicConfig(format='%(message)s', level=loglevel)

    # Open input data to check that band formatting is correct
    # Load reference set radiance
    reference_radiance_img = envi.open(reference_radiance_file + '.hdr', reference_radiance_file)
    n_reference_lines, n_radiance_bands, n_reference_columns = [int(reference_radiance_img.metadata[n])
                                                                for n in ('lines', 'bands', 'samples')]
    if n_reference_columns != 1:
        raise IndexError("Reference data should be a single-column list")

    # Load reference set reflectance
    reference_reflectance_img = envi.open(reference_reflectance_file + '.hdr', reference_reflectance_file)
    nrefr, nbr, srefr = [int(reference_reflectance_img.metadata[n]) for n in ('lines', 'bands', 'samples')]
    if nrefr != n_reference_lines or nbr != n_radiance_bands or srefr != n_reference_columns:
        raise IndexError("Reference file dimension mismatch (reflectance)")

    # Load reference set uncertainty, assuming reflectance uncertainty is
    # recoreded in the first n_radiance_bands channels of data
    reference_uncertainty_img = envi.open(reference_uncertainty_file + '.hdr', reference_uncertainty_file)
    nrefu, ns, srefu = [int(reference_uncertainty_img.metadata[n]) for n in ('lines', 'bands', 'samples')]
    if nrefu != n_reference_lines or ns < n_radiance_bands or srefu != n_reference_columns:
        raise IndexError("Reference file dimension mismatch (uncertainty)")

    # Load reference set locations
    reference_locations_img = envi.open(reference_locations_file + '.hdr', reference_locations_file)
    nrefl, lb, ls = [int(reference_locations_img.metadata[n]) for n in ('lines', 'bands', 'samples')]
    if nrefl != n_reference_lines or lb != 3:
        raise IndexError("Reference file dimension mismatch (locations)")

    input_radiance_img = envi.open(input_radiance_file + '.hdr', input_radiance_file)
    n_input_lines, n_input_bands, ns = [int(input_radiance_img.metadata[n])
                  for n in ('lines', 'bands', 'samples')]
    if n_radiance_bands != n_input_bands:
        msg = 'Number of channels mismatch: input (%i) vs. reference (%i)'
        raise IndexError(msg % (nbr, n_radiance_bands))

    input_locations_img = envi.open(input_locations_file + '.hdr', input_locations_file)
    nll, nlb, nls = [int(input_locations_img.metadata[n])
                     for n in ('lines', 'bands', 'samples')]
    if nll != n_input_lines or nlb != 3 or nls != ns:
        raise IndexError('Input location dimension mismatch')


    # Create output files
    output_reflectance_img = envi.create_image(output_reflectance_file + '.hdr', ext='',
                                               metadata=input_radiance_img.metadata, force=True)

    output_uncertainty_img = envi.create_image(output_uncertainty_file + '.hdr', ext='',
                                               metadata=input_radiance_img.metadata, force=True)

    # Now cleanup inputs and outputs, we'll write dynamically above
    del output_reflectance_img, output_uncertainty_img
    del reference_reflectance_img, reference_uncertainty_img, reference_locations_img, input_radiance_img, input_locations_img 

    # Determine the number of cores to use
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()
    n_cores = min(n_cores, n_input_lines)

    # Break data into sections
    line_sections = np.linspace(0, n_input_lines, num=n_cores+1, dtype=int)

    # Set up our pool
    pool = multiprocessing.Pool(processes=n_cores)
    start_time = time.time()
    logging.info('Beginning empirical line inversions using {} cores'.format(n_cores))

    # This shouldn't be necessary, but parallel write, despite significant effort with various locks, crashes the
    # system.  Possibly due to print statements in spectral.io.envi....consider trying with pure memmaps or gdal.
    # For now, if in parallel (n_cores != 1), run in-core, meaning substantial memory is required.  Keeping n_cores
    # at 1 preserves original out-of-core functionality
    out_of_core = n_cores == 1

    # Run the pool (or run serially)
    results = []
    for l in range(len(line_sections) - 1):
        args = (line_sections[l], line_sections[l+1], reference_radiance_file, reference_reflectance_file,
                reference_uncertainty_file, reference_locations_file, input_radiance_file,
                input_locations_file, segmentation_file, isofit_config, output_reflectance_file,
                output_uncertainty_file, radiance_factors, eps, nneighbors, nodata_value, out_of_core, )
        if n_cores != 1:
            results.append(pool.apply_async(run_chunk, args))
        else:
            run_chunk(*args)

    results = [p.get() for p in results]
    pool.close()
    pool.join()

    total_time = time.time() - start_time
    logging.info('Parallel empirical line inversions complete.  {} s total, {} spectra/s, {}/spectra/core'.format(
        total_time, line_sections[-1] / total_time, line_sections[-1] / total_time / n_cores))

    # Write results as necessary
    if n_cores > 1:
        logging.info('Beginning data write')
        output_reflectance_img = envi.open(output_reflectance_file + '.hdr', output_reflectance_file)
        output_uncertainty_img = envi.open(output_uncertainty_file + '.hdr', output_uncertainty_file)

        for output in results:
            # unpack
            start_line, stop_line, reflectance, uncertainty = output
            if start_line is None or stop_line is None or reflectance is None or uncertainty is None:
                logging.error('Chunk {}-{} failed to return valid results',format(start_line,stop_line))
            else:
                logging.info('Writing lines {}-{}'.format(start_line, stop_line))
                output_reflectance_mm = output_reflectance_img.open_memmap(interleave='source', writable=True)
                output_uncertainty_mm = output_uncertainty_img.open_memmap(interleave='source', writable=True)

                output_reflectance_mm[start_line:stop_line, ...] = reflectance
                output_uncertainty_mm[start_line:stop_line, ...] = uncertainty

                # Flush to keep memory from going too wild
                del output_reflectance_mm, output_uncertainty_mm


