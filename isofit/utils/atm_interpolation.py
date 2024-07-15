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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#
import logging
import multiprocessing
import time
from typing import List

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from spectral.io import envi

from isofit import ray
from isofit.core.common import envi_header
from isofit.core.fileio import write_bil_chunk
from isofit.core.forward import ForwardModel


@ray.remote(num_cpus=1)
def _run_chunk(
    start_line: int,
    stop_line: int,
    reference_state_file: str,
    reference_locations_file: str,
    input_locations_file: str,
    segmentation_file: str,
    output_atm_file: str,
    atm_band_names: list,
    nneighbors: list,
    nodata_value: float,
    loglevel: str,
    logfile: str,
) -> None:
    """
    Args:
        start_line:               line to start empirical line run at
        stop_line:                line to stop empirical line run at
        reference_state_file:     source file for retrieved superpixel state
        reference_locations_file: source file for file locations (lon, lat, elev), (interpolation built from this)
        input_locations_file:     input location file (interpolate over this)
        segmentation_file:        input file noting the per-pixel segmentation used
        output_atm_file:          output file for interpolated and smoothed per pixel atmospheric state
        atm_band_names:           names of atmospheric state parameters
        nneighbors:               number of neighbors to use for interpolation
        nodata_value:             nodata value of input and output
        loglevel:                 logging level
        logfile:                  logging file
    Returns:
        None
    """
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    # Load reference images
    reference_state_img = envi.open(
        envi_header(reference_state_file), reference_state_file
    )
    reference_locations_img = envi.open(
        envi_header(reference_locations_file), reference_locations_file
    )

    n_reference_lines, n_state_bands, n_reference_columns = [
        int(reference_state_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]

    # Load input images
    input_locations_img = envi.open(
        envi_header(input_locations_file), input_locations_file
    )
    n_location_bands = int(input_locations_img.metadata["bands"])
    n_input_samples = input_locations_img.shape[1]
    n_input_lines = input_locations_img.shape[0]

    # Load reference data
    reference_locations_mm = reference_locations_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_locations = np.array(reference_locations_mm[:, :, :]).reshape(
        (n_reference_lines, n_location_bands)
    )

    atm_bands = np.where(
        np.array(
            [x in atm_band_names for x in reference_state_img.metadata["band names"]]
        )
    )[0]
    n_atm_bands = len(atm_bands)
    reference_state_mm = reference_state_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_state = np.array(reference_state_mm[:, :, atm_bands]).reshape(
        (n_reference_lines, n_atm_bands)
    )

    # Load segmentation data
    if segmentation_file:
        segmentation_img = envi.open(envi_header(segmentation_file), segmentation_file)
        segmentation_img = segmentation_img.read_band(0)
    else:
        segmentation_img = None

    # Load Tree
    loc_scaling = np.array([1e6, 1e6, 0.01])
    scaled_ref_loc = reference_locations * loc_scaling
    tree = KDTree(scaled_ref_loc)
    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the
    # Third element, and the first two are latitude/longitude coordinates

    if len(nneighbors) == 1 and n_atm_bands != 1:
        logging.debug("assuming neighbors applies to all atm bands")
        nneighbors = [nneighbors[0] for n in range(n_atm_bands)]

    # Iterate through image
    hash_table = {}
    for row in np.arange(start_line, stop_line):
        if not np.all(segmentation_img[row, :] == 0):
            # Load inline input data
            input_locations_mm = input_locations_img.open_memmap(
                interleave="bip", writable=False
            )
            input_locations = np.array(input_locations_mm[row, :, :])
            output_atm_row = np.zeros((n_input_samples, len(atm_bands))) + nodata_value
            nspectra, start = 0, time.time()

            for col in np.arange(n_input_samples):
                x = input_locations[col, :]

                if np.isclose(x, nodata_value).all():
                    output_atm_row[col, :] = nodata_value
                    continue
                else:
                    x *= loc_scaling

                bhat = None
                hash_idx = segmentation_img[row, col]
                if hash_idx in hash_table:
                    bhat = hash_table[hash_idx]

                if bhat is None:
                    # Use the max number of neighbors...we'll zero out unwanted components below
                    dists, nn = tree.query(x, np.max(nneighbors))
                    xv = reference_locations[nn, :] * loc_scaling[np.newaxis, :]
                    yv = reference_state[nn, :]

                    # Zero out unwanted components
                    for _nneigh, nneigh in enumerate(nneighbors):
                        yv[nneigh:, _nneigh] = -10  # set to get filtered

                    bhat = np.zeros((n_atm_bands, xv.shape[1]))

                    for i in np.arange(n_atm_bands):
                        use = yv[:, i] > -5
                        n = sum(use)
                        # only use lat/lon here, ignore Z
                        X = np.concatenate((np.ones((n, 1)), xv[use, :-1]), axis=1)
                        W = np.diag(np.ones(n))  # /uv[use, i])
                        y = yv[use, i : i + 1]
                        try:
                            bhat[i, :] = (inv(X.T @ W @ X) @ X.T @ W @ y).T
                        except:
                            bhat[i, :] = 0

                        # if i == 0:
                        #    print(X, y, bhat)

                if (segmentation_img is not None) and not (hash_idx in hash_table):
                    hash_table[hash_idx] = bhat

                A = np.hstack((np.ones(1), x[:-1]))
                output_atm_row[col, :] = (bhat.T * A[:, np.newaxis]).sum(axis=0)

                nspectra = nspectra + 1

            elapsed = float(time.time() - start)
            logging.debug(
                "row {}/{}, ({}/{} local), {} spectra per second".format(
                    row,
                    n_input_lines,
                    int(row - start_line),
                    int(stop_line - start_line),
                    round(float(nspectra) / elapsed, 2),
                )
            )

            del input_locations_mm

            output_atm_row = output_atm_row.transpose((1, 0))
        else:
            output_atm_row = np.zeros((n_atm_bands, segmentation_img.shape[1])) - 9999

        write_bil_chunk(
            output_atm_row,
            output_atm_file,
            row,
            (n_input_lines, n_atm_bands, n_input_samples),
        )


def atm_interpolation(
    reference_state_file: str,
    reference_locations_file: str,
    input_locations_file: str,
    segmentation_file: str,
    output_atm_file: str,
    atm_band_names: list,
    nneighbors: list = None,
    nodata_value: float = -9999.0,
    loglevel: str = "INFO",
    logfile: str = None,
    n_cores: int = -1,
    gaussian_smoothing_sigma: list = None,
) -> None:
    """
    Perform an empirical line interpolation and gaussian smoothing to atmospheric parameters.
    Args:
        reference_state_file:     source file for retrieved superpixel state
        reference_locations_file: source file for file locations (lon, lat, elev), (interpolation built from this)
        input_locations_file:     input location file (interpolate over this)
        segmentation_file:        input file noting the per-pixel segmentation used
        output_atm_file:          output file for interpolated and smoothed per pixel atmospheric state
        atm_band_names:           names of atmospheric state parameters
        nneighbors:               number of neighbors to use for interpolation
        nodata_value:             nodata value of input and output
        loglevel:                 logging level
        logfile:                  logging file
        n_cores:                  number of cores to run on
        gaussian_smoothing_sigma: sigma value to apply to gaussian smoothing of atmospheric parameters
    Returns:
        None
    """
    if nneighbors is None:
        nneighbors = [400]
    if gaussian_smoothing_sigma is None:
        gaussian_smoothing_sigma = [2]
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    reference_state_img = envi.open(envi_header(reference_state_file))
    input_locations_img = envi.open(envi_header(input_locations_file))

    n_input_lines = int(input_locations_img.metadata["lines"])
    n_input_samples = int(input_locations_img.metadata["samples"])

    # Create output files
    output_metadata = reference_state_img.metadata
    output_metadata["interleave"] = "bil"
    output_metadata["lines"] = input_locations_img.metadata["lines"]
    output_metadata["samples"] = input_locations_img.metadata["samples"]

    output_metadata["band names"] = atm_band_names
    output_metadata["description"] = "Interpolated atmospheric state"
    output_metadata["bands"] = len(atm_band_names)
    output_metadata["bbl"] = (
        "{" + ",".join([str(1) for n in range(len(atm_band_names))]) + "}"
    )

    del output_metadata["fwhm"]
    del output_metadata["wavelength"]

    output_atm_img = envi.create_image(
        envi_header(output_atm_file), ext="", metadata=output_metadata, force=True
    )

    # Now cleanup inputs and outputs, we'll write dynamically above
    del output_atm_img, reference_state_img, input_locations_img

    # Initialize ray cluster
    start_time = time.time()
    ray.init(
        **{"ignore_reinit_error": True, "local_mode": n_cores == 1, "num_cpus": n_cores}
    )

    n_cores = min(n_cores, n_input_lines)

    logging.info(f"Beginning atmospheric interpolation {n_cores} cores")

    # Break data into sections
    line_sections = np.linspace(0, n_input_lines, num=int(n_cores + 1), dtype=int)

    start_time = time.time()

    # Run the pool (or run serially)
    args = (
        reference_state_file,
        reference_locations_file,
        input_locations_file,
        segmentation_file,
        output_atm_file,
        atm_band_names,
        nneighbors,
        nodata_value,
        loglevel,
        logfile,
    )
    jobs = [
        _run_chunk.remote(line_sections[l], line_sections[l + 1], *args)
        for l in range(len(line_sections) - 1)
    ]
    _ = [ray.get(jid) for jid in jobs]

    total_time = time.time() - start_time
    logging.info(
        "Parallel atmospheric interpolations complete.  {} s total, {} spectra/s, {}"
        " spectra/s/core".format(
            total_time,
            line_sections[-1] * n_input_samples / total_time,
            line_sections[-1] * n_input_samples / total_time / n_cores,
        )
    )

    atm_img = (
        envi.open(envi_header(output_atm_file)).open_memmap(interleave="bip").copy()
    )

    if len(gaussian_smoothing_sigma) == 1 and atm_img.shape[-1] != 1:
        logging.debug("assuming gaussian smoothing applies to all atm bands")
        gaussian_smoothing_sigma = [
            gaussian_smoothing_sigma[0] for n in range(atm_img.shape[-1])
        ]

    for n in range(atm_img.shape[-1]):
        if gaussian_smoothing_sigma[n] > 0:
            null = atm_img[..., n] == -9999
            V = atm_img[..., n]
            V[null] = 0
            VV = gaussian_filter(V, sigma=gaussian_smoothing_sigma[n])

            W = 0 * atm_img[..., n] + 1
            W[null] = 0
            WW = gaussian_filter(W, sigma=gaussian_smoothing_sigma[n])

            smoothed = VV / WW
            atm_img[..., n] = smoothed

    atm_img = atm_img.transpose((0, 2, 1))
    write_bil_chunk(atm_img, output_atm_file, 0, atm_img.shape)
