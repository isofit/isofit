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

import logging
import multiprocessing
import time
from copy import deepcopy

import click
import numpy as np
from scipy.linalg import inv
from scipy.spatial import KDTree
from spectral.io import envi

from isofit import configs, ray
from isofit.core.common import envi_header
from isofit.core.fileio import initialize_output, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.core.instrument import Instrument
from isofit.core.multistate import construct_full_state, update_config_for_surface


@ray.remote(num_cpus=1)
def _run_chunk(
    start_line: int,
    stop_line: int,
    reference_radiance_file: str,
    reference_reflectance_file: str,
    reference_uncertainty_file: str,
    reference_locations_file: str,
    input_radiance_file: str,
    input_locations_file: str,
    segmentation_file: str,
    isofit_config: str,
    output_reflectance_file: str,
    output_uncertainty_file: str,
    radiance_factors: np.array,
    nneighbors: int,
    nodata_value: float,
    loglevel: str,
    logfile: str,
    reference_class_file,
    input_obs_file: str = None,
    skyview_factor_file: str = None,
    bgrfl_file: str = None,
    output_tilt_file: str = None,
    tilt_fm: ForwardModel = None,
    subs_state_file: str = None,
) -> None:
    """
    Args:
        start_line: line to start empirical line run at
        stop_line:  line to stop empirical line run at
        reference_radiance_file: source file for radiance (interpolation built from this)
        reference_reflectance_file:  source file for reflectance (interpolation built from this)
        reference_uncertainty_file:  source file for uncertainty (interpolation built from this)
        reference_locations_file:  source file for file locations (lon, lat, elev), (interpolation built from this)
        input_radiance_file: input radiance file (interpolate over this)
        input_locations_file: input location file (interpolate over this)
        segmentation_file: input file noting the per-pixel segmentation used
        isofit_config: path to isofit configuration JSON file
        output_reflectance_file: location to write output reflectance to
        output_uncertainty_file: location to write output uncertainty to
        radiance_factors: radiance adjustment factors
        nneighbors: number of neighbors to use for interpolation
        nodata_value: nodata value of input and output
        loglevel: logging level
        logfile: logging file
        input_obs_file: input observation file, used for DEM-tilt coefficients
        skyview_factor_file: optional skyview factor file, used for DEM-tilt coefficients
        bgrfl_file: optional background reflectance file, used for DEM-tilt coefficients
        output_tilt_file: location to write DEM-tilt ratio PCA coefficients to
        tilt_fm: forward model, with tilt basis, used for DEM-tilt coefficients
        subs_state_file: superpixel state file, giving each pixel's atmosphere

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
    reference_radiance_img = envi.open(
        envi_header(reference_radiance_file), reference_radiance_file
    )
    reference_reflectance_img = envi.open(
        envi_header(reference_reflectance_file), reference_reflectance_file
    )
    reference_uncertainty_img = envi.open(
        envi_header(reference_uncertainty_file), reference_uncertainty_file
    )
    reference_locations_img = envi.open(
        envi_header(reference_locations_file), reference_locations_file
    )

    n_reference_lines, n_radiance_bands, n_reference_columns = [
        int(reference_radiance_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]
    n_reference_uncertainty_bands = int(reference_uncertainty_img.metadata["bands"])

    # Load input images
    input_radiance_img = envi.open(
        envi_header(input_radiance_file), input_radiance_file
    )
    n_input_lines, n_input_bands, n_input_samples = [
        int(input_radiance_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]

    input_locations_img = envi.open(
        envi_header(input_locations_file), input_locations_file
    )
    n_location_bands = int(input_locations_img.metadata["bands"])

    # Load output images
    output_reflectance_img = envi.open(
        envi_header(output_reflectance_file), output_reflectance_file
    )
    output_uncertainty_img = envi.open(
        envi_header(output_uncertainty_file), output_uncertainty_file
    )
    n_output_reflectance_bands = int(output_reflectance_img.metadata["bands"])
    n_output_uncertainty_bands = int(output_uncertainty_img.metadata["bands"])

    # Load reference data
    reference_locations_mm = reference_locations_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_locations = np.array(reference_locations_mm[:, :, :]).reshape(
        (n_reference_lines, n_location_bands)
    )

    reference_radiance_mm = reference_radiance_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_radiance = np.array(reference_radiance_mm[:, :, :]).reshape(
        (n_reference_lines, n_radiance_bands)
    )

    reference_reflectance_mm = reference_reflectance_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_reflectance = np.array(reference_reflectance_mm[:, :, :]).reshape(
        (n_reference_lines, n_radiance_bands)
    )

    reference_uncertainty_mm = reference_uncertainty_img.open_memmap(
        interleave="bip", writable=False
    )
    reference_uncertainty = np.array(reference_uncertainty_mm[:, :, :]).reshape(
        (n_reference_lines, n_reference_uncertainty_bands)
    )
    reference_uncertainty = reference_uncertainty[:, :n_radiance_bands].reshape(
        (n_reference_lines, n_radiance_bands)
    )

    if reference_class_file is not None:
        reference_class = np.squeeze(
            np.array(
                envi.open(
                    envi_header(reference_class_file), reference_class_file
                ).open_memmap(interleave="bip")
            )
        )
        un_reference_class = np.unique(reference_class)
        un_reference_class = un_reference_class[un_reference_class != -1]
        logging.info(f"Reference classes found: {un_reference_class}")
    else:
        reference_class = None

    # Load segmentation data
    if segmentation_file:
        segmentation_img = envi.open(envi_header(segmentation_file), segmentation_file)
        segmentation_img = segmentation_img.read_band(0)
    else:
        segmentation_img = None

    # Prepare instrument model, if available
    if isofit_config is not None:
        config = configs.create_new_config(isofit_config)
        instrument = Instrument(config)
        logging.info("Loading instrument")

        # Make sure the instrument is configured for single-pixel noise (no averaging)
        instrument.integrations = 1
    else:
        instrument = None

    # Load radiance factors
    if radiance_factors is None:
        radiance_adjustment = np.ones(
            n_radiance_bands,
        )
    else:
        radiance_adjustment = np.loadtxt(radiance_factors)

    # Load Tree
    loc_scaling = np.array([1e5, 1e5, 0.1])
    scaled_ref_loc = reference_locations * loc_scaling
    tree = KDTree(scaled_ref_loc)
    if reference_class is not None:
        trees = [
            KDTree(scaled_ref_loc[reference_class == _c, :])
            for _c in un_reference_class
        ]
    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the
    # Third element, and the first two are latitude/longitude coordinates

    # DEM-tilt coefficients use each pixel's superpixel atmospheric state
    if output_tilt_file:
        input_obs_img = envi.open(envi_header(input_obs_file), input_obs_file)
        subs_state = envi.open(envi_header(subs_state_file)).open_memmap(
            interleave="bip"
        )
        svf_mm = (
            envi.open(envi_header(skyview_factor_file)).open_memmap(interleave="bip")
            if skyview_factor_file
            else None
        )
        bgrfl_mm = (
            envi.open(envi_header(bgrfl_file)).open_memmap(interleave="bip")
            if bgrfl_file
            else None
        )
        n_tilt_bands = tilt_fm.tilt_basis.n_components

    # Iterate through image
    hash_table = {}

    for row in np.arange(start_line, stop_line):
        # Load inline input data
        input_radiance_mm = input_radiance_img.open_memmap(
            interleave="bip", writable=False
        )
        input_radiance = np.array(input_radiance_mm[row, :, :])
        input_radiance = input_radiance * radiance_adjustment

        input_locations_mm = input_locations_img.open_memmap(
            interleave="bip", writable=False
        )
        input_locations = np.array(input_locations_mm[row, :, :])

        output_reflectance_row = np.zeros(input_radiance.shape) + nodata_value
        output_uncertainty_row = np.zeros(input_radiance.shape) + nodata_value

        if output_tilt_file:
            input_obs_mm = input_obs_img.open_memmap(interleave="bip", writable=False)
            input_obs = np.array(input_obs_mm[row, :, :])
            del input_obs_mm
            output_tilt_row = np.zeros((n_input_samples, n_tilt_bands)) + nodata_value

        nspectra, start = 0, time.time()
        for col in np.arange(n_input_samples):
            x = input_radiance[col, :]
            if np.all(np.isclose(x, nodata_value)):
                output_reflectance_row[col, :] = nodata_value
                output_uncertainty_row[col, :] = nodata_value
                continue

            bhat = None
            if segmentation_img is not None:
                hash_idx = segmentation_img[row, col]
                if hash_idx in hash_table:
                    bhat, bmarg, bcov = hash_table[hash_idx]
                else:
                    loc = (
                        reference_locations[np.array(hash_idx, dtype=int), :]
                        * loc_scaling
                    )
            else:
                loc = input_locations[col, :] * loc_scaling

            if bhat is None:
                if reference_class is None:
                    dists, nn = tree.query(loc, nneighbors)
                    xv = reference_radiance[nn, :]
                    yv = reference_reflectance[nn, :]
                    uv = reference_uncertainty[nn, :]
                else:
                    dists, nn = tree.query(loc, 1)
                    loc_class = reference_class[nn]
                    tree_class = np.where(np.unique(reference_class) == loc_class)[0][0]
                    dists, nn = trees[tree_class].query(loc, nneighbors)
                    nn = nn[nn < np.sum(reference_class == loc_class)]
                    xv = reference_radiance[reference_class == loc_class, :][nn, :]
                    yv = reference_reflectance[reference_class == loc_class, :][nn, :]
                    uv = reference_uncertainty[reference_class == loc_class, :][nn, :]

                bhat = np.zeros((n_radiance_bands, 2))
                bmarg = np.zeros((n_radiance_bands, 2))
                bcov = np.zeros((n_radiance_bands, 2, 2))

                for i in np.arange(n_radiance_bands):
                    use = yv[:, i] > 0
                    n = sum(use)
                    X = np.concatenate((np.ones((n, 1)), xv[use, i : i + 1]), axis=1)
                    W = np.diag(np.ones(n))  # /uv[use, i])
                    y = yv[use, i : i + 1]
                    try:
                        bhat[i, :] = (inv(X.T @ W @ X) @ X.T @ W @ y).T
                        bcov[i, :, :] = inv(X.T @ W @ X)
                    except:
                        bhat[i, :] = 0
                        bcov[i, :, :] = 0
                    bmarg[i, :] = np.diag(bcov[i, :, :])

            if (segmentation_img is not None) and not (hash_idx in hash_table):
                hash_table[hash_idx] = bhat, bmarg, bcov

            A = np.array((np.ones(n_radiance_bands), x))
            output_reflectance_row[col, :] = np.multiply(bhat.T, A).sum(axis=0)

            # Calculate uncertainties.  Sy approximation rather than Seps for
            # speed, for now... but we do take into account instrument
            # radiometric uncertainties
            if instrument is None:
                output_uncertainty_row[col, :] = np.sqrt(
                    np.multiply(bmarg.T, A).sum(axis=0)
                )
            else:
                Sy = instrument.Sy(x, geom=None)
                calunc = instrument.bval[: instrument.n_chan]
                output_uncertainty_row[col, :] = (
                    np.sqrt(np.diag(Sy) + pow(calunc * x, 2)) * bhat[:, 1]
                )

            if output_tilt_file:
                if segmentation_img is not None:
                    subs_idx = int(hash_idx)
                else:
                    _, subs_idx = tree.query(input_locations[col, :] * loc_scaling, 1)
                state = subs_state[int(subs_idx), 0, tilt_fm.full_idx]

                geom = Geometry(
                    obs=input_obs[col, :],
                    loc=input_locations[col, :],
                    svf=svf_mm[row, col] if svf_mm is not None else 1,
                    bg_rfl=bgrfl_mm[row, col, :] if bgrfl_mm is not None else None,
                    coszen=tilt_fm.atmosphere.coszen,
                    full_config=config,
                )
                ratio = tilt_fm.calc_tilt_ratio(
                    output_reflectance_row[col, :],
                    state[tilt_fm.idx_atmosphere],
                    state[tilt_fm.idx_instrument],
                    geom,
                )
                output_tilt_row[col, :] = tilt_fm.tilt_basis.project(
                    ratio, fill_value=nodata_value
                )

            nspectra = nspectra + 1

        elapsed = float(time.time() - start)
        logging.info(
            "row {}/{}, ({}/{} local), {} spectra per second".format(
                row,
                n_input_lines,
                int(row - start_line),
                int(stop_line - start_line),
                round(float(nspectra) / max(elapsed, 1e-9), 2),
            )
        )

        del input_locations_mm
        del input_radiance_mm

        output_reflectance_row = output_reflectance_row.transpose((1, 0))
        output_uncertainty_row = output_uncertainty_row.transpose((1, 0))
        shp = output_reflectance_row.shape
        output_reflectance_row = output_reflectance_row.reshape((1, shp[0], shp[1]))
        shp = output_uncertainty_row.shape
        output_uncertainty_row = output_uncertainty_row.reshape((1, shp[0], shp[1]))

        write_bil_chunk(
            output_reflectance_row,
            output_reflectance_file,
            row,
            (n_input_lines, n_output_reflectance_bands, n_input_samples),
        )
        write_bil_chunk(
            output_uncertainty_row,
            output_uncertainty_file,
            row,
            (n_input_lines, n_output_uncertainty_bands, n_input_samples),
        )

        if output_tilt_file:
            write_bil_chunk(
                output_tilt_row.T.reshape((1, n_tilt_bands, n_input_samples)),
                output_tilt_file,
                row,
                (n_input_lines, n_tilt_bands, n_input_samples),
            )


def empirical_line(
    reference_radiance_file: str,
    reference_reflectance_file: str,
    reference_uncertainty_file: str,
    reference_locations_file: str,
    segmentation_file: str,
    input_radiance_file: str,
    input_locations_file: str,
    output_reflectance_file: str,
    output_uncertainty_file: str,
    nneighbors: int = 400,
    nodata_value: float = -9999.0,
    level: str = "INFO",
    logfile: str = None,
    radiance_factors: np.array = None,
    isofit_config: str = None,
    n_cores: int = -1,
    reference_class_file: str = None,
    segmentation_size: int = 40,
    input_obs_file: str = None,
    skyview_factor_file: str = None,
    bgrfl_file: str = None,
    output_tilt_file: str = None,
) -> None:
    """
    Perform an empirical line interpolation for reflectance and uncertainty extrapolation
    Args:
        reference_radiance_file: source file for radiance (interpolation built from this)
        reference_reflectance_file:  source file for reflectance (interpolation built from this)
        reference_uncertainty_file:  source file for uncertainty (interpolation built from this)
        reference_locations_file:  source file for file locations (lon, lat, elev), (interpolation built from this)
        segmentation_file: input file noting the per-pixel segmentation used
        input_radiance_file: input radiance file (interpolate over this)
        input_locations_file: input location file (interpolate over this)
        output_reflectance_file: location to write output reflectance to
        output_uncertainty_file: location to write output uncertainty to

        nneighbors: number of neighbors to use for interpolation
        nodata_value: nodata value of input and output
        level: logging level
        logfile: logging file
        radiance_factors: radiance adjustment factors
        isofit_config: path to isofit configuration JSON file
        n_cores: number of cores to run on
        reference_class_file: optional source file for sub-type-classifications, in order: [base, cloud, water]
        segmentation_size: Number of super pixels
        input_obs_file: input observation file, required for DEM-tilt coefficients
        skyview_factor_file: optional skyview factor file, used for DEM-tilt coefficients
        bgrfl_file: optional background reflectance file, used for DEM-tilt coefficients
        output_tilt_file: location to write DEM-tilt ratio PCA coefficients to. Uses
            the forward_model.atmosphere.tilt_basis_file and superpixel states of
            isofit_config
    Returns:
        None
    """

    loglevel = level

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    # Open input data to check that band formatting is correct
    # Load reference set radiance
    reference_radiance_img = envi.open(
        envi_header(reference_radiance_file), reference_radiance_file
    )
    n_reference_lines, n_radiance_bands, n_reference_columns = [
        int(reference_radiance_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]
    if n_reference_columns != 1:
        raise IndexError("Reference data should be a single-column list")

    # Load reference set reflectance
    reference_reflectance_img = envi.open(
        envi_header(reference_reflectance_file), reference_reflectance_file
    )
    nrefr, nbr, srefr = [
        int(reference_reflectance_img.metadata[n])
        for n in ("lines", "bands", "samples")
    ]
    if (
        nrefr != n_reference_lines
        or nbr != n_radiance_bands
        or srefr != n_reference_columns
    ):
        raise IndexError("Reference file dimension mismatch (reflectance)")

    # Load reference set uncertainty, assuming reflectance uncertainty is
    # recoreded in the first n_radiance_bands channels of data
    reference_uncertainty_img = envi.open(
        envi_header(reference_uncertainty_file), reference_uncertainty_file
    )
    nrefu, ns, srefu = [
        int(reference_uncertainty_img.metadata[n])
        for n in ("lines", "bands", "samples")
    ]
    if (
        nrefu != n_reference_lines
        or ns < n_radiance_bands
        or srefu != n_reference_columns
    ):
        raise IndexError("Reference file dimension mismatch (uncertainty)")

    # Load reference set locations
    reference_locations_img = envi.open(
        envi_header(reference_locations_file), reference_locations_file
    )
    nrefl, lb, ls = [
        int(reference_locations_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]
    if nrefl != n_reference_lines or lb != 3:
        raise IndexError("Reference file dimension mismatch (locations)")

    input_radiance_img = envi.open(
        envi_header(input_radiance_file), input_radiance_file
    )
    n_input_lines, n_input_bands, n_input_samples = [
        int(input_radiance_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]
    if n_radiance_bands != n_input_bands:
        msg = "Number of channels mismatch: input (%i) vs. reference (%i)"
        raise IndexError(msg % (nbr, n_radiance_bands))

    input_locations_img = envi.open(
        envi_header(input_locations_file), input_locations_file
    )
    nll, nlb, nls = [
        int(input_locations_img.metadata[n]) for n in ("lines", "bands", "samples")
    ]
    if nll != n_input_lines or nlb != 3 or nls != n_input_samples:
        raise IndexError("Input location dimension mismatch")

    # setup config
    if isofit_config is not None:
        iconfig = configs.create_new_config(isofit_config)
    else:
        # If none, create a temporary config to get default ray parameters
        iconfig = configs.Config({})
    if n_cores == -1:
        n_cores = iconfig.implementation.n_cores

    # Create output files
    output_metadata = input_radiance_img.metadata
    output_metadata["interleave"] = "bil"
    output_metadata["wavelength_unts"] = "Nanometers"
    isofit_version = iconfig.implementation.isofit_version
    engine_name = iconfig.forward_model.atmosphere.engine.name
    output_metadata["description"] = (
        f"L2A empirical line per-pixel surface retrieval (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
    )
    output_reflectance_img = initialize_output(
        output_metadata, output_reflectance_file, (nll, nlb, nls)
    )
    output_uncertainty_img = initialize_output(
        output_metadata, output_uncertainty_file, (nll, nlb, nls)
    )

    # Forward model and output for DEM-tilt ratio coefficients
    tilt_fm, subs_state_file = None, None
    if output_tilt_file:
        if isofit_config is None or input_obs_file is None:
            raise ValueError(
                "Empirical line tilt coefficients require isofit_config and "
                "input_obs_file"
            )
        issues = iconfig.get_tilt_issues()
        if issues:
            raise ValueError("; ".join(issues))

        # Only the atmosphere, instrument, and tilt basis are used, so the
        # first surface class serves for all pixels
        full_statevector, *_ = construct_full_state(deepcopy(iconfig))
        surface_class_str = next(iter(iconfig.forward_model.surface.Surfaces), None)
        tilt_fm = ForwardModel(
            update_config_for_surface(deepcopy(iconfig), surface_class_str)
        )
        tilt_fm.match_statevector(full_statevector)
        subs_state_file = str(iconfig.output.estimated_state_file)

        tilt_metadata = {
            "data type": 4,
            "file type": "ENVI Standard",
            "byte order": 0,
            "no data value": nodata_value,
            "lines": nll,
            "samples": nls,
            "bands": tilt_fm.tilt_basis.n_components,
            "interleave": "bil",
            "band names": tilt_fm.tilt_basis.band_names,
            "description": (
                f"L2A empirical line per-pixel DEM-tilt ratio PCA coefficients (tilt_basis_file={iconfig.forward_model.atmosphere.tilt_basis_file}, segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
            ),
        }
        if "map info" in output_metadata:
            tilt_metadata["map info"] = output_metadata["map info"]
        initialize_output(
            tilt_metadata,
            output_tilt_file,
            (nll, tilt_fm.tilt_basis.n_components, nls),
        )

    # Now cleanup inputs and outputs, we'll write dynamically above
    del output_reflectance_img, output_uncertainty_img
    del (
        reference_reflectance_img,
        reference_uncertainty_img,
        reference_locations_img,
        input_radiance_img,
        input_locations_img,
    )

    # Initialize ray cluster
    start_time = time.time()
    rayargs = {
        "ignore_reinit_error": iconfig.implementation.ray_ignore_reinit_error,
        "address": iconfig.implementation.ip_head,
        "_temp_dir": iconfig.implementation.ray_temp_dir,
        "include_dashboard": iconfig.implementation.ray_include_dashboard,
        "_redis_password": iconfig.implementation.redis_password,
    }

    # We can only set the num_cpus if running on a single-node
    if (
        iconfig.implementation.ip_head is None
        and iconfig.implementation.redis_password is None
    ):
        rayargs["num_cpus"] = n_cores

    ray.init(**rayargs)

    tilt_fm_ref = ray.put(tilt_fm) if tilt_fm is not None else None

    n_ray_cores = multiprocessing.cpu_count()
    n_cores = min(n_ray_cores, n_input_lines)

    logging.info(
        "Beginning empirical line inversions using {} cores".format(int(n_cores))
    )

    # Break data into sections
    line_sections = np.linspace(0, n_input_lines, num=int(n_cores + 1), dtype=int)

    start_time = time.time()

    # Run the pool (or run serially)
    results = []
    for l in range(len(line_sections) - 1):
        args = (
            line_sections[l],
            line_sections[l + 1],
            reference_radiance_file,
            reference_reflectance_file,
            reference_uncertainty_file,
            reference_locations_file,
            input_radiance_file,
            input_locations_file,
            segmentation_file,
            isofit_config,
            output_reflectance_file,
            output_uncertainty_file,
            radiance_factors,
            nneighbors,
            nodata_value,
            level,
            logfile,
            reference_class_file,
            input_obs_file,
            skyview_factor_file,
            bgrfl_file,
            output_tilt_file,
            tilt_fm_ref,
            subs_state_file,
        )
        results.append(_run_chunk.remote(*args))

    _ = ray.get(results)

    total_time = time.time() - start_time
    logging.info(
        "Parallel empirical line inversions complete.  {} s total, {} spectra/s, {}"
        " spectra/s/core".format(
            total_time,
            line_sections[-1] * n_input_samples / total_time,
            line_sections[-1] * n_input_samples / total_time / n_cores,
        )
    )


@click.command(name="empirical_line")
@click.argument("reference_radiance_file", type=str)
@click.argument("reference_reflectance_file", type=str)
@click.argument("reference_uncertainty_file", type=str)
@click.argument("reference_locations_file", type=str)
@click.argument("segmentation_file", type=str)
@click.argument("input_radiance_file", type=str)
@click.argument("input_locations_file", type=str)
@click.argument("output_reflectance_file", type=str)
@click.argument("output_uncertainty_file", type=str)
@click.option("--nneighbors", default=400, type=int, help="Number of neighbors")
@click.option("--nodata_value", default=-9999.0, type=float, help="Nodata value")
@click.option("--level", default="INFO", type=str, help="Logging level")
@click.option("--logfile", default=None, type=str, help="Log file path")
@click.option("--radiance_factors", default=None, type=str, help="Radiance factors")
@click.option("--isofit_config", default=None, type=str, help="Isofit config")
@click.option("--n_cores", default=-1, type=int, help="Number of cores")
@click.option(
    "--reference_class_file", default=None, type=str, help="Reference class file"
)
@click.option(
    "--input_obs_file", default=None, type=str, help="Obs file, for tilt coefficients"
)
@click.option(
    "--skyview_factor_file",
    default=None,
    type=str,
    help="Skyview factor file, for tilt coefficients",
)
@click.option(
    "--bgrfl_file",
    default=None,
    type=str,
    help="Background reflectance file, for tilt coefficients",
)
@click.option(
    "--output_tilt_file",
    default=None,
    type=str,
    help="Output file for DEM-tilt ratio PCA coefficients",
)
def cli(**kwargs):
    """Run empirical line"""
    empirical_line(**kwargs)
    click.echo("Done")
