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

import os
from types import SimpleNamespace

import click
import numpy as np
import scipy
from scipy.linalg import inv
from sklearn.cluster import KMeans
from spectral.io import envi

from isofit.core import units
from isofit.core.common import envi_header, expand_path, json_load_ascii, svd_inv


def next_diag_val(C: np.ndarray, starting_index, direction):
    if direction == 1:
        for i in range(starting_index, C.shape[0]):
            if np.isnan(C[i, i]) == False:
                return C[i, i]
    if direction == -1:
        for i in range(starting_index, -1, -1):
            if np.isnan(C[i, i]) == False:
                return C[i, i]

    return None


def surface_model(
    config_path: str,
    wavelength_path: str = None,
    output_path: str = None,
    seed: int = 13,
    multisurface: bool = False,
) -> None:
    """The surface model tool contains everything you need to build basic
    multicomponent (i.e. colleciton of Gaussian) surface priors for the
    multicomponent surface model.

    Args:
        config_path: path to a JSON formatted surface model configuration
        wavelength_path: optional path to a three-column wavelength file,
           overriding the configuration file settings
        output_path: optional path to the destination .mat file, overriding
           the configuration file settings
        seed: seed used for clustering
    Returns:
        None
    """

    # Set seed for random draws later
    np.random.seed(seed)

    # Load configuration JSON into a local dictionary
    configdir, _ = os.path.split(os.path.abspath(config_path))
    config = json_load_ascii(config_path, shell_replace=True)

    if wavelength_path:
        wavelength_file = wavelength_path
    else:
        if "wavelength_file" not in config:
            raise ValueError(
                "Missing config parameter 'wavelength_file', set via surface_model(wavelength_path=...)"
            )
        wavelength_file = expand_path(configdir, config["wavelength_file"])

    if output_path:
        outfile = output_path

    else:
        if "output_model_file" not in config:
            raise ValueError(
                "Missing config parameter 'output_model_file', set via surface_model(output_path=...)"
            )
        outfile = expand_path(configdir, config["output_model_file"])

    # Determine top level parameters
    for q in ["sources", "normalize"]:
        if q not in config:
            raise ValueError("Missing parameter: %s" % q)

    normalize = config["normalize"]
    reference_windows = config["reference_windows"]

    # Get selection metric if it exists
    selection_metric = config.get("selection_metric", "Euclidean")

    # load wavelengths file, and change units to nm if needed
    if os.path.splitext(wavelength_file)[-1] == ".hdr":
        ds = envi.open(wavelength_file)
        wl = np.array([float(x) for x in ds.metadata["wavelength"]])
    else:
        q = np.loadtxt(wavelength_file)
        if q.shape[1] > 2:
            q = q[:, 1:]
        wl = q[:, 0]

    if wl[0] < 100:
        wl = units.micron_to_nm(wl)
    nchan = len(wl)

    # build global reference windows
    refwl = []
    for wi, window in enumerate(reference_windows):
        active_wl = np.logical_and(wl >= window[0], wl < window[1])
        refwl.extend(wl[active_wl])
    normind = np.array([np.argmin(abs(wl - w)) for w in refwl])
    refwl = np.array(refwl, dtype=float)

    # create basic model template
    model = {
        "normalize": normalize,
        "selection_metric": selection_metric,
        "wl": wl,
        "means": [],
        "covs": [],
        "attribute_means": [],
        "attribute_covs": [],
        "attributes": [],
        "refwl": refwl,
        "surface_categories": [],
    }

    # each "source" (i.e. spectral library) is treated separately
    for si, source_config in enumerate(config["sources"]):
        # Determine source parameters
        for q in ["input_spectrum_files", "windows", "n_components", "windows"]:
            if q not in source_config:
                raise ValueError("Source %i is missing a parameter: %s" % (si, q))

        if multisurface:
            if "surface_category" not in source_config:
                raise ValueError(
                    "Source %i is missing a parameter: surface_category" % (si)
                )

        # Determine whether we should synthesize our own mixtures
        if "mixtures" in source_config:
            mixtures = source_config["mixtures"]
        elif "mixtures" in config:
            mixtures = config["mixtures"]
        else:
            mixtures = 0

        # open input files associated with this source
        infiles = [
            expand_path(configdir, fi) for fi in source_config["input_spectrum_files"]
        ]

        # associate attributes, if they exist. These will not be used
        # in the retrieval, but can be used in post-analysis
        if "input_attribute_files" in source_config:
            infiles_attributes = [
                expand_path(configdir, fi)
                for fi in source_config["input_attribute_files"]
            ]
            if len(infiles_attributes) != len(infiles):
                raise IndexError("spectrum / attribute file mismatch")
        else:
            infiles_attributes = [None for fi in source_config["input_spectrum_files"]]

        ncomp = int(source_config["n_components"])
        windows = source_config["windows"]

        # Surface model handling
        surface_category = source_config.get("surface_category", "")

        # load spectra
        spectra, attributes = [], []
        for infile, attribute_file in zip(infiles, infiles_attributes):
            rfl = envi.open(envi_header(infile), infile)
            nl, nb, ns = [int(rfl.metadata[n]) for n in ("lines", "bands", "samples")]
            swl = np.array([float(f) for f in rfl.metadata["wavelength"]])

            # Maybe convert to nanometers
            if swl[0] < 100:
                swl = units.micron_to_nm(swl)

            # Load library and adjust interleave, if needed
            rfl_mm = rfl.open_memmap(interleave="bip", writable=False)
            x = np.array(rfl_mm[:, :, :])
            x = x.reshape(nl * ns, nb)

            # import spectra and resample
            for x1 in x:
                p = scipy.interpolate.interp1d(
                    swl, x1, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                spectra.append(p(wl))

            # Load attributes
            if attribute_file is not None:
                attr = envi.open(envi_header(attribute_file), attribute_file)
                nla, nba, nsa = [
                    int(attr.metadata[n]) for n in ("lines", "bands", "samples")
                ]

                # Load library and adjust interleave, if needed
                attr_mm = attr.open_memmap(interleave="bip", writable=False)
                x = np.array(attr_mm[:, :, :])
                x = x.reshape(nla * nsa, nba)
                model["attributes"] = attr.metadata["band names"]

                # import spectra and resample
                for x1 in x:
                    attributes.append(x1)

        if len(attributes) > 0 and len(attributes) != len(spectra):
            raise IndexError("Mismatch in number of spectra vs. attributes")

        # calculate mixtures, if needed
        if len(attributes) > 0 and mixtures > 0:
            raise ValueError("Synthetic mixtures w/ attributes is not advised")

        n = float(len(spectra))
        nmix = int(n * mixtures)
        for mi in range(nmix):
            s1, m1 = spectra[int(np.random.rand() * n)], np.random.rand()
            s2, m2 = spectra[int(np.random.rand() * n)], 1.0 - m1
            spectra.append(m1 * s1 + m2 * s2)

        # Lists to arrays
        spectra = np.array(spectra)
        attributes = np.array(attributes)

        # Flag bad data
        use = np.all(np.isfinite(spectra), axis=1)
        spectra = spectra[use, :]
        if len(attributes) > 0:
            attributes = attributes[use, :]

        ## Accumulate total list of window indices
        # window_idx = -np.ones((nchan), dtype=int)
        # for wi, win in enumerate(windows):
        #    active_wl = np.logical_and(wl >= win['interval'][0], wl < win['interval'][1])
        #    window_idx[active_wl] = wi

        # Two step model generation.  First step is k-means clustering.
        # This is more "stable" than Expectation Maximization with an
        # unconstrained covariance matrix
        kmeans = KMeans(
            init="k-means++", n_clusters=ncomp, n_init=10, random_state=seed
        )
        kmeans.fit(spectra)
        Z = kmeans.predict(spectra)

        # Build a combined dataset of attributes and spectra
        if len(attributes) > 0:
            spectra_attr = np.concatenate((spectra, attributes), axis=1)

        # now fit the full covariance for each component
        for ci in range(ncomp):
            print(ci, source_config["input_spectrum_files"])

            m = np.mean(spectra[Z == ci, :], axis=0)
            C = np.cov(spectra[Z == ci, :], rowvar=False)
            C_base = C.copy()

            if len(attributes) > 0:
                m_attr = np.mean(spectra_attr[Z == ci, :], axis=0)
                C_attr = np.cov(spectra_attr[Z == ci, :], rowvar=False)

            # for i in range(nchan):
            for window in windows:
                window_idx = np.where(
                    np.logical_and(
                        wl >= window["interval"][0], wl < window["interval"][1]
                    )
                )[0]
                if len(window_idx) == 0:
                    continue
                window_range = slice(window_idx[0], window_idx[-1] + 1)

                # To minimize bias, leave the channels independent
                # and uncorrelated
                if window["correlation"] == "decorrelated":
                    c_diag = (
                        C[window_range, window_range] + float(window["regularizer"])
                    ) * np.eye(len(window_idx))
                    C[window_range, :] = 0
                    C[:, window_range] = 0
                    C[window_range, window_range] = c_diag

            for window in windows:
                window_idx = np.where(
                    np.logical_and(
                        wl >= window["interval"][0], wl < window["interval"][1]
                    )
                )[0]
                if len(window_idx) == 0:
                    continue
                window_range = slice(window_idx[0], window_idx[-1] + 1)

                # Each spectral interval, or window, is constructed
                # using one of several rules.  We can draw the covariance
                # directly from the data...
                if window["correlation"] in ["EM", "EM-gauss"]:
                    cdiag = (
                        C_base[window_range, window_range]
                        + np.eye(len(window_idx)) * float(window["regularizer"])
                    ).copy()
                    if "isolated" in list(window.keys()) and window["isolated"] == 1:
                        C[window_range, :] = 0
                        C[:, window_range] = 0
                    C[window_range, window_range] = cdiag
                window_idx = np.where(
                    np.logical_and(
                        wl >= window["interval"][0], wl < window["interval"][1]
                    )
                )[0]
                if len(window_idx) == 0:
                    continue
                window_range = slice(window_idx[0], window_idx[-1] + 1)

                if window["correlation"] == "GP":
                    for i in window_idx:
                        # Alternatively, we can use a band diagonal form,
                        # a Gaussian process that promotes local smoothness.
                        width = float(window["gp_width"])
                        magnitude = float(window["gp_magnitude"])
                        kernel = scipy.stats.norm.pdf((wl - wl[i]) / width)
                        kernel = kernel / kernel.sum() * magnitude
                        C[i, :] = kernel
                        C[:, i] = kernel
                        C[i, i] = C[i, i] + float(window["regularizer"])

                elif window["correlation"] in ["decorrelated", "EM"]:
                    # already handled
                    continue

                else:
                    raise ValueError(
                        "I do not recognize the method " + window["correlation"]
                    )

            # Now do any cross-block feathering by augmenting the precision matrix
            P = inv(C)
            for window in windows:
                window_idx = np.where(
                    np.logical_and(
                        wl >= window["interval"][0], wl < window["interval"][1]
                    )
                )[0]
                if len(window_idx) == 0:
                    continue

                # Look for the "feather_forward" or "feather_backward" options
                if window["correlation"] in ["EM", "decorrelated"]:
                    if "feather_backward" in list(window.keys()):
                        P[window_idx[0] - 1, window_idx[0]] -= (
                            1.0 / window["feather_backward"]
                        )
                        P[window_idx[0], window_idx[0] - 1] -= (
                            1.0 / window["feather_backward"]
                        )
                    if "feather_forward" in list(window.keys()):
                        P[window_idx[-1] + 1, window_idx[-1]] -= (
                            1.0 / window["feather_forward"]
                        )
                        P[window_idx[-1], window_idx[-1] + 1] -= (
                            1.0 / window["feather_forward"]
                        )
            C = inv(P)

            # Normalize the component spectrum if desired
            if normalize == "Euclidean":
                z = np.sqrt(np.sum(pow(m[normind], 2)))
            elif normalize == "RMS":
                z = np.sqrt(np.mean(pow(m[normind], 2)))
            elif normalize == "None":
                z = 1.0
            else:
                raise ValueError("Unrecognized normalization: %s\n" % normalize)
            m = m / z
            C = C / (z**2)

            try:
                Cinv = svd_inv(C)
            except:
                errstr = f"C from group {si, ci} is not invertible."
                raise AttributeError(errstr)

            model["means"].append(m)
            model["covs"].append(C)

            if len(attributes) > 0:
                model["attribute_means"].append(m_attr)
                model["attribute_covs"].append(C_attr)

            model["surface_categories"].append(surface_category)

    model["means"] = np.array(model["means"])
    model["covs"] = np.array(model["covs"])
    model["attribute_means"] = np.array(model["attribute_means"])
    model["attribute_covs"] = np.array(model["attribute_covs"])

    if multisurface:
        # Divide up model dict based on surface_type
        surface_categories = np.unique(model["surface_categories"])
        for surface_category in surface_categories:
            i = np.argwhere(np.array(model["surface_categories"]) == surface_category)

            type_model = model.copy()
            type_model["means"] = np.squeeze(model["means"][i])
            type_model["covs"] = np.squeeze(model["covs"][i])

            # To handle surfaces with 1 component, this check returns true
            if type_model["means"].ndim == 1 and type_model["covs"].ndim == 2:
                type_model["means"] = type_model["means"][np.newaxis, :]
                type_model["covs"] = type_model["covs"][np.newaxis, :]

            type_model["surface_categories"] = [
                str(stype[0]) for stype in np.array(model["surface_categories"])[i]
            ]
            if len(model["attribute_means"]):
                type_model["attribute_means"] = np.squeeze(model["attribute_means"][i])
                type_model["attribute_covs"] = np.squeeze(model["attribute_covs"][i])

            name, ext = os.path.splitext(outfile)
            type_outfile = f"{name}_{str(surface_category)}{ext}"
            scipy.io.savemat(type_outfile, type_model)

    else:
        scipy.io.savemat(outfile, model)


# Input arguments
@click.command(name="surface_model")
@click.argument("config_path")
@click.option(
    "--wavelength_path", help="Input wavelengths to three-column wavelength file to use"
)
@click.option("--output_path", help="Path to write the created surface model to")
@click.option("--seed", default=13, type=int, help="Seed for reproducibility")
def cli(**kwargs):
    """Build a new surface model to a block of data"""

    # SimpleNamespace converts a dict into dot-notational
    surface_model(**kwargs)
    click.echo("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "surface_model.py cannot be called this way.  Run as:\n isofit surface_model [CONFIG_PATH]"
    )
