#! /usr/bin/env python3
#
#  Copyright 2026 California Institute of Technology
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

import click
import numpy as np
from scipy.io import savemat
from spectral.io import envi

from isofit.core.common import envi_header


def read_ratio_spectra(path: str, max_samples: int, rng: np.random.Generator):
    """Read ratio spectra from an ENVI cube (bands = channels, with a
    wavelength header field) or a text file whose first row is the wavelength
    grid and whose other rows are spectra. At most max_samples spectra are
    drawn at random.

    Returns:
        (wl, spectra): wavelengths (n_wl) and spectra (n_samples x n_wl)
    """
    if path.endswith(".txt"):
        data = np.atleast_2d(np.loadtxt(path))
        wl, spectra = data[0], data[1:]
    else:
        img = envi.open(envi_header(path))
        if "wavelength" not in img.metadata:
            raise ValueError(f"No wavelength field in the header of {path}")
        wl = np.array([float(w) for w in img.metadata["wavelength"]])
        spectra = img.open_memmap(interleave="bip").reshape(-1, len(wl))

    if len(spectra) > max_samples:
        spectra = spectra[np.sort(rng.choice(len(spectra), max_samples, False))]

    return wl, np.array(spectra, dtype=float)


def build_tilt_basis(
    input_files: list,
    output_file: str,
    n_components: int = None,
    variance_fraction: float = 0.999,
    max_samples: int = 100000,
    nodata: float = -9999.0,
    seed: int = 0,
):
    """Fit a PCA basis to spectra of the ratio between DEM-tilted and flat
    reflectance, and write it to a .mat file for
    forward_model.atmosphere.tilt_basis_file.\n

    args:

        input_files:        ENVI cubes or .txt files of ratio spectra sharing
                            one wavelength grid.

        output_file:        Output .mat file, holding wl, mean, bases,
                            eigenvalues, explained_variance_ratio, n_samples.

        n_components:       Number of components to keep. If None, keep the
                            fewest reaching variance_fraction.

        variance_fraction:  Explained variance fraction used if n_components
                            is None.

        max_samples:        Maximum number of spectra used for the fit.

        nodata:             Spectra containing this value are dropped.

        seed:               Random seed for subsampling.

    returns:
        None
    """
    rng = np.random.default_rng(seed)

    wl, spectra = None, []
    for path in input_files:
        file_wl, file_spectra = read_ratio_spectra(path, max_samples, rng)
        if wl is None:
            wl = file_wl
        elif len(file_wl) != len(wl) or not np.allclose(file_wl, wl):
            raise ValueError(f"Wavelengths of {path} differ from {input_files[0]}")
        spectra.append(file_spectra)

    X = np.concatenate(spectra, axis=0)
    good = np.all(np.isfinite(X), axis=1) & ~np.any(np.isclose(X, nodata), axis=1)
    X = X[good]
    if len(X) < 2:
        raise ValueError(f"Only {len(X)} valid ratio spectra found")
    if len(X) > max_samples:
        X = X[np.sort(rng.choice(len(X), max_samples, False))]

    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    eigenvalues = s**2 / (len(X) - 1)
    total = eigenvalues.sum()
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    if n_components is None:
        n_components = int(np.searchsorted(np.cumsum(explained), variance_fraction))
        n_components += 1
    n_components = max(1, min(n_components, len(eigenvalues)))
    bases = Vt[:n_components]

    recon = mean + (X - mean) @ bases.T @ bases
    rms = np.sqrt(np.mean((recon - X) ** 2))
    logging.info(
        f"Tilt basis: {n_components} components from {len(X)} spectra explain "
        f"{explained[:n_components].sum():.6f} of the variance, "
        f"reconstruction RMS {rms:.3e}"
    )

    savemat(
        output_file,
        {
            "wl": wl,
            "mean": mean,
            "bases": bases,
            "eigenvalues": eigenvalues[:n_components],
            "explained_variance_ratio": explained[:n_components],
            "n_samples": len(X),
            "description": (
                "PCA basis for the ratio of DEM-tilted to flat reflectance, "
                f"built from {len(X)} spectra"
            ),
        },
    )


@click.command(
    name="build_tilt_basis", help=build_tilt_basis.__doc__, no_args_is_help=True
)
@click.argument("input_files", nargs=-1, required=True)
@click.argument("output_file")
@click.option("--n_components", type=int, default=None)
@click.option("--variance_fraction", type=float, default=0.999)
@click.option("--max_samples", type=int, default=100000)
@click.option("--nodata", type=float, default=-9999.0)
@click.option("--seed", type=int, default=0)
def cli(input_files, **kwargs):
    """Fit a PCA basis for DEM-tilt ratio coefficients"""

    logging.basicConfig(format="%(levelname)s:%(message)s", level="INFO")
    build_tilt_basis(list(input_files), **kwargs)
    click.echo("Done")
