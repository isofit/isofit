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
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from isofit.core import units

Logger = logging.getLogger(__file__)


class TiltBasis:
    """PCA model of the spectral ratio between DEM-tilted and flat
    (geoid-relative) reflectance, resampled to the instrument wavelengths.

    The .mat file contains:
        wl:    wavelength centers of the bases, in nm (or microns if < 100)
        bases: PCA components, one per row (n_components x n_wl)
        mean:  optional mean ratio spectrum (n_wl), zero if absent

    Coefficients are fit by least squares over the inversion-window channels,
    since resampled bases are no longer exactly orthonormal.
    """

    def __init__(self, path: str, wl: np.ndarray, windows: list = None):
        """
        Args:
            path:    tilt basis .mat file
            wl:      instrument wavelength centers (nm)
            windows: inversion windows [[lo, hi], ...] (nm) used for the fit;
                     all channels are used if None
        """
        D = loadmat(path)
        for key in ["wl", "bases"]:
            if key not in D:
                raise ValueError(f"Tilt basis file {path} is missing key '{key}'")

        wl_basis = np.asarray(D["wl"], dtype=float).ravel()
        if wl_basis[0] < 100:
            wl_basis = units.micron_to_nm(wl_basis)

        bases = np.atleast_2d(np.asarray(D["bases"], dtype=float))
        if bases.shape[1] != len(wl_basis) and bases.shape[0] == len(wl_basis):
            bases = bases.T
        if bases.shape[1] != len(wl_basis):
            raise ValueError(
                f"Tilt basis 'bases' shape {bases.shape} does not match "
                f"{len(wl_basis)} wavelengths in {path}"
            )

        mean = (
            np.asarray(D["mean"], dtype=float).ravel()
            if "mean" in D
            else np.zeros(len(wl_basis))
        )
        if len(mean) != len(wl_basis):
            raise ValueError(
                f"Tilt basis 'mean' has {len(mean)} values but there are "
                f"{len(wl_basis)} wavelengths in {path}"
            )

        if not (np.all(np.isfinite(bases)) and np.all(np.isfinite(mean))):
            raise ValueError(f"Non-finite values in tilt basis file {path}")

        self.wl = np.asarray(wl, dtype=float)
        if self.wl.min() < wl_basis.min() or self.wl.max() > wl_basis.max():
            Logger.warning(
                f"Instrument wavelengths extend beyond the tilt basis range "
                f"[{wl_basis.min()}, {wl_basis.max()}] nm; extrapolating"
            )

        # Resample to instrument wavelengths
        self.mean = interp1d(wl_basis, mean, fill_value="extrapolate")(self.wl)
        self.bases = interp1d(wl_basis, bases, axis=1, fill_value="extrapolate")(
            self.wl
        )

        # Channels used for the fit
        if windows is None:
            self.winidx = np.arange(len(self.wl))
        else:
            self.winidx = np.array((), dtype=int)
            for lo, hi in windows:
                idx = np.where(np.logical_and(self.wl > lo, self.wl < hi))[0]
                self.winidx = np.concatenate((self.winidx, idx), axis=0)

        if len(self.winidx) < self.n_components:
            raise ValueError(
                f"Only {len(self.winidx)} channels fall within the inversion "
                f"windows, fewer than the {self.n_components} tilt basis components"
            )

        self.pinv = np.linalg.pinv(self.bases[:, self.winidx].T)

    @property
    def n_components(self) -> int:
        return self.bases.shape[0]

    @property
    def band_names(self) -> list:
        return [f"Tilt PC {i + 1}" for i in range(self.n_components)]

    def project(self, ratio: np.ndarray, fill_value: float = -9999.0) -> np.ndarray:
        """Least-squares PCA coefficients for a tilted / flat reflectance ratio.

        Args:
            ratio:      ratio spectrum at instrument wavelengths, NaN where undefined
            fill_value: value for every coefficient if too few channels are usable

        Returns:
            coefficients, one per basis component
        """
        resid = ratio[self.winidx] - self.mean[self.winidx]
        good = np.isfinite(resid)

        if good.all():
            return self.pinv @ resid

        if good.sum() < self.n_components:
            return np.full(self.n_components, fill_value)

        A = self.bases[:, self.winidx[good]].T
        return np.linalg.lstsq(A, resid[good], rcond=None)[0]

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        """Ratio spectrum at instrument wavelengths from PCA coefficients."""
        return self.mean + np.asarray(coeffs) @ self.bases
