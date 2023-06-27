#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
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
# Author: Alberto Candela Garza, alberto.candela.garza@jpl.nasa.gov
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov


import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.io import loadmat
from scipy.linalg import block_diag, norm
from tensorflow import keras

from isofit.configs import Config
from isofit.core import common

from ..core.common import svd_inv
from .surface import Surface


class DCGMSurface(Surface):
    """A model of the surface based on a neural-network estimated
    mean and covariance distribution.
    """

    def __init__(self, full_config: Config, wavelengths=None):
        """."""

        super().__init__(full_config)

        config = full_config.forward_model.surface

        self.model = tf.keras.models.load_model(config.model_file)
        aux = np.load(config.model_aux_file)

        self.model_wl = aux["wavelengths"]

        if wavelengths is not None:
            self.wl = wavelengths
        elif config.wavelength_file is not None:
            self.wl, fwhm = common.load_wavelen(config.wavelength_file)
        else:
            self.wl, fwhm = common.load_wavelen(
                full_config.forward_model.instrument.wavelength_file
            )

        self.n_wl = len(self.wl)

        # initialize an empty cache
        self.cache = {}
        self.max_cache_size = 10

        self.select_on_init = config.select_on_init

        # Variables retrieved: each channel maps to a reflectance model parameter
        rmin, rmax = 0, 2.0
        self.statevec_names = ["RFL_%04i" % int(w) for w in self.wl]
        self.bounds = [[rmin, rmax] for w in self.wl]
        self.scale = [1.0 for w in self.wl]
        self.init = [0.15 * (rmax - rmin) + rmin for v in self.wl]
        self.idx_lamb = np.arange(self.n_wl)
        self.n_state = len(self.statevec_names)

    def interpolate(
        self,
        mean: np.array,
        covariance: np.array,
        in_wl: np.array = None,
        out_wl: np.array = None,
    ):
        """interpolate the mean and covariance

        Args:
            mean (np.array): input mean
            covariance (np.array): input covariance
            in_wl (np.array, optional): input wavelengths.  If none, use from self. Defaults to None.
            out_wl (np.array, optional): output wavelenghts.  If none, use from self. Defaults to None.

        Returns:
            np.array, np.array: interpolated mean and covariance
        """

        if in_wl is None:
            in_wl = self.model_wl
        if out_wl is None:
            out_wl = self.wl

        # do resmampling
        resampled_mean = interpolate.interp1d(in_wl, mean, kind="quadratic")(out_wl)
        resampled_covariance = interpolate.interp1d(
            in_wl, covariance, kind="quadratic"
        )(out_wl)

        return resampled_mean, resampled_covariance

    def get_meancov(self, x_surface: np.array):
        """Get the mean and covariance from model, resample if necessary, and cache

        Args:
            x_surface (np.array): reflectance to estimate the mean and covariance from

        Returns:
            surface_mean, surface_cov: mean and covariance of the estimated surface
        """

        hash_idx = tuple(x_surface)
        if hash_idx in self.cache:
            surface_mean_resamp, surface_cov_resamp = self.cache[hash_idx]
        else:
            surface_mean, surface_cov = self.model.predict(x_surface[np.newaxis, :])
            surface_mean = np.squeeze(surface_mean)
            surface_cov = np.squeeze(surface_cov)
            surface_mean_resamp, surface_cov_resamp = self.interpolate(
                surface_mean, surface_cov, self.model_wl, self.wl
            )
            if len(self.cache) > self.max_cache_size:
                del self.cache[next(iter(self.cache))]
            self.cache[hash_idx] = (surface_mean_resamp, surface_cov_resamp)

        return surface_mean_resamp, surface_cov_resamp

    def xa(self, x_surface, geom):
        """Mean of prior distribution, calculated at state x."""

        surface_mean, surface_cov = self.get_meancov(x_surface)
        return surface_mean

    def Sa(self, x_surface, geom):
        """Covariance of prior distribution, calculated at state x."""

        surface_mean, surface_cov = self.get_meancov(x_surface)
        return surface_cov

    def fit_params(self, rfl_meas, geom, *args):
        """Given a reflectance estimate, fit a state vector."""

        x_surface = np.zeros(len(self.statevec_names))
        if len(rfl_meas) != len(self.idx_lamb):
            raise ValueError("Mismatched reflectances")
        for i, r in zip(self.idx_lamb, rfl_meas):
            x_surface[i] = max(
                self.bounds[i][0] + 0.001, min(self.bounds[i][1] - 0.001, r)
            )
        return x_surface

    def calc_rfl(self, x_surface, geom):
        """Non-Lambertian reflectance."""

        return self.calc_lamb(x_surface, geom)

    def calc_lamb(self, x_surface, geom):
        """Lambertian reflectance."""

        return x_surface[self.idx_lamb]

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface."""

        return self.dlamb_dsurface(x_surface, geom)

    def dlamb_dsurface(self, x_surface, geom):
        """Partial derivative of Lambertian reflectance with respect to
        state vector, calculated at x_surface."""

        dlamb = np.eye(self.n_wl, dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = self.n_state - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dlamb, suffix), axis=1)

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros(self.n_wl, dtype=float)

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface."""

        dLs = np.zeros((self.n_wl, self.n_wl), dtype=float)
        nprefix = self.idx_lamb[0]
        nsuffix = len(self.statevec_names) - self.idx_lamb[-1] - 1
        prefix = np.zeros((self.n_wl, nprefix))
        suffix = np.zeros((self.n_wl, nsuffix))
        return np.concatenate((prefix, dLs, suffix), axis=1)

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return ""
