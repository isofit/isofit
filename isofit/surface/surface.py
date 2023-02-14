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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import logging

import numpy as np
from scipy.interpolate import interp1d

from isofit.configs import Config

from ..core.common import load_spectrum, load_wavelen


class Surface:
    """A model of the surface.

    Surface models are stored as MATLAB '.mat' format files.
    """

    def __init__(self, full_config: Config):
        config = full_config.forward_model.surface

        self.statevec_names = []
        self.bounds = np.array([])
        self.scale = np.array([])
        self.init = np.array([])
        self.bvec = []
        self.bval = np.array([])
        self.idx_lamb = np.empty(shape=0)
        self.emissive = False

        self.wl = None
        self.fwhm = None

        if config.wavelength_file is not None:
            self.wl, self.fwhm = load_wavelen(config.wavelength_file)
        elif full_config.implementation.mode == "simulation":
            logging.info(
                "No surface wavelength_file provided, getting wavelengths from"
                " input.reflectance_file"
            )
            _, self.wl = load_spectrum(full_config.input.reflectance_file)

        if self.wl is not None:
            self.n_wl = len(self.wl)

    def resample_reflectance(self):
        """Make sure model wavelengths align with the wavelength file."""

        if hasattr(self, "rwl"):
            p = interp1d(self.rwl, self.rfl, fill_value="extrapolate")
            self.rfl = p(self.wl)

    def xa(self, x_surface, geom):
        """Mean of prior state vector distribution calculated at state x."""

        return np.array(self.init)

    def Sa(self, x_surface, geom):
        """Covariance of prior state vector distribution calculated at state x."""

        return np.zeros((0, 0), dtype=float)

    def fit_params(self, rfl_meas, geom, *args):
        """Given a directional reflectance estimate and one or more emissive
        parameters, fit a state vector."""

        return np.array([])

    def calc_lamb(self, x_surface, geom):
        """Calculate a Lambertian surface reflectance for this state vector."""

        return self.rfl

    def calc_rfl(self, x_surface, geom):
        """Calculate the directed reflectance (specifically the HRDF) for this
        state vector."""

        return self.rfl

    def drfl_dsurface(self, x_surface, geom):
        """Partial derivative of reflectance with respect to state vector,
        calculated at x_surface. In the case that there are no free
        paramters our convention is to return the vector of zeros."""

        return np.zeros((self.n_wl, 1))

    def drfl_dsurfaceb(self, x_surface, geom):
        """Partial derivative of reflectance with respect to unmodeled
        variables, calculated at x_surface. In the case that there are no
        free paramters our convention is to return the vector of zeros."""

        return np.zeros((self.n_wl, 1))

    def calc_Ls(self, x_surface, geom):
        """Emission of surface, as a radiance."""

        return np.zeros((self.n_wl,))

    def dLs_dsurface(self, x_surface, geom):
        """Partial derivative of surface emission with respect to state vector,
        calculated at x_surface. In the case that there are no free paramters
        our convention is to return the vector of zeros."""

        return np.zeros((self.n_wl, 1))

    def summarize(self, x_surface, geom):
        """Summary of state vector."""

        return ""
