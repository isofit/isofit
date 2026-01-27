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
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.io import loadmat
from scipy.signal import convolve

from isofit.core import units
from isofit.core.common import (
    emissive_radiance,
    eps,
    load_wavelen,
    resample_spectrum,
    spectral_response_function,
    svd_inv_sqrt,
)

### Variables ###

# Max. wavelength difference (nm) that does not trigger expensive resampling
wl_tol = 0.01


### Classes ###


class Instrument:
    def __init__(self, full_config: Config):
        """A model of the spectrometer instrument, including spectral
        response and noise covariance matrices. Noise is typically calculated
        from a parametric model, fit for the specific instrument.  It is a
        function of the radiance level."""

        config = full_config.forward_model.instrument

        # If needed, skip first index column and/or convert to nanometers
        self.wl_init, self.fwhm_init = load_wavelen(config.wavelength_file)
        self.n_chan = len(self.wl_init)

        self.fast_resample = config.fast_resample

        self.bounds = config.statevector.get_all_bounds()
        self.scale = config.statevector.get_all_scales()
        self.init = config.statevector.get_all_inits()
        self.prior_mean = np.array(config.statevector.get_all_prior_means())
        self.prior_sigma = np.array(config.statevector.get_all_prior_sigmas())
        self.Sa_cached = np.diagflat(np.power(self.prior_sigma, 2))
        self.Sa_normalized = self.Sa_cached / np.mean(np.diag(self.Sa_cached))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            self.Sa_normalized
        )
        self.statevec_names = config.statevector.get_element_names()
        self.n_state = len(self.statevec_names)

        self.integrations = config.integrations

        if config.eof_path is not None:
            self.eof = np.loadtxt(config.eof_path)
            self.eof_idx = []
            for i, name in enumerate(sorted(self.statevec_names)):
                if "EOF" in name:
                    self.eof_idx.append(i)
        else:
            self.eof = None
            self.eof_idx = []

        self.dn_uncertainty_embedding = None
        if (
            config.unknowns is not None
            and config.unknowns.dn_uncertainty_file is not None
        ):
            dn_uncertainty_mat = loadmat(config.unknowns.dn_uncertainty_file)

            # Check validity of linearity file for dn-based noise
            keys = [
                "input_dn",
                "dn_ratio",
                "rcc",
                "rcc_wl",
            ]
            bad = [
                1 if np.any(~np.isfinite(dn_uncertainty_mat[key])) else 0
                for key in keys
            ]
            if np.sum(bad):
                er = f"""
                    Invalid value found in dn_uncertainty_mat keys: {[keys[i] for i in bad if i]}.
                    Check file at: {config.unknowns.dn_uncertainty_file}
                """
                logging.error(er)
                raise ValueError(er)

            input_dn = dn_uncertainty_mat["input_dn"].squeeze()
            dn_ratio = dn_uncertainty_mat["dn_ratio"].squeeze()
            rcc_in = dn_uncertainty_mat["rcc"].squeeze()
            rcc_wl = dn_uncertainty_mat["rcc_wl"].squeeze()

            rcc_interp = interp1d(rcc_wl, rcc_in, fill_value="extrapolate")
            self.dn_uncertainty_rcc = rcc_interp(self.wl_init)
            self.dn_uncertainty_interp = interp1d(
                input_dn, dn_ratio, fill_value="extrapolate"
            )
            self.dn_uncertainty_inflation = dn_uncertainty_mat.get(
                "inflation", [1.0]
            ).squeeze()
            self.dn_uncertainty_embedding = dn_uncertainty_mat.get(
                "embedding_location", "Sy"
            )

        if config.SNR is not None:
            self.model_type = "SNR"
            self.snr = config.SNR

        elif config.parametric_noise_file is not None:
            self.model_type = "parametric"
            self.noise_file = config.parametric_noise_file

            coeffs = np.loadtxt(self.noise_file, delimiter=" ", comments="#")
            p_a, p_b, p_c = [
                interp1d(coeffs[:, 0], coeffs[:, col], fill_value="extrapolate")
                for col in (1, 2, 3)
            ]
            self.noise = np.array([[p_a(w), p_b(w), p_c(w)] for w in self.wl_init])

        elif config.pushbroom_noise_file is not None:
            self.model_type = "pushbroom"
            self.noise_file = config.pushbroom_noise_file
            D = loadmat(self.noise_file)
            self.ncols = D["columns"][0, 0]
            if self.n_chan != np.sqrt(D["bands"][0, 0]):
                logging.error("Noise model mismatches wavelength # bands")
                raise ValueError("Noise model mismatches wavelength # bands")
            cshape = (self.ncols, self.n_chan, self.n_chan)
            self.covs = D["covariances"].reshape(cshape)
            self.integrations = config.integrations

        elif config.nedt_noise_file is not None:
            self.model_type = "NEDT"
            self.noise_file = config.nedt_noise_file
            self.noise_data = np.loadtxt(self.noise_file, delimiter=",", skiprows=8)
            noise_data_w_nm = units.micron_to_nm(self.noise_data[:, 0])
            noise_data_NEDT = self.noise_data[:, 1]
            nedt = interp1d(noise_data_w_nm, noise_data_NEDT)(self.wl_init)

            T, emis = 300.0, 0.95  # From Glynn Hulley, 2/18/2020
            _, drdn_dT = emissive_radiance(emis, T, self.wl_init)
            self.noise_NESR = nedt * drdn_dT

        else:
            raise IndexError("Please define the instrument noise.")
        # This should never be reached, as an error is designated in the config read

        # We track several unretrieved free variables, that are specified
        # in a fixed order (always start with relative radiometric
        # calibration)
        self.unknowns = config.unknowns
        self.bval = np.zeros(self.n_chan)
        self.bvec = ["Cal_Relative_%04i" % int(w) for w in self.wl_init]

        # self.unknowns should always exist via configs
        # but may not exist for manual configs
        if self.unknowns:
            # Now handle spectral calibration uncertainties
            if self.unknowns.wavelength_calibration_uncertainty is not None:
                self.bvec += ["Cal_Spectral"]
                self.cal_spectral_idx = len(self.bval)
                self.bval = np.hstack(
                    [self.bval, self.unknowns.wavelength_calibration_uncertainty]
                )

            if self.unknowns.stray_srf_uncertainty is not None:
                self.bvec += ["Cal_Stray_SRF"]
                self.cal_stray_idx = len(self.bval) + 1
                self.bval = np.hstack([self.bval, self.unknowns.stray_srf_uncertainty])

        # Determine whether the calibration is fixed.  If it is fixed,
        # and the wavelengths of radiative transfer modeling and instrument
        # are the same, then we can bypass computationally expensive sampling
        # operations later.
        self.calibration_fixed = True
        if (
            config.statevector.GROW_FWHM is not None
            or config.statevector.WL_SHIFT is not None
            or config.statevector.WL_SPACE is not None
        ):
            self.calibration_fixed = False

    def xa(self):
        """Mean of prior distribution, calculated at state x."""

        return self.init.copy()

    def Sa(self):
        """Covariance of prior distribution (diagonal)."""

        if self.n_state == 0:
            return np.zeros((0, 0), dtype=float)
        return self.Sa_cached

    def Sb(self, meas):
        """Uncertainty due to unmodeled variables."""
        bval = self.bval.copy()
        # First we take care of radiometric uncertainties, which add
        # in quadrature.  We sum their squared values.  Systematic
        # radiometric uncertainties account for differences in sampling
        # and radiative transfer that manifest predictably as a function
        # of wavelength.
        if self.unknowns:
            if self.unknowns.channelized_radiometric_uncertainty_file is not None:
                f = self.unknowns.channelized_radiometric_uncertainty_file
                u = np.loadtxt(f, comments="#")
                if len(u.shape) > 0 and u.shape[1] > 1:
                    u = u[:, 1]
                bval[: self.n_chan] = bval[: self.n_chan] + pow(u, 2)

            # Uncorrelated radiometric uncertainties are consistent and
            # independent in all channels.
            if self.unknowns.uncorrelated_radiometric_uncertainty:
                u = self.unknowns.uncorrelated_radiometric_uncertainty
                bval[: self.n_chan] = bval[: self.n_chan] + pow(
                    np.ones(self.n_chan) * u, 2
                )

            # Uncertainty due to imperfect knowledge of linearity correction
            if self.dn_uncertainty_embedding == "Sb":
                bval[: self.n_chan] += np.power(
                    self.DN_additive_uncertainty(
                        meas,
                        self.dn_uncertainty_rcc,
                        self.dn_uncertainty_interp,
                        self.dn_uncertainty_inflation,
                    ),
                    2,
                )

        # Radiometric uncertainties combine via Root Sum Square...
        # Be careful to avoid square roots of zero!
        small = np.ones(self.n_chan) * eps
        bval[: self.n_chan] = np.maximum(bval[: self.n_chan], small)
        bval[: self.n_chan] = np.sqrt(bval[: self.n_chan])

        return np.diagflat(np.power(bval, 2))

    def Sy(self, meas, geom):
        """Calculate measuremment error covariance.  Kelvin Man Yiu Leung and
            Jayanth Jagalur Mohan (MIT) developed the noise clipping strategy.

        Input: meas, the instrument measurement
        Returns: Sy, the measurement error covariance due to instrument noise
        """

        Sy = None
        if self.model_type == "SNR":
            nedl = (1.0 / self.snr) * meas
            minimum_noise = np.sqrt(1e-7)
            bad = nedl < minimum_noise
            if np.any(bad):
                logging.debug(
                    "SNR noise model found noise <= 0 - adjusting to slightly positive"
                    " to avoid /0."
                )
            nedl[bad] = minimum_noise
            Sy = np.diagflat(np.power(nedl, 2))

        elif self.model_type == "parametric":
            noise_plus_meas = self.noise[:, 1] + meas
            if np.any(noise_plus_meas <= 0):
                noise_plus_meas[noise_plus_meas <= 0] = 1e-5
                logging.debug(
                    "Parametric noise model found noise <= 0 - adjusting to slightly"
                    " positive to avoid /0."
                )
            nedl = np.abs(
                self.noise[:, 0] * np.sqrt(noise_plus_meas) + self.noise[:, 2]
            )
            nedl = nedl / np.sqrt(self.integrations)
            Sy = np.diagflat(np.power(nedl, 2))

        elif self.model_type == "pushbroom":
            C = np.squeeze(self.covs.mean(axis=0))
            Sy = C / np.sqrt(self.integrations)

        elif self.model_type == "NEDT":
            Sy = np.diagflat(np.power(self.noise_NESR, 2))

        if self.dn_uncertainty_embedding:
            # Uncertainty due to imperfect knowledge of linearity correction
            np.fill_diagonal(
                Sy,
                (
                    Sy.diagonal()
                    + self.DN_additive_uncertainty(
                        meas,
                        self.dn_uncertainty_rcc,
                        self.dn_uncertainty_interp,
                        self.dn_uncertainty_inflation,
                    )
                ),
            )

        return Sy

    def dmeas_dinstrument(self, x_instrument, wl_hi, rdn_hi):
        """Jacobian of measurement with respect to the instrument
        free parameter state vector. We use finite differences for now."""

        dmeas_dinstrument = np.zeros((self.n_chan, self.n_state), dtype=float)
        if self.n_state == 0:
            return dmeas_dinstrument

        meas = self.sample(x_instrument, wl_hi, rdn_hi) + self.eof_offset(x_instrument)
        for ind in range(self.n_state):
            x_instrument_perturb = x_instrument.copy()
            x_instrument_perturb[ind] = x_instrument_perturb[ind] + eps
            meas_perturb = self.sample(
                x_instrument_perturb, wl_hi, rdn_hi
            ) + self.eof_offset(x_instrument_perturb)

            dmeas_dinstrument[:, ind] = (meas_perturb - meas) / eps

        return dmeas_dinstrument

    def dmeas_dinstrumentb(self, x_instrument, wl_hi, rdn_hi):
        """Jacobian of radiance with respect to the instrument parameters
        that are unknown and not retrieved, i.e., the inevitable persisting
        uncertainties in instrument spectral and radiometric calibration.

        Input: meas, a vector of size n_chan
        Returns: Kb_instrument, a matrix of size [n_measurements x nb_instrument]
        """

        # Uncertainty due to radiometric calibration
        meas = self.sample(x_instrument, wl_hi, rdn_hi)
        dmeas_dinstrument = np.hstack(
            (
                np.diagflat(meas),
                np.zeros((self.n_chan, len(self.bvec) - len(self.wl_init))),
            )
        )

        # Uncertainty due to spectral calibration
        if self.unknowns:
            if self.unknowns.wavelength_calibration_uncertainty is not None:
                dmeas_dinstrument[:, self.cal_spectral_idx] = self.sample(
                    x_instrument, wl_hi, np.hstack((np.diff(rdn_hi), np.array([0])))
                )

            # Uncertainty due to spectral stray light
            if self.unknowns.stray_srf_uncertainty is not None:
                ssrf = spectral_response_function(np.arange(-10, 11), 0, 4)
                blur = convolve(meas, ssrf, mode="same")
                dmeas_dinstrument[:, self.cal_stray_idx] = blur - meas

        return dmeas_dinstrument

    def eof_offset(self, x_instrument):
        offset = np.zeros(len(self.wl_init))
        if len(self.eof_idx):
            for i in self.eof_idx:
                offset += self.eof[:, i] * x_instrument[i]
        return offset

    def sample(self, x_instrument, wl_hi, rdn_hi):
        """Apply instrument sampling to a radiance spectrum, returning predicted measurement."""

        if (
            self.calibration_fixed
            and (len(self.wl_init) == len(wl_hi))
            and all((self.wl_init - wl_hi) < wl_tol)
        ):

            return rdn_hi

        wl, fwhm = self.calibration(x_instrument)

        # If rdn_hi is a vector of length 1, return itself
        if rdn_hi.ndim == 1 and len(rdn_hi) <= 1:
            return rdn_hi

        # If rdn_hi is a vector of length > 1, return it resampled to instrument
        elif rdn_hi.ndim == 1 and len(rdn_hi) > 1:
            return resample_spectrum(rdn_hi, wl_hi, wl, fwhm)

        # If rdn_hi is a multidim array, do the multidim resampling
        else:
            resamp = []
            # The "fast resample" option approximates a complete resampling
            # by a convolution with a uniform FWHM.
            if self.fast_resample:
                for i, r in enumerate(rdn_hi):
                    ssrf = spectral_response_function(np.arange(-10, 11), 0, fwhm[0])
                    blur = convolve(r, ssrf, mode="same")
                    resamp.append(interp1d(wl_hi, blur)(wl))
            else:
                for i, r in enumerate(rdn_hi):
                    r2 = resample_spectrum(r, wl_hi, wl, fwhm)
                    resamp.append(r2)
            return np.array(resamp)

    def simulate_measurement(self, meas, geom):
        """Simulate a measurement by the given sensor, for a true radiance
        sampled to instrument wavelengths. This basically just means
        drawing a sample from the noise distribution."""

        Sy = self.Sy(meas, geom)
        mu = np.zeros(meas.shape)
        rdn_sim = meas + np.random.multivariate_normal(mu, Sy)
        return rdn_sim

    def calibration(self, x_instrument):
        """Calculate the measured wavelengths."""

        wl, fwhm = self.wl_init, self.fwhm_init
        space_orig = wl - wl[0]
        offset = wl[0]
        if "GROW_FWHM" in self.statevec_names:
            ind = self.statevec_names.index("GROW_FWHM")
            fwhm = fwhm + x_instrument[ind]
        elif any([v.startswith("FWHMSPL") for v in self.statevec_names]):
            # cubic spline perturbation
            channels, vals = [], []
            for i, v in enumerate(self.statevec_names):
                if v.startswith("FWHMSPL"):
                    chan = float(v.split("_")[1])
                    channels.append(chan)
                    vals.append(x_instrument[i])
            sp = splrep(channels, vals, s=0)
            xnew = np.arange(len(wl))
            fwhm = fwhm + splev(xnew, sp)

        if "WL_SPACE" in self.statevec_names:
            ind = self.statevec_names.index("WL_SPACE")
            space = x_instrument[ind]
        else:
            space = 1.0

        if "WL_SHIFT" in self.statevec_names:
            ind = self.statevec_names.index("WL_SHIFT")
            shift = x_instrument[ind]
        elif any([v.startswith("WLSPL") for v in self.statevec_names]):
            # cubic spline perturbation
            channels, vals = [], []
            for i, v in enumerate(self.statevec_names):
                if v.startswith("WLSPL"):
                    chan = int(v.split("_")[1])
                    channels.append(chan)
                    vals.append(x_instrument[i])
            sp = splrep(channels, vals, s=0)
            xnew = np.arange(len(wl))
            shift = splev(xnew, sp)
        else:
            shift = 0.0

        wl = offset + shift + space_orig * space
        return wl, fwhm

    @staticmethod
    def DN_additive_uncertainty(meas, rcc, interp, inflation):
        # Into DN space with rccs
        dn_est = np.maximum(meas / rcc, 0)
        noise_est = interp(dn_est)
        return np.abs(meas * (noise_est - 1) * inflation)

    def summarize(self, x_instrument, geom):
        """Summary of state vector."""

        if len(x_instrument) < 1:
            return ""
        return "Instrument: " + " ".join(["%5.3f" % xi for xi in x_instrument])
