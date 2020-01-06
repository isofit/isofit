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
import scipy as s
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.io import loadmat
from numpy.random import multivariate_normal as mvn

from .common import eps, srf, load_wavelen, resample_spectrum


### Variables ###

# Max. wavelength difference (nm) that does not trigger expensive resampling
wl_tol = 0.01


### Classes ###

class Instrument:

    def __init__(self, config):
        """A model of the spectrometer instrument, including spectral 
        response and noise covariance matrices. Noise is typically calculated
        from a parametric model, fit for the specific instrument.  It is a 
        function of the radiance level."""

        # If needed, skip first index column and/or convert to nanometers
        self.wl_init, self.fwhm_init = load_wavelen(config['wavelength_file'])
        self.n_chan = len(self.wl_init)
        self.bounds = []
        self.scale = []
        self.statevec = []
        self.init = []
        self.prior_sigma = []
        self.prior_mean = []
        self.fast_resample = True

        # The "fast resample" option approximates a complete resampling by a
        # convolution with a uniform FWHM.
        if 'fast_resample' in config:
            self.fast_resample = config['fast_resample']

        # Are there free parameters?
        if 'statevector' in config:
            for key in config['statevector']:
                self.statevec.append(key)
                for attr in config['statevector'][key]:
                    getattr(self, attr).append(
                        config['statevector'][key][attr])
        self.prior_sigma = s.array(self.prior_sigma)
        self.prior_mean = s.array(self.prior_mean)
        self.n_state = len(self.statevec)

        # Number of integrations comprising the measurement.  Noise diminishes
        # with the square root of this number.
        self.integrations = config['integrations']

        if 'SNR' in config:

            # We have several ways to define the instrument noise.  The
            # simplest model is based on a single uniform SNR number that
            # is signal-independnet and applied uniformly to all wavelengths
            self.model_type = 'SNR'
            self.snr = float(config['SNR'])

        elif 'parametric_noise_file' in config:

            # The second option is a parametric, signal- and wavelength-
            # dependent noise function. This is given by a four-column
            # ASCII Text file.  Rows represent, respectively, the reference
            # wavelength, and coefficients A, B, and C that define the
            # noise-equivalent radiance via NeDL = A * sqrt(B+L) + C
            # For the actual radiance L.
            self.noise_file = config['parametric_noise_file']
            self.model_type = 'parametric'
            coeffs = s.loadtxt(
                self.noise_file, delimiter=' ', comments='#')
            p_a, p_b, p_c = [interp1d(coeffs[:, 0], coeffs[:, col],
                                      fill_value='extrapolate') for col in (1, 2, 3)]
            self.noise = s.array([[p_a(w), p_b(w), p_c(w)]
                                  for w in self.wl_init])

        elif 'pushbroom_noise_file' in config:
            # The third option is a full pushbroom noise model that
            # specifies noise columns and covariances independently for
            # each cross-track location via an ENVI-format binary data file.
            self.model_type = 'pushbroom'
            self.noise_file = config['pushbroom_noise_file']
            D = loadmat(self.noise_file)
            self.ncols = D['columns'][0, 0]
            if self.n_chan != s.sqrt(D['bands'][0, 0]):
                logging.error('Noise model mismatches wavelength # bands')
                raise ValueError('Noise model mismatches wavelength # bands')
            cshape = ((self.ncols, self.n_chan, self.n_chan))
            self.covs = D['covariances'].reshape(cshape)

        else:
            logging.error('Instrument noise not defined.')
            raise IndexError('Please define the instrument noise.')

        # We track several unretrieved free variables, that are specified
        # in a fixed order (always start with relative radiometric
        # calibration)
        self.bvec = ['Cal_Relative_%04i' % int(w) for w in self.wl_init] + \
            ['Cal_Spectral', 'Cal_Stray_SRF']
        self.bval = s.zeros(self.n_chan+2)

        if 'unknowns' in config:

            # First we take care of radiometric uncertainties, which add
            # in quadrature.  We sum their squared values.  Systematic
            # radiometric uncertainties account for differences in sampling
            # and radiative transfer that manifest predictably as a function
            # of wavelength.
            unknowns = config['unknowns']
            if 'channelized_radiometric_uncertainty_file' in unknowns:
                f = unknowns['channelized_radiometric_uncertainty_file']
                u = s.loadtxt(f, comments='#')
                if (len(u.shape) > 0 and u.shape[1] > 1):
                    u = u[:, 1]
                self.bval[:self.n_chan] = self.bval[:self.n_chan] + pow(u, 2)

            # Uncorrelated radiometric uncertainties are consistent and
            # independent in all channels.
            if 'uncorrelated_radiometric_uncertainty' in unknowns:
                u = unknowns['uncorrelated_radiometric_uncertainty']
                self.bval[:self.n_chan] = self.bval[:self.n_chan] + \
                    pow(s.ones(self.n_chan) * u, 2)

            # Radiometric uncertainties combine via Root Sum Square...
            # Be careful to avoid square roots of zero!
            small = s.ones(self.n_chan)*eps
            self.bval[:self.n_chan] = s.maximum(self.bval[:self.n_chan], small)
            self.bval[:self.n_chan] = s.sqrt(self.bval[:self.n_chan])

            # Now handle spectral calibration uncertainties
            if 'wavelength_calibration_uncertainty' in unknowns:
                self.bval[-2] = unknowns['wavelength_calibration_uncertainty']
            if 'stray_srf_uncertainty' in unknowns:
                self.bval[-1] = unknowns['stray_srf_uncertainty']

        # Determine whether the calibration is fixed.  If it is fixed,
        # and the wavelengths of radiative transfer modeling and instrument
        # are the same, then we can bypass compputationally expensive sampling
        # operations later.
        self.calibration_fixed = (not ('FWHM_SCL' in self.statevec)) and \
            (not ('WL_SHIFT' in self.statevec)) and \
            (not ('WL_SPACE' in self.statevec))

    def xa(self):
        """Mean of prior distribution, calculated at state x."""

        return self.init.copy()

    def Sa(self):
        """Covariance of prior distribution (diagonal)."""

        if self.n_state == 0:
            return s.zeros((0, 0), dtype=float)
        return s.diagflat(pow(self.prior_sigma, 2))

    def Sy(self, meas, geom):
        """Calculate measurement error covariance.

        Input: meas, the instrument measurement
        Returns: Sy, the measurement error covariance due to instrument noise
        """

        if self.model_type == 'SNR':
            bad = meas < 1e-5
            meas[bad] = 1e-5
            nedl = (1.0 / self.snr) * meas
            return pow(s.diagflat(nedl), 2)

        elif self.model_type == 'parametric':
            nedl = abs(
                self.noise[:, 0]*s.sqrt(self.noise[:, 1]+meas)+self.noise[:, 2])
            nedl = nedl/s.sqrt(self.integrations)
            return pow(s.diagflat(nedl), 2)

        elif self.model_type == 'pushbroom':
            if geom.pushbroom_column is None:
                C = s.squeeze(self.covs.mean(axis=0))
            else:
                C = self.covs[geom.pushbroom_column, :, :]
            return C / s.sqrt(self.integrations)

    def dmeas_dinstrument(self, x_instrument, wl_hi, rdn_hi):
        """Jacobian of measurement with respect to the instrument 
           free parameter state vector. We use finite differences for now."""

        dmeas_dinstrument = s.zeros((self.n_chan, self.n_state), dtype=float)
        if self.n_state == 0:
            return dmeas_dinstrument

        meas = self.sample(x_instrument, wl_hi, rdn_hi)
        for ind in range(self.n_state):
            x_instrument_perturb = x_instrument.copy()
            x_instrument_perturb[ind] = x_instrument_perturb[ind]+eps
            meas_perturb = self.sample(x_instrument_perturb, wl_hi, rdn_hi)
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
        dmeas_dinstrument = s.hstack(
            (s.diagflat(meas), s.zeros((self.n_chan, 2))))

        # Uncertainty due to spectral calibration
        if self.bval[-2] > 1e-6:
            dmeas_dinstrument[:, -2] = self.sample(x_instrument, wl_hi,
                                                   s.hstack((s.diff(rdn_hi), s.array([0]))))

        # Uncertainty due to spectral stray light
        if self.bval[-1] > 1e-6:
            ssrf = srf(s.arange(-10, 11), 0, 4)
            blur = convolve(meas, ssrf, mode='same')
            dmeas_dinstrument[:, -1] = blur - meas

        return dmeas_dinstrument

    def sample(self, x_instrument, wl_hi, rdn_hi):
        """Apply instrument sampling to a radiance spectrum, returning predicted measurement."""

        if self.calibration_fixed and all((self.wl_init - wl_hi) < wl_tol):
            return rdn_hi
        wl, fwhm = self.calibration(x_instrument)
        if rdn_hi.ndim == 1:
            return resample_spectrum(rdn_hi, wl_hi, wl, fwhm)
        else:
            resamp = []
            # The "fast resample" option approximates a complete resampling
            # by a convolution with a uniform FWHM.
            if self.fast_resample:
                for i, r in enumerate(rdn_hi):
                    ssrf = srf(s.arange(-10, 11), 0, fwhm[0])
                    blur = convolve(r, ssrf, mode='same')
                    resamp.append(interp1d(wl_hi, blur)(wl))
            else:
                for i, r in enumerate(rdn_hi):
                    r2 = resample_spectrum(r, wl_hi, wl, fwhm)
                    resamp.append(r2)
            return s.array(resamp)

    def simulate_measurement(self, meas, geom):
        """Simulate a measurement by the given sensor, for a true radiance
        sampled to instrument wavelengths. This basically just means
        drawing a sample from the noise distribution."""

        Sy = self.Sy(meas, geom)
        mu = s.zeros(meas.shape)
        rdn_sim = meas + mvn(mu, Sy)
        return rdn_sim

    def calibration(self, x_instrument):
        """Calculate the measured wavelengths."""

        wl, fwhm = self.wl_init, self.fwhm_init
        space_orig = wl - wl[0]
        offset = wl[0]
        if 'GROW_FWHM' in self.statevec:
            ind = self.statevec.index('GROW_FWHM')
            fwhm = fwhm + x_instrument[ind]
        if 'WL_SPACE' in self.statevec:
            ind = self.statevec.index('WL_SPACE')
            space = x_instrument[ind]
        else:
            space = 1.0
        if 'WL_SHIFT' in self.statevec:
            ind = self.statevec.index('WL_SHIFT')
            shift = x_instrument[ind]
        else:
            shift = 0.0
        wl = offset + shift + space_orig * space
        return wl, fwhm

    def summarize(self, x_instrument, geom):
        """Summary of state vector."""

        if len(x_instrument) < 1:
            return ''
        return 'Instrument: '+' '.join(['%5.3f' % xi for xi in x_instrument])

    def reconfigure(self, config):
        """Reconfiguration not yet supported."""

        return
