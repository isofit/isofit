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

import scipy as s
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from scipy.signal import convolve
from common import srf, load_wavelen
from numpy.random import multivariate_normal as mvn


class Instrument:

    def __init__(self, config):
        """A model of the spectrometer instrument, including spectral 
        response and noise covariance matrices. Noise is typically calculated
        from a parametric model, fit for the specific instrument.  It is a 
        function of the radiance level."""

        # If needed, skip first index column and/or convert to nanometers
        self.wl_init, self.fwhm_init = load_wavelen(config['wavelength_file'])
        self.nchan = len(self.wl_init)
        self.bounds, self.scale, self.statevec, self.init_val = [], [], [], []
        if 'statevector' in config:
            for key in config['statevector']:
                 self.statevec.append(key)
                 self.init_val.append(config['statevector'][key]['init'])
                 self.bounds.append(config['statevector'][key]['bounds'])
                 self.scale.append(config['statevector'][key]['scale'])

        # noise specified as parametric model.
        if 'SNR' in config:

            self.model_type = 'SNR'
            self.snr = float(config['SNR'])

        else:

            self.noise_file = config['noise_file']

            if self.noise_file.endswith('.txt'):

                # parametric version
                self.model_type = 'parametric'
                coeffs = s.loadtxt(
                    self.noise_file, delimiter=' ', comments='#')
                p_a = interp1d(coeffs[:, 0], coeffs[:, 1],
                               fill_value='extrapolate')
                p_b = interp1d(coeffs[:, 0], coeffs[:, 2],
                               fill_value='extrapolate')
                p_c = interp1d(coeffs[:, 0], coeffs[:, 3],
                               fill_value='extrapolate')
                self.noise = s.array([[p_a(w), p_b(w), p_c(w)]
                                      for w in self.wl_init])

            elif self.noise_file.endswith('.mat'):

                self.model_type = 'pushbroom'
                D = loadmat(self.noise_file)
                self.ncols = D['columns'][0, 0]
                if self.nchan != s.sqrt(D['bands'][0, 0]):
                    raise ValueError(
                        'Noise model does not match wavelength # bands')
                cshape = ((self.ncols, self.nchan, self.nchan))
                self.covs = D['covariances'].reshape(cshape)

        self.integrations = config['integrations']

        # Variables not retrieved = always start with relative cal
        self.bvec = ['Cal_Relative_%04i' % int(w) for w in self.wl_init] + \
            ['Cal_Spectral', 'Cal_Stray_SRF']
        self.bval = s.zeros(self.nchan+2)

        if 'unknowns' in config:

            special_unknowns = ['wavelength_calibration_uncertainty']

            # Radiometric uncertainties combine via Root Sum Square...
            for key, val in config['unknowns'].items():
                if key in special_unknowns: 
                    continue 
                elif type(val) is str:
                    u = s.loadtxt(val, comments='#')
                    if (len(u.shape) > 0 and u.shape[1] > 1):
                        u = u[:, 1]
                else:
                    u = s.ones(self.nchan) * val
                self.bval[:self.nchan] = self.bval[:self.nchan] + pow(u,2)

            self.bval[:self.nchan] = s.sqrt(self.bval[:self.nchan])

            # Now handle spectral uncertainties
            if 'wavelength_calibration_uncertainty' in config['unknowns']:
                self.bval[-2] = \
                    config['unknowns']['wavelength_calibration_uncertainty']
            if 'stray_srf_uncertainty' in config:
                self.bval[-1] = config['unknowns']['stray_srf_uncertainty']

    def Sy(self, meas, geom):
        """ Calculate measurement error covariance.
           Input: meas, the instrument measurement
           Returns: Sy, the measurement error covariance due to instrument noise"""

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

    def Kb_instrument(self, meas):
        """Jacobian of radiance with respect to NOT RETRIEVED instrument 
           variables (relative miscalibration error).
           Input: meas, a vector of size nchan
           Returns: Kb_instrument, a matrix of size 
            [n_measurements x nb_instrument]"""

        # Uncertainty due to radiometric calibration
        Kb_instrument = s.hstack((s.diagflat(meas),
                s.zeros((self.nchan,2))))

        # Uncertainty due to spectral calibration - TODO: should do at high res
        if self.bval[-2] > 1e-6:
          Kb_instrument[:,-2] = s.hstack((s.diff(meas), s.array([0])))

        # Uncertainty due to spectral stray light 
        if self.bval[-1] > 1e-6:
          ssrf = srf(s.arange(-10,11), 0, 4) 
          blur = convolve(meas, ssrf, mode='same') 
          Kb_instrument[:,-1] = blur - meas
          
        return Kb_instrument

    def simulate_measurement(self, meas, geom):
        """ Simulate a measurement by the given sensor, for a true radiance."""
        Sy = self.Sy(meas, geom)
        mu = s.zeros(meas.shape)
        rdn_sim = meas + mvn(mu, Sy)
        return rdn_sim

    def calibration(self, x_instrument):
        """ Calculate the measured wavelengths"""
        wl, fwhm = self.wl_init, self.fwhm_init
        if 'FWHM_SCL' in self.statevec:
            ind = self.statevec.index('FWHM_SCL')
            fwhm = fwhm + x_instrument[ind]
        if 'WL_SHIFT' in self.statevec:
            ind = self.statevec.index('WL_SHIFT')
            wl = self.wl_init + x_instrument[ind]
        return wl, fwhm


