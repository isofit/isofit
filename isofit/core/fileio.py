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

import os
import numpy as np
import scipy.io
import scipy.interpolate
import pylab as plt
from spectral.io import envi
import logging
from collections import OrderedDict

from .common import load_spectrum, eps, resample_spectrum
from isofit.inversion.inverse_simple import invert_simple, invert_algebraic
from .geometry import Geometry
from isofit.configs import Config
from isofit.core.forward import ForwardModel
import time


### Variables ###

# Constants related to file I/O
typemap = {
    np.uint8: 1,
    np.int16: 2,
    np.int32: 3,
    np.float32: 4,
    np.float64: 5,
    np.complex64: 6,
    np.complex128: 9,
    np.uint16: 12,
    np.uint32: 13,
    np.int64: 14,
    np.uint64: 15
}

max_frames_size = 100
flush_rate = 10


### Classes ###

class SpectrumFile:
    """A buffered file object that contains configuration information about formatting, etc."""

    def __init__(self, fname, write=False, n_rows=None, n_cols=None, n_bands=None,
                 interleave=None, dtype=np.float32, wavelengths=None, fwhm=None,
                 band_names=None, bad_bands='[]', zrange='{0.0, 1.0}', flag=-9999.0,
                 ztitles='{Wavelength (nm), Magnitude}', map_info='{}'):
        """."""

        self.frames = OrderedDict()
        self.write = write
        self.fname = os.path.abspath(fname)
        self.wl = wavelengths
        self.band_names = band_names
        self.fwhm = fwhm
        self.flag = flag
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_bands = n_bands

        if self.fname.endswith('.txt'):

            # The .txt suffix implies a space-separated ASCII text file of
            # one or more data columns.  This is cheap to load and store, so
            # we do not defer read/write operations.
            logging.debug('Inferred ASCII file format for %s' % self.fname)
            self.format = 'ASCII'
            if not self.write:
                self.data,  self.wl = load_spectrum(self.fname)
                self.n_rows, self.n_cols, self.map_info = 1, 1, '{}'
                if self.wl is not None:
                    self.n_bands = len(self.wl)
                else:
                    self.n_bands = None
                self.meta = {}

        elif self.fname.endswith('.mat'):

            # The .mat suffix implies a matlab-style file, i.e. a dictionary
            # of 2D arrays and other matlab-like objects. This is typically
            # only used for specific output products associated with single
            # spectrum retrievals; there is no read option.
            logging.debug('Inferred MATLAB file format for %s' % self.fname)
            self.format = 'MATLAB'
            if not self.write:
                logging.error('Unsupported MATLAB file in input block')
                raise IOError('MATLAB format in input block not supported')

        else:

            # Otherwise we assume it is an ENVI-format file, which is
            # basically just a binary data cube with a detached human-
            # readable ASCII header describing dimensions, interleave, and
            # metadata.  We buffer this data in self.frames, reading and
            # writing individual rows of the cube on-demand.
            logging.debug('Inferred ENVI file format for %s' % self.fname)
            self.format = 'ENVI'

            if not self.write:

                # If we are an input file, the header must preexist.
                if not os.path.exists(self.fname+'.hdr'):
                    logging.error('Could not find %s' % (self.fname+'.hdr'))
                    raise IOError('Could not find %s' % (self.fname+'.hdr'))

                # open file and copy metadata
                self.file = envi.open(self.fname + '.hdr', fname)
                self.meta = self.file.metadata.copy()

                self.n_rows = int(self.meta['lines'])
                self.n_cols = int(self.meta['samples'])
                self.n_bands = int(self.meta['bands'])
                if 'data ignore value' in self.meta:
                    self.flag = float(self.meta['data ignore value'])
                else:
                    self.flag = -9999.0

            else:

                # If we are an output file, we may have to build the header
                # from scratch.  Hopefully the caller has supplied the
                # necessary metadata details.
                meta = {
                    'lines': n_rows,
                    'samples': n_cols,
                    'bands': n_bands,
                    'byte order': 0,
                    'header offset': 0,
                    'map info': map_info,
                    'file_type': 'ENVI Standard',
                    'sensor type': 'unknown',
                    'interleave': interleave,
                    'data type': typemap[dtype],
                    'wavelength units': 'nm',
                    'z plot range': zrange,
                    'z plot titles': ztitles,
                    'fwhm': fwhm,
                    'bbl': bad_bands,
                    'band names': band_names,
                    'wavelength': self.wl
                }

                for k, v in meta.items():
                    if v is None:
                        logging.error('Must specify %s' % (k))
                        raise IOError('Must specify %s' % (k))

                if os.path.isfile(fname+'.hdr') is False:
                    self.file = envi.create_image(fname+'.hdr', meta, ext='',
                                                  force=True)
                else:
                    self.file = envi.open(fname+'.hdr')

            self.open_map_with_retries()

    def open_map_with_retries(self):
        """Try to open a memory map, handling Beowulf I/O issues."""

        self.memmap = None
        for attempt in range(10):
            self.memmap = self.file.open_memmap(interleave='bip',
                                                writable=self.write)
            if self.memmap is not None:
                return
        raise IOError('could not open memmap for '+self.fname)

    def get_frame(self, row):
        """The frame is a 2D array, essentially a list of spectra. The 
        self.frames list acts as a hash table to avoid storing the 
        entire cube in memory. So we read them or create them on an
        as-needed basis.  When the buffer flushes via a call to 
        flush_buffers, they will be deleted."""

        if row not in self.frames:
            if not self.write:
                d = self.memmap[row, :, :]
                self.frames[row] = d.copy()
            else:
                self.frames[row] = np.nan * np.zeros((self.n_cols, self.n_bands))
        return self.frames[row]

    def write_spectrum(self, row, col, x):
        """We write a spectrum. If a binary format file, we simply change
        the data cached in self.frames and defer file I/O until 
        flush_buffers is called."""

        if self.format == 'ASCII':

            # Multicolumn output for ASCII products
            np.savetxt(self.fname, x, fmt='%10.6f')

        elif self.format == 'MATLAB':

            # Dictionary output for MATLAB products
            scipy.io.savemat(self.fname, x)

        else:

            # Omit wavelength column for spectral products
            frame = self.get_frame(row)
            if x.ndim == 2:
                x = x[:, -1]
            frame[col, :] = x

    def read_spectrum(self, row, col):
        """Get a spectrum from the frame list or ASCII file. Note that if
        we are an ASCII file, we have already read the single spectrum and 
        return it as-is (ignoring the row/column indices)."""

        if self.format == 'ASCII':
            return self.data
        else:
            frame = self.get_frame(row)
            return frame[col]

    def flush_buffers(self):
        """Write to file, and refresh the memory map object."""

        if self.format == 'ENVI':
            if self.write:
                for row, frame in self.frames.items():
                    valid = np.logical_not(np.isnan(frame[:, 0]))
                    self.memmap[row, valid, :] = frame[valid, :]
            self.frames = OrderedDict()
            del self.file
            self.file = envi.open(self.fname+'.hdr', self.fname)
            self.open_map_with_retries()


class IO:
    """..."""

    def __init__(self, config: Config, forward: ForwardModel, inverse):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions."""

        self.config = config

        self.iv = inverse
        self.fm = forward
        #fm = forward
        self.bbl = '[]'
        self.radiance_correction = None
        self.meas_wl = forward.instrument.wl_init
        self.meas_fwhm = forward.instrument.fwhm_init
        self.writes = 0
        self.n_rows = 1
        self.n_cols = 1
        self.n_sv = len(self.fm.statevec)
        self.n_chan = len(self.fm.instrument.wl_init)

        self.simulation_mode = config.implementation.mode == 'simulation'

        self.total_time = 0

        # Names of either the wavelength or statevector outputs
        wl_names = [('Channel %i' % i) for i in range(self.n_chan)]
        sv_names = self.fm.statevec.copy()

        self.input_datasets, self.output_datasets, self.map_info = {}, {}, '{}'

        # Load input files and record relevant metadata
        for element, element_name in zip(*self.config.input.get_elements()):
            self.input_datasets[element_name] = SpectrumFile(element)

            if (self.input_datasets[element_name].n_rows > self.n_rows) or \
               (self.input_datasets[element_name].n_cols > self.n_cols):
                self.n_rows = self.input_datasets[element_name].n_rows
                self.n_cols = self.input_datasets[element_name].n_cols

            for inherit in ['map info', 'bbl']:
                if inherit in self.input_datasets[element_name].meta:
                    setattr(self, inherit.replace(' ', '_'),
                            self.input_datasets[element_name].meta[inherit])

        for element, element_header, element_name in zip(*self.config.output.get_output_files()):
            band_names, ztitle, zrange = element_header

            if band_names == 'statevector':
                band_names = sv_names
            elif band_names == 'wavelength':
                band_names = wl_names
            elif band_names == 'atm_coeffs':
                band_names = wl_names*5
            else:
                band_names = '{}'

            n_bands = len(band_names)
            self.output_datasets[element_name] = SpectrumFile(
                element,
                write=True,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                n_bands=n_bands,
                interleave='bip',
                dtype=np.float32,
                wavelengths=self.meas_wl,
                fwhm=self.meas_fwhm,
                band_names=band_names,
                bad_bands=self.bbl,
                map_info=self.map_info,
                zrange=zrange,
                ztitles=ztitle
            )

        # Do we apply a radiance correction?
        if self.config.input.radiometry_correction_file is not None:
            filename = self.config.input.radiometry_correction_file
            self.radiance_correction, wl = load_spectrum(filename)

    def get_components_at_index(self, row: int, col: int) -> (bool, np.array, Geometry):
        """
        Load data from input files at the specified (row, col) index.

        Args:
            row: reference location for iter_inds
            col: reference location for iter_inds

        Returns:
            bool: flag indicating if data present
            np.array: measured radiance file
            Geometry: geometry object
        """
        # Determine the appropriate row, column index. and initialize the
        # data dictionary with empty entries.
        data = dict([(i, None) for i in self.config.input.get_all_element_names()])
        logging.debug(f'Row {row} Column {col}')

        # Read data from any of the input files that are defined.
        #import ipdb; ipdb.set_trace()
        for source in self.input_datasets:
            data[source] = self.input_datasets[source].read_spectrum(row, col)
            self.input_datasets[source].flush_buffers()
            #TODO
            #if (index % flush_rate) == 0:
            #    self.infiles[source].flush_buffers()

        if self.simulation_mode:
            # If solving the inverse problem, the measurment is the surface reflectance
            meas = data['reflectance_file'].copy()
        else:
            # If solving the inverse problem, the measurment is the radiance
            # We apply the calibration correciton here for simplicity.
            meas = data['measured_radiance_file']
            if meas is not None:
                meas = meas.copy()
            if data["radiometry_correction_file"] is not None:
                meas *= data['radiometry_correction_file']

        if meas is None or np.all(meas < -49):
            return False, None, None

        ## Check for any bad data flags
        for source in self.input_datasets:
            if np.all(abs(data[source] - self.input_datasets[source].flag) < eps):
                return False, None, None

        # We build the geometry object for this spectrum.  For files not
        # specified in the input configuration block, the associated entries
        # will be 'None'. The Geometry object will use reasonable defaults.
        geom = Geometry(obs=data['obs_file'],
                        glt=data['glt_file'],
                        loc=data['loc_file'],
                        bg_rfl=data['background_reflectance_file'])

        return True, meas, geom


#    ## TODO: - revise
#    def __iter__(self):
#        """ Reset the iterator"""
#
#        self.iter = 0
#        return self
#
#    # TODO: - revise
#    def __next__(self):
#        """ Get the next spectrum from the file.  Turn the iteration number
#            into row/column indices and read from all input products."""
#
#        # Try to read data until we hit the end or find good values
#        success = False
#        while not success:
#            if self.iter == len(self.iter_inds):
#                self.flush_buffers()
#                raise StopIteration
#
#            # Determine the appropriate row, column index. and initialize the
#            # data dictionary with empty entries.
#            success, r, c, meas, geom = self.get_components_at_index(
#                self.iter)
#            self.iter = self.iter + 1
#
#        return r, c, meas, geom
#
#    def check_wavelengths(self, wl):
#        """Make sure an input wavelengths align to the instrument definition."""
#
#        return (len(wl) == self.fm.instrument.wl) and \
#            all((wl-self.fm.instrument.wl) < 1e-2)

    def flush_buffers(self):
        """Write all buffered output data to disk, and erase read buffers."""

        for file_dictionary in [self.input_datasets, self.output_datasets]:
            for name, fi in file_dictionary.items():
                fi.flush_buffers()


    def write_spectrum(self, row, col, states, meas, geom, flush_immediately=False):
        """Write data from a single inversion to all output buffers."""

        self.writes = self.writes + 1

        if len(states) == 0:

            # Write a bad data flag
            atm_bad = np.zeros(len(self.fm.instrument.n_chan)*5) * -9999.0
            state_bad = np.zeros(len(self.fm.statevec)) * -9999.0
            data_bad = np.zeros(self.fm.instrument.n_chan) * -9999.0
            to_write = {
                'estimated_state_file': state_bad,
                'estimated_reflectance_file': data_bad,
                'estimated_emission_file': data_bad,
                'modeled_radiance_file': data_bad,
                'apparent_reflectance_file': data_bad,
                'path_radiance_file': data_bad,
                'simulated_measurement_file': data_bad,
                'algebraic_inverse_file': data_bad,
                'atmospheric_coefficients_file': atm_bad,
                'radiometry_correction_file': data_bad,
                'spectral_calibration_file': data_bad,
                'posterior_uncertainty_file': state_bad
            }

        else:
            to_write = {}

            # The inversion returns a list of states, which are
            # intepreted either as samples from the posterior (MCMC case)
            # or as a gradient descent trajectory (standard case). For
            # gradient descent the last spectrum is the converged solution.
            if self.config.implementation.mode == 'inversion_mcmc':
                state_est = states.mean(axis=0)
            else:
                state_est = states[-1, :]

            ############ Start with all of the 'independent' calculations
            if 'estimated_state_file' in self.output_datasets:
                to_write['estimated_state_file'] = state_est

            if 'path_radiance_file' in self.output_datasets:
                path_est = self.fm.calc_meas(state_est, geom, rfl=np.zeros(self.meas_wl.shape))
                to_write['path_radiance_file'] = np.column_stack((self.fm.instrument.wl, path_est))

            if 'spectral_calibration_file' in self.output_datasets:
                # Spectral calibration
                wl, fwhm = self.fm.calibration(state_est)
                cal = np.column_stack(
                    [np.arange(0, len(wl)), wl / 1000.0, fwhm / 1000.0])
                to_write['spectral_calibration_file'] = cal

            if 'posterior_uncertainty_file' in self.output_datasets:
                S_hat, K, G = self.iv.calc_posterior(state_est, geom, meas)
                to_write['posterior_uncertainty_file'] = np.sqrt(np.diag(S_hat))

            ############ Now proceed to the calcs where they may be some overlap
            if any(item in ['modeled_radiance_file', 'simulated_measurement_file'] for item in self.output_datasets):
                meas_est = self.fm.calc_meas(state_est, geom, rfl=np.zeros(self.meas_wl.shape))

                if 'modeled_radiance_file' in self.output_datasets:
                    to_write['modeled_radiance_file'] = np.column_stack((self.fm.instrument.wl, meas_est))

                if 'simulated_measurement_file' in self.output_datasets:
                    meas_sim = self.fm.instrument.simulate_measurement(meas_est, geom)
                    to_write['simulated_measurement_file'] = np.column_stack((self.meas_wl, meas_sim))

            if any(item in ['estimated_emission_file', 'apparent_reflectance_file'] for item in self.output_datasets):
                Ls_est = self.fm.calc_Ls(state_est, geom)

            if any(item in ['estimated_reflectance_file', 'apparent_reflectance_file'] for item in self.output_datasets):
                lamb_est = self.fm.calc_lamb(state_est, geom)

            if 'estimated_emission_file' in self.output_datasets:
                to_write['estimated_emission_file'] = np.column_stack((self.meas_wl, Ls_est))

            if 'estimated_reflectance_file' in self.output_datasets:
                to_write['estimated_reflectance_file'] = np.column_stack((self.meas_wl, lamb_est))

            if 'apparent_reflectance_file' in self.output_datasets:
                # Upward emission & glint and apparent reflectance
                apparent_rfl_est = lamb_est + Ls_est
                to_write['apparent_reflectance_file'] = np.column_stack((self.meas_wl, apparent_rfl_est))

            x_surface, x_RT, x_instrument = self.fm.unpack(state_est)
            if any(item in ['algebraic_inverse_file', 'atmospheric_coefficients_file'] for item in self.output_datasets):
                rfl_alg_opt, Ls, coeffs = invert_algebraic(self.fm.surface,
                                                           self.fm.RT, self.fm.instrument, x_surface, x_RT,
                                                           x_instrument,
                                                           meas, geom)

            if 'algebraic_inverse_file' in self.output_datasets:
                to_write['algebraic_inverse_file'] = np.column_stack((self.meas_wl, rfl_alg_opt))

            if 'atmospheric_coefficients_file' in self.output_datasets:
                rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs
                atm = np.column_stack(list(coeffs[:4]) +
                                      [np.ones((len(self.meas_wl), 1)) * coszen])
                atm = atm.T.reshape((len(self.meas_wl) * 5,))
                to_write['atmospheric_coefficients_file'] = atm

            if 'radiometry_correction_file' in self.output_datasets:
                factors = np.ones(len(self.meas_wl))
                if 'reference_reflectance_file' in self.input_datasets:
                    reference_file = self.input_datasets['reference_reflectance_file']
                    reference_reflectance = reference_file.read_spectrum(row, col)
                    reference_wavelengths = reference_file.wl
                    w, fw = self.fm.instrument.calibration(x_instrument)
                    resamp = resample_spectrum(reference_reflectance, reference_wavelengths,
                                               w, fw, fill=True)
                    meas_est = self.fm.calc_meas(state_est, geom, rfl=resamp)
                    factors = meas_est / meas
                else:
                    logging.warning('No reflectance reference')
                to_write['radiometry_correction_file'] = factors

        for product in self.output_datasets:
            logging.debug('IO: Writing '+product)
            self.output_datasets[product].write_spectrum(row, col, to_write[product])
            if (self.writes % flush_rate) == 0 or flush_immediately:
                self.output_datasets[product].flush_buffers()

        # Special case! samples file is matlab format.
        if self.config.output.mcmc_samples_file is not None:
            logging.debug('IO: Writing mcmc_samples_file')
            mdict = {'samples': states}
            scipy.io.savemat(self.config.output.mcmc_samples_file, mdict)

