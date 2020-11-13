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
import pylab as plt
from spectral.io import envi
import logging
from collections import OrderedDict

from .common import load_spectrum, eps, resample_spectrum
from isofit.inversion.inverse_simple import invert_simple, invert_algebraic
from .geometry import Geometry
from isofit.configs import Config
from isofit.core.forward import ForwardModel


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

    def __init__(self, config: Config, forward: ForwardModel, inverse, active_rows, active_cols):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions."""

        self.input = config.input
        self.output = config.output

        self.iv = inverse
        self.fm = forward
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

        # Names of either the wavelength or statevector outputs
        wl_names = [('Channel %i' % i) for i in range(self.n_chan)]
        sv_names = self.fm.statevec.copy()

        self.defined_outputs, self.defined_inputs = {}, {}
        self.infiles, self.outfiles, self.map_info = {}, {}, '{}'

        # Load input files and record relevant metadata
        for element, element_name in zip(*self.input.get_elements()):
            self.infiles[element_name] = SpectrumFile(element)

            if (self.infiles[element_name].n_rows > self.n_rows) or \
               (self.infiles[element_name].n_cols > self.n_cols):
                self.n_rows = self.infiles[element_name].n_rows
                self.n_cols = self.infiles[element_name].n_cols

            for inherit in ['map info', 'bbl']:
                if inherit in self.infiles[element_name].meta:
                    setattr(self, inherit.replace(' ', '_'),
                            self.infiles[element_name].meta[inherit])

        for element, element_header, element_name in zip(*self.output.get_output_files()):
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
            self.outfiles[element_name] = SpectrumFile(
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
        if self.input.radiometry_correction_file is not None:
            filename = self.input.radiometry_correction_file
            self.radiance_correction, wl = load_spectrum(filename)

        # Last thing is to define the active image area
        if active_rows is None:
            active_rows = np.arange(self.n_rows)
        if active_cols is None:
            active_cols = np.arange(self.n_cols)
        self.iter_inds = []
        for row in active_rows:
            for col in active_cols:
                self.iter_inds.append([row, col])
        self.iter_inds = np.array(self.iter_inds)

        if self.simulation_mode:
            self.fm.surface.rwl = np.array([float(x) for x in self.infiles['reflectance_file'].meta['wavelength']])

    def get_components_at_index(self, index: int) -> (int, int, np.array, Geometry):
        """
        Get the spectrum from the file at the specified index.  Helper/
        parallel enabling function.

        :param index: reference location for iter_inds

        :return: success: boolean flag indicating if data present
        :return: r: row index
        :return: c: column index
        :return: meas: measured radiance file
        :return: geom: set up specified geometry files
        """
        # Determine the appropriate row, column index. and initialize the
        # data dictionary with empty entries.
        r, c = self.iter_inds[index]
        data = dict([(i, None) for i in self.input.get_all_element_names()])
        logging.debug('Row %i Column %i' % (r, c))

        # Read data from any of the input files that are defined.
        for source in self.infiles:
            data[source] = self.infiles[source].read_spectrum(r, c)
            if (index % flush_rate) == 0:
                self.infiles[source].flush_buffers()

        # Check for any bad data flags
        for source in self.infiles:
            if np.all(abs(data[source] - self.infiles[source].flag) < eps):
                return False, r, c, None, None

        if self.simulation_mode:
            # If solving the inverse problem, the measurment is the surface reflectance
            self.fm.surface.rfl = data['reflectance_file'].copy()
            if self.fm.surface.wl is not None:
                self.fm.surface.resample_reflectance()
            meas = self.fm.surface.rfl
        else:
            # If solving the inverse problem, the measurment is the radiance
            # We apply the calibration correciton here for simplicity.
            meas = data['measured_radiance_file']
            if meas is not None:
                meas = meas.copy()
            if data["radiometry_correction_file"] is not None:
                meas *= data['radiometry_correction_file']

        # We build the geometry object for this spectrum.  For files not
        # specified in the input configuration block, the associated entries
        # will be 'None'. The Geometry object will use reasonable defaults.
        geom = Geometry(obs=data['obs_file'],
                        glt=data['glt_file'],
                        loc=data['loc_file'],
                        bg_rfl=data['background_reflectance_file'])

        return True, r, c, meas, geom

    def __iter__(self):
        """ Reset the iterator"""

        self.iter = 0
        return self

    def __next__(self):
        """ Get the next spectrum from the file.  Turn the iteration number
            into row/column indices and read from all input products."""

        # Try to read data until we hit the end or find good values
        success = False
        while not success:
            if self.iter == len(self.iter_inds):
                self.flush_buffers()
                raise StopIteration

            # Determine the appropriate row, column index. and initialize the
            # data dictionary with empty entries.
            success, r, c, meas, geom = self.get_components_at_index(
                self.iter)
            self.iter = self.iter + 1

        return r, c, meas, geom

    def check_wavelengths(self, wl):
        """Make sure an input wavelengths align to the instrument definition."""

        return (len(wl) == self.fm.instrument.wl) and \
            all((wl-self.fm.instrument.wl) < 1e-2)

    def flush_buffers(self):
        """Write all buffered output data to disk, and erase read buffers."""

        for file_dictionary in [self.infiles, self.outfiles]:
            for name, fi in file_dictionary.items():
                fi.flush_buffers()

    def write_spectrum(self, row, col, states, meas, geom, flush_immediately=False):
        """Write data from a single inversion to all output buffers."""

        self.writes = self.writes + 1

        if len(states) == 0:

            # Write a bad data flag
            atm_bad = np.zeros(len(self.fm.n_chan)*5) * -9999.0
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

            # The inversion returns a list of states, which are
            # intepreted either as samples from the posterior (MCMC case)
            # or as a gradient descent trajectory (standard case). For
            # gradient descent the last spectrum is the converged solution.
            if self.iv.mode == 'mcmc':
                state_est = states.mean(axis=0)
            else:
                state_est = states[-1, :]

            # Spectral calibration
            wl, fwhm = self.fm.calibration(state_est)
            cal = np.column_stack(
                [np.arange(0, len(wl)), wl / 1000.0, fwhm / 1000.0])

            # If there is no actual measurement, we use the simulated version
            # in subsequent calculations.  Naturally in these cases we're
            # mostly just interested in the simulation result.
            if meas is None:
                meas = self.fm.calc_rdn(state_est, geom)

            # Rodgers diagnostics
            lamb_est, meas_est, path_est, S_hat, K, G = \
                self.iv.forward_uncertainty(state_est, meas, geom)

            # Simulation with noise
            meas_sim = self.fm.instrument.simulate_measurement(meas_est, geom)

            # Algebraic inverse and atmospheric optical coefficients
            x_surface, x_RT, x_instrument = self.fm.unpack(state_est)
            rfl_alg_opt, Ls, coeffs = invert_algebraic(self.fm.surface,
                                                       self.fm.RT, self.fm.instrument, x_surface, x_RT, x_instrument,
                                                       meas, geom)
            rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs

            L_atm = self.fm.RT.get_L_atm(x_RT, geom)
            L_down_transmitted = self.fm.RT.get_L_down_transmitted(x_RT, geom)

            atm = np.column_stack(list(coeffs[:4]) +
                                  [np.ones((len(wl), 1)) * coszen])
            atm = atm.T.reshape((len(wl)*5,))

            # Upward emission & glint and apparent reflectance
            Ls_est = self.fm.calc_Ls(state_est, geom)
            apparent_rfl_est = lamb_est + Ls_est

            # Radiometric calibration
            factors = np.ones(len(wl))
            if 'radiometry_correction_file' in self.outfiles:
                if 'reference_reflectance_file' in self.infiles:
                    reference_file = self.infiles['reference_reflectance_file']
                    self.rfl_ref = reference_file.read_spectrum(row, col)
                    self.wl_ref = reference_file.wl
                    w, fw = self.fm.instrument.calibration(x_instrument)
                    resamp = resample_spectrum(self.rfl_ref, self.wl_ref,
                                               w, fw, fill=True)
                    meas_est = self.fm.calc_meas(state_est, geom, rfl=resamp)
                    factors = meas_est / meas
                else:
                    logging.warning('No reflectance reference')

            # Assemble all output products
            to_write = {
                'estimated_state_file': state_est,
                'estimated_reflectance_file': np.column_stack((self.fm.surface.wl, lamb_est)),
                'estimated_emission_file': np.column_stack((self.fm.surface.wl, Ls_est)),
                'modeled_radiance_file': np.column_stack((wl, meas_est)),
                'apparent_reflectance_file': np.column_stack((self.fm.surface.wl, apparent_rfl_est)),
                'path_radiance_file': np.column_stack((wl, path_est)),
                'simulated_measurement_file': np.column_stack((wl, meas_sim)),
                'algebraic_inverse_file': np.column_stack((self.fm.surface.wl, rfl_alg_opt)),
                'atmospheric_coefficients_file': atm,
                'radiometry_correction_file': factors,
                'spectral_calibration_file': cal,
                'posterior_uncertainty_file': np.sqrt(np.diag(S_hat))
            }

        for product in self.outfiles:
            logging.debug('IO: Writing '+product)
            self.outfiles[product].write_spectrum(row, col, to_write[product])
            if (self.writes % flush_rate) == 0 or flush_immediately:
                self.outfiles[product].flush_buffers()

        # Special case! samples file is matlab format.
        if self.output.mcmc_samples_file is not None:
            logging.debug('IO: Writing mcmc_samples_file')
            mdict = {'samples': states}
            scipy.io.savemat(self.output.mcmc_samples_file, mdict)

        # Special case! Data dump file is matlab format.
        if self.output.data_dump_file is not None:

            logging.debug('IO: Writing data_dump_file')
            x = state_est
            xall = states
            Seps_inv, Seps_inv_sqrt = self.iv.calc_Seps(x, meas, geom)
            meas_est_window = meas_est[self.iv.winidx]
            meas_window = meas[self.iv.winidx]
            xa, Sa, Sa_inv, Sa_inv_sqrt = self.iv.calc_prior(x, geom)
            prior_resid = (x - xa).dot(Sa_inv_sqrt)
            rdn_est = self.fm.calc_rdn(x, geom)
            rdn_est_all = np.array([self.fm.calc_rdn(xtemp, geom)
                                    for xtemp in states])

            x_surface, x_RT, x_instrument = self.fm.unpack(x)
            Kb = self.fm.Kb(x, geom)
            xinit = invert_simple(self.fm, meas, geom)
            Sy = self.fm.instrument.Sy(meas, geom)
            cost_jac_prior = np.diagflat(x - xa).dot(Sa_inv_sqrt)
            cost_jac_meas = Seps_inv_sqrt.dot(K[self.iv.winidx, :])
            meas_Cov = self.fm.Seps(x, meas, geom)
            lamb_est, meas_est, path_est, S_hat, K, G = \
                self.iv.forward_uncertainty(state_est, meas, geom)
            A = np.matmul(K, G)

            # Form the MATLAB dictionary object and write to file
            mdict = {
                'K': K,
                'G': G,
                'S_hat': S_hat,
                'prior_mu': xa,
                'Ls': Ls,
                'prior_Cov': Sa,
                'meas': meas,
                'rdn_est': rdn_est,
                'rdn_est_all': rdn_est_all,
                'x': x,
                'xall': xall,
                'x_surface': x_surface,
                'x_RT': x_RT,
                'x_instrument': x_instrument,
                'meas_Cov': meas_Cov,
                'wl': wl,
                'fwhm': fwhm,
                'lamb_est': lamb_est,
                'coszen': coszen,
                'cost_jac_prior': cost_jac_prior,
                'Kb': Kb,
                'A': A,
                'cost_jac_meas': cost_jac_meas,
                'winidx': self.iv.winidx,
                'windows': self.iv.windows,
                'prior_resid': prior_resid,
                'noise_Cov': Sy,
                'xinit': xinit,
                'rhoatm': rhoatm,
                'sphalb': sphalb,
                'transm': transm,
                'transup': transup,
                'solar_irr': solar_irr,
                'L_atm': L_atm,
                'L_down_transmitted': L_down_transmitted
            }
            scipy.io.savemat(self.output.data_dump_file, mdict)

        # Write plots, if needed
        if len(states) > 0 and self.output.plot_directory is not None:

            if 'reference_reflectance_file' in self.infiles:
                reference_file = self.infiles['reference_reflectance_file']
                self.rfl_ref = reference_file.read_spectrum(row, col)
                self.wl_ref = reference_file.wl

            for i, x in enumerate(states):

                # Calculate intermediate solutions
                lamb_est, meas_est, path_est, S_hat, K, G = \
                    self.iv.forward_uncertainty(state_est, meas, geom)

                plt.cla()
                red = [0.7, 0.2, 0.2]
                wl, fwhm = self.fm.calibration(x)
                xmin, xmax = min(wl), max(wl)
                fig = plt.subplots(1, 2, figsize=(10, 5))
                plt.subplot(1, 2, 1)
                meas_est = self.fm.calc_meas(x, geom)
                for lo, hi in self.iv.windows:
                    idx = np.where(np.logical_and(wl > lo, wl < hi))[0]
                    p1 = plt.plot(wl[idx], meas[idx], color=red, linewidth=2)
                    p2 = plt.plot(wl, meas_est, color='k', linewidth=1)
                plt.title("Radiance")
                plt.title("Measurement (Scaled DN)")
                ymax = max(meas)*1.25
                ymax = max(meas)+0.01
                ymin = min(meas)-0.01
                plt.text(500, ymax*0.92, "Measured", color=red)
                plt.text(500, ymax*0.86, "Model", color='k')
                plt.ylabel(r"$\mu$W nm$^{-1}$ sr$^{-1}$ cm$^{-2}$")
                plt.ylabel("Intensity")
                plt.xlabel("Wavelength (nm)")
                plt.ylim([-0.001, ymax])
                plt.ylim([ymin, ymax])
                plt.xlim([xmin, xmax])

                plt.subplot(1, 2, 2)
                lamb_est = self.fm.calc_lamb(x, geom)
                ymax = min(max(lamb_est)*1.25, 0.10)
                for lo, hi in self.iv.windows:

                    # black line
                    idx = np.where(np.logical_and(wl > lo, wl < hi))[0]
                    p2 = plt.plot(wl[idx], lamb_est[idx], 'k', linewidth=2)
                    ymax = max(max(lamb_est[idx]*1.2), ymax)

                    # red line
                    if 'reference_reflectance_file' in self.infiles:
                        idx = np.where(np.logical_and(
                            self.wl_ref > lo, self.wl_ref < hi))[0]
                        p1 = plt.plot(self.wl_ref[idx], self.rfl_ref[idx],
                                      color=red, linewidth=2)
                        ymax = max(max(self.rfl_ref[idx]*1.2), ymax)

                    # green and blue lines - surface components
                    if hasattr(self.fm.surface, 'components') and \
                            self.output.plot_surface_components:
                        idx = np.where(np.logical_and(self.fm.surface.wl > lo,
                                                      self.fm.surface.wl < hi))[0]
                        p3 = plt.plot(self.fm.surface.wl[idx],
                                      self.fm.xa(x, geom)[idx], 'b', linewidth=2)
                        for j in range(len(self.fm.surface.components)):
                            z = self.fm.surface.norm(
                                lamb_est[self.fm.surface.idx_ref])
                            mu = self.fm.surface.components[j][0] * z
                            plt.plot(self.fm.surface.wl[idx], mu[idx], 'g:',
                                     linewidth=1)
                plt.text(500, ymax*0.86, "Remote estimate", color='k')
                if 'reference_reflectance_file' in self.infiles:
                    plt.text(500, ymax*0.92, "In situ reference", color=red)
                if hasattr(self.fm.surface, 'components') and \
                        self.output.plot_surface_components:
                    plt.text(500, ymax*0.80, "Prior mean state ",
                             color='b')
                    plt.text(500, ymax*0.74, "Surface components ",
                             color='g')
                plt.ylim([-0.0010, ymax])
                plt.xlim([xmin, xmax])
                plt.title("Reflectance")
                plt.title("Source Model")
                plt.xlabel("Wavelength (nm)")
                fn = os.path.join(self.output.plot_directory, ('frame_%i.png' % i))
                plt.savefig(fn)
                plt.close()
