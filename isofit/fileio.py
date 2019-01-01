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

import os, sys
import scipy as s
import pylab as plt
from scipy.linalg import inv, norm, sqrtm, det
from scipy.io import savemat
from inverse_simple import invert_simple, invert_algebraic
from spectral.io import envi
from common import load_spectrum, resample_spectrum, expand_all_paths
import logging.config
from logging import getLogger
from collections import OrderedDict
from geometry import Geometry


# Constants related to file I/O
typemap = {s.uint8: 1, s.int16: 2, s.int32: 3, s.float32:4, s.float64: 5,
    s.complex64: 6, s.complex128: 9, s.uint16: 12, s.uint32: 13, s.int64: 14,
    s.uint64: 15}
max_frames_size = 100
flush_rate = 5


class SpectrumFile:
    """A buffered file object that contains configuration information about
        formatting, etc."""

    def __init__(self, fname, write=False, n_rows=None, n_cols=None, 
            n_bands=None, interleave=None, dtype=s.float32, 
            wavelengths=None, fwhm=None, band_names=None):

        self.logger = getLogger(__name__)
        self.frames = OrderedDict()
        self.write = write
        self.fname = fname
        self.wl = wavelengths
        self.band_names = band_names
        self.fwhm = fwhm
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_bands = n_bands

        if self.fname.endswith('.txt'):
        
            # The .txt suffix implies a space-separated ASCII text file of 
            # one or more data columns.  This is cheap to load and store, so
            # we do not defer read/write operations.  
            self.logger.info('Inferred ASCII file format for %s'%self.fname)
            self.format = 'ASCII'
            if not self.write:
                self.data,  self.wl  = load_spectrum(self.fname)
                self.n_rows, self.n_cols, self.n_bands = 1,1,len(self.wl)

        elif self.fname.endswith('.mat'):

            # The .mat suffix implies a matlab-style file, i.e. a dictionary
            # of 2D arrays and other matlab-like objects. This is typically
            # only used for specific output products associated with single 
            # spectrum retrievals; there is no read option.
            self.logger.info('Inferred MATLAB file format for %s'%self.fname)
            self.format = 'MATLAB'
            if not self.write:
                self.logger.error('Unsupported MATLAB file in input block')
                raise IOError('MATLAB format in input block not supported')
         
        else:

            # Otherwise we assume it is an ENVI-format file, which is 
            # basically just a binary data cube with a detached human-
            # readable ASCII header describing dimensions, interleave, and
            # metadata.  We buffer this data in self.frames, reading and 
            # writing individual rows of the cube on-demand.
            self.logger.info('Inferred ENVI file format for %s'%self.fname)
            self.format = 'ENVI'

            if not self.write:
                
                # If we are an input file, the header must preexist.
                if not os.path.exists(self.fname+'.hdr'):
                    self.logger.error('Could not find %s'%(self.fname+'.hdr'))
                    raise IOError('Could not find %s'%(self.fname+'.hdr'))

                # open file and copy metadata, checking interleave format
                self.file = envi.open(self.fname + '.hdr', fname)
                self.meta = self.file.metadata.copy()
                if self.meta['interleave'] not in ['bil','bip']:
                    self.logger.error('Unsupported interleave format.')
                    raise IOError('Unsupported interleave format.')

                self.n_rows = int(self.meta['lines'])
                self.n_cols = int(self.meta['samples'])
                self.n_bands = int(self.meta['bands'])

            else:

                # If we are an output file, we may have to build the header
                # from scratch.  Hopefully the caller has supplied the 
                # necessary metadata details.
                meta = {'lines': n_rows, 'samples': n_cols, 'bands':n_bands,
                    'byte_order': 0, 'header_offset':0,
                    'file_type':'ENVI Standard', 'sensor_type': 'Unknown',
                    'interleave': interleave, 'data type': typemap[dtype]}
                for k,v in meta.items():
                    if v is None:
                        self.logger.error('Must specify %s'%(k))
                        raise IOError('Must specify %s'%(k))

                # Create the image.
                for k in ['fwhm','band_names','wavelengths']:
                    lcl = locals()
                    if lcl[k] is not None:
                      meta[k] = lcl[k]
                self.file = envi.create_image(fname+'.hdr', meta, ext='',
                            force=True)

            # Open a memory map
            self.memmap = self.file.open_memmap(interleave='source', 
                   writable=self.write)

    def get_frame(self, row):
        """The frame is a 2D array, essentially a list of spectra.  The 
            self.frames list acts as a hash table to avoid storing the 
            entire cube in memory.  So we read them or create them on an
            as-needed basis.  When the buffer flushes via a call to 
            flush_buffers, they will be deleted."""

        if not row in self.frames:
            if not self.write:
                d = self.memmap[row,:,:]
                if self.file.metadata['interleave'] == 'bil':
                     d = d.T
                self.frames[row] = d
            else:
                self.frames[row] = s.nan * s.zeros((self.n_cols, self.n_bands))
        return self.frames[row]
        
    def write_spectrum(self, row, col, x):
        """We write a spectrum.  If a binary format file, we simply change
           the data cached in self.frames and defer file I/O until 
           flush_buffers is called."""

        if self.format == 'ASCII':
        
            # Multicolumn output for ASCII products
            s.savetxt(self.fname, x, fmt='%10.6f')

        elif self.format == 'MATLAB':

            # Dictionary output for MATLAB products
            savemat(self.fname, x)

        else:

            # Omit wavelength column for spectral products
            frame = self.get_frame(row)
            if x.ndim == 2:
                x = x[:,-1] 
            frame[col,:] = x

    def read_spectrum(self, row, col):
        """Get a spectrum from the frame list or ASCII file.  Note that if
           we are an ASCII file, we have already read the single spectrum and 
           return it as-is (ignoring the row/column indices)."""

        if self.format == 'ASCII':
            return self.data
        else:
            frame = self.get_frame(row)
            return frame[col]

    def flush_buffers(self):
        """Write to file, and refresh the memory map object"""
        if self.format == 'ENVI':
            if self.write:
                for row, frame in self.frames.items():
                    valid = s.logical_not(s.isnan(frame[:,0]))
                    if self.file.metadata['interleave'] == 'bil':
                        print(self.memmap[row,:,valid].shape)
                        print(frame[valid,:].shape)
                        self.memmap[row,:,valid] = frame[valid,:].T
                    else:
                        self.memmap[row,valid,:] = frame[valid,:]
            self.frames = OrderedDict()
            del self.file
            self.file = envi.open(self.fname+'.hdr',self.fname)
            self.memmap = self.file.open_memmap(interleave='source', 
                            writable=self.write)


class IO:

    def check_wavelengths(self, wl):
        """Make sure an input wavelengths align to the instrument 
            definition"""

        return (len(wl) == self.fm.instrument.wl) and \
            all((wl-self.fm.instrument.wl) < 1e-2)

    def __init__(self, config, forward, inverse, active_rows, active_cols):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions"""

        self.iv = inverse
        self.fm = forward
        self.input = {}
        self.output = {}
        self.radiance_correction = None 
        self.meas_wl   = forward.instrument.wl_init
        self.meas_fwhm = forward.instrument.fwhm_init
        self.logger = logging.getLogger(__name__)
        self.writes = 0
        self.n_rows = 0
        self.n_cols = 0
        self.n_sv   = len(self.fm.statevec)
        self.n_chan = len(self.fm.instrument.wl_init)
        
        if 'input' in config:
            self.input.update(config['input'])
        if 'output' in config:
            self.output.update(config['output'])
        if 'logging' in config:
            logging.config.dictConfig(config)

        # A list of all possible input data sources
        self.possible_inputs = ["measured_radiance_file",
                   "reference_reflectance_file",
                   "obs_file",
                   "glt_file",
                   "loc_file",
                   "surface_prior_file",
                   "atmosphere_prior_file",
                   "instrument_prior_file",
                   "radiometry_correction_file"]

        # A list of all possible outputs.  There are several special cases
        # that we handle differently - the "plot_directory", "summary_file",
        # "Data dump file", etc.  
        wl_names = [('Channel %i' % i) for i in range(self.n_chan)]
        sv_names = self.fm.statevec.copy()
        self.output_bands = {"estimated_state_file": sv_names,
                             "estimated_reflectance_file": wl_names,
                             "estimated_emission_file": wl_names,
                             "modeled_radiance_file": wl_names,
                             "apparent_reflectance_file": wl_names,
                             "path_radiance_file": wl_names,
                             "simulated_measurement_file": wl_names,
                             "algebraic_inverse_file": wl_names,
                             "atmospheric_coefficients_file": wl_names,
                             "radiometry_correction_file":  wl_names,
                             "spectral_calibration_file":  wl_names,
                             "posterior_uncertainty_file": sv_names}

        self.defined_outputs, self.defined_inputs = {}, {}
        self.infiles, self.outfiles = {}, {}

        for q in self.input:
            if q in self.possible_inputs:
                self.infiles[q] = SpectrumFile(self.input[q])

                if (self.infiles[q].n_rows > self.n_rows) or \
                   (self.infiles[q].n_cols > self.n_cols):
                    self.n_rows = self.infiles[q].n_rows
                    self.n_cols = self.infiles[q].n_cols

        for q in self.output:
            if q in self.output_bands:
                nb = len(self.output_bands[q])
                self.outfiles[q] = SpectrumFile(self.output[q], write=True, 
                        n_rows=self.n_rows, n_cols=self.n_cols, 
                        n_bands=nb, interleave='bip', dtype=s.float32, 
                        wavelengths=self.meas_wl, fwhm=self.meas_fwhm, 
                        band_names=self.output_bands[q])

        # Do we apply a radiance correction?
        if 'radiometry_correction_file' in self.input:
            filename = self.input['radiometry_correction_file']
            self.radiance_correction, wl = load_spectrum(filename)
            if not self.same_wavelengths(wl):
                self.logger.warn('Radiometry correction wavelength mismatch')

        # Last thing is to define the active image area
        if active_rows is None: 
            active_rows = s.arange(self.n_rows)
        if active_cols is None:
            active_cols = s.arange(self.n_cols)
        self.iter_inds = s.fliplr(s.array([q.flatten() for q in \
                s.meshgrid(active_rows, active_cols)]).T)

    def flush_buffers(self):
        """ Write all buffered output data to disk, and erase read buffers."""

        for fi in self.infiles + self.outfiles:
            fi.flush_buffers()

    def __iter__(self):
        """ Reset the iterator"""

        self.iter = 0
        return self

    def __next__(self):
        """ Get the next spectrum from the file.  Turn the iteration number
            into row/column indices and read from all input products."""

        if self.iter == len(self.iter_inds):
            raise StopIteration

        # Determine the appropriate row, column index. and initialize the
        # data dictionary with empty entries.
        r, c = self.iter_inds[self.iter]
        self.iter = self.iter + 1
        data = dict([(i, None) for i in self.possible_inputs])

        # Read data from any of the input files that are defined.
        for source in self.infiles:
            data[source] = self.infiles[source].read_spectrum(r,c)
            if (self.iter % flush_rate) == 0:
                self.infiles[source].flush_buffers()

        # We apply the calibration correciton here for simplicity.
        meas = data['measured_radiance_file'] 
        if data["radiometry_correction_file"] is not None:
            meas = meas.copy() * data['radiometry_correction_file']

        # We build the geometry object for this spectrum.  For files not
        # specified in the input configuration block, the associated entries
        # will be 'None'. The Geometry object will use reasonable defaults.
        geom = Geometry(obs = data['obs_file'], 
                        glt = data['glt_file'], 
                        loc = data['loc_file'])

        # Updates are simply serialized prior distribution vectors for this
        # spectrum (or 'None' if the file was not specified in the input
        # configuration block)
        updates = ({'serialized_prior': data['surface_prior_file']}, 
                   {'serialized_prior': data['atmosphere_prior_file']}, 
                   {'serialized_prior': data['instrument_prior_file']})

        return r, c, meas, geom, updates
         
    def write_spectrum(self, row, col, states, meas, geom):
        """Write data from a single inversion to all output buffers."""

        self.writes = self.writes + 1

        if len(states) == 0:

            # Write a bad data flag
            state_bad = s.zeros(len(self.fm.statevec)) * -9999.0
            data_bad = s.zeros(len(self.fm.instrument.n_chan)) * -9999.0
            to_write = {'estimated_state_file': state_bad,
                    'estimated_reflectance_file': data_bad,
                    'estimated_emission_file': data_bad,
                    'modeled_radiance_file': data_bad,
                    'apparent_reflectance_file': data_bad,
                    'path_radiance_file': data_bad,
                    'simulated_measurement_file': data_bad,
                    'algebraic_inverse_file': data_bad,
                    'atmospheric_coefficients_file': atm_bad,
                    'data_dump_file': {},
                    'radiometry_correction_file': data_bad,
                    'spectral_calibration_file': data_bad,
                    'posterior_uncertainty_file': state_bad}

        else:
 
            # The inversion returns a list of states, which are 
            # intepreted either as samples from the posterior (MCMC case)
            # or as a gradient descent trajectory (standard case). For 
            # gradient descent the last spectrum is the converged solution.
            if self.iv.method == 'MCMC':
                state_est = states.mean(axis=0)
            else:
                state_est = states[-1,:]

            # Spectral calibration
            wl, fwhm = self.fm.calibration(state_est) 
            cal = s.column_stack([s.arange(0,len(wl)), wl / 1000.0, fwhm / 1000.0])
            
            # Rodgers diagnostics
            lamb_est, meas_est, path_est, S_hat, K, G = \
                self.iv.forward_uncertainty(state_est, meas, geom)
            
            # Simulation with noise
            meas_sim = self.fm.instrument.simulate_measurement(meas_est, geom)
            
            # Algebraic inverse and atmospheric optical coefficients
            x_surface, x_RT, x_instrument = self.fm.unpack(state_est)
            rfl_alg_opt, Ls, coeffs = invert_algebraic(self.fm.surface, 
                    self.fm.RT, self.fm.instrument, x_surface, x_RT, 
                    x_instrument, meas, geom)
            rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs
            atm = s.column_stack(list(coeffs[:4]) + 
                    [s.ones((len(wl),1)) * coszen])
            
            # Upward emission & glint and apparent reflectance
            Ls_est = self.fm.calc_Ls(state_est, geom)
            apparent_rfl_est = lamb_est + Ls_est
            
            # radiometric calibration
            factors = s.ones(len(wl))
            if 'radiometry_correction_file' in self.outfiles:
                if 'reference_reflectance_file' in self.infiles:
                    reference_file = infiles['reference_reflectance_file']
                    self.rfl_ref = reference_file.read_spectrum(row, col)
                    self.wl_ref = reference_file.wl
                    resamp = resample_spectrum(self.rfl_ref, self.wl_ref, 
                        self.fm.surface.wl, self.fm.surface.fwhm, fill=True)
                    meas_est = self.fm.calc_meas(x, geom, rfl=resamp)
                    factors = meas_est / meas
                else:
                    self.logger.warning('No reflectance reference')

        # Assemble all output products
        to_write = {'estimated_state_file': state_est,
                    'estimated_reflectance_file': 
                        s.column_stack((self.fm.surface.wl, lamb_est)),
                    'estimated_emission_file': 
                        s.column_stack((self.fm.surface.wl, Ls_est)),
                    'estimated_reflectance_file': 
                        s.column_stack((self.fm.surface.wl, lamb_est)),
                    'modeled_radiance_file': 
                        s.column_stack((wl, meas_est)),
                    'apparent_reflectance_file': 
                        s.column_stack((wl, apparent_rfl_est)),
                    'path_radiance_file': 
                        s.column_stack((wl, path_est)),
                    'simulated_measurement_file': 
                        s.column_stack((wl, meas_sim)),
                    'algebraic_inverse_file': 
                        s.column_stack((self.fm.surface.wl, rfl_alg_opt)),
                    'mcmc_samples_file': {'samples': states},
                    'state_trajectory_file': {'trajectory': states},
                    'atmospheric_coefficients_file': atm,
                    'radiometry_correction_file': factors,
                    'spectral_calibration_file': cal,
                    'posterior_uncertainty_file': s.sqrt(s.diag(S_hat))}

        # Special case! Data dump file calculations are intensive so we do them 
        # on-demand.
        if 'data_dump_file' in self.outfiles:

             Seps_inv, Seps_inv_sqrt = self.iv.calc_Seps(x, meas, geom)
             meas_est_window = meas_est[self.iv.winidx]
             meas_window = meas[self.iv.winidx]
             meas_resid = (rdn_est_window-meas_window).dot(Seps_inv_sqrt)
             xa, Sa, Sa_inv, Sa_inv_sqrt = self.iv.calc_prior(x, geom)
             prior_resid = (x - xa).dot(Sa_inv_sqrt)
             x_surface, x_RT, x_instrument = self.fm.unpack(x)
             Kb = self.fm.Kb(x, geom)
             xinit = invert_simple(self.fm, meas, geom)
             Sy = self.fm.instrument.Sy(meas, geom)
             cost_jac_prior = s.diagflat(x - xa).dot(Sa_inv_sqrt)
             cost_jac_meas = Seps_inv_sqrt.dot(K[self.iv.winidx, :])
             meas_Cov = self.fm.Seps(x, meas, geom)
          
             # Form the MATLAB dictionary object and write to file
             to_write.extend({'data_dump_file':{'K': K, 'G': G, 'S_hat': S_hat, 
                'prior_mu': xa, 'Ls': Ls, 'prior_Cov': Sa, 'meas': meas, 
                'rdn_est': rdn_est, 'x': x, 'x_surface':x_surface, 'x_RT':x_RT,
                'x_instrument':x_instrument, 'meas_Cov': meas_Cov, 'wl': wl,
                'fwhm': fwhm, 'lamb_est': lamb_est, 'coszen': coszen,
                'cost_jac_prior': cost_jac_prior,'Kb': Kb, 'A':A,
                'cost_jac_meas': cost_jac_meas, 'winidx': self.iv.winidx, 
                'meas_resid': meas_resid, 'prior_resid': prior_resid,
                'noise_Cov': Sy, 'xinit': xinit, 'rhoatm': rhoatm,
                'sphalb': sphalb, 'transm': transm, 'solar_irr': solar_irr}})
             
        for product in self.outfiles: 
            self.logger.info('Writing '+product)
            self.outfiles[product].write_spectrum(row, col, to_write[product])
            if (self.writes % flush_rate) == 0:
                self.outfiles[product].flush_buffers()

        # Write plots, if needed
        if len(states)>0 and 'plot_directory' in self.output:

            if 'reference_reflectance_file' in self.infiles:
                reference_file = self.infiles['reference_reflectance_file']
                self.ref_rfl = reference_file.read_spectrum(row, col)
                self.ref_wl = reference_file.wl

            for i, x in enumerate(states):

                # Calculate intermediate solutions
                lamb_est, meas_est, path_est, S_hat, K, G = \
                     self.iv.forward_uncertainty(state_est, meas, geom)

                plt.cla()
                wl, fwhm = self.fm.calibration(x)
                xmin, xmax = min(wl), max(wl)
                fig = plt.subplots(1, 2, figsize=(10, 5))
                plt.subplot(1, 2, 1)
                meas_est = self.fm.calc_meas(x, geom)
                for lo, hi in self.iv.windows:
                    idx = s.where(s.logical_and(wl>lo, wl<hi))[0]
                    p1 = plt.plot(wl[idx], meas[idx], color=[0.7, 0.2, 0.2], 
                            linewidth=2)
                    plt.hold(True)
                    p2 = plt.plot(wl, meas_est, color='k', linewidth=1)
                plt.title("Radiance")
                plt.title("Measurement (Scaled DN + Floating Bias)")
                ymax = max(meas)*1.25
                ymax = max(meas)+0.01
                ymin = min(meas)-0.01
                plt.text(500, ymax*0.92, "Measured", color=[0.7, 0.2, 0.2])
                plt.text(500, ymax*0.86, "Model", color='k')
                plt.ylabel("$\mu$W nm$^{-1}$ sr$^{-1}$ cm$^{-2}$")
                plt.ylabel("Intensity")
                plt.xlabel("Wavelength (nm)")
                plt.ylim([-0.001, ymax])
                plt.ylim([ymin, ymax])
                plt.xlim([xmin, xmax])
                
                plt.subplot(1, 2, 2)
                lamb_est = self.fm.calc_lamb(x, geom)
                ymax = min(max(lamb_est)*1.25, 0.10)
                for lo, hi in self.iv.windows:
                        
                    # red line
                    if 'reference_reflectance_file' in self.infiles:
                        idx = s.where(s.logical_and(
                            self.ref_wl > lo, self.ref_wl < hi))[0]
                        p1 = plt.plot(self.ref_wl[idx], self.ref_rfl[idx],
                                      color=[0.7, 0.2, 0.2], linewidth=2)
                        ymax = max(max(self.ref_rfl[idx]*1.2), ymax)
                        plt.hold(True)

                    # black line
                    idx = s.where(s.logical_and(wl > lo, wl < hi))[0]
                    p2 = plt.plot(wl[idx], lamb_est[idx], 'k', linewidth=2)
                    ymax = max(max(lamb_est[idx]*1.2), ymax)

                    # green and blue lines - surface components
                    if hasattr(self.fm.surface, 'components'):
                        idx = s.where(s.logical_and(self.fm.surface.wl > lo, 
                                self.fm.surface.wl < hi))[0]
                        p3 = plt.plot(self.fm.surface.wl[idx], self.fm.xa(x, geom)[idx],
                                      'b', linewidth=2)
                        for j in range(len(self.fm.surface.components)):
                            z = self.fm.surface.norm(
                                lamb_est[self.fm.surface.idx_ref])
                            mu = self.fm.surface.components[j][0] * z
                            plt.plot(self.fm.surface.wl[idx], mu[idx], 'g:', linewidth=1)
                plt.ylim([-0.0010, ymax])
                plt.xlim([xmin, xmax])
                plt.title("Reflectance")
                plt.title("Source Model (Smooth GP)")
                plt.xlabel("Wavelength (nm)")
                if self.ref_rfl is not None:
                    plt.text(500, ymax*0.92, "In situ reference",
                             color=[0.7, 0.2, 0.2])
                    plt.text(500, ymax*0.86, "Remote estimate", color='k')
                    plt.text(500, ymax*0.80, "Prior mean state ", color='b')
                    plt.text(500, ymax*0.74, "Surface components ", color='g')
                
                fn = self.output['plot_directory'] + ('/frame_%i.png' % i)
                plt.savefig(fn)
                plt.close()

