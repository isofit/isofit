#! /usr/bin/env python
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
import sys
import json
import argparse
import scipy as s
from spectral.io import envi
from os.path import expandvars, split, abspath
from scipy.io import savemat
from common import expand_all_paths, load_spectrum
from forward import ForwardModel
from inverse import Inversion
from inverse_mcmc import MCMCInversion
from geometry import Geometry
from output import Output
import cProfile


class OOBError(Exception):
    """Spectrum is out of bounds or has bad data"""
    def __init__(self):
        super(OOBError, self).__init__("")


def main():

    description = 'Spectroscopic Surface & Atmosphere Fitting'
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--row_column', default='')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--simulation', action='store_true')
    args = parser.parse_args()
    simulation = args.simulation
    
    # Load the configuration file
    config = json.load(open(args.config_file, 'r'))
    configdir, f = split(abspath(args.config_file))
    config = expand_all_paths(config, configdir)

    # Build the forward model, inversion, and output
    fm = ForwardModel(config['forward_model'])
    iv = Inversion(config['inversion'], fm)
    out = Output(config, fm, iv)
    if 'mcmc_inversion' in config:
        iv = MCMCInversion(config['mcmc_inversion'], fm)

    # Do we apply a radiance correction?
    radiance_correction = None
    if (not simulation) and ('radiometry_correction_file' in \
            config['input']):
        filename = config['input']['radiometry_correction_file']
        radiance_correction, wl = load_spectrum(filename)

    # Text mode operates on just one spectrum.  Special options that are only 
    # available in text mode include MCMC sampling, forward simulation of
    # measured spectra with noise, and performance profiling.  We determine
    # from filename suffixes (.txt) whether we are in text or binary mode
    if simulation or config['input']['measured_radiance_file'].endswith('txt'):

        # Build the geometry object.
        geom, obs, loc, glt = None, None, None, None
        if 'input' in config:
            for m, f in [(glt,'glt_file'),(obs,'obs_file'),(loc,'loc_file')]:
                if f in config['input']:
                    m = s.loadtxt(config['input'][f])
            geom = Geometry(obs=obs, glt=glt, loc=loc)

        # Simulation mode calculates a simulated measurement.
        if simulation:
            state_est = fm.init.copy()
            rdn_est = fm.calc_rdn(state_est, geom)
            meas = rdn_est.copy()
            rdn_sim = fm.instrument.simulate_measurement(meas, geom)
            rfl_est, rdn_est, path_est, S_hat, K, G =\
                iv.forward_uncertainty(state_est, meas, geom)
            out.write_spectrum(state_est, rfl_est, rdn_est, path_est,
                           meas, rdn_sim, geom)
            sys.exit(0)

        # Retrieval mode inverts a radiance spectrum.
        meas, wl = load_spectrum(config['input']['measured_radiance_file'])
        if radiance_correction is not None:
            meas = meas * radiance_correction
        rdn_sim = None

        # MCMC Sampler
        if 'mcmc_inversion' in config:
            state_est, samples = iv.invert(meas, geom, out=out)
            if 'mcmc_samples_file' in config['output']:
                D = {'samples': samples}
                savemat(config['output']['mcmc_samples_file'], D)
            rfl_est, rdn_est, path_est, S_hat, K, G =\
                iv.forward_uncertainty(state_est, meas, geom)

        # Inversion by conjugate gradient descent, profile output
        elif args.profile:
            cProfile.runctx('iv.invert(meas, geom, None)',
                globals(), locals())

        # Inversion by conjugate gradient descent
        else:
            state_est = iv.invert(meas, geom, out=out)
            rfl_est, rdn_est, path_est, S_hat, K, G =\
                iv.forward_uncertainty(state_est, meas, geom)
            out.write_spectrum(state_est, rfl_est, rdn_est, path_est,
                           meas, rdn_sim, geom)
 
    # Binary mode operates on ENVI-format data cubes.
    else:

        # The measurement file is strictly required for all inversions.
        meas_file = config['input']['measured_radiance_file']
        meas = envi.open(meas_file + '.hdr', meas_file)
        nl, nb, ns = [int(meas.metadata[n])
                      for n in ('lines', 'bands', 'samples')]

        # Do we apply a flatfield correction?
        flatfield = None
        if 'flatfield_correction_file' in config['input']:
            ffile = config['input']['flatfield_correction_file']
            fcor = envi.open(ffile+'.hdr', ffile)
            fcor_mm = fcor.open_memmap(interleave='source',  writable=False)
            flatfield = s.array(fcor_mm[0, :, :])

        # We start with all input and output objects initialized to an empty
        # list, and fill in whichever are specified by the configuration file.
        # We open each in the local scope, and check to make sure that the 
        # specified number of bands matches our prior expectations.
        ins = [('obs', [10,11]), ('glt', [2]), ('loc', [3])]
        for name, bands in ins:
            if name+'_file' in config['input']:
                lcl = locals()
                fname = config['input'][name+'_file']
                lcl[name+'_file'] = fname
                lcl[name] = envi.open(fname + '.hdr', fname)
                if not any([lcl[name].metadata['bands'] == q for q in bands]):
                    raise ValueError('Channel number mismatch in '+fname)
        
        # We create output objects. These come in several flavors.  Some 
        # output cubes have one band per instrument channel.  These are 
        # initialized based on the template provided by our measurement.
        # Another batch of output objects has one channel per state vector
        # element.  A third option is simply the index of the surface model 
        # component (which only makes sense for a multicomponent surface).
        channel_outs = ['rfl', 'path', 'mdl']
        state_outs = ['post','state']
        comp_outs = ['comp']
        for name in channel_outs + state_outs + comp_outs:
            lcl = locals()
            if name+'_file' in config['output']:
                fname = config['output'][name+'_file']
                meta = meas.metadata.copy()
                if name in state_outs:
                    meta['bands'] = len(fm.statevec)
                    meta['band names'] = fm.statevec[:] 
                elif name in comp_outs:
                    meta['bands'] = 1
                    meta['band names'] = '{Surface Model Component}'
                if not os.path.exists(fname+'.hdr'):
                    lcl[name] = envi.create_image(fname+'.hdr', meta, ext='', 
                            force=True)
                lcl[name] = envi.open(fname+'.hdr', fname)
            else:
                lcl[name] = None
            lcl[name+'_mm'] = None

        # We set the row and column range of our analysis. The user can 
        # specify: a single number, in which case it is interpreted as a row; 
        # a comma-separated pair, in which case it is interpreted as a 
        # row/column tuple (i.e. a single spectrum); or a comma-separated 
        # quartet, in which case it is interpreted as a row, column range in the
        # order (line_start, line_end, sample_start, sample_end) - all values are
        # inclusive. If none of the above, we will analyze the whole cube.
        lines, samps = range(nl), range(ns)
        if len(args.row_column) < 1:
            ranges = args.row_column.split(',')
            if len(ranges) == 1:
                lines, samps = [int(ranges[0])], range(ns)
            if len(ranges) == 2:
                line_start, line_end = ranges
                lines, samps = range(int(line_start), int(line_end)), range(ns)
            elif len(ranges) == 4:
                line_start, line_end, samp_start, samp_end = ranges
                lines = range(int(line_start), int(line_end))
                samps = range(int(samp_start), int(samp_end))

        nl = int(rfl_meta['lines'])
        ns = int(rfl_meta['samples'])
        nb = int(rfl_meta['bands'])
        nsv = int(state_meta['bands'])
        ins = ['meas','obs','glt','loc']
        outs = ['rfl','path','comp','state','post']

        # Analyze the image, loading and writing one frame (i.e. line of data)
        # at a time.  
        for i in lines:

            print('line %i/%i' % (i, nl))
            lcl = locals()

            # Flush cache every once in a while
            if meas_mm is None or i % 100 == 0:
                for name in outs + ins: 
                    if lcl[name] is not None:
                       eval('del '+name)
                       eval('del '+name+'_mm')
                       f = envi.open(name+'_file.hdr', name+'_file')
                       w = (name in outs)
                       m = f.open_memmap(interleave='source', writable=w)
                       lcl[name] = f
                       lcl[name+'_mm'] = m
                    
            # Input data, translating the BIL measurement cube to BIP
            meas_frame = s.array(meas_mm[i, :, :]).T
            for name in ['obs','glt','loc']:
                lcl[name+'_frame'] = s.array(lcl[name+'_mm'][i,:,:])
            init = None
            if comp is not None:
                comp_frame = s.zeros((1, ns))
            for name in channel_outs:
                lcl[name+'_frame'] = s.zeros((nb, ns), dtype=s.float32)
            for name in state_outs:
                lcl[name+'_frame'] = s.zeros((nsv, ns), dtype=s.float32)

            # Use AVIRIS-C convention, translating meters to km?
            convert_km = (loc is not None and "t01p00r" in loc_file)

            for j in samps:

                # We must specify the pushbroom column for FPA indexing.
                # For orthorectified data this is specified in the glt 
                # cube.  By default, we will simply use the column number
                pc = j 

                try:

                    # get the radiance spectrum, watching for bad data flags
                    # and applying any calibration corrections
                    meas = meas_frame[j, :]
                    if all(meas < -49.0):
                        raise OOBError()
                    if radiance_correction is not None:
                        meas = meas * radiance_correction
 
                    # Next we build our geometry object.  It accepts many
                    # parameter options so there is some flexibility here.
                    obs_spectrum, glt_spectrum, loc_spectrum = None, None, None
                    if obs is not None:
                        obs_spectrum = obs_frame[j, :]
                    if glt is not None:
                        pc = abs(glt_frame[j, 0])-1
                        if pc < 0:
                            raise OOBError()
                        glt_spectrum = glt_frame[j, :]
                    if loc is not None:
                        loc_spectrum = loc_frame[j, :]
                        if convert_km:
                           loc_spectrum[2] = loc_spectrum[2] / 1000.0
                    geom = Geometry(obs_spectrum, glt_spectrum, loc_spectrum,
                                    pushbroom_column=pc)

                    # We now know the true pusbroom column and can apply 
                    # any FPA-column-indexed flat field corrections
                    if flatfield is not None:
                        meas = meas * flatfield[:, pc]

                    # Inversion (magic happens here).  Calculate posterior
                    # uncertainty predictions, and unpack the optimal state
                    # vector.
                    state_est = iv.invert(meas, geom, None, init=init)
                    rfl_est, rdn_est, path_est, S_hat, K, G =\
                            iv.forward_uncertainty(state_est, meas, geom)

                    # Write all output spectra to our local (RAM) data frame.
                    rfl_frame[:, j] = rfl_est
                    state_frame[:, j] = state_est
                    path_frame[:, j] = path_est
                    mdl_frame[:, j] = rdn_est

                    # The "component" file is a special case, since it only 
                    # works for multicomponent surface models.
                    surf = state_est[iv.fm.surface_inds]
                    if comp is not None:
                        comp_frame[:, j] = iv.fm.surface.component(surf, geom)

                    # The posterior file holds marginal standard deviations of
                    # posterior uncertainty predictions for all state variables
                    # (i.e. the square root of the S_hat diagonal)
                    post_frame[:, j] = s.sqrt(s.diag(S_hat))

                # Flag any bad data or ill-convergence due to numerical errors, 
                # et cetera.
                except OOBError:
                    post_frame[:, j] = -9999*s.ones((nsv))
                    rfl_frame[:, j] = -9999*s.ones((nb))
                    state_frame[:, j] = -9999*s.ones((nsv))
                    path_frame[:, j] = -9999*s.ones((nb))
                    mdl_frame[:, j] = -9999*s.ones((nb))
                    if comp is not None:
                        comp_frame[:, j] = -9999

            # We have reached the end of our line of data.  Write to file.
            lcl = locals()
            for name in channel_outs + state_outs + ['comp']:
                if lcl[name] is not None:
                    lcl[name+'_mm'][i,:,:] = lcl[name+'_frame'].copy()

        # Clean up before exit, to ensure that all output buffers are flushed 
        # and all output files are closed.  This would probably happen auto-
        # matically when they go out of scope....
        for name in channel_outs + state_outs + ['comp']:
            eval('del '+name+'_mm')


if __name__ == '__main__':
    main()
