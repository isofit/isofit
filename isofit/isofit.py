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

from common import expand_all_paths, spectrumLoad
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
    parser.add_argument('--profile', default='')
    args = parser.parse_args()

    # Setup
    config = json.load(open(args.config_file, 'r'))
    configdir, f = split(abspath(args.config_file))
    config = expand_all_paths(config, configdir)

    # Forward Model
    fm = ForwardModel(config['forward_model'])

    # Inversion method
    if 'mcmc_inversion' in config:
        iv = MCMCInversion(config['mcmc_inversion'], fm)
    else:
        iv = Inversion(config['inversion'], fm)

    # Output object
    out = Output(config, iv)

    # Simulation mode? Binary or text mode?
    simulation_mode = (not ('input' in config)) or \
        (not ('measured_radiance_file' in config['input']))
    text_mode = simulation_mode or \
        config['input']['measured_radiance_file'].endswith('txt')

    # Do we apply a radiance correction?
    if (not simulation_mode) and \
            'radiometry_correction_file' in config['input']:
        radiance_correction_file = config['input']['radiometry_correction_file']
        radiance_correction, wl = spectrumLoad(radiance_correction_file)
    else:
        radiance_correction = None

    if text_mode:

        # ------------------------------- Text mode

        # build geometry object
        if 'input' in config:
            obs, loc, glt = None, None, None
            if 'glt_file' in config['input']:
                glt = s.loadtxt(config['input']['glt_file'])
            if 'obs_file' in config['input']:
                obs = s.loadtxt(config['input']['obs_file'])
            if 'loc_file' in config['input']:
                loc = s.loadtxt(config['input']['loc_file'])
            geom = Geometry(obs=obs, glt=glt, loc=loc)
        else:
            geom = None

        if simulation_mode:

            # Get our measurement from instrument data or the initial state vector
            state_est = fm.init_val.copy()
            rdn_est = fm.calc_rdn(state_est, geom)
            rdn_meas = rdn_est.copy()
            rdn_sim = fm.instrument.simulate_measurement(rdn_meas, geom)
            rfl_est, rdn_est, path_est, S_hat, K, G =\
                iv.forward_uncertainty(state_est, rdn_meas, geom)

        else:

            # Get radiance
            rdn_meas, wl = spectrumLoad(
                config['input']['measured_radiance_file'])
            if radiance_correction is not None:
                rdn_meas = rdn_meas * radiance_correction
            rdn_sim = None

            if len(args.profile) > 0:
                cProfile.runctx('iv.invert(rdn_meas, geom, None)',
                                globals(), locals())
                sys.exit(0)

            elif 'mcmc_inversion' in config:

                # MCMC Sampler
                state_est, samples = iv.invert(rdn_meas, geom, out=out)
                if 'mcmc_samples_file' in config['output']:
                    D = {'samples': samples}
                    savemat(config['output']['mcmc_samples_file'], D)
                rfl_est, rdn_est, path_est, S_hat, K, G =\
                    iv.forward_uncertainty(state_est, rdn_meas, geom)

            else:

                # Conjugate Gradient
                state_est = iv.invert(rdn_meas, geom, out=out)
                rfl_est, rdn_est, path_est, S_hat, K, G =\
                    iv.forward_uncertainty(state_est, rdn_meas, geom)

        out.write_spectrum(state_est, rfl_est, rdn_est, path_est,
                           rdn_meas, rdn_sim, geom)

    else:

        # ------------------------------ Binary mode

        meas_file = config['input']['measured_radiance_file']
        meas_hdr = meas_file + '.hdr'
        meas = envi.open(meas_hdr, meas_file)
        nl, nb, ns = [int(meas.metadata[n])
                      for n in ('lines', 'bands', 'samples')]

        # Do we apply a flatfield correction?
        if 'flatfield_correction_file' in config['input']:
            ffile = config['input']['flatfield_correction_file']
            fcor = envi.open(ffile+'.hdr', ffile)
            fcor_mm = fcor.open_memmap(interleave='source',  writable=False)
            flatfield = s.array(fcor_mm[0, :, :])
        else:
            flatfield = None

        if 'obs_file' in config['input']:
            obs_file = config['input']['obs_file']
            obs_hdr = obs_file + '.hdr'
            obs = envi.open(obs_hdr,  obs_file)
            if int(obs.metadata['bands']) != 11 and \
               int(obs.metadata['bands']) != 10:
                raise ValueError('Expected 10 or 11 bands in OBS file')
            if int(obs.metadata['lines']) != nl or \
               int(obs.metadata['samples']) != ns:
                raise ValueError('obs file dimensions do not match radiance')
        else:
            obs_file, obs_hdr = None, None

        if 'glt_file' in config['input']:
            glt_file = config['input']['glt_file']
            glt_hdr = glt_file + '.hdr'
            glt = envi.open(glt_hdr,  glt_file)
            if int(glt.metadata['bands']) != 2:
                raise ValueError('Expected two bands in GLT file')
            if int(glt.metadata['lines']) != nl or \
               int(glt.metadata['samples']) != ns:
                raise ValueError('GLT file dimensions do not match radiance')
        else:
            glt_file, glt_hdr = None, None

        if 'loc_file' in config['input']:
            loc_file = config['input']['loc_file']
            loc_hdr = loc_file + '.hdr'
            loc = envi.open(loc_hdr,  loc_file)
            if int(loc.metadata['bands']) != 3:
                raise ValueError('Expected three bands in LOC file')
            if int(loc.metadata['lines']) != nl or \
               int(loc.metadata['samples']) != ns:
                raise ValueError('loc file dimensions do not match radiance')
        else:
            loc_file, loc_hdr = None, None

        rfl_file = config['output']['estimated_reflectance_file']
        rfl_hdr = rfl_file+'.hdr'
        rfl_meta = meas.metadata.copy()
        if not os.path.exists(rfl_hdr):
            rfl = envi.create_image(rfl_hdr, rfl_meta, ext='', force=True)
            del rfl
        rfl = envi.open(rfl_hdr, rfl_file)

        state_file = config['output']['estimated_state_file']
        state_hdr = state_file+'.hdr'
        state_meta = meas.metadata.copy()
        state_meta['bands'] = len(fm.statevec)
        state_meta['band names'] = fm.statevec[:]
        if not os.path.exists(state_hdr):
            state = envi.create_image(
                state_hdr, state_meta, ext='', force=True)
            del state
        state = envi.open(state_hdr, state_file)

        mdl_file = config['output']['modeled_radiance_file']
        mdl_hdr = mdl_file+'.hdr'
        mdl_meta = meas.metadata.copy()
        if not os.path.exists(mdl_hdr):
            mdl = envi.create_image(mdl_hdr, mdl_meta, ext='', force=True)
            del mdl
        mdl = envi.open(mdl_hdr, mdl_file)

        path_file = config['output']['path_radiance_file']
        path_hdr = path_file+'.hdr'
        path_meta = meas.metadata.copy()
        if not os.path.exists(path_hdr):
            path = envi.create_image(path_hdr, path_meta, ext='', force=True)
            del path
        path = envi.open(path_hdr, path_file)

        post_file = config['output']['posterior_uncertainty_file']
        post_hdr = post_file+'.hdr'
        post_meta = state_meta.copy()
        if not os.path.exists(post_hdr):
            post = envi.create_image(post_hdr, post_meta, ext='', force=True)
            del post
        post = envi.open(post_hdr, post_file)

        if 'component_file' in config['output']:
            comp_file = config['output']['component_file']
            comp_hdr = comp_file+'.hdr'
            comp_meta = state_meta.copy()
            comp_meta['bands'] = 1
            comp_meta['band names'] = '{Surface Model Component}'
            if not os.path.exists(comp_hdr):
                comp = envi.create_image(
                    comp_hdr, comp_meta, ext='', force=True)
                del comp
            comp = envi.open(comp_hdr, comp_file)
        else:
            comp = None

        meas_mm, obs_mm, state_mm, rfl_mm, path_mm, mdl_mm = \
            None, None, None, None, None, None

        # Did the user specify a row,column tuple, or a row?
        if len(args.row_column) < 1:
            lines, samps = range(nl), range(ns)
        else:
            # Restrict the range of the retrieval if overridden.
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

        # analyze the image one frame at a time
        for i in lines:

            # Flush cache every once in a while
            print('line %i/%i' % (i, nl))
            if meas_mm is None or i % 100 == 0:

                # Refresh Radiance buffer
                del meas
                meas = envi.open(meas_hdr,  meas_file)
                meas_mm = meas.open_memmap(
                    interleave='source',  writable=False)

                # Refresh OBS buffer
                if obs_hdr is not None:
                    del obs
                    obs = envi.open(obs_hdr,   obs_file)
                    obs_mm = obs.open_memmap(
                        interleave='source',   writable=False)

                # Refresh GLT buffer
                if glt_hdr is not None:
                    del glt
                    glt = envi.open(glt_hdr,   glt_file)
                    glt_mm = glt.open_memmap(
                        interleave='source',   writable=False)

                # Refresh LOC buffer
                if loc_hdr is not None:
                    del loc
                    loc = envi.open(loc_hdr,   loc_file)
                    loc_mm = loc.open_memmap(
                        interleave='source',   writable=False)

                # Refresh Output buffers
                del rfl
                rfl = envi.open(rfl_hdr,   rfl_file)
                rfl_mm = rfl.open_memmap(interleave='source',   writable=True)

                del state
                state = envi.open(state_hdr, state_file)
                state_mm = state.open_memmap(
                    interleave='source', writable=True)

                del path
                path = envi.open(path_hdr,  path_file)
                path_mm = path.open_memmap(interleave='source',  writable=True)

                del mdl
                mdl = envi.open(mdl_hdr,  mdl_file)
                mdl_mm = mdl.open_memmap(interleave='source',  writable=True)

                del post
                post = envi.open(post_hdr,  post_file)
                post_mm = post.open_memmap(interleave='source',  writable=True)

                if comp is not None:
                    del comp
                    comp = envi.open(comp_hdr,  comp_file)
                    comp_mm = comp.open_memmap(
                        interleave='source',  writable=True)

            # translate to BIP
            meas_frame = s.array(meas_mm[i, :, :]).T

            if obs_hdr is not None:
                obs_frame = s.array(obs_mm[i, :, :])

            if glt_hdr is not None:
                glt_frame = s.array(glt_mm[i, :, :])

            if loc_hdr is not None:
                loc_frame = s.array(loc_mm[i, :, :])

            init = None

            nl = int(rfl_meta['lines'])
            ns = int(rfl_meta['samples'])
            nb = int(rfl_meta['bands'])
            nsv = int(state_meta['bands'])

            if comp is not None:
                comp_frame = s.zeros((1, ns))
            post_frame = s.zeros((nsv, ns), dtype=s.float32)
            rfl_frame = s.zeros((nb, ns), dtype=s.float32)
            state_frame = s.zeros((nsv, ns), dtype=s.float32)
            path_frame = s.zeros((nb, ns), dtype=s.float32)
            mdl_frame = s.zeros((nb, ns), dtype=s.float32)

            for j in samps:
                try:

                    # Use AVIRIS-C convention?
                    if loc_hdr is not None and "t01p00r" in loc_file:
                        loc_frame[:, 2] = loc_frame[:, 2] / \
                            1000.0  # translate to km

                    rdn_meas = meas_frame[j, :]

                    # Bad data test
                    if all(rdn_meas < -49.0):
                        raise OOBError()

                    if obs_hdr is not None:
                        obs_spectrum = obs_frame[j, :]
                    else:
                        obs_spectrum = None

                    if glt_hdr is not None:
                        pc = abs(glt_frame[j, 0])-1
                        if pc < 0:
                            raise OOBError()
                        glt_spectrum = glt_frame[j, :]
                    else:
                        pc = j
                        glt_spectrum = None

                    if loc_hdr is not None:
                        loc_spectrum = loc_frame[j, :]
                    else:
                        loc_spectrum = None

                    if flatfield is not None:
                        rdn_meas = rdn_meas * flatfield[:, pc]
                    if radiance_correction is not None:
                        rdn_meas = rdn_meas * radiance_correction
                    geom = Geometry(obs_spectrum, glt_spectrum, loc_spectrum,
                                    pushbroom_column=pc)

                    # Inversion
                    if len(args.profile) > 0:
                        cProfile.runctx('iv.invert(rdn_meas, geom, None)',
                                        globals(), locals())
                        sys.exit(0)
                    else:
                        state_est = iv.invert(rdn_meas, geom, None, init=init)
                        rfl_est, rdn_est, path_est, S_hat, K, G =\
                            iv.forward_uncertainty(state_est, rdn_meas, geom)

                    # write spectrum
                    state_surf = state_est[iv.fm.surface_inds]
                    post_frame[:, j] = s.sqrt(s.diag(S_hat))
                    rfl_frame[:, j] = rfl_est
                    state_frame[:, j] = state_est
                    path_frame[:, j] = path_est
                    mdl_frame[:, j] = rdn_est
                    if comp is not None:
                        comp_frame[:, j] = iv.fm.surface.component(
                            state_surf, geom)

                except OOBError:
                    post_frame[:, j] = -9999*s.ones((nsv))
                    rfl_frame[:, j] = -9999*s.ones((nb))
                    state_frame[:, j] = -9999*s.ones((nsv))
                    path_frame[:, j] = -9999*s.ones((nb))
                    mdl_frame[:, j] = -9999*s.ones((nb))
                    if comp is not None:
                        comp_frame[:, j] = -9999

            post_mm[i, :, :] = post_frame.copy()
            rfl_mm[i, :, :] = rfl_frame.copy()
            state_mm[i, :, :] = state_frame.copy()
            path_mm[i, :, :] = path_frame.copy()
            mdl_mm[i, :, :] = mdl_frame.copy()
            if comp is not None:
                comp_mm[i, :, :] = comp_frame.copy()

        del post_mm
        del rfl_mm
        del state_mm
        del path_mm
        del mdl_mm
        if comp is not None:
            del comp_mm


if __name__ == '__main__':
    main()
