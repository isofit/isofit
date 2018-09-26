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

import argparse
from scipy import logical_and as aand
from os.path import realpath, split, abspath, expandvars
import scipy as s
from spectral.io import envi
from scipy.io import savemat
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

from common import find_header, expand_path, json_load_ascii


# parse the command line
def main():

    parser = argparse.ArgumentParser(description="Create a surface model")
    parser.add_argument('config', type=str, metavar='INPUT')
    args = parser.parse_args()
    config = json_load_ascii(args.config, shell_replace=True)
    configdir, configfile = split(abspath(args.config))

    # Determine top level parameters
    for q in ['output_model_file', 'sources', 'normalize', 'wavelength_file']:
        if q not in config:
            raise ValueError('Missing parameter: %s' % q)

    wavelength_file = expand_path(configdir, config['wavelength_file'])
    normalize = config['normalize']
    reference_windows = config['reference_windows']
    outfile = expand_path(configdir, config['output_model_file'])
    if 'mixtures' in config:
        mixtures = config['mixtures']
    else:
        mixtures = 0

    # load wavelengths file
    q = s.loadtxt(wavelength_file)
    if q.shape[1] > 2:
        q = q[:, 1:]
    if q[0, 0] < 100:
        q = q * 1000.0
    wl = q[:, 0]
    nchan = len(wl)

    # build global reference windows
    refwl = []
    for wi, window in enumerate(reference_windows):
        active_wl = aand(wl >= window[0], wl < window[1])
        refwl.extend(wl[active_wl])
    normind = s.array([s.argmin(abs(wl-w)) for w in refwl])
    refwl = s.array(refwl, dtype=float)

    # create basic model template
    model = {'normalize': normalize, 'wl': wl,
             'means': [], 'covs': [], 'refwl': refwl}

    for si, source_config in enumerate(config['sources']):

        # Determine source parameters
        for q in ['input_spectrum_files', 'windows', 'n_components', 'windows']:
            if q not in source_config:
                raise ValueError(
                    'Source %i is missing a parameter: %s' % (si, q))

        infiles = [expand_path(configdir, fi) for fi in
                   source_config['input_spectrum_files']]
        ncomp = int(source_config['n_components'])
        windows = source_config['windows']

        # load spectra
        spectra = []
        for infile in infiles:
            hdrfile = infile + '.hdr'
            rfl = envi.open(hdrfile, infile)
            nl, nb, ns = [int(rfl.metadata[n])
                          for n in ('lines', 'bands', 'samples')]
            swl = s.array([float(f) for f in rfl.metadata['wavelength']])
            rfl_mm = rfl.open_memmap(interleave='source', writable=True)
            if rfl.metadata['interleave'] == 'bip':
                x = s.array(rfl_mm[:, :, :])
            if rfl.metadata['interleave'] == 'bil':
                x = s.array(rfl_mm[:, :, :]).transpose((0, 2, 1))
            x = x.reshape(nl*ns, nb)
            swl = s.array([float(f) for f in rfl.metadata['wavelength']])

            # import spectra and resample
            for x1 in x:
                p = interp1d(swl, x1, kind='linear', bounds_error=False,
                             fill_value='extrapolate')
                spectra.append(p(wl))

            # calculate mixtures, if needed
            n = float(len(spectra))
            nmix = int(n * mixtures)
            for mi in range(nmix):
                s1, m1 = spectra[int(s.rand()*n)], s.rand()
                s2, m2 = spectra[int(s.rand()*n)], 1.0-m1
                spectra.append(m1 * s1 + m2 * s2)

        spectra = s.array(spectra)
        use = s.all(s.isfinite(spectra), axis=1)
        spectra = spectra[use, :]

        # accumulate total list of window indices
        window_idx = -s.ones((nchan), dtype=int)
        for wi, win in enumerate(windows):
            active_wl = aand(wl >= win['interval'][0], wl < win['interval'][1])
            window_idx[active_wl] = wi

        # Two step model.  First step is k-means initialization
        kmeans = KMeans(init='k-means++', n_clusters=ncomp, n_init=10)
        kmeans.fit(spectra)
        Z = kmeans.predict(spectra)

        for ci in range(ncomp):

            m = s.mean(spectra[Z == ci, :], axis=0)
            C = s.cov(spectra[Z == ci, :], rowvar=False)

            for i in range(nchan):
                window = windows[window_idx[i]]
                if window['correlation'] == 'EM':
                    C[i, i] = C[i, i] + float(window['regularizer'])
                elif window['correlation'] == 'decorrelated':
                    ci = C[i, i]
                    C[:, i] = 0
                    C[i, :] = 0
                    C[i, i] = ci + float(window['regularizer'])
                else:
                    raise ValueError(
                        'I do not recognize the source '+window['correlation'])

            # Normalize the component spectrum if desired
            if normalize == 'Euclidean':
                z = s.sqrt(s.sum(pow(m[normind], 2)))
            elif normalize == 'RMS':
                z = s.sqrt(s.mean(pow(m[normind], 2)))
            elif normalize == 'None':
                z = 1.0
            else:
                raise ValueError(
                    'Unrecognized normalization: %s\n' % normalize)
            m = m/z
            C = C/(z**2)

            model['means'].append(m)
            model['covs'].append(C)

    model['means'] = s.array(model['means'])
    model['covs'] = s.array(model['covs'])

    savemat(outfile, model)
    print('saving results to '+outfile)


if __name__ == "__main__":
    main()
