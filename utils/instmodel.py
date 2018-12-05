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
import sys
import spectral
import argparse
import scipy as s
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter1d
from scipy.linalg import svd
from os.path import abspath, split

from common import find_header, expand_path, json_load_ascii

hdr_template = '''ENVI
samples = {samples}
lines   = {lines}
bands   = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 0
'''


def percentile(X, p):
    S = sorted(X)
    return S[int(s.floor(len(S)*(p/100.0)))]


# Return the header associated with an image file
def find_header(imgfile):
    if os.path.exists(imgfile+'.hdr'):
        return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    raise IOError('No header found for file {0}'.format(imgfile))


def high_frequency_vert(X, sigma=4.0):
    nl, nb, nr = X.shape
    Xvert = X.copy()
    for r in range(nr):
        for b in range(nb):
            filt = gaussian_filter1d(Xvert[:, b, r], sigma, mode='nearest')
            Xvert[:, b, r] = X[:, b, r] - filt
    return Xvert


def low_frequency_horiz(X, sigma=4.0):
    nl, nb, nr = X.shape
    Xhoriz = X.copy()
    for l in range(nl):
        for b in range(nb):
            Xhoriz[l, b, :] = gaussian_filter1d(
                Xhoriz[l, b, :], sigma, mode='nearest')
    return Xhoriz


def flat_field(X, uniformity_thresh):
    Xhoriz = low_frequency_horiz(X, sigma=4.0)
    Xhorizp = low_frequency_horiz(X, sigma=3.0)
    nl, nb, nc = X.shape
    FF = s.zeros((nb, nc))
    use_ff = s.ones((X.shape[0], X.shape[2])) > 0
    for b in range(nb):
        xsub = Xhoriz[:, b, :]
        xsubp = Xhorizp[:, b, :]
        mu = xsub.mean(axis=0)
        dists = abs(xsub - mu)
        distsp = abs(xsubp - mu)
        thresh = percentile(dists.flatten(), 90.0)
        uthresh = dists * uniformity_thresh
        #use       = s.logical_and(dists<thresh, abs(dists-distsp) < uthresh)
        use = dists < thresh
        FF[b, :] = ((xsub*use).sum(axis=0)/use.sum(axis=0)) / \
            ((X[:, b, :]*use).sum(axis=0)/use.sum(axis=0))
        use_ff = s.logical_and(use_ff, use)
    return FF, Xhoriz, Xhorizp, s.array(use_ff)


def column_covariances(X, uniformity_thresh):
    Xvert = high_frequency_vert(X, sigma=4.0)
    Xvertp = high_frequency_vert(X, sigma=3.0)
    models = []
    use_C = []
    for i in range(X.shape[2]):
        xsub = Xvert[:, :, i]
        xsubp = Xvertp[:, :, i]
        mu = xsub.mean(axis=0)
        dists = s.sqrt(pow((xsub - mu), 2).sum(axis=1))
        distsp = s.sqrt(pow((xsubp - mu), 2).sum(axis=1))
        thresh = percentile(dists, 95.0)
        uthresh = dists * uniformity_thresh
        #use       = s.logical_and(dists<thresh, abs(dists-distsp) < uthresh)
        use = dists < thresh
        C = s.cov(xsub[use, :], rowvar=False)
        [U, V, D] = svd(C)
        V[V < 1e-8] = 1e-8
        C = U.dot(s.diagflat(V)).dot(D)
        models.append(C)
        use_C.append(use)
    return s.array(models), Xvert, Xvertp, s.array(use_C).T


# parse the command line (perform the correction on all command line arguments)
def main():

    parser = argparse.ArgumentParser(description="Create an instrument model")
    parser.add_argument('config', type=str, metavar='INPUT')
    args = parser.parse_args(sys.argv[1:])
    config = json_load_ascii(args.config, shell_replace=True)
    configdir, configfile = split(abspath(args.config))

    infile = expand_path(configdir, config['input_radiance_file'])
    outfile = expand_path(configdir, config['output_model_file'])
    flatfile = expand_path(configdir, config['output_flatfield_file'])
    uniformity_thresh = float(config['uniformity_threshold'])

    infile_hdr = infile + '.hdr'
    img = spectral.io.envi.open(infile_hdr, infile)
    inmm = img.open_memmap(interleave='source', writable=False)
    if img.interleave != 1:
        raise ValueError('I need BIL interleave')
    X = s.array(inmm[:, :, :], dtype=s.float32)
    nr, nb, nc = X.shape

    FF, Xhoriz, Xhorizp, use_ff = flat_field(X, uniformity_thresh)
    s.array(FF, dtype=s.float32).tofile(flatfile)
    with open(flatfile+'.hdr', 'w') as fout:
        fout.write(hdr_template.format(lines=nb, samples=nc))

    C, Xvert, Xvertp, use_C = column_covariances(X, uniformity_thresh)
    cshape = (C.shape[0], C.shape[1]**2)
    out = s.array(C, dtype=s.float32).reshape(cshape)
    mdict = {'columns': out.shape[0], 'bands': out.shape[1],
             'covariances': out, 'Xvert': Xvert, 'Xhoriz': Xhoriz,
             'Xvertp': Xvertp, 'Xhorizp': Xhorizp, 'use_ff': use_ff,
             'use_C': use_C}
    savemat(outfile, mdict)


if __name__ == "__main__":
    main()
