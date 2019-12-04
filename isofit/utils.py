#! /usr/bin/env python3
#
#  Copyright 2019 California Institute of Technology
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
from os.path import realpath, split, abspath, expandvars
import sys
import time
import logging
from numpy.random import multivariate_normal
import scipy as s
from scipy import logical_and as aand
from scipy.linalg import eigh, norm, inv
from scipy.spatial import KDTree
from scipy.stats import linregress
from scipy.linalg import svd
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.cluster import KMeans
from spectral.io import envi
from skimage.segmentation import slic
import matplotlib
import pylab as plt

from .common import expand_all_paths, load_spectrum, \
    find_header, expand_path, json_load_ascii
from .instrument import Instrument
from .geometry import Geometry

# EMPLINE

def empline(reference_radiance, reference_reflectance, reference_uncertainty,
            reference_locations, hashfile,
            input_radiance, input_locations, output_reflectance, output_uncertainty,
            nneighbors=15, flag=-9999.0, skip=0, level='INFO', 
            radiance_factors=None):
    """..."""

    def plot_example(xv, yv, b, predx, predy):
        """Plot for debugging purposes."""
        matplotlib.rcParams['font.family'] = "serif"
        matplotlib.rcParams['font.sans-serif'] = "Times"
        matplotlib.rcParams["legend.edgecolor"] = "None"
        matplotlib.rcParams["axes.spines.top"] = False
        matplotlib.rcParams["axes.spines.bottom"] = True
        matplotlib.rcParams["axes.spines.left"] = True
        matplotlib.rcParams["axes.spines.right"] = False
        matplotlib.rcParams['axes.grid'] = True
        matplotlib.rcParams['axes.grid.axis'] = 'both'
        matplotlib.rcParams['axes.grid.which'] = 'major'
        matplotlib.rcParams['legend.edgecolor'] = '1.0'
        plt.plot(xv[:, 113], yv[:, 113], 'ko')
        plt.plot(xv[:, 113], xv[:, 113]*b[113, 1] + b[113, 0], 'k')
        plt.plot(predx[113], predx[113]*b[113, 1] + b[113, 0], 'ro')
        plt.plot(predx[113], predy[113], 'bx')
        plt.grid(True)
        plt.xlabel('Radiance, $\mu{W }nm^{-1} sr^{-1} cm^{-2}$')
        plt.ylabel('Reflectance')
        plt.show(block=True)
        plt.savefig('empirical_line.pdf')

    eps = 1e-6
    k = nneighbors
    loglevel = level

    # Open input data, get dimensions
    logging.basicConfig(format='%(message)s', level=loglevel)
    ref_rdn_file = reference_radiance
    ref_rfl_file = reference_reflectance
    ref_unc_file = reference_uncertainty
    ref_loc_file = reference_locations
    inp_hash_file = hashfile
    inp_rdn_file = input_radiance
    inp_loc_file = input_locations
    out_rfl_file = output_reflectance
    out_unc_file = output_uncertainty

    # Load reference set radiance
    ref_rdn_img = envi.open(ref_rdn_file+'.hdr', ref_rdn_file)
    meta = ref_rdn_img.metadata
    nref, nb, sref = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if sref != 1:
        raise IndexError("Reference data should be a single-column list")
    ref_rdn_mm = ref_rdn_img.open_memmap(interleave='source', writable=False)
    ref_rdn = s.array(ref_rdn_mm[:, :, :]).reshape((nref, nb))

    # Load reference set reflectance
    ref_rfl_img = envi.open(ref_rfl_file+'.hdr', ref_rfl_file)
    meta = ref_rfl_img.metadata
    nrefr, nbr, srefr = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrefr != nref or nbr != nb or srefr != sref:
        raise IndexError("Reference file dimension mismatch (reflectance)")
    ref_rfl_mm = ref_rfl_img.open_memmap(interleave='source', writable=False)
    ref_rfl = s.array(ref_rfl_mm[:, :, :]).reshape((nref, nb))

    # Load reference set uncertainty, assuming reflectance uncertainty is
    # recoreded in the first nbr channels of data
    ref_unc_img = envi.open(ref_unc_file+'.hdr', ref_unc_file)
    meta = ref_unc_img.metadata
    nrefu, ns, srefu = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrefu != nref or ns < nb or srefu != sref:
        raise IndexError("Reference file dimension mismatch (uncertainty)")
    ref_unc_mm = ref_unc_img.open_memmap(interleave='source', writable=False)
    ref_unc = s.array(ref_unc_mm[:, :, :]).reshape((nref, ns))
    ref_unc = ref_unc[:, :nbr].reshape((nref, nbr))

    # Load reference set locations
    ref_loc_img = envi.open(ref_loc_file+'.hdr', ref_loc_file)
    meta = ref_loc_img.metadata
    nrefl, lb, ls = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrefl != nref or lb != 3:
        raise IndexError("Reference file dimension mismatch (locations)")
    ref_loc_mm = ref_loc_img.open_memmap(interleave='source', writable=False)
    ref_loc = s.array(ref_loc_mm[:, :, :]).reshape((nref, lb))

    # Prepare radiance adjustment
    if radiance_factors is None:
      rdn_factors = s.ones(nb,)
    else:
      rdn_factors = s.loadtxt(radiance_factors)

    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the
    # Third element, and the first two are latitude/longitude coordinates
    loc_scaling = s.array([1e5, 1e5, 0.1])
    scaled_ref_loc = ref_loc * loc_scaling
    tree = KDTree(scaled_ref_loc)

    inp_rdn_img = envi.open(inp_rdn_file+'.hdr', inp_rdn_file)
    inp_rdn_meta = inp_rdn_img.metadata
    nl, nb, ns = [int(inp_rdn_meta[n])
                  for n in ('lines', 'bands', 'samples')]
    if nb != nbr:
        msg = 'Number of channels mismatch: input (%i) vs. reference (%i)'
        raise IndexError(msg % (nbr, nb))
    inp_rdn_mm = inp_rdn_img.open_memmap(interleave='source', writable=False)

    inp_loc_img = envi.open(inp_loc_file+'.hdr', inp_loc_file)
    inp_loc_meta = inp_loc_img.metadata
    nll, nlb, nls = [int(inp_loc_meta[n])
                     for n in ('lines', 'bands', 'samples')]
    if nll != nl or nlb != 3 or nls != ns:
        raise IndexError('Input location dimension mismatch')
    inp_loc_mm = inp_loc_img.open_memmap(interleave='source', writable=False)
    inp_loc = s.array(inp_loc_mm).reshape((nl, nlb, ns))

    if inp_hash_file:
        inp_hash_img = envi.open(inp_hash_file+'.hdr', inp_hash_file)
        hash_img = inp_hash_img.read_band(0)
    else:
        hash_img = None

    out_rfl_img = envi.create_image(out_rfl_file+'.hdr', ext='',
                                    metadata=inp_rdn_img.metadata, force=True)
    out_rfl_mm = out_rfl_img.open_memmap(interleave='source', writable=True)

    out_unc_img = envi.create_image(out_unc_file+'.hdr', ext='',
                                    metadata=inp_rdn_img.metadata, force=True)
    out_unc_mm = out_unc_img.open_memmap(interleave='source', writable=True)

    # Iterate through image
    hash_table = {}

    for row in s.arange(nl):
        del inp_loc_mm
        del inp_rdn_mm
        del out_rfl_mm
        del out_unc_mm

        # Extract data
        inp_rdn_mm = inp_rdn_img.open_memmap(
            interleave='source', writable=False)
        inp_rdn = s.array(inp_rdn_mm[row, :, :])
        if inp_rdn_meta['interleave'] == 'bil':
            inp_rdn = inp_rdn.transpose((1, 0))
        inp_rdn = inp_rdn * rdn_factors

        inp_loc_mm = inp_loc_img.open_memmap(
            interleave='source', writable=False)
        inp_loc = s.array(inp_loc_mm[row, :, :])
        if inp_loc_meta['interleave'] == 'bil':
            inp_loc = inp_loc.transpose((1, 0))

        out_rfl = s.zeros(inp_rdn.shape)
        out_unc = s.zeros(inp_rdn.shape)

        nspectra, start = 0, time.time()
        for col in s.arange(ns):

            x = inp_rdn[col, :]
            if s.all(abs(x-flag) < eps):
                out_rfl[col, :] = flag
                out_unc[col, :] = flag
                continue

            bhat = None
            if hash_img is not None:
                hash_idx = hash_img[row, col]
                if hash_idx in hash_table:
                    bhat, bmarg, bcov = hash_table[hash_idx]
                else:
                    loc = ref_loc[s.array(
                        hash_idx, dtype=int), :] * loc_scaling
            else:
                loc = inp_loc[col, :] * loc_scaling

            if bhat is None:
                dists, nn = tree.query(loc, k)
                xv = ref_rdn[nn, :]
                yv = ref_rfl[nn, :]
                uv = ref_unc[nn, :]
                bhat = s.zeros((nb, 2))
                bmarg = s.zeros((nb, 2))
                bcov = s.zeros((nb, 2, 2))
                
                for i in s.arange(nb):
                    use = yv[:, i] > 0
                    n = sum(use)
                    X = s.concatenate((s.ones((n, 1)), xv[use, i:i+1]), axis=1)
                    W = s.diag(s.ones(n))#/uv[use, i])
                    y = yv[use, i:i+1]
                    bhat[i, :] =  (inv(X.T @ W @ X) @ X.T @ W @ y).T
                    bcov[i, :, :] = inv(X.T @ W @ X)
                    bmarg[i, :] = s.diag(bcov[i, :, :])

            if (hash_img is not None) and not (hash_idx in hash_table):
                hash_table[hash_idx] = bhat, bmarg, bcov

            A = s.array((s.ones(nb), x))
            out_rfl[col, :] = (s.multiply(bhat.T, A).sum(axis=0))
            out_unc[col, :] = s.sqrt(s.multiply(bmarg.T, A).sum(axis=0))
            if loglevel == 'DEBUG':
                plot_example(xv, yv, bhat, x, out_rfl[col, :])

            nspectra = nspectra+1

        elapsed = float(time.time()-start)
        logging.info('%5.1f spectra per second' % (float(nspectra)/elapsed))

        out_rfl_mm = out_rfl_img.open_memmap(interleave='source',
                                             writable=True)
        if inp_rdn_meta['interleave'] == 'bil':
            out_rfl = out_rfl.transpose((1, 0))
        out_rfl_mm[row, :, :] = out_rfl

        out_unc_mm = out_unc_img.open_memmap(interleave='source',
                                             writable=True)
        if inp_rdn_meta['interleave'] == 'bil':
            out_unc = out_unc.transpose((1, 0))
        out_unc_mm[row, :, :] = out_unc


# EXTRACT

def extract(inputfile, labels, output, chunksize, flag):
    """..."""
    in_file = inputfile
    lbl_file = labels
    out_file = output
    nchunk = chunksize

    dtm = {'4': s.float32, '5': s.float64}

    # Open input data, get dimensions
    in_img = envi.open(in_file+'.hdr', in_file)
    meta = in_img.metadata

    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    img_mm = in_img.open_memmap(interleave='source', writable=False)

    lbl_img = envi.open(lbl_file+'.hdr', lbl_file)
    labels = lbl_img.read_band(0)
    nout = len(s.unique(labels))

    # reindex from zero to n
    #lbl     = s.sort(s.unique(labels.flat))
    #idx     = s.arange(len(lbl))
    #nout    = len(lbl)
    # for i, L in enumerate(lbl):
    #    labels[labels==L] = i

    # Iterate through image "chunks," segmenting as we go
    next_label = 1
    extracted = s.zeros(nout) > 1
    out = s.zeros((nout, nb))
    counts = s.zeros((nout))

    for lstart in s.arange(0, nl, nchunk):

        del img_mm
        img_mm = in_img.open_memmap(interleave='source', writable=False)

        # Which labels will we extract? ignore zero index
        lend = min(lstart+nchunk, nl)
        active = s.unique(labels[lstart:lend, :])
        active = active[active >= 1]

        # Handle labels extending outside our chunk by expanding margins
        active_area = s.zeros(labels.shape)
        lstart_adjust, lend_adjust = lstart, lend
        for i in active:
            active_area[labels == i] = True
        active_locs = s.where(active_area)
        lstart_adjust = min(active_locs[0])
        lend_adjust = max(active_locs[0])+1

        chunk_inp = s.array(img_mm[lstart_adjust:lend_adjust, :, :])
        if meta['interleave'] == 'bil':
            chunk_inp = chunk_inp.transpose((0, 2, 1))
        chunk_lbl = s.array(labels[lstart_adjust:lend_adjust, :])

        for i in active:
            idx = int(i)
            out[idx, :] = 0
            locs = s.where(chunk_lbl == i)
            for row, col in zip(locs[0], locs[1]):
                out[idx, :] = out[idx, :] + s.squeeze(chunk_inp[row, col, :])
            counts[idx] = len(locs[0])

    out = s.array((out.T / counts[s.newaxis, :]).T, dtype=s.float32)
    out[s.logical_not(s.isfinite(out))] = flag
    meta["lines"] = str(nout)
    meta["bands"] = str(nb)
    meta["samples"] = '1'
    meta["interleave"] = "bil"
    out_img = envi.create_image(out_file+'.hdr',  metadata=meta,
                                ext='', force=True)
    out_mm = s.memmap(out_file, dtype=dtm[meta['data type']], mode='w+',
                      shape=(nout, 1, nb))
    if dtm[meta['data type']] == s.float32:
        out_mm[:, 0, :] = s.array(out, s.float32)
    else:
        out_mm[:, 0, :] = s.array(out, s.float64)


# GENNOISE

def gennoise(config):
    """Add noise to a radiance spectrum or image."""
    config = json_load_ascii(config, shell_replace=True)
    configdir, configfile = split(abspath(config))

    infile = expand_path(configdir, config['input_radiance_file'])
    outfile = expand_path(configdir, config['output_radiance_file'])
    instrument = Instrument(config['instrument_model'])
    geom = Geometry()

    if infile.endswith('txt'):
        rdn, wl = load_spectrum(infile)
        Sy = instrument.Sy(rdn, geom)
        rdn_noise = rdn + multivariate_normal(s.zeros(rdn.shape), Sy)
        with open(outfile, 'w') as fout:
            for w, r in zip(wl, rdn_noise):
                fout.write('%8.5f %8.5f' % (w, r))
    else:
        raise ValueError("Image cubes not yet implemented.")


# INSTMODEL

def percentile(X, p):
    """..."""
    S = sorted(X)
    return S[int(s.floor(len(S)*(p/100.0)))]


def find_header(imgfile):
    """Return the header associated with an image file."""
    if os.path.exists(imgfile+'.hdr'):
        return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    raise IOError("No header found for file {0}".format(imgfile))


def high_frequency_vert(X, sigma=4.0):
    """..."""
    nl, nb, nr = X.shape
    Xvert = X.copy()
    for r in range(nr):
        for b in range(nb):
            filt = gaussian_filter1d(Xvert[:, b, r], sigma, mode='nearest')
            Xvert[:, b, r] = X[:, b, r] - filt
    return Xvert


def low_frequency_horiz(X, sigma=4.0):
    """..."""
    nl, nb, nr = X.shape
    Xhoriz = X.copy()
    for l in range(nl):
        for b in range(nb):
            Xhoriz[l, b, :] = gaussian_filter1d(
                Xhoriz[l, b, :], sigma, mode='nearest')
    return Xhoriz


def flat_field(X, uniformity_thresh):
    """..."""
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
    """."""
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


def instmodel(config):
    """."""
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
    config = json_load_ascii(config, shell_replace=True)
    configdir, configfile = split(abspath(config))

    infile = expand_path(configdir, config['input_radiance_file'])
    outfile = expand_path(configdir, config['output_model_file'])
    flatfile = expand_path(configdir, config['output_flatfield_file'])
    uniformity_thresh = float(config['uniformity_threshold'])

    infile_hdr = infile + '.hdr'
    img = envi.open(infile_hdr, infile)
    inmm = img.open_memmap(interleave='source', writable=False)
    if img.interleave != 1:
        raise ValueError("I need BIL interleave.")
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
    s.io.savemat(outfile, mdict)


# REMAP

def remap(inputfile, labels, outputfile, flag, chunksize):
    """."""
    ref_file = inputfile
    lbl_file = labels
    out_file = outputfile
    nchunk = chunksize

    ref_img = envi.open(ref_file+'.hdr', ref_file)
    ref_meta = ref_img.metadata
    ref_mm = ref_img.open_memmap(interleave='source', writable=False)
    ref = s.array(ref_mm[:, :])

    lbl_img = envi.open(lbl_file+'.hdr', lbl_file)
    lbl_meta = lbl_img.metadata
    labels = lbl_img.read_band(0)
    nl = int(lbl_meta['lines'])
    ns = int(lbl_meta['samples'])
    nb = int(ref_meta['bands'])

    out_meta = dict([(k, v) for k, v in ref_meta.items()])
    out_meta["samples"] = ns
    out_meta["bands"] = nb
    out_meta["lines"] = nl
    out_meta['data type'] = ref_meta['data type']
    out_meta["interleave"] = "bil"
    out_img = envi.create_image(out_file+'.hdr',  metadata=out_meta,
                                ext='', force=True)
    out_mm = out_img.open_memmap(interleave='source', writable=True)

    # Iterate through image "chunks," restoring as we go
    for lstart in s.arange(0, nl, nchunk):
        print(lstart)
        del out_mm
        out_mm = out_img.open_memmap(interleave='source', writable=True)

        # Which labels will we extract? ignore zero index
        lend = min(lstart+nchunk, nl)

        lbl = labels[lstart:lend, :]
        out = flag * s.ones((lbl.shape[0], nb, lbl.shape[1]))
        for row in range(lbl.shape[0]):
            for col in range(lbl.shape[1]):
                out[row, :, col] = s.squeeze(ref[int(lbl[row, col]), :])

        out_mm[lstart:lend, :, :] = out


# SEGMENT

def segment(spectra, flag, npca, segsize, nchunk):
    """."""
    in_file = spectra[0]
    if len(spectra) > 1:
        lbl_file = spectra[1]
    else:
        lbl_file = spectra + '_lbl'

    # Open input data, get dimensions
    in_img = envi.open(in_file+'.hdr', in_file)
    meta = in_img.metadata
    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    img_mm = in_img.open_memmap(interleave='source', writable=False)
    if meta['interleave'] != 'bil':
        raise ValueError('I need BIL interleave.')

    # Iterate through image "chunks," segmenting as we go
    next_label = 1
    all_labels = s.zeros((nl, ns))
    for lstart in s.arange(0, nl, nchunk):

        del img_mm
        print(lstart)

        # Extract data
        lend = min(lstart+nchunk, nl)
        img_mm = in_img.open_memmap(interleave='source', writable=False)
        x = s.array(img_mm[lstart:lend, :, :]).transpose((0, 2, 1))
        nc = x.shape[0]
        x = x.reshape((nc * ns, nb))

        # Excluding bad locations, calculate top PCA coefficients
        use = s.all(abs(x-flag) > 1e-6, axis=1)
        mu = x[use, :].mean(axis=0)
        C = s.cov(x[use, :], rowvar=False)
        [v, d] = eigh(C)

        # Determine segmentation compactness scaling based on eigenvalues
        cmpct = norm(s.sqrt(v[-npca:]))

        # Project, redimension as an image with "npca" channels, and segment
        x_pca = (x-mu) @ d[:, -npca:]
        x_pca[use < 1, :] = 0.0
        x_pca = x_pca.reshape([nc, ns, npca])
        valid = use.reshape([nc, ns, 1])
        seg_in_chunk = int(sum(use) / float(segsize))
        labels = slic(x_pca, n_segments=seg_in_chunk, compactness=cmpct,
                      max_iter=10, sigma=0, multichannel=True,
                      enforce_connectivity=True, min_size_factor=0.5,
                      max_size_factor=3)

        # Reindex the subscene labels and place them into the larger scene
        labels = labels.reshape([nc * ns])
        labels[s.logical_not(use)] = 0
        labels[use] = labels[use] + next_label
        next_label = max(labels) + 1
        labels = labels.reshape([nc, ns])
        all_labels[lstart:lend, :] = labels

    # Reindex
    labels_sorted = s.sort(s.unique(all_labels))
    lbl = s.zeros((nl, ns))
    for i, val in enumerate(labels_sorted):
        lbl[all_labels == val] = i

    # Final file I/O
    del img_mm
    lbl_meta = {"samples": str(ns), "lines": str(nl), "bands": "1",
                "header offset": "0", "file type": "ENVI Standard",
                "data type": "4", "interleave": "bil"}
    lbl_img = envi.create_image(lbl_file+'.hdr', lbl_meta, ext='', force=True)
    lbl_mm = lbl_img.open_memmap(interleave='source', writable=True)
    lbl_mm[:, :] = s.array(lbl, dtype=s.float32).reshape((nl, 1, ns))
    del lbl_mm


# SURFMODEL

def surfmodel(config):
    """."""
    configdir, configfile = split(abspath(config))
    config = json_load_ascii(config, shell_replace=True)

    # Determine top level parameters
    for q in ['output_model_file', 'sources', 'normalize', 'wavelength_file']:
        if q not in config:
            raise ValueError("Missing parameter: %s" % q)

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
    model = {
        'normalize': normalize,
        'wl': wl,
        'means': [],
        'covs': [],
        'refwl': refwl
    }

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

            # Maybe convert to nanometers
            if swl[0] < 100:
                swl = swl * 1000.0

            rfl_mm = rfl.open_memmap(interleave='source', writable=True)
            if rfl.metadata['interleave'] == 'bip':
                x = s.array(rfl_mm[:, :, :])
            if rfl.metadata['interleave'] == 'bil':
                x = s.array(rfl_mm[:, :, :]).transpose((0, 2, 1))
            x = x.reshape(nl*ns, nb)

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

    s.io.savemat(outfile, model)
    print("saving results to", outfile)
