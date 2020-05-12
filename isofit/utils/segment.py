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

import scipy as s
from scipy.linalg import eigh, norm
from spectral.io import envi
from skimage.segmentation import slic


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
