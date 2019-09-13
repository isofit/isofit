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

import argparse
from scipy import logical_and as aand
from os.path import realpath, split, abspath, expandvars
import scipy as s
from spectral.io import envi
from skimage.segmentation import slic
from scipy.linalg import eigh


# parse the command line
def main():

    parser = argparse.ArgumentParser(description="Representative subset")
    parser.add_argument('spectra', type=str)
    parser.add_argument('--flag', type=float, default=-9999)
    parser.add_argument('--npca', type=int, default=5)
    args = parser.parse_args()
    in_file  = args.spectra
    sub_file = args.spectra + '_sub'
    lbl_file = args.spectra + '_lbl'
    npca     = args.npca
    
    in_img = envi.open(infile+'.hdr', infile)
    meta   = in_img.metadata
    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    all_labels = s.zeros((nl,ns))
    
    img_mm = in_img.open_memmap(interleave='source', writable=True)
    if meta['interleave'] != 'bil':
       raise ValueError('I need BIL interleave.')
    chunksize = 256
    next_label = 1
    for lstart in range(0, s.floor((nl-1)/chunksize)*chunksize+1, chunksize):
        lend = lstart+chunksize

        del img_mm
        img_mm = in_img.open_memmap(interleave='source', writable=True)
        x = s.array(img_mm[lstart:lend, :, :]).transpose((0, 2, 1))
        x = x.reshape(chunksize*ns, nb)
        use = s.all(abs(x-flag)>1e-6,axis=0)
        mu = x[use,:].mean(axis=0)
        C = s.cov(x[use,:], rowvar=False)
        [v,d] = eigh(C)
        x_pca = (x-mu) @ d[:,npca]
        x_pca[use<1] = 0.0
        x_pca = x_pca.reshape([nl,ns,npca])
        valid = use.reshape([nl,ns,1])
        labels = slic(x_pca, n_segments=100, compactness=10., max_iter=10, sigma=0,
            spacing=None, multichannel=True, convert2lab=None,
            enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
            slic_zero=False)
        labels = labels.reshape([nl*ns])  
        labels[s.logical_not(use)]=0
        labels[use] = labels[use] + next_label
        next_label = max(labels) + 1
        labels = labels.reshape([nl,ns])  
        all_labels[lstart:lend,:] = labels

    # reindex
    labels_sorted = s.sort(all_labels)
    lbl = s.zeros((nl,ns))
    for i, val in enumerate(labels_sorted):
        lbl[all_labels==val] = i

    del img_mm
    lbl_meta = {"samples":str(ns), "lines":str(nl), "bands":"1",
                "header offset":"0","file type":"ENVI Standard",
                "data type":"4", "interleave":"bil"}
    lbl_img = envi.create_image(lbl_file+'.hdr', lbl_meta, ext='', force=True)
    lbl_mm = lbl_img.open_memmap(interleave='source', writable=True)
    lbl_mm[:,:] = s.array(all_labels, dtype=s.float32)
    del lbl_mm

if __name__ == "__main__":
    main()
