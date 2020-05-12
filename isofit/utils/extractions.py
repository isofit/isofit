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
from spectral.io import envi


def extractions(inputfile, labels, output, chunksize, flag):
    """..."""

    in_file = inputfile
    lbl_file = labels
    out_file = output
    nchunk = chunksize

    dtm = {
        '4': s.float32,
        '5': s.float64
    }

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
