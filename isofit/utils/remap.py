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

import numpy as np
from spectral.io import envi


def remap(inputfile, labels, outputfile, flag, chunksize):
    """."""

    ref_file = inputfile
    lbl_file = labels
    out_file = outputfile
    nchunk = chunksize

    ref_img = envi.open(ref_file+'.hdr', ref_file)
    ref_meta = ref_img.metadata
    ref_mm = ref_img.open_memmap(interleave='source', writable=False)
    ref = np.array(ref_mm[:, :])

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
    for lstart in np.arange(0, nl, nchunk):
        print(lstart)
        del out_mm
        out_mm = out_img.open_memmap(interleave='source', writable=True)

        # Which labels will we extract? ignore zero index
        lend = min(lstart+nchunk, nl)

        lbl = labels[lstart:lend, :]
        out = flag * np.ones((lbl.shape[0], nb, lbl.shape[1]))
        for row in range(lbl.shape[0]):
            for col in range(lbl.shape[1]):
                out[row, :, col] = np.squeeze(ref[int(lbl[row, col]), :])

        out_mm[lstart:lend, :, :] = out
