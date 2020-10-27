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

import scipy
from spectral.io import envi
from skimage.segmentation import slic
import numpy as np
import ray
import atexit
import logging

@ray.remote
def segment_chunk(lstart, lend, in_file, nodata_value, npca, segsize, logfile=None, loglevel='INFO'):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel, filename=logfile)

    logging.info(f'{lstart}: starting')

    in_img = envi.open(in_file + '.hdr', in_file)
    meta = in_img.metadata
    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    img_mm = in_img.open_memmap(interleave='bip', writable=False)

    # Do quick single-band screen before reading all bands
    use = np.logical_not(np.isclose(np.array(img_mm[lstart:lend, :, 0]), nodata_value))
    if np.sum(use) == 0:
        logging.info(f'{lstart}: no non null data present, returning early')
        return lstart, lend, np.zeros((use.shape[0],ns))


    x = np.array(img_mm[lstart:lend, :, :]).astype(np.float32)
    nc = x.shape[0]
    x = x.reshape((nc * ns, nb))
    logging.info(f'{lstart}: read and reshaped data')

    # Excluding bad locations, calculate top PCA coefficients
    use = np.all(abs(x - nodata_value) > 1e-6, axis=1)

    # If this chunk is empty, return immediately
    if np.sum(use) == 0:
        logging.info(f'{lstart}: no non null data present, returning early')
        return lstart, lend, np.zeros((nc,ns))

    mu = x[use, :].mean(axis=0)
    C = np.cov(x[use, :], rowvar=False)
    [v, d] = scipy.linalg.eigh(C)

    # Determine segmentation compactness scaling based on eigenvalues
    cmpct = scipy.linalg.norm(np.sqrt(v[-npca:]))

    # Project, redimension as an image with "npca" channels, and segment
    x_pca_subset = (x[use,:] - mu) @ d[:, -npca:]
    del x, mu, d
    x_pca = np.zeros((nc,ns,npca))
    x_pca[use.reshape(nc,ns),:] = x_pca_subset
    del x_pca_subset
    
    x_pca = x_pca.reshape([nc, ns, npca])
    seg_in_chunk = int(sum(use) / float(segsize))

    logging.info(f'{lstart}: starting slic')
    labels = slic(x_pca, n_segments=seg_in_chunk, compactness=cmpct,
                  max_iter=10, sigma=0, multichannel=True,
                  enforce_connectivity=True, min_size_factor=0.5,
                  max_size_factor=3, mask=use.reshape(nc,ns))

    # Reindex the subscene labels and place them into the larger scene
    labels = labels.reshape([nc * ns])
    labels[np.logical_not(use)] = 0
    labels = labels.reshape([nc, ns])

    logging.info(f'{lstart}: completing')
    return lstart, lend, labels


def segment(spectra, nodata_value, npca, segsize, nchunk, n_cores=1, ray_address=None, ray_redis_password=None,
            ray_temp_dir=None, ray_ip_head=None, logfile=None, loglevel='INFO'):
    """."""

    logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel, filename=logfile)

    in_file = spectra[0]
    if len(spectra) > 1:
        lbl_file = spectra[1]
    else:
        lbl_file = spectra + '_lbl'

    # Open input data, get dimensions
    in_img = envi.open(in_file+'.hdr', in_file)
    meta = in_img.metadata
    nl, nb, ns = [int(meta[n]) for n in ('lines', 'bands', 'samples')]

    # Start up a ray instance for parallel work
    rayargs = {'ignore_reinit_error': True,
               'local_mode': n_cores == 1,
               "address": ray_address,
               "redis_password": ray_redis_password}

    if rayargs['local_mode']:
        rayargs['temp_dir'] = ray_temp_dir
        # Used to run on a VPN
        ray.services.get_node_ip_address = lambda: '127.0.0.1'

    # We can only set the num_cpus if running on a single-node
    if ray_ip_head is None and ray_redis_password is None:
        rayargs['num_cpus'] = n_cores

    ray.init(**rayargs)
    atexit.register(ray.shutdown)


    # Iterate through image "chunks," segmenting as we go
    next_label = 1
    all_labels = np.zeros((nl, ns))
    jobs = []
    for lstart in np.arange(0, nl, nchunk):

        # Extract data
        lend = min(lstart+nchunk, nl)

        jobs.append(segment_chunk.remote(lstart, lend, in_file, nodata_value, npca, segsize, logfile=logfile, loglevel=loglevel))

    # Collect results, making sure each chunk is distinct (note, label indexing is arbitrary,
    # so not attempt at sorting is made)
    rreturn = [ray.get(jid) for jid in jobs]
    for lstart, lend, ret in rreturn:
        if ret is not None:
            chunk_label = ret.copy()
            chunk_label[chunk_label != 0] += np.max(all_labels.astyp(np.float32)) + 1
            all_labels[lstart:lend,...] = chunk_label

    ray.shutdown()
    all_labels = all_labels.astype(int)

    # Reindex
    labels_sorted = np.sort(np.unique(all_labels))
    lbl = np.zeros((nl, ns))
    for i, val in enumerate(labels_sorted):
        lbl[all_labels == val] = i

    # Final file I/O
    lbl_meta = {"samples": str(ns), "lines": str(nl), "bands": "1",
                "header offset": "0", "file type": "ENVI Standard",
                "data type": "4", "interleave": "bil"}
    lbl_img = envi.create_image(lbl_file+'.hdr', lbl_meta, ext='', force=True)
    lbl_mm = lbl_img.open_memmap(interleave='source', writable=True)
    lbl_mm[:, :] = np.array(lbl, dtype=np.float32).reshape((nl, 1, ns))
    del lbl_mm
