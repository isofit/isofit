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

import logging

import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.core.common import envi_header
from isofit.core.fileio import write_bil_chunk


@ray.remote(num_cpus=1)
def extract_chunk(
    lstart: int,
    lend: int,
    in_file: str,
    labels: np.array,
    flag: float,
    logfile=None,
    loglevel="INFO",
):
    """
    Extract a small chunk of the image

    Args:
        lstart: line to start extraction at
        lend: line to end extraction at
        in_file: file to read image from
        labels: labels to use for data read
        flag: nodata value of image
        logfile: logging file name
        loglevel: logging level

    Returns:
        out_index: array of output indices (based on labels)
        out_data: array of output data
    """

    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.info(f"{lstart}: starting")

    in_img = envi.open(envi_header(in_file))
    img_mm = in_img.open_memmap(interleave="bip", writable=False)

    # Which labels will we extract? ignore zero index
    active = labels[lstart:lend, :]
    active = active[active >= 1]
    active = np.unique(active)
    logging.debug(f"{lstart}: found {len(active)} unique labels")
    if len(active) == 0:
        return None, None

    # Handle labels extending outside our chunk by expanding margins
    cs = lend - lstart
    boundary_min = max(lstart - cs, 0)
    boundary_max = min(lend + cs, labels.shape[0])

    active_area = np.zeros((boundary_max - boundary_min, labels.shape[1]))
    for i in active:
        active_area[labels[boundary_min:boundary_max, :] == i] = True
    active_locs = np.where(active_area)

    lstart_adjust = min(active_locs[0]) + boundary_min
    lend_adjust = max(active_locs[0]) + boundary_min + 1

    cstart_adjust = min(active_locs[1])
    cend_adjust = max(active_locs[1]) + 1

    logging.debug(
        f"{lstart} area subset: {lstart_adjust}, {lend_adjust} :::: {cstart_adjust},"
        f" {cend_adjust}"
    )

    chunk_lbl = np.array(labels[lstart_adjust:lend_adjust, cstart_adjust:cend_adjust])
    chunk_inp = np.array(
        img_mm[lstart_adjust:lend_adjust, cstart_adjust:cend_adjust, :]
    )

    out_data = np.zeros((len(active), img_mm.shape[-1])) + flag

    logging.debug(f"{lstart}: running extraction from local array")
    for _lab, lab in enumerate(active):
        out_data[_lab, :] = 0
        locs = np.where(chunk_lbl == lab)
        for row, col in zip(locs[0], locs[1]):
            out_data[_lab, :] += np.squeeze(chunk_inp[row, col, :])
        out_data[_lab, :] /= float(len(locs[0]))

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 1]
    if unique_labels[0] != 0:
        unique_labels = np.hstack([np.zeros(1), unique_labels])

    match_idx = np.searchsorted(unique_labels, active)

    out_data[np.logical_not(np.isfinite(out_data))] = flag
    logging.debug(f"{lstart}: complete")

    return match_idx, out_data


def extractions(
    inputfile,
    labels,
    output,
    chunksize,
    flag,
    n_cores: int = 1,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir: str = None,
    ray_ip_head=None,
    logfile: str = None,
    loglevel: str = "INFO",
):
    """..."""

    in_file = inputfile
    lbl_file = labels
    out_file = output
    nchunk = chunksize

    dtm = {"4": np.float32, "5": np.float64}

    # Open input data, get dimensions
    in_img = envi.open(envi_header(in_file), in_file)
    meta = in_img.metadata

    nl, nb, ns = [int(meta[n]) for n in ("lines", "bands", "samples")]
    img_mm = in_img.open_memmap(interleave="bip", writable=False)

    lbl_img = envi.open(envi_header(lbl_file), lbl_file)
    labels = lbl_img.read_band(0)
    un_labels = np.unique(labels).tolist()
    if 0 not in un_labels:
        un_labels.insert(0, 0)
    nout = len(un_labels)

    # Start up a ray instance for parallel work
    rayargs = {
        "ignore_reinit_error": True,
        "local_mode": n_cores == 1,
        "address": ray_address,
        "include_dashboard": False,
        "_temp_dir": ray_temp_dir,
        "_redis_password": ray_redis_password,
    }

    # We can only set the num_cpus if running on a single-node
    if ray_ip_head is None and ray_redis_password is None:
        rayargs["num_cpus"] = n_cores

    ray.init(**rayargs)

    labelid = ray.put(labels)
    jobs = []
    for lstart in np.arange(0, nl, nchunk):
        lend = min(lstart + nchunk, nl)
        jobs.append(
            extract_chunk.remote(
                lstart, lend, in_file, labelid, flag, logfile=logfile, loglevel=loglevel
            )
        )

    # Collect results
    rreturn = [ray.get(jid) for jid in jobs]

    ## Iterate through image "chunks," segmenting as we go
    out = np.zeros((nout, nb, 1))
    for idx, ret in rreturn:
        if ret is not None:
            out[idx, :, 0] = ret
    del rreturn

    meta["lines"] = str(nout)
    meta["bands"] = str(nb)
    meta["samples"] = "1"
    meta["interleave"] = "bil"

    out_img = envi.create_image(
        envi_header(out_file), metadata=meta, ext="", force=True
    )
    del out_img
    if dtm[meta["data type"]] == np.float32:
        type = "float32"
    else:
        type = "float64"

    write_bil_chunk(out, out_file, 0, out.shape, dtype=type)
