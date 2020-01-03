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

import time
import logging
import scipy as s
from scipy.spatial import KDTree
from scipy.stats import linregress
from spectral.io import envi
import matplotlib
import pylab as plt


plt.switch_backend('Agg')


def empirical_line(reference_radiance, reference_reflectance, reference_locations, hashfile,
                   input_radiance, input_locations, output_reflectance, output_uncertainty,
                   nneighbors, flag, skip, level):
    """..."""

    def plot_example(xv, yv, b):
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
        #plt.plot(x[113], x[113]*b[113, 1] + b[113, 0], 'ro')
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
    nrrf, nbr, srrf = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrrf != nref or nbr != nb or srrf != sref:
        raise IndexError("Reference file dimension mismatch (reflectance)")
    ref_rfl_mm = ref_rfl_img.open_memmap(interleave='source', writable=False)
    ref_rfl = s.array(ref_rfl_mm[:, :, :]).reshape((nref, nb))

    # Load reference set locations
    ref_loc_img = envi.open(ref_loc_file+'.hdr', ref_loc_file)
    meta = ref_loc_img.metadata
    nrrf, lb, ls = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrrf != nref or lb != 3:
        raise IndexError("Reference file dimension mismatch (locations)")
    ref_loc_mm = ref_loc_img.open_memmap(interleave='source', writable=False)
    ref_loc = s.array(ref_loc_mm[:, :, :]).reshape((nref, lb))

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

            if hash_img is not None:
                hash_idx = hash_img[row, col]
                if hash_idx in hash_table:
                    b, unc = hash_table[hash_idx]
                else:
                    loc = ref_loc[s.array(
                        hash_idx, dtype=int), :] * loc_scaling
                    dists, nn = tree.query(loc, k)
                    xv = ref_rdn[nn, :]
                    yv = ref_rfl[nn, :]
                    b = s.zeros((nb, 2))
                    unc = s.zeros(nb,)

                    for i in s.arange(nb):
                        b[i, 1], b[i, 0], q1, q2, q3 = linregress(
                            xv[:, i], yv[:, i])
                        unc[i] = s.std(xv[:, i]*b[i, 1]+b[i, 0]-yv[:, i])

                    hash_table[hash_idx] = b, unc
            else:
                loc = inp_loc[col, :] * loc_scaling
                dists, nn = tree.query(loc, k)
                xv = ref_rdn[nn, :]
                yv = ref_rfl[nn, :]
                b = s.zeros((nb, 2))
                unc = s.zeros(nb,)
                for i in s.arange(nb):
                    b[i, 1], b[i, 0], q1, q2, q3 = linregress(
                        xv[:, i], yv[:, i])
                    unc[i] = s.std(xv[:, i]*b[i, 1]+b[i, 0]-yv[:, i])

            A = s.array((s.ones(nb), x))
            out_rfl[col, :] = (b.T * A).sum(axis=0)
            out_unc[col, :] = unc
            if loglevel == 'DEBUG':
                plot_example(xv, yv, b)

            nspectra = nspectra+1

        elapsed = float(time.time()-start)
        logging.info('%5.1f spectra per second' % (float(nspectra)/elapsed))

        out_rfl_mm = out_rfl_img.open_memmap(interleave='source',
                                             writable=True)
        if inp_rdn_meta['interleave'] == 'bil':
            out_rfl = out_rfl.transpose((1, 0))

        out_rfl_mm[row, :, :] = out_rfl

        out_unc_mm = out_rfl_img.open_memmap(interleave='source',
                                             writable=True)
        if inp_rdn_meta['interleave'] == 'bil':
            out_unc = out_rfl.transpose((1, 0))

        out_unc_mm[row, :, :] = out_rfl
