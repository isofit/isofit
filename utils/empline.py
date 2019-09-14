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
from scipy.linalg import eigh, norm, inv
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from numba import jit

@jit
def linregress(xv,yv,nb,k):
    b   = s.zeros((nb,2))
    for i in s.arange(nb):
        A = s.array((s.ones(k), xv[:,i])).T
        Y = yv[:,i:i+1]
        b[i,:] = (inv(A.T @ A) @ A.T @ Y).T
    return b

eps = 1e-6

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Local empirical line")
    parser.add_argument('reference_radiance',  type=str)
    parser.add_argument('reference_reflectance',  type=str)
    parser.add_argument('reference_locations',  type=str)
    parser.add_argument('input_radiance',  type=str)
    parser.add_argument('input_locations',  type=str)
    parser.add_argument('output_reflectance',  type=str)
    parser.add_argument('--flag',   type=float, default=-9999)
    parser.add_argument('--nneighbors',   type=int, default=10)
    args = parser.parse_args()
    flag     = args.flag
    k        = args.nneighbors
    
    # Open input data, get dimensions

    ref_rdn_file = args.reference_radiance 
    ref_rfl_file = args.reference_reflectance
    ref_loc_file = args.reference_locations
    inp_rdn_file = args.input_radiance
    inp_loc_file = args.input_locations
    out_rfl_file = args.output_reflectance

    ref_rdn_img    = envi.open(ref_rdn_file+'.hdr', ref_rdn_file)
    meta           = ref_rdn_img.metadata
    nref, nb, sref = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if sref != 1:
        raise IndexError('Reference data should be a single-column list')
    ref_rdn_mm  = ref_rdn_img.open_memmap(interleave='source', writable=False)
    ref_rdn     = s.array(ref_rdn_mm[:,:,:]).reshape((nref, nb))

    ref_rfl_img     = envi.open(ref_rfl_file+'.hdr', ref_rfl_file)
    meta            = ref_rfl_img.metadata
    nrrf, nbr, srrf = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrrf != nref or nbr != nb or srrf != sref:
        raise IndexError('Reference file dimension mismatch (reflectance)')
    ref_rfl_mm  = ref_rfl_img.open_memmap(interleave='source', writable=False)
    ref_rfl     = s.array(ref_rfl_mm[:,:,:]).reshape((nref, nb))

    ref_loc_img     = envi.open(ref_loc_file+'.hdr', ref_loc_file)
    meta            = ref_loc_img.metadata
    nrrf, lb, ls    = [int(meta[n]) for n in ('lines', 'bands', 'samples')]
    if nrrf != nref or lb != 3:
        raise IndexError('Reference file dimension mismatch (locations)')
    ref_loc_mm  = ref_loc_img.open_memmap(interleave='source', writable=False)
    ref_loc     = s.array(ref_loc_mm[:,:,:]).reshape((nref, lb))

    # Assume (heuristically) that, for distance purposes, 1 m vertically is
    # comparable to 10 m horizontally, and that there are 100 km per latitude
    # degree.  This is all approximate of course.  Elevation appears in the 
    # Third element, and the first two are latitude/longitude coordinates
    loc_scaling = s.array([1e5,1e5,0.1])
    scaled_ref_loc = ref_loc * loc_scaling
    tree = KDTree(scaled_ref_loc)

    inp_rdn_img    = envi.open(inp_rdn_file+'.hdr', inp_rdn_file)
    inp_rdn_meta   = inp_rdn_img.metadata
    nl, nb, ns     = [int(inp_rdn_meta[n]) \
                        for n in ('lines', 'bands', 'samples')]
    if nb != nbr:
        msg = 'Number of channels mismatch: input (%i) vs. reference (%i)'
        raise IndexError(msg % (nbr, nb))
    inp_rdn_mm  = inp_rdn_img.open_memmap(interleave='source', writable=False)

    inp_loc_img   = envi.open(inp_loc_file+'.hdr', inp_loc_file)
    inp_loc_meta  = inp_loc_img.metadata
    nll, nlb, nls = [int(inp_loc_meta[n]) \
                        for n in ('lines', 'bands', 'samples')]
    if nll != nl or nlb != 3 or nls != ns:
        raise IndexError('Input location dimension mismatch')
    inp_loc_mm  = inp_loc_img.open_memmap(interleave='source', writable=False)
    inp_loc     = s.array(inp_loc_mm).reshape((nl, nlb, ns))

    out_rfl_img = envi.create_image(out_rfl_file+'.hdr', ext='',
                        metadata=inp_rdn_img.metadata, force=True)
    out_rfl_mm  = out_rfl_img.open_memmap(interleave='source', writable=True)

    # Iterate through image 
    for row in s.arange(nl):

        del inp_loc_mm
        del inp_rdn_mm
        del out_rfl_mm
        print(row)

        # Extract data
        inp_rdn_mm = inp_rdn_img.open_memmap(interleave='source', 
                            writable=False)
        inp_rdn = s.array(inp_rdn_mm[row, :, :])
        if inp_rdn_meta['interleave'] == 'bil':
            inp_rdn = inp_rdn.transpose((1, 0))
    
        inp_loc_mm = inp_loc_img.open_memmap(interleave='source', 
                            writable=False)
        inp_loc = s.array(inp_loc_mm[row, :, :])
        if inp_loc_meta['interleave'] == 'bil':
            inp_loc = inp_loc.transpose((1, 0))

        out_rfl = s.zeros(inp_rdn.shape)

        for col in s.arange(ns):

            x   = inp_rdn[col,:]
            if s.all(abs(x-flag)<eps):
                out_rfl[col,:] = flag
                continue

            loc = inp_loc[col,:] * loc_scaling
            dists, nn  = tree.query(loc, k)
            xv  = ref_rdn[nn,:] 
            yv  = ref_rfl[nn,:]
            b = linregress(xv,yv,nb,k)
            A = s.array((s.ones(nb), x))
            out_rfl[col,:] = (b.T * A).sum(axis=0)

        out_rfl_mm = out_rfl_img.open_memmap(interleave='source', 
                writable=True)
        if inp_rdn_meta['interleave'] == 'bil':
            out_rfl = out_rfl.transpose((1, 0))
        out_rfl_mm[row, :, :]  = out_rfl


if __name__ == "__main__":
    main()
